#!/usr/bin/env python3
"""Exercise hard-to-reach repair routes with real LLM calls and injected gates.

This runs the production compute graphs without persistence. Gate injection is
restricted to this explicit live test command; no fake result reaches SFT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_sft.agents.contracts import CriticResult  # noqa: E402
from cuda_sft.compile import CompileResult  # noqa: E402
from cuda_sft.config import get_settings  # noqa: E402
from cuda_sft.knowledge.graph import (  # noqa: E402
    build_knowledge_compute_graph,
    knowledge_recursion_limit,
)
from cuda_sft.llm import reset_llm_client  # noqa: E402
from cuda_sft.pipeline.common import set_print_stream  # noqa: E402
from cuda_sft.refval.spec import RefvalReport  # noqa: E402
from cuda_sft.runtime import deps, scheduler, trace  # noqa: E402

QUESTION = (
    "Implement a CUDA vector addition kernel C[i] = A[i] + B[i] with bounds "
    "checking and an extern C host launcher. Return a complete solution.cu."
)
KNOWLEDGE_QUESTION = (
    "Explain CUDA occupancy, how registers and shared memory limit active "
    "warps, and why high occupancy need not imply high utilization."
)


def _configure(out_dir: Path, *, semantic: bool = False, knowledge: bool = False) -> None:
    overrides = {
        "DATA_DIR": str(out_dir / "data"),
        "WORK_DIR": str(out_dir / "work"),
        "TRACE_ENABLED": "true",
        "MAX_CANDIDATES": "1",
        "MAX_REPAIRS": "1",
        "KNOWLEDGE_MAX_CANDIDATES": "1",
        "KNOWLEDGE_MAX_REPAIRS": "1",
        "DIFFICULTY_AWARE": "false",
        "KERNEL_FAST_MODE": "false",
        "ASYNC_LLM_ENABLED": "false",
        "REFVAL_ENABLED": "true",
        "REFVAL_STRICT": "false",
        "KERNEL_LLM_CRITIC": "always" if semantic else "off",
        "COT_ENABLED": "false",
        "JUDGE_ENABLED": "false",
        "KNOWLEDGE_JUDGE_ENABLED": "false" if knowledge else "true",
    }
    os.environ.update(overrides)
    get_settings.cache_clear()
    reset_llm_client()


def _kernel_probe(kind: str, qid: int, out_dir: Path) -> dict[str, object]:
    _configure(out_dir, semantic=kind == "semantic")
    compile_calls = 0
    refval_calls = 0
    critic_calls = 0

    def compile_fn(_dialect, _code, workdir, _settings):
        nonlocal compile_calls
        compile_calls += 1
        workdir.mkdir(parents=True, exist_ok=True)
        bad = kind == "compile" and compile_calls == 1
        return CompileResult(
            ok=not bad, command=[], used_rdc=False,
            output="error: expected a declaration" if bad else "",
        )

    def refval_fn(**_kwargs):
        nonlocal refval_calls
        refval_calls += 1
        bad = kind == "numeric" and refval_calls == 1
        return RefvalReport(
            status="fail" if bad else "pass", dialect="cuda",
            cases_run=3, manifest_hash="injected-probe",
            error_class="numeric_mismatch" if bad else "",
            evidence="output differs from CPU reference" if bad else "",
        )

    def critic_fn(_self, **_kwargs):
        nonlocal critic_calls
        critic_calls += 1
        bad = critic_calls == 1
        return CriticResult(
            passed=not bad, skipped=False,
            must_fix=["correct the host launch semantics"] if bad else [],
        )

    init = {
        "question_id": qid, "question": QUESTION, "kind": "kernel",
        "dialect": "cuda", "source": "live_probe", "status": "running",
    }
    with deps.use(deps.Deps(compile_fn=compile_fn, refval_fn=refval_fn)):
        if kind == "semantic":
            with patch("cuda_sft.agents.critic.KernelCritic.evaluate", critic_fn):
                result = scheduler.solve_job_for_tests(init)
        else:
            result = scheduler.solve_job_for_tests(init)
    return {
        "status": result.get("status"), "compile_calls": compile_calls,
        "refval_calls": refval_calls, "critic_calls": critic_calls,
        "repairs": max((int(item.get("repair") or 0) for item in result.get("attempts") or []), default=0),
    }


def _knowledge_probe(qid: int, out_dir: Path) -> dict[str, object]:
    _configure(out_dir, knowledge=True)
    from cuda_sft.knowledge import graph as knowledge_graph

    gate_calls = 0

    def gate_fn(*_args, **_kwargs):
        nonlocal gate_calls
        gate_calls += 1
        return ["probe: revise the explanation"] if gate_calls == 1 else []

    with patch.object(knowledge_graph, "hard_gate", gate_fn):
        result = build_knowledge_compute_graph().invoke(
            {
                "question_id": qid, "question": KNOWLEDGE_QUESTION,
                "kind": "knowledge", "topic": "execution",
                "track": "knowledge:execution", "source": "live_probe", "status": "running",
            },
            {"recursion_limit": knowledge_recursion_limit()},
        )
    return {"status": result.get("status"), "gate_calls": gate_calls}


def _read_calls(path: Path) -> tuple[dict[str, int], dict[str, list[str]], dict[str, list[str]]]:
    counts: dict[str, int] = {}
    routes: dict[str, set[str]] = {}
    by_job: dict[str, set[str]] = {}
    for file in path.glob("trace-*.jsonl"):
        for line in file.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "llm.call" and event.get("ok"):
                role = str(event.get("role") or "")
                counts[role] = counts.get(role, 0) + 1
                routes.setdefault(role, set()).add(
                    f"{event.get('provider')}/{event.get('model')}"
                )
                by_job.setdefault(str(event.get("job_key") or ""), set()).add(role)
    return (
        counts,
        {role: sorted(values) for role, values in routes.items()},
        {job: sorted(values) for job, values in by_job.items()},
    )


def _probe_failures(records: dict[str, object], by_job: dict[str, list[str]]) -> list[str]:
    expected = {
        "compile": ("q101:cuda", {"generator", "repair.compile"}, {"compile_calls": 2, "refval_calls": 1, "repairs": 1}),
        "numeric": ("q102:cuda", {"generator", "repair.numeric"}, {"compile_calls": 2, "refval_calls": 2, "repairs": 1}),
        "semantic": ("q103:cuda", {"generator", "repair.semantic"}, {"compile_calls": 2, "refval_calls": 2, "critic_calls": 2, "repairs": 1}),
        "knowledge": ("q104:knowledge:execution", {"knowledge_generator", "knowledge_repair"}, {"gate_calls": 2}),
    }
    failures: list[str] = []
    for name, (job_key, required, counts) in expected.items():
        record = records.get(name)
        if not isinstance(record, dict) or record.get("status") != "success":
            failures.append(f"{name}: compute graph did not succeed")
            continue
        for field, value in counts.items():
            if record.get(field) != value:
                failures.append(f"{name}: {field}={record.get(field)!r}, expected {value}")
        for role in required - set(by_job.get(job_key, [])):
            failures.append(f"{name}: missing live {role} call")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output-dir must be empty; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    set_print_stream(False)
    trace.set_sink(trace.FileSink(output / "trace"))
    records: dict[str, object] = {}
    try:
        for index, kind in enumerate(("compile", "numeric", "semantic"), start=101):
            records[kind] = _kernel_probe(kind, index, output / kind)
        records["knowledge"] = _knowledge_probe(104, output / "knowledge")
    finally:
        trace.get_sink().flush()
        trace.set_sink(None)
        deps.reset()
    roles, routes, by_job = _read_calls(output / "trace")
    required = {"generator", "repair.compile", "repair.numeric", "repair.semantic", "knowledge_generator", "knowledge_repair"}
    result = {
        "records": records, "roles": roles, "routes": routes,
        "roles_by_job": by_job, "missing_roles": sorted(required - roles.keys()),
        "failures": _probe_failures(records, by_job),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 1 if result["missing_roles"] or result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
