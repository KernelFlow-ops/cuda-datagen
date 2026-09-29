"""YAML scenario driver for component tests.

Usage from pytest (see tests/component/test_scenarios.py):

    sc = load_scenario(path)
    result = run_scenario(sc, tmp_path, monkeypatch, schedule="serial")
    assert_expectations(result, sc)
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from cuda_sft.runtime import deps, trace
from cuda_sft.testing.fakes import FakeCompiler, FakeLLM, FakeRefval, FakeSleep, strip_marker

FIXTURES = Path(__file__).resolve().parents[3] / "tests" / "fixtures"


def load_scenario(path: Path) -> dict[str, Any]:
    sc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(sc, dict) or "id" not in sc:
        raise ValueError(f"bad scenario file {path}")
    if not Path(path).name.startswith(str(sc["id"]) + "_"):
        raise ValueError(f"scenario id {sc['id']} does not match file name {Path(path).name}")
    return sc


@dataclass
class ScenarioResult:
    finals: list[dict[str, Any]]
    samples: list[dict[str, Any]]
    abandoned: list[dict[str, Any]]
    llm: FakeLLM
    compiler: FakeCompiler
    refval: FakeRefval
    sleeper: FakeSleep
    events: list[dict[str, Any]] = field(default_factory=list)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _init_state(sc: dict[str, Any], dialect: str) -> dict[str, Any]:
    job = sc["job"]
    base = {
        "question_id": int(job.get("question_id", 1)),
        "question": job["question"],
        "input_metadata": {"source_line": job["source_line"]} if "source_line" in job else {},
        "source": "scenario",
        "status": "running",
    }
    if job.get("kind", "kernel") == "knowledge":
        topic = job.get("topic") or "general"
        return {**base, "kind": "knowledge", "topic": topic, "track": f"knowledge:{topic}"}
    return {**base, "kind": "kernel", "dialect": dialect}


def _run_one(init: dict[str, Any], *, schedule: str) -> dict[str, Any]:
    """Run the requested compute schedule and persist its test result."""
    if schedule == "parallel":
        from cuda_sft.config import get_settings
        from cuda_sft.runtime.scheduler import solve_job_for_tests

        final = dict(solve_job_for_tests(init))
        from cuda_sft.store import get_store

        if final.get("status") == "success":
            get_store().write_success(final, model_name=get_settings().resolved_model)
        elif final.get("status") == "abandoned":
            get_store().write_abandoned(final)
        return final
    if init["kind"] == "knowledge":
        from cuda_sft.knowledge.graph import build_knowledge_graph, knowledge_recursion_limit

        return dict(
            build_knowledge_graph().invoke(init, {"recursion_limit": knowledge_recursion_limit()})
        )
    from cuda_sft.graph import build_graph, recursion_limit

    return dict(build_graph().invoke(init, {"recursion_limit": recursion_limit()}))


def run_scenario(
    sc: dict[str, Any],
    tmp_path: Path,
    monkeypatch: Any,
    *,
    schedule: str = "serial",
) -> ScenarioResult:
    data_dir, work_dir = tmp_path / "data", tmp_path / "work"
    env = {
        "DATA_DIR": str(data_dir),
        "WORK_DIR": str(work_dir),
        "TRACE_ENABLED": "false",
        "ASYNC_LLM_ENABLED": "false",
        "CUDA_ARCH": "sm_86",
        "GPU_NAME": "FakeGPU",
        "CUDA_HOME": str(tmp_path / "cuda"),
        **{str(k): str(v) for k, v in (sc.get("settings") or {}).items()},
    }
    if schedule == "serial":
        env.update(
            {
                "EARLY_STOP": "false",
                "BUDGET_WAVES_SIMPLE": "1,1,1,1",
                "BUDGET_WAVES_MEDIUM": "1,1,1,1",
                "BUDGET_WAVES_HARD": "1,1,1,1",
            }
        )
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    from cuda_sft import config

    monkeypatch.setattr(config, "detect_cuda_version", lambda *_a, **_k: "12.4")
    config.get_settings.cache_clear()
    from cuda_sft import store

    store.init_store(data_dir, allow_test_sources=True)

    llm = FakeLLM(script=dict(sc.get("llm") or {}), fixtures_root=FIXTURES)
    compiler, refval, sleeper = FakeCompiler(), FakeRefval(), FakeSleep()
    sink = trace.MemorySink()
    previous_sink = trace.set_sink(sink)
    _apply_patches(sc.get("patches") or {}, monkeypatch, llm)
    finals: list[dict[str, Any]] = []
    try:
        with deps.use(
            deps.Deps(
                llm_factory=llm.factory, compile_fn=compiler, refval_fn=refval, sleep_fn=sleeper
            )
        ):
            for dialect in sc["job"].get("dialects", ["cuda"]):
                finals.append(_run_one(_init_state(sc, dialect), schedule=schedule))
    finally:
        trace.set_sink(previous_sink)
        config.get_settings.cache_clear()
    return ScenarioResult(
        finals=finals,
        samples=_read_jsonl(data_dir / "sft.jsonl"),
        abandoned=_read_jsonl(data_dir / "abandoned.jsonl"),
        llm=llm,
        compiler=compiler,
        refval=refval,
        sleeper=sleeper,
        events=list(sink.events),
    )


def _apply_patches(patches: dict[str, Any], monkeypatch: Any, llm: FakeLLM) -> None:
    allowed = {"judge_optimized_code", "critic_raise"}
    bad = set(patches) - allowed
    if bad:
        raise ValueError(f"unsupported patches: {bad}")
    if "judge_optimized_code" in patches:
        code = (FIXTURES / patches["judge_optimized_code"]).read_text(encoding="utf-8")
        from cuda_sft.judge import JudgeResult

        orig_init = JudgeResult.__init__

        def _init(self: Any, *a: Any, **kw: Any) -> None:
            orig_init(self, *a, **kw)
            object.__setattr__(self, "optimized_code", code)

        monkeypatch.setattr(JudgeResult, "__init__", _init)
    if patches.get("critic_raise"):
        llm.script["critic"] = {**llm.script.get("critic", {}), "default": {"raise": "timeout"}}


# ------------------------------------------------------------------ assertions

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
CODE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.S)
TRAINING = {"TRAINING_ZH": "TRAINING_SYSTEM_PROMPT_ZH", "TRAINING_EN": "TRAINING_SYSTEM_PROMPT_EN"}


def _dig(obj: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return obj


def _count_check(actual: int, spec: dict[str, int], label: str) -> None:
    if "eq" in spec:
        assert actual == spec["eq"], f"{label}: expected == {spec['eq']}, got {actual}"
    if "max" in spec:
        assert actual <= spec["max"], f"{label}: expected <= {spec['max']}, got {actual}"
    if "min" in spec:
        assert actual >= spec["min"], f"{label}: expected >= {spec['min']}, got {actual}"


def _msg(sample: dict[str, Any], role: str) -> str | None:
    for m in sample["messages"]:
        if m["role"] == role:
            return str(m["content"])
    return None


def assert_expectations(res: ScenarioResult, sc: dict[str, Any]) -> None:
    exp = sc.get("expect") or {}
    last = res.finals[-1] if res.finals else {}
    if "status" in exp:
        assert last.get("status") == exp["status"], (
            f"status {last.get('status')} != {exp['status']}"
        )
    if "abandon_reason" in exp:
        assert res.abandoned, "no abandoned record written"
        assert res.abandoned[-1].get("abandon_reason") == exp["abandon_reason"], res.abandoned[-1]
    sample = res.samples[-1] if res.samples else None
    if "selected" in exp:
        assert sample is not None, "no sample written"
        md = sample["metadata"]
        sel = exp["selected"]
        pool = md.get("candidate_pool", {})
        if "candidate" in sel:
            got = md.get("candidate", pool.get("selected_candidate"))
            assert got == sel["candidate"], f"selected candidate {got} != {sel['candidate']}"
        if "repairs" in sel:
            assert md.get("repairs") == sel["repairs"], (
                f"repairs {md.get('repairs')} != {sel['repairs']}"
            )
        if "code_file" in sel:
            assistant = _msg(sample, "assistant") or ""
            blocks = CODE_RE.findall(assistant)
            selected_code = blocks[-1] if blocks else assistant.split("</think>", 1)[-1].strip()
            assert selected_code, "assistant has no code"
            want = strip_marker((FIXTURES / sel["code_file"]).read_text(encoding="utf-8"))
            assert strip_marker(selected_code) == want, "selected code differs from fixture"
    if "sample" in exp:
        assert sample is not None, "no sample written"
        _assert_sample(sample, exp["sample"], res)
    llm_exp = exp.get("llm") or {}
    for role, spec in (llm_exp.get("calls") or {}).items():
        _count_check(len(res.llm.calls_for(role)), spec, f"llm.calls[{role}]")
    for check in llm_exp.get("call_checks") or []:
        calls = [
            c
            for c in res.llm.calls_for(check["role"])
            if check.get("key") in (None, c.key) and check.get("purpose") in (None, c.purpose)
        ]
        assert calls, f"no calls matched {check}"
        for c in calls:
            for s in check.get("user_contains", []):
                assert s in c.user_text, f"{c.role}/{c.key}: user lacks {s!r}"
            for s in check.get("user_absent", []):
                assert s not in c.user_text, f"{c.role}/{c.key}: user contains {s!r}"
            for s in check.get("system_contains", []):
                assert s in c.system, f"{c.role}/{c.key}: system lacks {s!r}"
            if "messages_len" in check:
                assert len(c.messages) == check["messages_len"], (
                    f"{c.role}/{c.key}: {len(c.messages)} messages"
                )
    if "refval" in exp:
        _count_check(len(res.refval.calls), exp["refval"]["calls"], "refval.calls")
    if "sleeps" in exp:
        assert res.sleeper.calls == [float(x) for x in exp["sleeps"]], res.sleeper.calls
    if "trace" in exp:
        _assert_trace(res.events, exp["trace"])
    if "abandoned" in exp:
        rec = res.abandoned[-1]
        if "candidates_len" in exp["abandoned"]:
            assert len(rec.get("candidates", [])) == exp["abandoned"]["candidates_len"], rec
        for cand, gate in (exp["abandoned"].get("last_gate") or {}).items():
            entry = next(c for c in rec["candidates"] if int(c["candidate"]) == int(cand))
            for k, v in gate.items():
                assert entry["last_gate"][k] == v, (cand, k, entry["last_gate"])
    if "timing" in exp and "cancel_within_s" in exp["timing"]:
        cancelled = [c for c in res.llm.calls if c.cancelled]
        assert cancelled, "expected at least one cancelled LLM call"
        # TODO(T3.12): measure from sched.cancel event ts to c.t_end


def _assert_sample(sample: dict[str, Any], spec: dict[str, Any], res: ScenarioResult) -> None:
    system = _msg(sample, "system")
    if "system_equals" in spec:
        want = spec["system_equals"]
        if want == "ABSENT":
            assert system is None, "system message should be absent"
        elif want in TRAINING:
            from cuda_sft import prompt

            assert system == getattr(prompt, TRAINING[want]), "system is not the training prompt"
        else:
            assert system == want
    for pat in spec.get("system_absent_patterns", []):
        assert pat.lower() not in (system or "").lower(), f"system contains {pat!r}"
    think = "\n".join(THINK_RE.findall(_msg(sample, "assistant") or ""))
    for pat in spec.get("cot_absent_patterns", []):
        assert pat.lower() not in think.lower(), f"CoT contains {pat!r}"
    for marker in spec.get("cot_absent_markers", []):
        assert marker not in (_msg(sample, "assistant") or ""), f"assistant leaks marker {marker!r}"
    if "user_endswith_contract" in spec:
        user = _msg(sample, "user") or ""
        tail = user.rstrip()[-2000:]
        has = ("## 接口约定" in tail) or ("## Interface contract" in tail)
        assert has == bool(spec["user_endswith_contract"]), "contract suffix mismatch"
    for path, want in (spec.get("metadata") or {}).items():
        got = _dig(sample["metadata"], path)
        assert got == want, f"metadata.{path}: {got!r} != {want!r}"


def _assert_trace(events: list[dict[str, Any]], spec: dict[str, Any]) -> None:
    nodes = [e["node"] for e in events if e.get("event") == "node.end"]
    want = list(spec.get("nodes_in_order", []))
    it = iter(nodes)
    for name in want:
        assert any(n == name for n in it), f"node {name!r} missing/out of order in {nodes}"
    for name, cnt in (spec.get("events") or {}).items():
        _count_check(sum(1 for e in events if e.get("event") == name), cnt, f"trace.events[{name}]")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
