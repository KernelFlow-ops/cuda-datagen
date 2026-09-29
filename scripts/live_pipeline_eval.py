#!/usr/bin/env python3
"""Run real-provider production graphs and controlled repair probes in isolation."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import suppress
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_sft.config import get_settings  # noqa: E402
from cuda_sft.refval.cross_dialect import canonical_tasks, make_manifest  # noqa: E402

DIALECTS = ("cuda", "cutlass", "triton", "tilelang")
REQUIRED_ROLES = {
    "generator", "repair.compile", "repair.numeric", "repair.semantic",
    "refval_extract", "critic", "cot_editor", "knowledge_generator",
    "knowledge_repair", "knowledge_judge",
}


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_inputs(root: Path) -> dict[str, Path]:
    task = canonical_tasks()["elementwise_add"]
    manifests = {
        dialect: make_manifest(task, question_id=1, dialect=dialect).to_dict()
        for dialect in DIALECTS
    }
    kernel_question = (
        "Implement strided elementwise addition c[i*c_stride] = "
        "a[i*a_stride] + b[i*b_stride] for 0 <= i < n. Handle n=0 and "
        "non-unit strides. For CUDA/CUTLASS emit extern C void "
        "launch_add(const float* a, const float* b, float* c, int n, "
        "int a_stride, int b_stride, int c_stride). For Triton/TileLang "
        "emit def launch_add(a, b, c, n, a_stride, b_stride, c_stride). "
        "For Python dialects, tensor arguments are logical Torch views whose "
        "tensor strides equal the corresponding stride scalars; packing those "
        "logical views and copying the result back is allowed. The callable "
        "must launch the GPU kernel and write c."
    )
    knowledge_question = (
        "Explain CUDA occupancy versus GPU utilization. Describe how "
        "registers, shared memory, threads per block, and hardware limits "
        "determine active warps. Explain why maximum occupancy need not "
        "maximize performance, with a concrete latency-hiding example."
    )
    paths = {
        "kernel": root / "kernel_questions.jsonl",
        "model_oracle": root / "model_oracle_questions.jsonl",
        "knowledge": root / "knowledge_questions.jsonl",
    }
    paths["kernel"].write_text(
        json.dumps({"question": kernel_question, "task": "kernel", "oracle_manifests": manifests}) + "\n",
        encoding="utf-8",
    )
    paths["model_oracle"].write_text(
        json.dumps({"question": kernel_question, "task": "kernel"}) + "\n",
        encoding="utf-8",
    )
    paths["knowledge"].write_text(
        json.dumps({"question": knowledge_question, "task": "knowledge", "topic": "execution"}) + "\n",
        encoding="utf-8",
    )
    return paths


def _phase_env(root: Path, *, phase: str, candidates: int, repairs: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "WORK_DIR": str(root / phase / "work"),
        "TRACE_ENABLED": "true",
        "TRACE_DIR": "trace",
        "KERNEL_FAST_MODE": "false",
        "DIFFICULTY_AWARE": "false",
        "MAX_CANDIDATES": str(candidates),
        "MAX_REPAIRS": str(repairs),
        "KNOWLEDGE_MAX_CANDIDATES": "1",
        "KNOWLEDGE_MAX_REPAIRS": str(repairs),
        "KERNEL_LLM_CRITIC": "always",
        "COT_ENABLED": "true",
        "COT_AGENT_ENABLED": "true",
        "KNOWLEDGE_JUDGE_ENABLED": "true",
        "REFVAL_ENABLED": "true",
        "REFVAL_STRICT": "true",
        "REFVAL_CASES": "standard",
        "ASYNC_LLM_ENABLED": "true",
        "MAX_INFLIGHT_JOBS": "2",
        "LLM_CONCURRENCY": "2",
    })
    return env


def _descendants(pid: int) -> set[int]:
    children: dict[int, set[int]] = {}
    for path in Path("/proc").glob("[0-9]*/status"):
        try:
            parent_line = next(
                line for line in path.read_text(encoding="utf-8").splitlines()
                if line.startswith("PPid:")
            )
            children.setdefault(int(parent_line.split()[1]), set()).add(int(path.parent.name))
        except (OSError, StopIteration, ValueError):
            continue
    found: set[int] = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        for child in children.get(parent, ()):
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _stop_process_tree(process: subprocess.Popen[Any]) -> None:
    groups = {process.pid}
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGSTOP)
    for _ in range(3):
        found = set()
        for pid in _descendants(process.pid):
            try:
                found.add(os.getpgid(pid))
            except ProcessLookupError:
                continue
        for group in found - groups:
            try:
                os.killpg(group, signal.SIGSTOP)
            except ProcessLookupError:
                continue
        if found <= groups:
            break
        groups.update(found)
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            continue
    process.wait()


def _invoke(command: list[str], *, env: dict[str, str], timeout: int) -> dict[str, Any]:
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        return {"returncode": process.wait(timeout=timeout), "timed_out": False}
    except subprocess.TimeoutExpired:
        _stop_process_tree(process)
        return {"returncode": 124, "timed_out": True}


def _trace_summary(data_dir: Path) -> dict[str, Any]:
    events = [
        row for file in (data_dir / "trace").glob("trace-*.jsonl")
        for row in _rows(file)
    ]
    calls = [row for row in events if row.get("event") == "llm.call"]
    roles = Counter(str(row.get("role") or "") for row in calls if row.get("ok"))
    routes = {
        role: sorted({f"{row.get('provider')}/{row.get('model')}" for row in calls if row.get("role") == role})
        for role in sorted(roles)
    }
    intervals: list[tuple[datetime, datetime, dict[str, Any]]] = []
    for row in events:
        if row.get("event") not in {"llm.call", "node.end"}:
            continue
        try:
            ended = datetime.fromisoformat(str(row["ts"]))
            started = ended - timedelta(seconds=float(row["elapsed_s"]))
        except (KeyError, TypeError, ValueError):
            continue
        intervals.append((started, ended, row))

    def peak(items: list[tuple[datetime, datetime, dict[str, Any]]]) -> int:
        edges = [(start, 1) for start, _end, _row in items]
        edges.extend((end, -1) for _start, end, _row in items)
        count = maximum = 0
        for _instant, delta in sorted(edges, key=lambda item: (item[0], item[1])):
            count += delta
            maximum = max(maximum, count)
        return maximum

    llm_intervals = [item for item in intervals if item[2].get("event") == "llm.call"]
    key_peaks = {
        key: peak([item for item in llm_intervals if item[2].get("key_label") == key])
        for key in {str(item[2].get("key_label") or "") for item in llm_intervals}
    }
    candidate_nodes = [
        item for item in intervals
        if item[2].get("event") == "node.end"
        and item[2].get("node") in {"generate", "compile"}
    ]
    candidate_overlap = any(
        left[2].get("job_key") == right[2].get("job_key")
        and left[2].get("candidate") != right[2].get("candidate")
        and left[0] < right[1] and right[0] < left[1]
        for index, left in enumerate(candidate_nodes)
        for right in candidate_nodes[index + 1:]
    )
    return {
        "roles": dict(roles), "routes": routes,
        "llm_calls": len(calls),
        "llm_errors": sum(not row.get("ok") for row in calls),
        "max_llm_inflight_by_key": key_peaks,
        "candidate_overlap": candidate_overlap,
        "job_starts": sum(row.get("event") == "job.start" for row in events),
        "job_ends": sum(row.get("event") == "job.end" for row in events),
    }


def _phase_summary(data_dir: Path) -> dict[str, Any]:
    progress = _rows(data_dir / "progress.jsonl")
    samples = _rows(data_dir / "sft.jsonl")
    return {
        "progress": [{"dialect": row.get("dialect"), "status": row.get("status")} for row in progress],
        "samples": [
            {
                "dialect": (row.get("metadata") or {}).get("dialect"),
                "origin": (row.get("metadata") or {}).get("provenance", {}).get("origin"),
                "verification_tier": (row.get("metadata") or {}).get("verification_tier", "none"),
                "oracle_origin": (row.get("metadata") or {}).get("oracle_origin", "none"),
                "release_tier": (row.get("metadata") or {}).get("release_tier"),
                "candidate_count": ((row.get("metadata") or {}).get("candidate_pool") or {}).get("count"),
                "selected_candidate": ((row.get("metadata") or {}).get("candidate_pool") or {}).get("selected_candidate"),
                "critic_status": ((row.get("metadata") or {}).get("critic") or {}).get("status"),
                "knowledge_judge_pass": ((row.get("metadata") or {}).get("knowledge_judge") or {}).get("pass"),
                "cot_chars": len(str(((row.get("metadata") or {}).get("cot") or {}).get("text") or "")),
            }
            for row in samples
        ],
        **_trace_summary(data_dir),
    }


def _run_production(
    root: Path, inputs: dict[str, Path], *, candidates: int, repairs: int,
    timeout: int, provider: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for phase in ("kernel", "model_oracle", "knowledge"):
        data_dir = root / phase / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, str(ROOT / "run.py"), "--input", str(inputs[phase]),
            "--data-dir", str(data_dir), "--workers", "1",
            "--providers", provider, "--quiet",
        ]
        if phase == "kernel":
            command += ["--task", "kernel", "--dialects", ",".join(DIALECTS), "--kernel-mode", "all"]
        elif phase == "model_oracle":
            command += ["--task", "kernel", "--dialects", "cuda", "--kernel-mode", "single"]
        else:
            command += ["--task", "knowledge"]
        run = _invoke(
            command, env=_phase_env(root, phase=phase, candidates=candidates, repairs=repairs),
            timeout=timeout,
        )
        results[phase] = {**run, **_phase_summary(data_dir)}
    return results


def _run_gpu_oracle(root: Path, *, timeout: int) -> dict[str, Any]:
    report_path = root / "canonical_gpu.json"
    command = [
        sys.executable, str(ROOT / "scripts" / "cross_dialect_oracle.py"),
        "--tasks", "elementwise_add", "--dialects", ",".join(DIALECTS),
        "--cases", "smoke", "--mutants", "--report", str(report_path),
        "--work-dir", str(root / "canonical_gpu_work"),
    ]
    env = os.environ.copy()
    env.update(DATA_DIR=str(root / "canonical_gpu_data"), TRACE_ENABLED="false")
    run = _invoke(command, env=env, timeout=timeout)
    if not report_path.exists():
        return run
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    case_hashes = {str(row.get("cases_hash") or "") for row in rows}
    group_hashes = {str(row.get("group_hash") or "") for row in rows}
    return {
        **run,
        "good_pass": payload.get("good_pass", 0),
        "mutants_caught": payload.get("mutants_caught", 0),
        "unavailable": payload.get("unavailable", 0),
        "shared_cases_hash": next(iter(case_hashes)) if len(case_hashes) == 1 else "",
        "shared_group_hash": next(iter(group_hashes)) if len(group_hashes) == 1 else "",
        "records": [
            {"dialect": row.get("dialect"), "variant": row.get("variant"), "status": row.get("status")}
            for row in rows
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-candidates", type=int, default=2)
    parser.add_argument("--max-repairs", type=int, default=1)
    parser.add_argument("--timeout-sec", type=int, default=1800)
    args = parser.parse_args()
    if args.max_candidates < 2 or args.max_repairs < 1 or args.timeout_sec < 1:
        parser.error("invalid candidates, repairs, or timeout")
    root = (args.output_dir or Path(tempfile.mkdtemp(prefix="cuda-sft-live-"))).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("output-dir must be empty; use a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    missing = settings.missing_provider_secrets(
        [settings.llm_provider], roles=sorted(REQUIRED_ROLES)
    )
    if missing:
        parser.error(f"missing configured provider secrets: {', '.join(missing)}")
    inputs = _write_inputs(root)
    production = _run_production(
        root, inputs, candidates=args.max_candidates, repairs=args.max_repairs,
        timeout=args.timeout_sec, provider=settings.llm_provider,
    )
    probes_dir = root / "probes"
    probe = _invoke(
        [sys.executable, str(ROOT / "scripts" / "live_agent_probes.py"),
         "--output-dir", str(probes_dir)],
        env=os.environ.copy(), timeout=args.timeout_sec,
    )
    probe_report = json.loads((probes_dir / "report.json").read_text(encoding="utf-8")) if (probes_dir / "report.json").exists() else {}
    gpu = _run_gpu_oracle(root, timeout=args.timeout_sec)
    role_counts = Counter()
    for phase in production.values():
        role_counts.update(phase.get("roles") or {})
    role_counts.update(probe_report.get("roles") or {})
    missing_roles = sorted(REQUIRED_ROLES - role_counts.keys())
    independent = production["kernel"]["samples"]
    expected_dialects = set(DIALECTS)
    independent_pass = {
        str(sample.get("dialect")) for sample in independent
        if sample.get("verification_tier") == "independent" and sample.get("origin") == "live_api"
    }
    model_pass = any(
        sample.get("verification_tier") == "model_consistency"
        for sample in production["model_oracle"]["samples"]
    )
    knowledge_pass = bool(production["knowledge"]["samples"])
    failures: list[str] = []
    if missing_roles:
        failures.append(f"missing live roles: {', '.join(missing_roles)}")
    if expected_dialects - independent_pass:
        failures.append("independent GPU samples missing for some dialects")
    if not model_pass:
        failures.append("no model_consistency sample")
    if not knowledge_pass:
        failures.append("no knowledge sample")
    expected_jobs = {"kernel": 4, "model_oracle": 1, "knowledge": 1}
    actual_routes: dict[str, set[str]] = {}
    for phase_name, phase in production.items():
        if phase["returncode"] != 0:
            failures.append(f"{phase_name}: CLI exited {phase['returncode']}")
        if len(phase["progress"]) != expected_jobs[phase_name]:
            failures.append(f"{phase_name}: incomplete progress records")
        if any(item.get("status") != "success" for item in phase["progress"]):
            failures.append(f"{phase_name}: abandoned or failed job")
        if phase["job_starts"] != expected_jobs[phase_name] or phase["job_ends"] != expected_jobs[phase_name]:
            failures.append(f"{phase_name}: incomplete trace job lifecycle")
        if any(peak > 2 for peak in phase["max_llm_inflight_by_key"].values()):
            failures.append(f"{phase_name}: per-key LLM concurrency exceeded 2")
        for role, routes in phase["routes"].items():
            actual_routes.setdefault(role, set()).update(routes)
    for role, routes in (probe_report.get("routes") or {}).items():
        actual_routes.setdefault(role, set()).update(routes)
    route_mismatches = {}
    for role, routes in actual_routes.items():
        resolved = settings.for_role(role)
        expected = f"{resolved.llm_provider}/{resolved.resolved_model}"
        if routes != {expected}:
            route_mismatches[role] = {"expected": expected, "actual": sorted(routes)}
    if route_mismatches:
        failures.append("role provider/model routing mismatch")
    if not production["kernel"]["candidate_overlap"]:
        failures.append("kernel candidates did not overlap")
    for phase_name in ("kernel", "model_oracle"):
        for sample in production[phase_name]["samples"]:
            if sample.get("candidate_count") != args.max_candidates or not sample.get("selected_candidate"):
                failures.append(f"{phase_name}: candidate pool or selection incomplete")
            if sample.get("critic_status") != "verified":
                failures.append(f"{phase_name}: selected candidate lacks verified critic")
            if sample.get("cot_chars", 0) == 0:
                failures.append(f"{phase_name}: selected candidate has empty CoT")
    for sample in production["knowledge"]["samples"]:
        if sample.get("knowledge_judge_pass") is not True or sample.get("cot_chars", 0) == 0:
            failures.append("knowledge judge or CoT incomplete")
    if probe["returncode"] != 0 or probe_report.get("failures"):
        failures.append("controlled repair probes failed")
    if (
        gpu["returncode"] != 0 or gpu.get("good_pass") != len(DIALECTS)
        or gpu.get("mutants_caught") != len(DIALECTS) or gpu.get("unavailable") != 0
        or not gpu.get("shared_cases_hash") or not gpu.get("shared_group_hash")
    ):
        failures.append("canonical GPU oracle or mutant check failed")
    report = {
        "status": "fail" if failures else "pass", "output_dir": str(root),
        "production": production, "probes": {**probe, **probe_report},
        "canonical_gpu": gpu, "role_calls": dict(role_counts),
        "missing_roles": missing_roles,
        "missing_independent_dialects": sorted(expected_dialects - independent_pass),
        "model_consistency_pass": model_pass, "knowledge_pass": knowledge_pass,
        "route_mismatches": route_mismatches, "failures": failures,
    }
    (root / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
