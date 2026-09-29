#!/usr/bin/env python3
"""Run and audit a fresh, fixed 25-question live generation batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_sft.config import ROLE_CONFIG_PREFIXES, get_settings  # noqa: E402
from cuda_sft.formats import (  # noqa: E402
    extract_system,
    iter_sft_rows,
    resolve_export_assistant,
    split_user_assistant,
    to_ms_swift,
    to_openrlhf,
)
from cuda_sft.parse import extract_cuda_source  # noqa: E402
from cuda_sft.tasks.kinds import question_hash  # noqa: E402


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_digest() -> str:
    digest = hashlib.sha256()
    source_root = ROOT / "src" / "cuda_sft"
    for path in sorted(source_root.rglob("*.py")):
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _role_config() -> dict[str, Any]:
    settings = get_settings()
    return {
        role: {
            "provider": settings.for_role(role).llm_provider,
            "model": settings.for_role(role).resolved_model,
            "thinking_level": settings.for_role(role).thinking_level,
        }
        for role in ROLE_CONFIG_PREFIXES
    }


def _configuration(task: str, workers: int, concurrency: int, inflight: int) -> dict[str, Any]:
    settings = get_settings()
    return {
        "task": task,
        "workers": workers,
        "llm_concurrency": concurrency,
        "max_inflight_jobs": inflight,
        "compile_concurrency": settings.compile_concurrency,
        "max_candidates": settings.max_candidates,
        "max_repairs": settings.max_repairs,
        "knowledge_max_candidates": settings.knowledge_max_candidates,
        "knowledge_max_repairs": settings.knowledge_max_repairs,
        "refval_enabled": settings.refval_enabled,
        "refval_strict": settings.refval_strict,
        "refval_cases": settings.refval_cases,
        "knowledge_judge_enabled": settings.knowledge_judge_enabled,
        "cot_enabled": settings.cot_enabled,
        "cot_agent_enabled": settings.cot_agent_enabled,
        "roles": _role_config(),
    }


def _valid_row(task: str, sample: dict[str, Any], source: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    metadata = sample.get("metadata") or {}
    source_hash = question_hash(str(source["question"]))
    if metadata.get("question_hash") != source_hash:
        errors.append("question_hash")
    if int(metadata.get("source_line") or 0) != int(source["source_line"]):
        errors.append("source_line")
    if (metadata.get("provenance") or {}).get("origin") != "live_api":
        errors.append("live_api_origin")
    pair = split_user_assistant(sample)
    assistant = pair[1] if pair else ""
    cot = metadata.get("cot") or {}
    if not str(cot.get("text") or "").strip():
        errors.append("cot_empty")
    if task == "kernel":
        refval = metadata.get("refval") or {}
        if refval.get("status") != "pass" or int(refval.get("cases_run") or 0) <= 0:
            errors.append("refval_pass")
        if metadata.get("verification_tier") not in {"model_consistency", "independent"}:
            errors.append("verification_tier")
        code = extract_cuda_source(assistant)
        if hashlib.sha256(code.encode()).hexdigest() != metadata.get("selected_code_sha256"):
            errors.append("selected_code_hash")
        if not re.search(r"\b(__global__|cudaLaunchKernel)\b|<<<", code):
            errors.append("kernel_source")
    else:
        judge = metadata.get("knowledge_judge") or {}
        if judge.get("pass") is not True or judge.get("skipped_llm") or judge.get("unavailable"):
            errors.append("judge_evidence")
        if judge.get("hard_gate_failed") or judge.get("answer_sha256") != metadata.get("code_hash"):
            errors.append("answer_integrity")
        if metadata.get("task") != "knowledge" or not assistant.strip():
            errors.append("knowledge_answer")
    return errors


def audit(task: str, input_path: Path, data_dir: Path, *, elapsed_s: float, returncode: int,
          count: int = 25) -> dict[str, Any]:
    inputs = _rows(input_path)[:count]
    samples = list(iter_sft_rows(data_dir / "sft.jsonl")) if (data_dir / "sft.jsonl").exists() else []
    progress = _rows(data_dir / "progress.jsonl")
    abandoned = _rows(data_dir / "abandoned.jsonl")
    trace_events = [
        event for path in sorted((data_dir / "trace").glob("trace-*.jsonl"))
        for event in _rows(path)
    ]
    expected = {index: row for index, row in enumerate(inputs, 1)}
    by_id: dict[int, list[dict[str, Any]]] = {}
    for row in samples:
        by_id.setdefault(int(row.get("id") or 0), []).append(row)
    issues: dict[str, list[str]] = {}
    for qid, source in expected.items():
        if source.get("question_hash") and source["question_hash"] != question_hash(str(source["question"])):
            issues.setdefault(str(qid), []).append("input_question_hash")
        sample_rows = by_id.get(qid, [])
        if len(sample_rows) != 1:
            issues[str(qid)] = [f"sample_count={len(sample_rows)}"]
        else:
            errors = _valid_row(task, sample_rows[0], source)
            if errors:
                issues[str(qid)] = errors
            job_calls = [event for event in trace_events if event.get("event") == "llm.call"
                         and str(event.get("job_key") or "").startswith(f"q{qid}:") and event.get("ok")]
            required_roles = {"generator", "refval_extract"} if task == "kernel" else {
                "knowledge_generator", "knowledge_judge"
            }
            if not required_roles <= {str(event.get("role") or "") for event in job_calls}:
                issues.setdefault(str(qid), []).append("live_role_calls")
    extra_ids = sorted(set(by_id) - set(expected))
    if extra_ids:
        issues["extra_ids"] = [str(value) for value in extra_ids]
    progress_success = Counter(int(row.get("id") or 0) for row in progress if row.get("status") == "success")
    for qid in expected:
        if progress_success[qid] != 1:
            issues.setdefault(str(qid), []).append(f"progress_success={progress_success[qid]}")
    export_counts = {}
    exports: dict[str, list[dict[str, Any]]] = {}
    for name in ("sft_ms_swift.jsonl", "sft_openrlhf.jsonl"):
        path = data_dir / name
        exports[name] = _rows(path)
        export_counts[name] = len(exports[name])
        if export_counts[name] != len(expected):
            issues.setdefault("exports", []).append(f"{name}={export_counts[name]}")
    if len(samples) == len(exports["sft_ms_swift.jsonl"]) == len(exports["sft_openrlhf.jsonl"]):
        for index, row in enumerate(samples):
            pair = split_user_assistant(row)
            if pair is None:
                issues.setdefault("exports", []).append(f"sample {index + 1} has no user/assistant")
                continue
            user, _assistant = pair
            assistant = resolve_export_assistant(
                row, cot_in_assistant=bool(get_settings().cot_in_assistant)
            )
            system = extract_system(row)
            if exports["sft_ms_swift.jsonl"][index] != to_ms_swift(user, assistant, system=system):
                issues.setdefault("exports", []).append(f"ms-swift row {index + 1} differs")
            if exports["sft_openrlhf.jsonl"][index] != to_openrlhf(user, assistant, system=system):
                issues.setdefault("exports", []).append(f"openrlhf row {index + 1} differs")
    limit_s = count * (60 if task == "kernel" else 30)
    llm_calls = [event for event in trace_events if event.get("event") == "llm.call"]
    committed = {
        str(event.get("job_key") or ""): event
        for event in trace_events if event.get("event") == "job.commit"
    }
    per_question: list[dict[str, Any]] = []
    for qid, source in expected.items():
        prefix = f"q{qid}:"
        outcome = next((event for key, event in committed.items() if key.startswith(prefix)), {})
        if outcome.get("status") != "success":
            issues.setdefault(str(qid), []).append("commit_not_success")
        row = by_id.get(qid, [{}])[0]
        metadata = row.get("metadata") or {}
        per_question.append({
            "id": qid,
            "source_line": source["source_line"],
            "question_hash": question_hash(str(source["question"])),
            "status": outcome.get("status") or "missing_commit",
            "elapsed_s": round(float(outcome.get("elapsed_s") or 0), 3),
            "llm_calls": sum(str(event.get("job_key") or "").startswith(prefix) for event in llm_calls),
            "repairs": int(metadata.get("repairs") or 0),
            "issues": issues.get(str(qid), []),
        })
    elapsed_by_node: dict[str, float] = {}
    for event in trace_events:
        if event.get("event") == "node.end":
            node = str(event.get("node") or "unknown")
            elapsed_by_node[node] = elapsed_by_node.get(node, 0.0) + float(event.get("elapsed_s") or 0)
    wait_by_stage: dict[str, float] = {}
    for event in trace_events:
        if event.get("event") == "stage.wait":
            stage = str(event.get("stage") or "unknown")
            wait_by_stage[stage] = wait_by_stage.get(stage, 0.0) + float(event.get("elapsed_s") or 0)
    quality_pass = not issues and returncode == 0 and len(expected) == count
    return {
        "task": task,
        "mode": "acceptance" if count == 25 else "probe",
        "input_sha256": _digest(input_path),
        "expected": len(expected),
        "sample_count": len(samples),
        "progress_success": sum(progress_success.values()),
        "abandoned_count": len(abandoned),
        "export_counts": export_counts,
        "llm_calls": len(llm_calls),
        "llm_errors": sum(event.get("ok") is False for event in llm_calls),
        "llm_calls_by_role": dict(Counter(str(event.get("role") or "") for event in llm_calls)),
        "node_elapsed_s": {key: round(value, 3) for key, value in sorted(elapsed_by_node.items())},
        "stage_wait_s": {key: round(value, 3) for key, value in sorted(wait_by_stage.items())},
        "repair_attempts": sum(int((row.get("metadata") or {}).get("repairs") or 0) for row in samples),
        "questions": per_question,
        "returncode": returncode,
        "elapsed_s": round(elapsed_s, 3),
        "seconds_per_question": round(elapsed_s / len(expected), 3)
        if len(samples) == len(expected) and not issues else None,
        "seconds_per_published": round(elapsed_s / len(samples), 3) if samples else None,
        "limit_s": limit_s,
        "quality_pass": quality_pass,
        "performance_pass": elapsed_s <= limit_s,
        "accepted": count == 25 and quality_pass and elapsed_s <= limit_s,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("kernel", "knowledge"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--llm-concurrency", type=int, default=2)
    parser.add_argument("--inflight", type=int, default=2)
    parser.add_argument("--count", type=int, default=25,
                        help="number of fixed input rows; counts below 25 are diagnostic probes")
    args = parser.parse_args()
    input_path = args.input.resolve()
    out = args.out.resolve()
    if len(_rows(input_path)) != 25 or not 1 <= args.count <= 25:
        parser.error("input must contain 25 fixed questions and count must be 1..25")
    if out.exists() and any(out.iterdir()):
        parser.error(f"output directory must be fresh: {out}")
    out.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data"
    env = os.environ.copy()
    env.update({
        "DATA_DIR": str(data_dir), "WORK_DIR": str(out / "work"),
        "TASK_MODE": args.task, "KERNEL_DIALECTS": "cuda", "KERNEL_MODE": "single",
        "KERNEL_FAST_MODE": "false", "REFVAL_ENABLED": "true", "REFVAL_STRICT": "true",
        "KNOWLEDGE_JUDGE_ENABLED": "true", "COT_ENABLED": "true",
        "COT_AGENT_ENABLED": "true", "TRACE_ENABLED": "true", "TRACE_DIR": "trace",
        "LLM_CONCURRENCY": str(args.llm_concurrency),
        "MAX_INFLIGHT_JOBS": str(args.inflight),
        "COMPILE_CONCURRENCY": "1",
    })
    env.pop("CUDA_SFT_LLM_REPLAY", None)
    os.environ.update({key: env[key] for key in (
        "DATA_DIR", "WORK_DIR", "TASK_MODE", "KERNEL_DIALECTS", "KERNEL_MODE",
        "KERNEL_FAST_MODE", "REFVAL_ENABLED", "REFVAL_STRICT", "KNOWLEDGE_JUDGE_ENABLED",
        "COT_ENABLED", "COT_AGENT_ENABLED", "TRACE_ENABLED", "TRACE_DIR", "LLM_CONCURRENCY",
        "MAX_INFLIGHT_JOBS", "COMPILE_CONCURRENCY",
    )})
    get_settings.cache_clear()
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "input": str(input_path), "input_sha256": _digest(input_path),
        "source_sha256": _source_digest(),
        "started_at": started_at,
        "count": args.count,
        "configuration": _configuration(args.task, args.workers, args.llm_concurrency, args.inflight),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, str(ROOT / "run.py"), "--input", str(input_path),
               "--task", args.task, "--data-dir", str(data_dir), "--workers", str(args.workers),
               "--limit", str(args.count), "--quiet"]
    started = time.monotonic()
    with (out / "console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - started
    report = audit(args.task, input_path, data_dir, elapsed_s=elapsed,
                   returncode=result.returncode, count=args.count)
    report["started_at"] = started_at
    report["ended_at"] = datetime.now(timezone.utc).isoformat()
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["quality_pass"] and report["performance_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
