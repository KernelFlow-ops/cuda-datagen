"""Run and report the fixed M0 baseline benchmark matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KERNEL_IDS = (1, 5, 8, 20, 26, 27, 38, 42, 75, 90)
KNOWLEDGE_IDS = tuple(range(1, 11))
PIPELINES = ("cuda", "cutlass", "triton", "tilelang", "knowledge")
SAFE_ENV_KEYS = (
    "LLM_PROVIDER", "LLM_PROVIDERS", "WORKERS", "WORKERS_PER_PROVIDER",
    "MODEL", "OPENROUTER_MODEL", "NVIDIA_MODEL", "THINKING_LEVEL",
    "MAX_CANDIDATES", "MAX_REPAIRS", "COT_ENABLED", "COT_AGENT_ENABLED",
    "KNOWLEDGE_MAX_CANDIDATES", "KNOWLEDGE_MAX_REPAIRS",
    "JUDGE_ENABLED", "KNOWLEDGE_JUDGE_ENABLED", "KNOWLEDGE_MIN_SCORE",
    "REFVAL_ENABLED", "REFVAL_TIMEOUT_SEC", "REFVAL_CASES", "REFVAL_STRICT",
    "KNOWLEDGE_FACTUAL_MIN", "LLM_TIMEOUT_SEC", "MAX_INPUT_TOKENS",
    "MAX_OUTPUT_TOKENS", "MAX_TOKENS", "NVCC_TIMEOUT_SEC", "KERNEL_DIALECT",
    "KERNEL_DIALECTS", "KERNEL_MODE", "TASK_MODE",
)
SECRET_NAME_RE = re.compile(r"(?i)(?:api[_-]?key|token|secret|authorization)")
AUTH_RE = re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;\"']+")


def _utc_now() -> str:
    """Return the current UTC time in an unambiguous ISO representation."""
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str | None:
    """Return a file SHA-256, or ``None`` when the file is absent."""
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_text(argv: Sequence[str]) -> str | None:
    """Return stdout from a diagnostic command without raising."""
    try:
        result = subprocess.run(argv, cwd=PROJECT_ROOT, text=True, capture_output=True,
                                timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def snapshot() -> dict[str, Any]:
    """Capture reproducibility metadata without serializing credentials."""
    status = _command_text(["git", "status", "--porcelain=v1"]) or ""
    dirty_digest = hashlib.sha256(status.encode())
    dirty_digest.update((_command_text(["git", "diff", "HEAD", "--binary"]) or "").encode())
    untracked = _command_text(["git", "ls-files", "--others", "--exclude-standard"]) or ""
    for filename in untracked.splitlines():
        dirty_digest.update(filename.encode())
        dirty_digest.update((_sha256(PROJECT_ROOT / filename) or "").encode())
    return {
        "git": {
            "sha": _command_text(["git", "rev-parse", "HEAD"]),
            "dirty": bool(status),
            "dirty_hash": dirty_digest.hexdigest() if status else None,
        },
        "inputs": {
            "question.jsonl": _sha256(PROJECT_ROOT / "question.jsonl"),
            "knowledge_questions.jsonl": _sha256(PROJECT_ROOT / "knowledge_questions.jsonl"),
        },
        "business_source_hashes": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in sorted((PROJECT_ROOT / "src" / "cuda_sft").rglob("*.py"))
            if path.name != "benchmark.py"
        },
        "settings": _safe_settings(),
        "environment": {
            "python": sys.version.split()[0], "python_executable": sys.executable,
            "platform": platform.platform(),
            "gpu": _command_text(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
            "cuda": _command_text(["nvcc", "--version"]),
            "compute_capability": _command_text(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]),
        },
    }


def _dotenv_values() -> dict[str, str]:
    """Read simple dotenv values for snapshot/redaction without exporting them."""
    from dotenv import dotenv_values

    path = PROJECT_ROOT / ".env"
    if not path.is_file():
        return {}
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def _safe_settings() -> dict[str, Any]:
    """Return only explicitly permitted, non-secret settings with env precedence."""
    dotenv = _dotenv_values()
    from cuda_sft.config import Settings

    settings = Settings()
    return {key: getattr(settings, key.lower(), os.environ.get(key, dotenv.get(key)))
            for key in SAFE_ENV_KEYS}


def _secret_values() -> list[str]:
    """Return configured secret values used solely to redact subprocess output."""
    merged = {**_dotenv_values(), **os.environ}
    return sorted({part for key, value in merged.items() if SECRET_NAME_RE.search(key)
                   and not re.search(r"(?i)(max.*tokens|tokens.*max)", key)
                   for part in [value, *re.split(r"[\s,;]+", value)] if len(part) >= 6},
                  key=len, reverse=True)


def _redact(text: str, secrets: Sequence[str]) -> str:
    """Remove exact configured secrets and authorization header credentials."""
    for secret in secrets:
        text = text.replace(secret, "[REDACTED]")
    return AUTH_RE.sub(r"\1[REDACTED]", text)


def _sanitize(value: Any, secrets: Sequence[str]) -> Any:
    """Redact strings recursively while preserving JSON numeric types."""
    if isinstance(value, str):
        return _redact(value, secrets)
    if isinstance(value, dict):
        return {key: _sanitize(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, secrets) for item in value]
    return value


def _json_safe(value: Any) -> Any:
    """Coerce an artifact value into JSON primitives for report fields."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return str(value)


def _quality_fields(metadata: dict[str, Any], detail: dict[str, Any]) -> dict[str, Any]:
    """Extract serializable judge/critic quality evidence from an artifact."""
    judge = metadata.get("judge") or metadata.get("knowledge_judge")
    critic = metadata.get("critic")
    refval = metadata.get("refval")
    if not isinstance(judge, dict):
        judge = {}
    if not isinstance(critic, dict):
        critic = {}
    if not isinstance(refval, dict):
        refval = {}
    score = judge.get("quality_score", metadata.get("judge_score"))
    # Critic status was added after the original store schema. Infer it for
    # old rows so reports remain comparable across benchmark runs.
    critic_status = critic.get("status")
    if not critic_status:
        if critic.get("skipped"):
            critic_status = "skipped"
        elif any(str(item).startswith(("critic_error:", "critic_invalid_response"))
                 for item in critic.get("issues") or []):
            critic_status = "unverified"
        elif critic:
            critic_status = "verified" if critic.get("passed") else "failed"
    return _json_safe({
        "judge_score": score,
        "judge": judge,
        "critic": critic,
        "critic_status": critic_status,
        "refval": refval,
        "verified": critic_status not in {"unverified"},
        "source": "sft_metadata" if metadata else "progress_or_abandoned",
    })


def _failure_fields(
    *,
    status: str,
    error_category: str | None,
    error: Any,
    quality: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a structured failure record while retaining the raw message."""
    critic_status = quality.get("critic_status")
    refval = quality.get("refval") if isinstance(quality.get("refval"), dict) else {}
    refval_status = str(refval.get("status") or "").lower()
    if critic_status == "unverified":
        return {
            "kind": "critic_unverified",
            "category": "quality_gate",
            "message": "semantic critic did not return a verifiable result",
        }
    if critic_status == "failed":
        return {
            "kind": "critic_rejected",
            "category": "quality_gate",
            "message": "semantic critic reported blocking issues",
        }
    if refval_status in {"fail", "reference_error"}:
        return {
            "kind": "refval_" + refval_status,
            "category": "quality_gate",
            "message": str(refval.get("reason") or refval.get("evidence") or ""),
        }
    if status in {"abandoned", "crashed", "incomplete"} or error_category:
        return {
            "kind": error_category or status,
            "category": error_category or status,
            "message": str(error or ""),
        }
    return None


def pipeline_dir(module: str, pipeline: str, run_id: str) -> Path:
    """Return the isolated output directory for one pipeline."""
    return PROJECT_ROOT / "data" / f"e2e_{module}_{pipeline}_{run_id}"


def build_command(module: str, pipeline: str, run_id: str) -> list[str]:
    """Build the generation CLI argv for exactly ten fixed jobs."""
    del module, run_id
    if pipeline == "knowledge":
        return [sys.executable, str(PROJECT_ROOT / "run.py"), "--task", "knowledge",
                "--input", str(PROJECT_ROOT / "knowledge_questions.jsonl"), "--ids",
                ",".join(map(str, KNOWLEDGE_IDS)), "--quiet"]
    return [sys.executable, str(PROJECT_ROOT / "run.py"), "--task", "kernel",
            "--input", str(PROJECT_ROOT / "question.jsonl"), "--ids",
            ",".join(map(str, KERNEL_IDS)), "--dialects", pipeline,
            "--kernel-mode", "single", "--quiet"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read valid JSON-object lines, tolerating interrupted final writes."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _jsonl_issues(path: Path) -> int:
    """Count invalid nonempty JSON lines in an artifact."""
    if not path.is_file():
        return 0
    invalid = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            invalid += not isinstance(json.loads(line), dict)
        except json.JSONDecodeError:
            invalid += 1
    return invalid


def _trace_metrics(path: Path) -> dict[str, Any]:
    """Extract optional future trace timing fields without assuming a schema."""
    rows = _read_jsonl(path)
    return {
        "available": bool(rows),
        "reason": None if rows else "trace.jsonl not produced by current pipeline",
        "events": len(rows),
        "node_timings": [row for row in rows if row.get("event") == "node_timing"],
        "call_timings": [row for row in rows if row.get("event") == "llm_call"],
    }


def _log_job_metrics(log: str) -> dict[int, dict[str, Any]]:
    """Derive worker/provider, second-resolution elapsed time and compile events."""
    workers: dict[str, str] = {}
    jobs: dict[int, dict[str, Any]] = {}
    for line in log.splitlines():
        worker_match = re.search(r"worker (\d+) provider=([^\s]+)", line)
        if worker_match:
            workers[worker_match[1]] = worker_match[2]
        q_match = re.search(r"\bQ(\d+)\s", line)
        if not q_match:
            continue
        qid = int(q_match[1])
        job = jobs.setdefault(qid, {"compile_results": []})
        slot = re.search(r"\[w(\d+)\]", line)
        if slot and slot[1] in workers:
            label = workers[slot[1]]
            job.update({"provider": label.split("#")[0], "provider_label": label,
                        "worker": int(slot[1])})
        clock = re.match(r"(\d{2}):(\d{2}):(\d{2})", line)
        seconds = sum(int(clock[index]) * scale for index, scale in ((1, 3600), (2, 60), (3, 1))) \
            if clock else None
        if seconds is not None and "calling model" in line:
            job.setdefault("start_clock_seconds", seconds)
        if seconds is not None and ("saved SFT sample" in line or "abandoned" in line):
            job["end_clock_seconds"] = seconds
        compile_match = re.search(r"compile (PASS|FAIL)\b", line)
        if compile_match:
            job["compile_results"].append(compile_match[1].lower())
    for job in jobs.values():
        start = job.pop("start_clock_seconds", None)
        end = job.pop("end_clock_seconds", None)
        job["derived_elapsed_seconds"] = (end - start) % 86400 if start is not None and end is not None else None
        job["elapsed_measurement_method"] = "log clock difference, seconds precision, modulo 86400; not monotonic"
        job["compile_attempts"] = len(job["compile_results"])
    return jobs


def collect_pipeline(module: str, pipeline: str, run_id: str) -> dict[str, Any]:
    """Collect ten expected job outcomes from append-only pipeline artifacts."""
    directory = pipeline_dir(module, pipeline, run_id)
    progress = _read_jsonl(directory / "progress.jsonl")
    abandoned = _read_jsonl(directory / "abandoned.jsonl")
    samples = _read_jsonl(directory / "sft.jsonl")
    metadata = {}
    metadata_path = directory / "benchmark_runtime.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {"metadata_error": "invalid benchmark_runtime.json"}
    expected = KNOWLEDGE_IDS if pipeline == "knowledge" else KERNEL_IDS
    input_path = PROJECT_ROOT / ("knowledge_questions.jsonl" if pipeline == "knowledge" else "question.jsonl")
    input_rows = _read_jsonl(input_path)
    expected_tracks = {qid: pipeline for qid in expected}
    if pipeline == "knowledge" and len(input_rows) >= max(expected):
        from cuda_sft.tasks.classify import classify_row
        from cuda_sft.tasks.kinds import QuestionRow

        expected_tracks = {
            qid: f"knowledge:{classify_row(QuestionRow(qid, str(input_rows[qid - 1].get('question', '')), input_rows[qid - 1]), 'knowledge').topic}"
            for qid in expected
        }
    integrity_errors: list[str] = []
    artifact_stats = {}
    for filename, rows in (("progress.jsonl", progress), ("sft.jsonl", samples),
                           ("abandoned.jsonl", abandoned)):
        keys = [(row.get("id"), row.get("dialect") or
                 (row.get("metadata") or {}).get("dialect") or "cuda") for row in rows]
        duplicates = len(keys) - len(set(keys))
        extraneous = sum(qid not in expected_tracks or expected_tracks.get(qid) != track
                         for qid, track in keys)
        corrupt = _jsonl_issues(directory / filename)
        artifact_stats[filename] = {"rows": len(rows), "duplicates": duplicates,
                                    "extraneous": extraneous, "corrupt": corrupt}
        if duplicates or extraneous or corrupt:
            integrity_errors.append(f"{filename}: duplicate={duplicates}, extraneous={extraneous}, corrupt={corrupt}")
    by_id: dict[int, dict[str, Any]] = {}
    for row in progress:
        qid = row.get("id")
        if isinstance(qid, int) and qid in expected:
            if row.get("dialect", "cuda") == expected_tracks[qid]:
                by_id[qid] = row
    abandoned_by_id = {row.get("id"): row for row in abandoned}
    sample_by_id = {row.get("id"): row for row in samples}
    for qid in sample_by_id:
        if by_id.get(qid, {}).get("status") != "success":
            integrity_errors.append(f"id={qid}: sft without matching success progress")
    for qid in abandoned_by_id:
        if by_id.get(qid, {}).get("status") != "abandoned":
            integrity_errors.append(f"id={qid}: abandoned record without matching abandoned progress")
    secrets = _secret_values()
    log = (directory / "console.log").read_text(encoding="utf-8", errors="replace") \
        if (directory / "console.log").is_file() else ""
    log_metrics = _log_job_metrics(log)
    return_code = metadata.get("return_code")
    try:
        dt.datetime.fromisoformat(str(metadata.get("finished_at")).replace("Z", "+00:00"))
        valid_finished_at = True
    except ValueError:
        valid_finished_at = False
    jobs = []
    for qid in expected:
        row = by_id.get(qid)
        status = row.get("status") if row else None
        if status not in {"success", "abandoned"}:
            status = "crashed" if return_code not in (None, 0) else "incomplete"
        if status == "success" and qid not in sample_by_id:
            integrity_errors.append(f"id={qid}: success progress without sft")
            status = "incomplete"
        if status == "abandoned" and qid not in abandoned_by_id:
            integrity_errors.append(f"id={qid}: abandoned progress without abandoned record")
            status = "incomplete"
        detail = abandoned_by_id.get(qid, {}) if status == "abandoned" else sample_by_id.get(qid, {})
        sample_meta = detail.get("metadata") or {}
        if not isinstance(sample_meta, dict):
            sample_meta = {}
        error = detail.get("last_error") or detail.get("reason")
        if isinstance(error, str):
            error = _redact(error, secrets)
        reason = str(detail.get("reason") or "")
        if status == "abandoned":
            error_category = ("judge_unavailable" if "judge_unavailable" in reason else
                              "parse" if "parse" in reason else
                              "infra" if "infra" in reason or "timeout" in reason else
                              "quality_gate" if reason in {"all_candidates_failed", "knowledge_quality"}
                              else "unknown")
        else:
            error_category = ("process_crash" if status == "crashed" else
                              "missing_terminal_artifact" if status == "incomplete" else None)
        quality = _quality_fields(sample_meta, detail)
        failure = _failure_fields(
            status=status,
            error_category=error_category,
            error=error,
            quality=quality,
        )
        jobs.append({
            "id": qid, "status": status,
            "candidate": (row or {}).get("candidate", sample_meta.get("candidate")),
            "repairs": (row or {}).get("repairs", sample_meta.get("repairs")),
            "error": error,
            "error_category": error_category,
            "judge_score": sample_meta.get("judge_score"),
            "judge": sample_meta.get("knowledge_judge") or sample_meta.get("judge"),
            # Keep the legacy flat judge fields above and add structured,
            # JSON-safe quality/failure records for downstream analysis.
            "quality": quality,
            "failure": failure,
            "model": sample_meta.get("model"),
            "provider": sample_meta.get("provider") or log_metrics.get(qid, {}).get("provider"),
            "provider_unavailable_reason": None if log_metrics.get(qid, {}).get("provider") else
                "no provider evidence in current store/log",
            "log_metrics": log_metrics.get(qid, {}),
            "attempts": json.loads(_redact(json.dumps(detail.get("attempts")), secrets)),
            "ttft_seconds": None,
            "ttft_unavailable_reason": "current pipeline does not emit TTFT",
            "tokens": None,
            "tokens_unavailable_reason": "current pipeline does not emit token usage",
        })
    counts = {name: sum(job["status"] == name for job in jobs)
              for name in ("success", "abandoned", "crashed", "incomplete")}
    for filename in ("sft_ms_swift.jsonl", "sft_openrlhf.jsonl"):
        rows = _read_jsonl(directory / filename)
        corrupt = _jsonl_issues(directory / filename)
        artifact_stats[filename] = {"rows": len(rows), "corrupt": corrupt}
        if len(rows) != len(samples) or corrupt:
            integrity_errors.append(f"{filename}: row count differs from sft or corrupt rows")
    return {
        "pipeline": pipeline, "data_dir": str(directory.relative_to(PROJECT_ROOT)),
        "expected_jobs": 10, "observed_terminal_jobs": counts["success"] + counts["abandoned"],
        "complete": counts["crashed"] == 0 and counts["incomplete"] == 0 and not integrity_errors
                    and metadata.get("return_code") == 0 and valid_finished_at,
        "process_complete": metadata.get("return_code") == 0 and valid_finished_at,
        "integrity_errors": integrity_errors, "artifacts": artifact_stats,
        "counts": counts, "runtime": metadata,
        "retry_log_matches": len(re.findall(r"(?i)\bretr(?:y|ies|ying)\b", log)),
        "retry_log_matches_note": "keyword matches, not measured HTTP retries",
        "llm_call_count": None,
        "llm_call_count_unavailable_reason": "current pipeline does not emit complete per-call traces",
        "http_retry_count": None,
        "http_retry_count_unavailable_reason": "log keyword matches cannot establish HTTP retry count",
        "artifact_hashes": {filename: _sha256(directory / filename)
                            for filename in artifact_stats},
        "jobs": jobs, "trace": _trace_metrics(directory / "trace.jsonl"),
    }


def write_report(module: str, run_id: str) -> tuple[Path, Path]:
    """Write machine-readable JSON and a compact Markdown summary."""
    report_dir = PROJECT_ROOT / "data" / f"e2e_{module}_{run_id}"
    report_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = report_dir / "snapshot.json"
    if not snapshot_path.exists():
        initial_snapshot = snapshot()
        initial_snapshot["reconstructed_at"] = _utc_now()
        initial_snapshot["reconstruction_reason"] = "snapshot captured after manual baseline began"
        snapshot_path.write_text(json.dumps(initial_snapshot, indent=2) + "\n", encoding="utf-8")
    immutable_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    pipelines = [collect_pipeline(module, name, run_id) for name in PIPELINES]
    starts = [item["runtime"].get("started_at") for item in pipelines
              if item["runtime"].get("started_at")]
    finishes = [item["runtime"].get("finished_at") for item in pipelines
                if item["runtime"].get("finished_at")]
    measured = [item["runtime"].get("monotonic_wall_seconds") for item in pipelines
                if isinstance(item["runtime"].get("monotonic_wall_seconds"), (int, float))]
    report = {
        "schema_version": 1, "module": module, "run_id": run_id,
        "started_at": min(starts) if starts else None,
        "finished_at": max(finishes) if len(finishes) == len(PIPELINES) else None,
        "monotonic_wall_seconds": None,
        "sum_pipeline_monotonic_seconds": sum(measured) if measured else None,
        "report_generated_at": _utc_now(),
        "snapshot": immutable_snapshot, "pipelines": pipelines,
    }
    json_path = report_dir / "benchmark.json"
    md_path = report_dir / "benchmark.md"
    json_path.write_text(json.dumps(_sanitize(report, _secret_values()), indent=2,
                                   ensure_ascii=False) + "\n", encoding="utf-8")
    def percentile(values: list[float], fraction: float) -> float | None:
        """Return nearest-rank percentile for compact human reporting."""
        if not values:
            return None
        ordered = sorted(values)
        return ordered[max(0, min(len(ordered) - 1, int((len(ordered) * fraction + 0.999999) - 1)))]

    lines = [f"# M0 baseline: {module} / {run_id}", "",
             "| Pipeline | Success | Abandoned | Crashed | Incomplete | Complete | Wall seconds | p50 derived | p95 derived | First pass | Compile fails |",
             "|---|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|"]
    for item in pipelines:
        c = item["counts"]
        elapsed = [job["log_metrics"].get("derived_elapsed_seconds") for job in item["jobs"]
                   if isinstance(job["log_metrics"].get("derived_elapsed_seconds"), (int, float))]
        first_pass = sum(job.get("candidate") == 1 and job.get("repairs") == 0
                         for job in item["jobs"])
        compile_fails = sum(job["log_metrics"].get("compile_results", []).count("fail")
                            for job in item["jobs"])
        runtime = item["runtime"]
        wall = runtime.get("monotonic_wall_seconds")
        wall_note = "measured"
        if wall is None and runtime.get("derived_wall_seconds") is not None:
            wall, wall_note = runtime["derived_wall_seconds"], "derived"
        wall_text = f"{wall:.1f} ({wall_note})" if isinstance(wall, (int, float)) else "null"
        p50, p95 = percentile(elapsed, 0.50), percentile(elapsed, 0.95)
        lines.append(f"| {item['pipeline']} | {c['success']} | {c['abandoned']} | "
                     f"{c['crashed']} | {c['incomplete']} | {'yes' if item['complete'] else 'no'} | "
                     f"{wall_text} | {p50 if p50 is not None else 'null'} | "
                     f"{p95 if p95 is not None else 'null'} | {first_pass}/10 | {compile_fails} |")
    lines += ["", "Wall time is subprocess monotonic measurement except CUDA, whose derived value uses "
              "the manual start and console last-write time. Per-job times are log-clock differences "
              "at one-second precision and are not monotonic.", "",
              "| Pipeline/ID | Status | Provider | Candidate | Repairs | Derived seconds | Judge | Compile | Error |",
              "|---|---|---|---:|---:|---:|---:|---|---|"]
    for item in pipelines:
        for job in item["jobs"]:
            lm = job["log_metrics"]
            compile_text = ",".join(lm.get("compile_results", [])) or "-"
            error = str(job.get("error") or "-").replace("|", "\\|").replace("\n", " ")[:160]
            lines.append(f"| {item['pipeline']}/{job['id']} | {job['status']} | "
                         f"{job.get('provider') or '-'} | {job.get('candidate') or '-'} | "
                         f"{job.get('repairs') if job.get('repairs') is not None else '-'} | "
                         f"{lm.get('derived_elapsed_seconds') if lm.get('derived_elapsed_seconds') is not None else '-'} | "
                         f"{job.get('judge_score') if job.get('judge_score') is not None else '-'} | "
                         f"{compile_text} | {error} |")
    lines += ["", "API call count, HTTP retry count, TTFT, and token usage are null because the current "
              "pipeline does not emit complete measurements. Retry keyword matches are diagnostic only. "
              "Optional trace.jsonl node/call timings are consumed when present.", ""]
    md_text = _redact("\n".join(lines), _secret_values())
    md_path.write_text(md_text, encoding="utf-8")
    return json_path, md_path


def run_pipeline(module: str, pipeline: str, run_id: str) -> int:
    """Run one isolated pipeline and persist subprocess runtime metadata."""
    directory = pipeline_dir(module, pipeline, run_id)
    if any((directory / name).exists() for name in
           ("progress.jsonl", "sft.jsonl", "abandoned.jsonl", "benchmark_runtime.json")):
        raise ValueError(f"Refusing to rerun existing benchmark directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    command = build_command(module, pipeline, run_id) + ["--data-dir", str(directory)]
    work_dir = PROJECT_ROOT / "work" / directory.name
    environment = os.environ.copy()
    environment["WORK_DIR"] = str(work_dir)
    if pipeline != "knowledge":
        environment["KERNEL_DIALECT"] = pipeline
    started_at, start = _utc_now(), time.monotonic()
    runtime = {"command": command, "started_at": started_at, "finished_at": None,
               "return_code": None, "snapshot": snapshot()}
    (directory / "benchmark_runtime.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
    secrets = _secret_values()
    return_code = 1
    process = None
    try:
        with (directory / "console.log").open("a", encoding="utf-8") as console:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace", bufsize=1,
                                       start_new_session=True)
            assert process.stdout is not None
            for line in process.stdout:
                console.write(_redact(line, secrets))
                console.flush()
            return_code = process.wait()
    except KeyboardInterrupt:
        return_code = 130
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        runtime.update({"command": command, "started_at": started_at, "finished_at": _utc_now(),
                   "monotonic_wall_seconds": time.monotonic() - start,
                   "return_code": return_code, "work_dir": str(work_dir.relative_to(PROJECT_ROOT))})
        (directory / "benchmark_runtime.json").write_text(
            json.dumps(runtime, indent=2) + "\n", encoding="utf-8")
        write_report(module, run_id)
    return return_code


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark command-line parser."""
    parser = argparse.ArgumentParser(description="Run the fixed 50-job M0 benchmark")
    parser.add_argument("--module", default="baseline")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pipeline", action="append", choices=PIPELINES,
                        help="pipeline to run; repeat for multiple (default: all)")
    parser.add_argument("--report-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run selected pipelines sequentially and always refresh reports."""
    args = build_parser().parse_args(argv)
    started_at, start = _utc_now(), time.monotonic()
    return_code = 0
    if not args.report_only:
        report_dir = PROJECT_ROOT / "data" / f"e2e_{args.module}_{args.run_id}"
        snapshot_path = report_dir / "snapshot.json"
        if not snapshot_path.exists() and not any(
                pipeline_dir(args.module, name, args.run_id).exists() for name in PIPELINES):
            report_dir.mkdir(parents=True, exist_ok=True)
            initial_snapshot = snapshot()
            initial_snapshot["captured_at"] = started_at
            snapshot_path.write_text(json.dumps(initial_snapshot, indent=2) + "\n", encoding="utf-8")
        for pipeline in args.pipeline or PIPELINES:
            result_code = run_pipeline(args.module, pipeline, args.run_id)
            return_code = max(return_code, 1 if result_code < 0 else result_code)
    json_path, md_path = write_report(args.module, args.run_id)
    print(json_path.relative_to(PROJECT_ROOT))
    print(md_path.relative_to(PROJECT_ROOT))
    incomplete = any(not collect_pipeline(args.module, pipeline, args.run_id)["complete"]
                     for pipeline in PIPELINES)
    return max(return_code, int(incomplete))


if __name__ == "__main__":
    raise SystemExit(main())
