#!/usr/bin/env python3
"""Compute run metrics from trace JSONL (T0.9). Definitions: 05_testing/09_metrics_gates.md §1.

Usage:
    python scripts/metrics_report.py runs/v0/trace --out runs/v0/report [--compare runs/prev/report/report.json]
T0.9 moves the computation into src/cuda_sft/observability/report.py and keeps this CLI as a thin wrapper.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_events(trace_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not trace_dir.exists():
        raise FileNotFoundError(f"trace path does not exist: {trace_dir}")
    files = [trace_dir] if trace_dir.is_file() else sorted(trace_dir.glob("trace-*.jsonl"))
    if not files:
        raise ValueError(f"no trace JSONL files found in {trace_dir}")
    for f in files:
        for line_number, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid trace JSON at {f}:{line_number}") from exc
                if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                    raise ValueError(f"invalid trace event at {f}:{line_number}")
                events.append(event)
    if not events:
        raise ValueError(f"trace contains no events: {trace_dir}")
    return events


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vs = sorted(values)
    position = q * (len(vs) - 1)
    lower = int(position)
    upper = min(lower + 1, len(vs) - 1)
    return vs[lower] + (vs[upper] - vs[lower]) * (position - lower)


def _wall_hours(events: list[dict[str, Any]]) -> float:
    times: list[datetime] = []
    for event in events:
        if event.get("event") not in {"job.start", "job.end", "job.commit"} or not event.get("ts"):
            continue
        times.append(datetime.fromisoformat(event["ts"]))
    return (max(times) - min(times)).total_seconds() / 3600.0 if len(times) > 1 else 0.0


def compute(events: list[dict[str, Any]]) -> dict[str, Any]:
    commits = [e for e in events if e.get("event") == "job.commit"]
    jobs = commits or [e for e in events if e.get("event") == "job.end"]
    llm = [e for e in events if e.get("event") == "llm.call"]
    n_jobs = len(jobs)
    n_success = sum(1 for j in jobs if j.get("status") == "success")
    strict = sum(1 for j in jobs if j.get("release_tier") in {"strict", "strict_perf"})
    in_tok = sum(int(e.get("input_tokens") or 0) for e in llm)
    out_tok = sum(int(e.get("output_tokens") or 0) for e in llm)
    cost = [float(e["cost_usd"]) for e in llm if e.get("cost_usd") is not None]
    roles = Counter(e.get("role") for e in llm)

    # infra_triggered_repairs: node.end(gate in compile/refval, owner ORACLE/INFRA) followed by node.start(repair)
    by_thread: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for e in events:
        if e.get("event") in {"node.end", "node.start", "node.error"}:
            by_thread[(e.get("run_id"), e.get("job_key"), e.get("candidate"))].append(e)
    infra_repairs = 0
    for seq in by_thread.values():
        previous_end: dict[str, Any] | None = None
        for e in seq:
            if e["event"] == "node.end":
                previous_end = e
            elif e["event"] == "node.start":
                outcome = (previous_end or {}).get("outcome") or {}
                if (
                    e.get("node") == "repair"
                    and outcome.get("gate") in {"compile", "refval"}
                    and outcome.get("owner") in {"ORACLE", "INFRA"}
                ):
                    infra_repairs += 1
                previous_end = None
            else:
                previous_end = None

    hours = _wall_hours(events)
    lat = [float(j.get("elapsed_s") or 0) for j in jobs]
    writes = [
        float(e.get("elapsed_s") or 0) * 1000 for e in events if e.get("event") == "store.write"
    ]
    finished = sum(1 for j in jobs if j.get("status") in {"success", "abandoned"})
    return {
        "jobs": n_jobs,
        "outcome_source": "commit" if commits else "legacy_compute",
        "computed_jobs": sum(e.get("event") == "job.end" for e in events),
        "success": n_success,
        "success_rate": round(n_success / n_jobs, 4) if n_jobs else None,
        "strict_yield": round(strict / n_jobs, 4) if n_jobs else None,
        "llm_calls": len(llm),
        "llm_calls_by_role": dict(roles),
        "llm_calls_per_success": round(len(llm) / n_success, 3) if n_success else None,
        "input_tokens_per_success": round(in_tok / n_success, 1) if n_success else None,
        "output_tokens_per_success": round(out_tok / n_success, 1) if n_success else None,
        "cost_per_success": round(sum(cost) / n_success, 4)
        if n_success and llm and len(cost) == len(llm)
        else None,
        "infra_triggered_repairs": infra_repairs,
        "refval_extract_calls_per_job": round(roles.get("refval_extract", 0) / n_jobs, 3)
        if n_jobs
        else None,
        "throughput_jobs_per_hour": round(finished / hours, 2) if hours > 0 else None,
        "job_latency_p50": _pct(lat, 0.5),
        "job_latency_p95": _pct(lat, 0.95),
        "store_write_p99_ms": _pct(writes, 0.99),
        "gpu_utilization": None,
        "false_accept_taskspec": None,
        "false_accept_legacy": None,
        "mutation_score": None,
        "cot_consistency_pass": None,
        "abandon_reasons": dict(
            Counter(
                j.get("abandon_reason") or "unknown" for j in jobs if j.get("status") == "abandoned"
            )
        ),
        "mean_job_latency": round(statistics.mean(lat), 2) if lat else None,
    }


def to_markdown(m: dict[str, Any], prev: dict[str, Any] | None) -> str:
    rows = ["| 指标 | 本次 | 对比 | 变化 |", "|---|---|---|---|"]
    for k, v in m.items():
        if isinstance(v, dict):
            continue
        p = (prev or {}).get(k)
        delta = ""
        if isinstance(v, (int, float)) and isinstance(p, (int, float)) and p:
            delta = f"{(v - p) / p * 100:+.1f}%"
        rows.append(f"| {k} | {v} | {'' if p is None else p} | {delta} |")
    return "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path)
    args = ap.parse_args(argv)
    events = load_events(args.trace)
    if not any(event["event"] in {"job.end", "job.commit"} for event in events):
        raise ValueError(f"trace contains no completed jobs: {args.trace}")
    metrics = compute(events)
    prev = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else None
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "report.md").write_text(to_markdown(metrics, prev), encoding="utf-8")
    print(to_markdown(metrics, prev))
    return 0


if __name__ == "__main__":
    sys.exit(main())
