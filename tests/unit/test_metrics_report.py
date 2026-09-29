"""Metric definitions use trace events with explicit denominators."""

import json

import pytest

from cuda_sft.observability.report import compute, main


def test_metric_denominators_and_cost() -> None:
    events = [
        {
            "event": "job.start",
            "ts": "2026-01-01T00:00:00+00:00",
            "job_key": "a:cuda",
            "question_hash": "a",
        },
        {
            "event": "llm.call",
            "ts": "2026-01-01T00:00:01+00:00",
            "job_key": "a:cuda",
            "role": "generator",
            "input_tokens": 10,
            "output_tokens": 5,
            "cost_usd": 0.03,
        },
        {
            "event": "llm.call",
            "ts": "2026-01-01T00:00:02+00:00",
            "job_key": "a:cuda",
            "role": "refval_extract",
            "input_tokens": 20,
            "output_tokens": 10,
            "cost_usd": 0.02,
        },
        {
            "event": "job.end",
            "ts": "2026-01-01T00:00:03+00:00",
            "job_key": "a:cuda",
            "status": "success",
            "release_tier": "strict",
            "elapsed_s": 3,
        },
        {
            "event": "job.end",
            "ts": "2026-01-01T00:00:04+00:00",
            "job_key": "b:cuda",
            "status": "abandoned",
            "release_tier": "quarantine",
            "elapsed_s": 4,
        },
    ]
    metrics = compute(events)
    assert metrics["success_rate"] == 0.5
    assert metrics["strict_yield"] == 0.5
    assert metrics["llm_calls_per_success"] == 2
    assert metrics["refval_extract_calls_per_job"] == 0.5
    assert metrics["input_tokens_per_success"] == 30
    assert metrics["cost_per_success"] == 0.05
    assert metrics["job_latency_p50"] == 3.5
    assert metrics["throughput_jobs_per_hour"] == 1800.0
    assert metrics["gpu_utilization"] is None
    assert metrics["false_accept_taskspec"] is None
    assert metrics["false_accept_legacy"] is None
    assert metrics["mutation_score"] is None
    assert metrics["cot_consistency_pass"] is None


def test_commit_outcome_overrides_compute_success() -> None:
    events = [
        {"event": "job.start", "ts": "2026-01-01T00:00:00+00:00"},
        {"event": "job.end", "status": "success", "release_tier": "strict", "elapsed_s": 2},
        {"event": "job.commit", "ts": "2026-01-01T00:00:03+00:00", "status": "failed", "release_tier": "quarantine", "elapsed_s": 3},
    ]
    metrics = compute(events)
    assert metrics["outcome_source"] == "commit"
    assert metrics["computed_jobs"] == 1
    assert metrics["jobs"] == 1
    assert metrics["success"] == 0
    assert metrics["strict_yield"] == 0


def test_infra_repair_requires_immediately_preceding_gate() -> None:
    def event(name, node, **fields):
        return {"event": name, "node": node, "job_key": "q1:cuda", "candidate": 1, **fields}

    events = [
        event("node.end", "compile", outcome={"gate": "compile", "owner": "INFRA"}),
        event("node.end", "select", outcome={"route": "repair"}),
        event("node.start", "repair"),
        event("node.end", "refval", outcome={"gate": "refval", "owner": "ORACLE"}),
        event("node.start", "repair"),
        event("node.end", "compile", outcome={"gate": "compile", "owner": "INFRA"}),
        event("node.error", "compile"),
        event("node.start", "repair"),
    ]
    assert compute(events)["infra_triggered_repairs"] == 1


def test_cost_is_unknown_when_any_call_has_no_price() -> None:
    events = [
        {"event": "job.end", "status": "success"},
        {"event": "llm.call", "cost_usd": 0.01},
        {"event": "llm.call", "cost_usd": None},
    ]
    assert compute(events)["cost_per_success"] is None


def test_overlapping_worker_jobs_use_global_wall_clock() -> None:
    events = []
    for worker, start, end in (("a", 0, 10), ("b", 2, 12), ("c", 4, 14)):
        events.extend(
            [
                {
                    "event": "job.start",
                    "run_id": worker,
                    "ts": f"2026-01-01T00:00:{start:02d}+00:00",
                },
                {
                    "event": "job.end",
                    "run_id": worker,
                    "ts": f"2026-01-01T00:00:{end:02d}+00:00",
                    "status": "success",
                },
            ]
        )
    assert compute(events)["throughput_jobs_per_hour"] == round(3 / (14 / 3600), 2)


@pytest.mark.parametrize("case", ["missing", "empty_dir", "empty_file", "bad_json", "no_jobs"])
def test_cli_rejects_invalid_or_incomplete_trace(tmp_path, case: str) -> None:
    trace_dir = tmp_path / "trace"
    if case != "missing":
        trace_dir.mkdir()
    if case in {"empty_file", "bad_json", "no_jobs"}:
        content = {
            "empty_file": "",
            "bad_json": '{"event": "job.end", "status": "success"}\n{broken\n',
            "no_jobs": '{"event": "job.start"}\n',
        }[case]
        (trace_dir / "trace-20260101.jsonl").write_text(content, encoding="utf-8")
    out = tmp_path / "report"
    with pytest.raises((FileNotFoundError, ValueError)):
        main([str(trace_dir), "--out", str(out)])
    assert not out.exists()


def test_cli_writes_reports_and_compare(tmp_path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    (trace_dir / "trace-20260101.jsonl").write_text(
        "\n".join(
            json.dumps(e)
            for e in [
                {"event": "job.start", "ts": "2026-01-01T00:00:00+00:00"},
                {
                    "event": "job.end",
                    "ts": "2026-01-01T00:00:02+00:00",
                    "status": "success",
                    "release_tier": "strict",
                    "elapsed_s": 2,
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"success_rate": 0.5}), encoding="utf-8")
    out = tmp_path / "report"
    assert main([str(trace_dir), "--out", str(out), "--compare", str(old)]) == 0
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["success_rate"] == 1
    assert "+100.0%" in (out / "report.md").read_text(encoding="utf-8")
