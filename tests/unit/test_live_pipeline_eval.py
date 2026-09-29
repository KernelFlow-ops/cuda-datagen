"""Live acceptance reporting must not turn incomplete work into a pass."""

from __future__ import annotations

import json
from pathlib import Path

from cuda_sft.refval.spec import RefvalReport
from scripts.cross_dialect_oracle import mutation_caught
from scripts.live_agent_probes import _probe_failures
from scripts.live_pipeline_eval import _trace_summary


def test_trace_summary_measures_real_candidate_and_key_overlap(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    events = [
        {"event": "llm.call", "ts": "2026-09-25T00:00:10+00:00", "elapsed_s": 10,
         "role": "generator", "provider": "openai", "model": "model", "key_label": "openai#1", "ok": True},
        {"event": "llm.call", "ts": "2026-09-25T00:00:12+00:00", "elapsed_s": 10,
         "role": "generator", "provider": "openai", "model": "model", "key_label": "openai#1", "ok": True},
        {"event": "node.end", "ts": "2026-09-25T00:00:10+00:00", "elapsed_s": 10,
         "node": "generate", "job_key": "q1:cuda", "candidate": 1},
        {"event": "node.end", "ts": "2026-09-25T00:00:12+00:00", "elapsed_s": 10,
         "node": "generate", "job_key": "q1:cuda", "candidate": 2},
    ]
    (trace_dir / "trace-test.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    summary = _trace_summary(tmp_path)
    assert summary["candidate_overlap"] is True
    assert summary["max_llm_inflight_by_key"] == {"openai#1": 2}


def test_probe_check_requires_correct_role_for_each_fault() -> None:
    records = {
        "compile": {"status": "success", "compile_calls": 2, "refval_calls": 1, "repairs": 1},
        "numeric": {"status": "success", "compile_calls": 2, "refval_calls": 2, "repairs": 1},
        "semantic": {"status": "success", "compile_calls": 2, "refval_calls": 2,
                     "critic_calls": 2, "repairs": 1},
        "knowledge": {"status": "success", "gate_calls": 2},
    }
    by_job = {
        "q101:cuda": ["generator", "repair.compile"],
        "q102:cuda": ["generator", "repair.numeric"],
        "q103:cuda": ["generator", "repair.semantic"],
        "q104:knowledge:execution": ["knowledge_generator", "knowledge_repair"],
    }
    assert _probe_failures(records, by_job) == []
    by_job["q102:cuda"] = ["generator", "repair.compile"]
    assert "numeric: missing live repair.numeric call" in _probe_failures(records, by_job)


def test_mutant_count_requires_numeric_evidence() -> None:
    assert not mutation_caught(RefvalReport(
        status="fail", dialect="tilelang", error_class="launch_runtime", cases_run=0,
    ))
    assert not mutation_caught(RefvalReport(
        status="fail", dialect="cuda", error_class="numeric_mismatch", cases_run=0,
    ))
    assert mutation_caught(RefvalReport(
        status="fail", dialect="cuda", error_class="numeric_mismatch", cases_run=3,
    ))
