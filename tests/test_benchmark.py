"""Tests for the fixed M0 benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path

from cuda_sft import benchmark


def test_build_command_has_fixed_ten_jobs() -> None:
    """Kernel and knowledge commands select their immutable ID sets."""
    cuda = benchmark.build_command("baseline", "cuda", "run")
    knowledge = benchmark.build_command("baseline", "knowledge", "run")
    assert cuda[cuda.index("--ids") + 1] == "1,5,8,20,26,27,38,42,75,90"
    assert knowledge[knowledge.index("--ids") + 1] == "1,2,3,4,5,6,7,8,9,10"
    assert cuda[cuda.index("--kernel-mode") + 1] == "single"


def test_collect_marks_missing_results_incomplete(tmp_path: Path, monkeypatch) -> None:
    """A report always contains ten jobs and preserves missing outcomes."""
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    directory = benchmark.pipeline_dir("baseline", "cuda", "r1")
    directory.mkdir(parents=True)
    (directory / "progress.jsonl").write_text(
        json.dumps({"id": 1, "dialect": "cuda", "status": "success", "candidate": 2}) + "\n",
        encoding="utf-8",
    )
    result = benchmark.collect_pipeline("baseline", "cuda", "r1")
    assert len(result["jobs"]) == 10
    assert result["counts"] == {"success": 0, "abandoned": 0, "crashed": 0, "incomplete": 10}
    assert result["integrity_errors"] == ["id=1: success progress without sft"]
    assert result["jobs"][0]["tokens"] is None


def test_nonzero_process_marks_missing_results_crashed(tmp_path: Path, monkeypatch) -> None:
    """Missing artifacts are crashed when the subprocess exited nonzero."""
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    directory = benchmark.pipeline_dir("baseline", "triton", "r2")
    directory.mkdir(parents=True)
    (directory / "benchmark_runtime.json").write_text('{"return_code": 1}', encoding="utf-8")
    result = benchmark.collect_pipeline("baseline", "triton", "r2")
    assert result["counts"]["crashed"] == 10


def test_redaction_preserves_json_types(monkeypatch) -> None:
    """Report sanitization preserves numbers and replaces nested credentials."""
    monkeypatch.setenv("NVIDIA_API_KEY", "secret-one,secret-two")
    monkeypatch.setenv("MAX_TOKENS", "50000")
    secrets = benchmark._secret_values()
    assert "secret-one" in secrets and "secret-two" in secrets
    assert "50000" not in secrets
    assert benchmark._sanitize({"tokens": 50000, "nested": ["secret-one"]}, secrets) == {
        "tokens": 50000,
        "nested": ["[REDACTED]"],
    }


def test_safe_settings_env_only(monkeypatch) -> None:
    """Environment-only settings do not evaluate a missing dotenv fallback."""
    monkeypatch.setattr(benchmark, "_dotenv_values", lambda: {})
    monkeypatch.setenv("WORKERS", "3")
    assert benchmark._safe_settings()["WORKERS"] == 3


def test_duplicate_and_wrong_track_are_integrity_errors(tmp_path: Path, monkeypatch) -> None:
    """Duplicate and unrelated terminal rows cannot satisfy completeness."""
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    directory = benchmark.pipeline_dir("baseline", "cuda", "bad")
    directory.mkdir(parents=True)
    rows = [{"id": 1, "dialect": "triton", "status": "success"}] * 2
    (directory / "progress.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\ninvalid\n", encoding="utf-8"
    )
    result = benchmark.collect_pipeline("baseline", "cuda", "bad")
    assert result["artifacts"]["progress.jsonl"] == {
        "rows": 2,
        "duplicates": 1,
        "extraneous": 2,
        "corrupt": 1,
    }
    assert not result["complete"]


def test_existing_results_refuse_rerun(tmp_path: Path, monkeypatch) -> None:
    """An existing result directory cannot silently become a resume run."""
    import pytest

    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    directory = benchmark.pipeline_dir("baseline", "cuda", "reuse")
    directory.mkdir(parents=True)
    (directory / "progress.jsonl").touch()
    with pytest.raises(ValueError, match="Refusing"):
        benchmark.run_pipeline("baseline", "cuda", "reuse")


def test_report_only_incomplete_is_nonzero(tmp_path: Path, monkeypatch) -> None:
    """Reporting a missing benchmark never returns a successful CLI status."""
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "snapshot", lambda: {})
    assert benchmark.main(["--run-id", "missing", "--report-only"]) == 1


def test_negative_subprocess_exit_is_failure(monkeypatch, tmp_path: Path) -> None:
    """A signal-killed subprocess cannot be swallowed by max(0, exitcode)."""
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "run_pipeline", lambda *args: -9)
    monkeypatch.setattr(benchmark, "collect_pipeline", lambda *args: {"complete": True})
    monkeypatch.setattr(
        benchmark,
        "write_report",
        lambda *args: (tmp_path / "benchmark.json", tmp_path / "benchmark.md"),
    )
    assert benchmark.main(["--run-id", "kill", "--pipeline", "cuda"]) == 1


def test_report_rebuild_preserves_runtime_evidence(tmp_path: Path, monkeypatch) -> None:
    """Report generation derives benchmark timing only from pipeline runtime files."""
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "snapshot", lambda: {})
    for index, pipeline in enumerate(benchmark.PIPELINES):
        directory = benchmark.pipeline_dir("baseline", pipeline, "timing")
        directory.mkdir(parents=True)
        (directory / "benchmark_runtime.json").write_text(
            json.dumps(
                {
                    "started_at": f"2026-09-17T15:{index:02d}:00+00:00",
                    "finished_at": f"2026-09-17T15:{index + 1:02d}:00+00:00",
                    "return_code": 0,
                    "monotonic_wall_seconds": 60,
                }
            ),
            encoding="utf-8",
        )
    json_path, _ = benchmark.write_report("baseline", "timing")
    first = json.loads(json_path.read_text())
    json_path, _ = benchmark.write_report("baseline", "timing")
    second = json.loads(json_path.read_text())
    assert first["started_at"] == second["started_at"] == "2026-09-17T15:00:00+00:00"
    assert first["finished_at"] == second["finished_at"] == "2026-09-17T15:05:00+00:00"
    assert second["monotonic_wall_seconds"] is None
    assert second["sum_pipeline_monotonic_seconds"] == 300


def test_terminal_artifacts_require_finished_zero_exit(tmp_path: Path, monkeypatch) -> None:
    """Ten abandoned jobs are complete only with a finished successful process."""
    monkeypatch.setattr(benchmark, "PROJECT_ROOT", tmp_path)
    directory = benchmark.pipeline_dir("baseline", "cuda", "terminal")
    directory.mkdir(parents=True)
    progress = [
        {"id": qid, "dialect": "cuda", "status": "abandoned"} for qid in benchmark.KERNEL_IDS
    ]
    abandoned = [
        {"id": qid, "dialect": "cuda", "reason": "knowledge_judge_unavailable"}
        for qid in benchmark.KERNEL_IDS
    ]
    for filename, rows in (("progress.jsonl", progress), ("abandoned.jsonl", abandoned)):
        (directory / filename).write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
    assert not benchmark.collect_pipeline("baseline", "cuda", "terminal")["complete"]
    runtime_path = directory / "benchmark_runtime.json"
    runtime_path.write_text(
        '{"return_code":1,"finished_at":"2026-09-17T15:00:00Z"}', encoding="utf-8"
    )
    assert not benchmark.collect_pipeline("baseline", "cuda", "terminal")["complete"]
    runtime_path.write_text(
        '{"return_code":0,"finished_at":"2026-09-17T15:00:00Z"}', encoding="utf-8"
    )
    result = benchmark.collect_pipeline("baseline", "cuda", "terminal")
    assert result["complete"]
    assert result["jobs"][0]["error_category"] == "judge_unavailable"


def test_log_metrics_provider_compile_and_midnight() -> None:
    """Log evidence maps providers and handles second-resolution midnight wrap."""
    log = "\n".join(
        [
            "23:59:50 INFO [w2] cuda_sft: worker 2 provider=nvidia#2 model=m",
            "23:59:55 INFO [w2] cuda_sft.graph: [Q1 cuda candidate=1] calling model",
            "[Q1 cuda] compile FAIL",
            "[Q1 cuda] compile PASS",
            "00:00:10 INFO [w2] cuda_sft.graph: Q1 cuda saved SFT sample",
        ]
    )
    result = benchmark._log_job_metrics(log)[1]
    assert result["provider"] == "nvidia"
    assert result["derived_elapsed_seconds"] == 15
    assert result["compile_results"] == ["fail", "pass"]
