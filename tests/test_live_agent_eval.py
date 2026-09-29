"""Offline contract tests for the opt-in live evaluation command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuda_sft.config import Settings
from scripts import live_agent_eval as live


class _Completion:
    text = "ok"
    reasoning = ""


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def stream_completion(self, **kwargs: object) -> _Completion:
        self.calls += 1
        return _Completion()


def _events(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_fixed_matrix_has_required_strata() -> None:
    tasks = live.FIXED_TASKS
    assert len(tasks) == 10
    assert sum(t.kind == "knowledge" for t in tasks) == 2
    assert sum(t.dialect == "cuda" and t.difficulty == "simple" for t in tasks) == 2
    assert sum(t.dialect == "cuda" and t.task_id.startswith("cuda_reduction") for t in tasks) == 1
    assert sum(t.dialect == "cuda" and t.task_id.startswith("cuda_transpose") for t in tasks) == 1
    assert sum(t.dialect == "cuda" and t.difficulty == "hard" for t in tasks) == 1
    assert sum(t.dialect == "cutlass" and t.difficulty == "hard" for t in tasks) == 1
    assert sum(t.dialect == "triton" for t in tasks) == 1
    assert sum(t.dialect == "tilelang" for t in tasks) == 1


def test_budget_journal_records_before_after_and_exhaustion(tmp_path: Path) -> None:
    journal_path = tmp_path / "journal.jsonl"
    budget = live.BudgetClient(_Client(), 1, live.Journal(journal_path))
    with budget.context(component="generate", task_id="q1"):
        assert (
            budget.stream_completion(messages=[], system="", temperature=0, print_stream=False).text
            == "ok"
        )
        with pytest.raises(live._BudgetExceeded):
            budget.stream_completion(messages=[], system="", temperature=0, print_stream=False)
    events = _events(journal_path)
    assert [event["event"] for event in events] == [
        "request_before",
        "request_after",
        "budget_exhausted",
    ]
    assert budget.attempted == 1
    assert budget.completed == 1
    assert budget.budget_exhausted == 1


def test_budget_journal_records_failed_request(tmp_path: Path) -> None:
    class Failing:
        def stream_completion(self, **kwargs: object) -> _Completion:
            raise TimeoutError("provider timeout")

    path = tmp_path / "journal.jsonl"
    budget = live.BudgetClient(Failing(), 2, live.Journal(path))
    with pytest.raises(TimeoutError):
        budget.stream_completion(messages=[], system="", temperature=0, print_stream=False)
    events = _events(path)
    assert events[-1]["event"] == "request_after"
    assert events[-1]["status"] == "error"
    assert events[-1]["error_category"] == "timeout"
    assert budget.failed == 1


def test_refval_without_oracle_is_explicitly_not_evaluated() -> None:
    called = False

    def forbidden(**kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("refval runner must not be called without an oracle")

    task = live.FIXED_TASKS[0]
    result = live._run_refval_component(
        task,
        "code",
        object(),
        budget=live.BudgetClient(_Client(), 1),
        runner=forbidden,
    )
    assert result == {"status": "not_evaluated", "reason": "oracle_not_injected"}
    assert called is False


def test_cot_consistency_rejects_code_leak() -> None:
    assert live.check_cot_consistency("1. Mapping\n2. Bounds", "kernel")["status"] == "pass"
    leaked = live.check_cot_consistency("```cuda\n__global__ void k() {}\n```", "kernel")
    assert leaked["status"] == "fail"
    assert leaked["reason"] == "code_leak"


def test_strict_rejects_not_evaluated_components() -> None:
    record = {
        "components": {
            "generate": {"status": "pass"},
            "compile": {"status": "not_evaluated"},
        }
    }
    assert live._strict_ok([record], {"generate", "compile"}) is False
    record["components"]["compile"] = {"status": "not_applicable"}
    assert live._strict_ok([record], {"generate", "compile"}) is True


def test_thinking_only_response_is_not_an_answer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class ThinkingOnly:
        def stream_completion(self, **_kwargs: object) -> object:
            return SimpleNamespace(text="", reasoning="internal thinking")

    monkeypatch.setattr(
        live, "get_settings", lambda: Settings(llm_provider="openai", openai_api_key="test")
    )
    monkeypatch.setattr(live, "get_llm_client", lambda _settings: ThinkingOnly())
    report = tmp_path / "report.json"
    args = live.build_parser().parse_args(
        [
            "--smoke", "--max-calls", "1", "--candidate-pool", "1",
            "--skip-compile", "--agent-components", "generate", "--strict",
            "--journal", str(tmp_path / "journal.jsonl"), "--report", str(report),
        ]
    )
    assert live.run(args) == 1
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["records"][0]["components"]["generate"] == {
        "status": "fail", "reason": "empty_visible_answer"
    }
