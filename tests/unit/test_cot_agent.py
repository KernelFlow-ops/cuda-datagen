"""Selected-candidate CoT modes and consistency fallbacks."""

from __future__ import annotations

import pytest

from cuda_sft.config import Settings
from cuda_sft.cot import CotAgent
from cuda_sft.llm import LLMCompletion
from cuda_sft.prompt import (
    COT_NO_REPAIR_RULES_EN,
    COT_NO_REPAIR_RULES_ZH,
    build_cot_user_prompt,
    cot_system_for,
)

VALID_COT = (
    "1. Problem restatement\nThe input contains two arrays and the output has the same length.\n"
    "2. Algorithm\nEach output is the sum of one corresponding input pair.\n"
    "3. Thread/block mapping\nEach thread computes one output index independently.\n"
    "4. Memory and sync\nInputs and outputs use global memory; no shared memory is needed.\n"
    "5. Bounds and edge cases\nA bounds guard handles the final partial block.\n"
    "6. Implementation checklist\nUse the selected host entry and a bounded kernel launch."
)


class SequenceClient:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.calls: list[dict] = []

    def stream_completion(self, **kwargs) -> LLMCompletion:
        self.calls.append(kwargs)
        return LLMCompletion(self.texts.pop(0))


def state(*, mode: str = "polish", raw: str = "teacher reasoning") -> dict:
    return {
        "question_id": 7,
        "question": "vector add",
        "dialect": "cuda",
        "code": "stale_code",
        "raw_reasoning": raw,
        "compile_error": "secret compile log",
        "repair_idx": 9,
        "candidate_idx": 9,
        "cot_mode": mode,
        "selected": {
            "code": "__global__ void selected_kernel() {}",
            "candidate": 2,
            "repairs": 1,
            "judge": {},
        },
    }


def settings(**kwargs) -> Settings:
    return Settings(
        cot_agent_enabled=True,
        cot_on_agent_fail="raw",
        cot_on_empty="empty",
        cot_max_chars=8000,
        **kwargs,
    )


def test_refine_requires_selected() -> None:
    with pytest.raises(RuntimeError, match="selected"):
        CotAgent(settings(), SequenceClient(VALID_COT)).refine({"question": "add"})


def test_drop_never_calls_editor() -> None:
    client = SequenceClient(VALID_COT)
    result = CotAgent(settings(), client).refine(state(mode="drop", raw="repair error"))
    assert (result.cot, result.source, result.raw_reasoning, result.error) == (
        "",
        "empty",
        "",
        "dropped_by_policy",
    )
    assert result.consistency_issues == []
    assert not client.calls


def test_synthetic_uses_selected_code_without_repair_history() -> None:
    client = SequenceClient(VALID_COT)
    result = CotAgent(settings(), client).refine(
        state(mode="synthetic", raw="REPAIR_REASONING_MARKER")
    )
    assert result.source == "synthetic" and result.raw_reasoning == ""
    call = client.calls[0]
    user = call["messages"][0]["content"]
    assert "selected_kernel" in user and "stale_code" not in user
    assert "(none — derive the reasoning from the problem and the final code)" in user
    assert "REPAIR_REASONING_MARKER" not in user
    assert "repairs=" not in user and "last_error_summary" not in user
    assert "secret compile log" not in user
    assert (call["meta"].candidate, call["meta"].repair, call["meta"].purpose) == (
        2,
        1,
        "synthetic",
    )


def test_synthetic_failure_does_not_fall_back_to_teacher_reasoning() -> None:
    client = SequenceClient()
    result = CotAgent(settings(), client).refine(
        state(mode="synthetic", raw="REPAIR_REASONING_MARKER")
    )
    assert result.source == "empty" and result.cot == ""
    assert result.raw_reasoning == ""
    assert len(client.calls) == 1


def test_consistency_retry_appends_issues_and_accepts_clean_draft() -> None:
    client = SequenceClient(VALID_COT + "\nThe previous version had a compile error.", VALID_COT)
    result = CotAgent(settings(), client).refine(state())
    assert result.source == "agent" and result.cot == VALID_COT
    assert result.consistency_issues == []
    assert len(client.calls) == 2
    assert "The previous draft violated these rules:" in client.calls[1]["messages"][0]["content"]
    assert "repair leak:" in client.calls[1]["messages"][0]["content"]


def test_inconsistent_raw_fallback_becomes_empty() -> None:
    bad = VALID_COT + "\nThe previous version had a compile error."
    client = SequenceClient(bad, bad)
    result = CotAgent(settings(cot_max_calls=2), client).refine(state(raw="I fixed a compile error."))
    assert result.source == "empty" and result.cot == ""
    assert any(issue.startswith("repair leak:") for issue in result.consistency_issues)
    assert len(client.calls) == 2


def test_clean_raw_fallback_preserves_rejected_draft_issues() -> None:
    bad = VALID_COT + "\nI fixed a compile error."
    client = SequenceClient(bad, bad)
    result = CotAgent(settings(cot_max_calls=2), client).refine(state(raw=VALID_COT))
    assert result.source == "raw" and result.cot == VALID_COT
    assert any(issue.startswith("repair leak:") for issue in result.consistency_issues)


def test_late_editor_failure_keeps_earlier_consistency_issues() -> None:
    bad = VALID_COT + "\nThe previous version had a compile error."
    client = SequenceClient(bad)
    result = CotAgent(settings(cot_max_calls=2), client).refine(state(raw=VALID_COT))
    assert result.source == "raw" and result.cot == VALID_COT
    assert any(issue.startswith("repair leak:") for issue in result.consistency_issues)


def test_prompt_appends_language_rules_and_omits_repair_fields() -> None:
    assert cot_system_for(dialect="cuda").endswith(COT_NO_REPAIR_RULES_ZH)
    assert cot_system_for(dialect="triton").endswith(COT_NO_REPAIR_RULES_EN)
    user = build_cot_user_prompt(question="add", code="void add() {}", raw_reasoning="")
    assert "repairs=" not in user and "last_error_summary" not in user
    assert "(none — derive the reasoning from the problem and the final code)" in user
