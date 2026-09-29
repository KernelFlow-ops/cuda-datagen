"""Critic retries only unavailable or invalid reviewer responses."""

from __future__ import annotations

import pytest

from cuda_sft.agents.critic import KernelCritic, critic_result_to_dict
from cuda_sft.config import Settings
from cuda_sft.judge import JudgeResult
from cuda_sft.llm import LLMCompletion
from cuda_sft.runtime.meta import CallMeta


class SequenceClient:
    def __init__(self, *replies: str | Exception) -> None:
        self.replies = list(replies)
        self.calls: list[dict] = []

    def stream_completion(self, **kwargs) -> LLMCompletion:
        self.calls.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMCompletion(text=reply)


def evaluate(client: SequenceClient, *, retries: int = 1):
    critic = KernelCritic(
        settings=Settings(kernel_llm_critic="always", critic_retry_on_error=retries),
        llm_client=client,
    )
    return critic.evaluate(
        question="vector add",
        code="__global__ void add() {}",
        dialect="cuda",
        heuristic=JudgeResult(quality_score=5, issues=[]),
        meta=CallMeta("critic", "q7:cuda", 7, "cuda", candidate=2),
    )


def test_invalid_json_then_valid_response_recovers() -> None:
    client = SequenceClient("not json", '{"pass": true, "must_fix": [], "issues": []}')
    result = evaluate(client)
    assert critic_result_to_dict(result)["status"] == "verified"
    assert result.passed and len(client.calls) == 2
    assert [call["meta"].attempt for call in client.calls] == [1, 2]
    assert all(call["thinking_level"] == "low" and call["max_output_tokens"] >= 2048 for call in client.calls)


def test_exception_then_valid_must_fix_is_failed_without_extra_retry() -> None:
    client = SequenceClient(
        TimeoutError("critic unavailable"),
        '{"pass": false, "must_fix": ["wrong host signature"], "issues": []}',
    )
    result = evaluate(client, retries=3)
    assert critic_result_to_dict(result)["status"] == "failed"
    assert not result.passed and result.must_fix == ["wrong host signature"]
    assert len(client.calls) == 2


@pytest.mark.parametrize(
    ("replies", "kind"),
    [
        (("not json", "still not json"), "invalid_response"),
        ((TimeoutError("first"), TimeoutError("second")), "critic_error"),
        (('{"pass": true, "must_fix": "bad"}', '{"pass": "yes", "must_fix": []}'), "invalid_response"),
    ],
)
def test_exhausted_retries_return_unverified(replies, kind: str) -> None:
    client = SequenceClient(*replies)
    record = critic_result_to_dict(evaluate(client))
    assert record["status"] == "unverified"
    assert record["failure"]["kind"] == kind
    assert len(client.calls) == 2


def test_zero_retry_budget_calls_once() -> None:
    client = SequenceClient(RuntimeError("offline"))
    record = critic_result_to_dict(evaluate(client, retries=0))
    assert record["status"] == "unverified" and len(client.calls) == 1


def test_valid_rejection_does_not_retry() -> None:
    client = SequenceClient('{"pass": false, "must_fix": ["wrong algorithm"], "issues": []}')
    result = evaluate(client, retries=3)
    assert critic_result_to_dict(result)["status"] == "failed"
    assert result.must_fix == ["wrong algorithm"] and len(client.calls) == 1


def test_critic_decision_can_be_parsed_from_reasoning_channel() -> None:
    class SplitChannelClient:
        def stream_completion(self, **_kwargs) -> LLMCompletion:
            return LLMCompletion(
                text="The implementation has a semantic defect.",
                reasoning='{"pass": false, "must_fix": ["wrong indexing"], "issues": []}',
            )

    critic = KernelCritic(
        settings=Settings(kernel_llm_critic="always", critic_retry_on_error=0),
        llm_client=SplitChannelClient(),  # type: ignore[arg-type]
    )
    result = critic.evaluate(
        question="vector add",
        code="__global__ void add() {}",
        dialect="cuda",
        heuristic=JudgeResult(quality_score=5, issues=[]),
    )
    assert not result.passed
    assert result.must_fix == ["wrong indexing"]
