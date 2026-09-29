"""Scenario fake behavior used by component tests."""

import openai
import pytest

from cuda_sft.llm import approx_tokens, is_retryable_llm_error
from cuda_sft.runtime.meta import CallMeta
from cuda_sft.testing.fakes import Cancelled, FakeLLM, UnscriptedCall, parse_fake_markers


def test_marker_parser_and_default_reply(tmp_path) -> None:
    assert parse_fake_markers("// @fake compile=fail:syntax refval=pass\nint x;") == {
        "compile": "fail:syntax",
        "refval": "pass",
    }
    fake = FakeLLM({"generator": {"default": {"text": "ok"}}}, tmp_path)
    meta = CallMeta("generator", "q1:cuda", 1, "cuda", candidate=1)
    result = fake.factory("generator").stream_completion(
        messages=[{"role": "user", "content": "add"}], system="test", meta=meta
    )
    assert result.text == "ok"


def test_unscripted_call_fails(tmp_path) -> None:
    fake = FakeLLM({}, tmp_path)
    meta = CallMeta("generator", "q1:cuda", 1, "cuda", candidate=1)
    user = "prefix " + "x" * 300 + "UNIQUE_TAIL"
    with pytest.raises(UnscriptedCall, match="role=generator key=c1r0") as error:
        fake.factory("generator").stream_completion(
            messages=[{"role": "user", "content": user}], system="test", meta=meta
        )
    assert "purpose=''" in str(error.value)
    assert f"user={user[:300]!r}" in str(error.value)
    assert "UNIQUE_TAIL" not in str(error.value)
    assert fake.calls[0].t_end >= fake.calls[0].t_start


@pytest.mark.parametrize(
    ("kind", "error_type", "status", "retryable"),
    [
        ("rate_limit", openai.RateLimitError, 429, True),
        ("bad_request", openai.BadRequestError, 400, False),
    ],
)
def test_sdk_shaped_errors(tmp_path, kind, error_type, status, retryable) -> None:
    fake = FakeLLM({"generator": {"default": {"raise": kind}}}, tmp_path)
    meta = CallMeta("generator", "q1:cuda", 1, "cuda", candidate=1)
    with pytest.raises(error_type) as error:
        fake.factory("generator").stream_completion(
            messages=[{"role": "user", "content": "add"}], system="test", meta=meta
        )
    assert error.value.status_code == status
    assert error.value.response.status_code == status
    assert is_retryable_llm_error(error.value) is retryable
    assert fake.calls[0].t_end >= fake.calls[0].t_start


def test_scripted_and_estimated_usage(tmp_path) -> None:
    meta = CallMeta("generator", "q1:cuda", 1, "cuda", candidate=1)
    messages = [{"role": "user", "content": "add"}]
    fake = FakeLLM(
        {
            "generator": {
                "default": {
                    "text": "ok",
                    "reasoning": "think",
                    "usage": {"input_tokens": 17, "output_tokens": 9},
                }
            }
        },
        tmp_path,
    )
    result = fake.factory("generator").stream_completion(
        messages=messages, system="test", meta=meta
    )
    assert result.usage == {"input_tokens": 17, "output_tokens": 9}
    assert not result.tokens_estimated
    assert result.reasoning_source == "api"

    estimated = FakeLLM({"generator": {"default": {"text": "ok"}}}, tmp_path)
    result = estimated.factory("generator").stream_completion(
        messages=messages, system="test", meta=meta
    )
    assert result.usage == {
        "input_tokens": approx_tokens("test") + approx_tokens("add"),
        "output_tokens": approx_tokens("ok"),
    }
    assert result.tokens_estimated


def test_scripted_cancelled_records_call(tmp_path) -> None:
    fake = FakeLLM({"generator": {"default": {"raise": "cancelled"}}}, tmp_path)
    meta = CallMeta("generator", "q1:cuda", 1, "cuda", candidate=1)
    with pytest.raises(Cancelled):
        fake.factory("generator").stream_completion(
            messages=[{"role": "user", "content": "add"}], system="test", meta=meta
        )
    assert fake.calls[0].cancelled
    assert fake.calls[0].t_end >= fake.calls[0].t_start
