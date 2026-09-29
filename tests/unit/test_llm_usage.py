"""SDK stream usage is recorded without contacting a provider."""

from types import SimpleNamespace as Obj

import pytest

from cuda_sft.config import get_settings
from cuda_sft.llm import AnthropicOpenRouterClient, NvidiaOpenAIClient
from cuda_sft.runtime import trace
from cuda_sft.runtime.meta import CallMeta

USER_TURN = [{"role": "user", "content": "hello"}]


class AnthropicStream:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        yield Obj(type="content_block_delta", delta=Obj(type="text_delta", text="code"))

    def get_final_message(self):
        return Obj(
            content=[Obj(type="text", text="code")], usage=Obj(input_tokens=7, output_tokens=3)
        )


def test_anthropic_usage_and_job_aggregation() -> None:
    client = AnthropicOpenRouterClient.__new__(AnthropicOpenRouterClient)
    client.settings = get_settings()
    client._client = Obj(messages=Obj(stream=lambda **_kw: AnthropicStream()))
    meta = CallMeta("generator", "q1:cuda", 1, "cuda", candidate=1)
    sink = trace.MemorySink()
    trace.set_sink(sink)
    result = client.stream_completion(
        messages=[{"role": "user", "content": "add"}],
        system="kernel",
        temperature=0.2,
        print_stream=False,
        meta=meta,
    )
    trace.emit("job.end", job_key="q1:cuda", status="success")
    assert result.usage == {"input_tokens": 7, "output_tokens": 3, "reasoning_tokens": 0}
    assert result.origin == "live_api"
    call = sink.of("llm.call")[0]
    assert call["role"] == "generator" and not call["tokens_estimated"]
    assert sink.of("job.end")[0]["input_tokens"] == 7


def test_nvidia_usage_chunk_and_estimate() -> None:
    client = NvidiaOpenAIClient.__new__(NvidiaOpenAIClient)
    client.settings = get_settings()
    client._stream_usage_supported = True
    seen = []

    def create(**kwargs):
        seen.append(kwargs)
        yield Obj(choices=[Obj(delta=Obj(content="code", reasoning_content=""))], usage=None)
        if len(seen) == 1:
            yield Obj(choices=[], usage=Obj(prompt_tokens=11, completion_tokens=4))

    client._client = Obj(chat=Obj(completions=Obj(create=create)))
    sink = trace.MemorySink()
    trace.set_sink(sink)
    meta = CallMeta("generator", "q2:cuda", 2, "cuda")
    first = client.stream_completion(
        messages=USER_TURN, system="kernel", temperature=0, print_stream=False, meta=meta
    )
    second = client.stream_completion(
        messages=USER_TURN, system="kernel", temperature=0, print_stream=False, meta=meta
    )
    assert first.usage["input_tokens"] == 11 and not first.tokens_estimated
    assert second.tokens_estimated
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert [e["tokens_estimated"] for e in sink.of("llm.call")] == [False, True]


def test_anthropic_partial_stream_failure_is_metered() -> None:
    class BrokenStream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def __iter__(self):
            yield Obj(type="content_block_delta", delta=Obj(type="text_delta", text="partial"))
            raise RuntimeError("stream ended early")

    client = AnthropicOpenRouterClient.__new__(AnthropicOpenRouterClient)
    client.settings = get_settings()
    client._client = Obj(messages=Obj(stream=lambda **_kw: BrokenStream()))
    sink = trace.MemorySink()
    previous = trace.set_sink(sink)
    try:
        with pytest.raises(RuntimeError, match="stream ended early"):
            client.stream_completion(
                messages=USER_TURN, system="", temperature=0, print_stream=False
            )
    finally:
        trace.set_sink(previous)
    call = sink.of("llm.call")[0]
    assert not call["ok"] and call["error_type"] == "RuntimeError"
    assert call["output_tokens"] > 0 and call["tokens_estimated"]
