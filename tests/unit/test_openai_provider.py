"""Offline OpenAI-compatible provider request and stream contract."""

from __future__ import annotations

import json
from types import SimpleNamespace as Obj

import httpx
import pytest
from openai import OpenAI

import cuda_sft.llm as llm_module
from cuda_sft.config import Settings
from cuda_sft.llm import (
    NvidiaOpenAIClient,
    OpenAIChatClient,
    get_llm_client,
    is_retryable_llm_error,
    reset_llm_client,
)
from cuda_sft.runtime import trace
from cuda_sft.runtime.meta import CallMeta

USER_TURN = [{"role": "user", "content": "hello"}]


def _settings(**overrides: object) -> Settings:
    values = {
        "llm_provider": "openai",
        "llm_providers": "openai",
        "openai_api_key": "test-key",
        "openai_model": "gpt-6-luna",
        "model": "",
        "thinking_level": "high",
    }
    values.update(overrides)
    return Settings(**values)


def test_openai_config_and_worker_slot() -> None:
    settings = _settings()
    assert settings.provider_pool() == ["openai"]
    assert settings.resolved_api_key == "test-key"
    assert settings.resolved_base_url == "https://www.poke2api.com/v1"
    assert settings.resolved_model == "gpt-6-luna"
    assert settings.build_worker_assignments(workers=1)[0].api_key == "test-key"
    assert _settings(openai_api_key="").missing_provider_secrets() == ["OPENAI_API_KEY"]
    assert (
        _settings(api_base_url="https://www.poke2api.com/v1/").resolved_base_url
        == settings.resolved_base_url
    )


def test_three_openai_workers_share_selected_key() -> None:
    settings = _settings()
    slots = settings.build_worker_assignments(workers=3)
    assert len(slots) == 3
    assert [(slot.provider, slot.api_key, slot.label) for slot in slots] == [
        ("openai", "test-key", "openai")
    ] * 3


def test_openai_sdk_request_and_stream_without_network() -> None:
    settings = _settings()
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        stream = (
            'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"gpt-6-luna",'
            '"choices":[{"index":0,"delta":{"reasoning_content":"plan"}}]}\n\n'
            'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"gpt-6-luna",'
            '"choices":[{"index":0,"delta":{"content":"answer"}}]}\n\n'
            'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"gpt-6-luna",'
            '"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=stream)

    client = OpenAIChatClient(settings)
    assert str(client._client.base_url) == "https://www.poke2api.com/v1/"
    client._client.close()
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client._client = OpenAI(
            api_key="test-key", base_url=settings.resolved_base_url, http_client=http_client
        )
        result = client.stream_completion(
            messages=[{"role": "user", "content": "hello"}],
            system="system",
            temperature=0.2,
            print_stream=False,
        )
    assert result.text == "answer"
    assert result.reasoning == "plan"
    assert result.reasoning_source == "openai_delta"
    assert result.origin == "live_api"
    assert result.usage == {"input_tokens": 7, "output_tokens": 3, "reasoning_tokens": 0}
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://www.poke2api.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "gpt-6-luna"
    assert payload["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]
    assert payload["reasoning_effort"] == "high"
    assert payload["max_completion_tokens"] == settings.resolved_max_output_tokens
    assert "chat_template_kwargs" not in payload
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_openai_client_dispatch() -> None:
    reset_llm_client()
    try:
        client = get_llm_client(_settings())
        assert type(client) is OpenAIChatClient
        client._client.close()
    finally:
        reset_llm_client()


def test_nvidia_keeps_nim_request_options() -> None:
    client = NvidiaOpenAIClient.__new__(NvidiaOpenAIClient)
    client.settings = Settings(llm_provider="nvidia", model="", thinking_level="medium")
    client._stream_usage_supported = False
    requests = []

    def create(**kwargs: object):
        requests.append(kwargs)
        return iter([Obj(choices=[Obj(delta=Obj(content="ok", reasoning_content=""))], usage=None)])

    client._client = Obj(chat=Obj(completions=Obj(create=create)))
    assert (
        client.stream_completion(
            messages=USER_TURN, system="", temperature=0.3, print_stream=False
        ).text
        == "ok"
    )
    request = requests[0]
    assert request["max_tokens"] == client.settings.resolved_max_output_tokens
    assert request["temperature"] == 0.3
    assert request["top_p"] == client.settings.top_p
    assert request["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "reasoning_effort": "medium",
    }
    assert "max_completion_tokens" not in request


def test_openai_stream_options_400_falls_back_once() -> None:
    seen: list[dict[str, object]] = []
    client = OpenAIChatClient.__new__(OpenAIChatClient)
    client.settings = _settings()
    client._stream_usage_supported = True

    def create(**kwargs: object):
        seen.append(kwargs)
        if len(seen) == 1:
            raise _bad_request()
        return iter([Obj(choices=[Obj(delta=Obj(content="ok", reasoning_content=""))], usage=None)])

    client._client = Obj(chat=Obj(completions=Obj(create=create)))
    sink = trace.MemorySink()
    previous = trace.set_sink(sink)
    try:
        result = client.stream_completion(
            messages=[{"role": "user", "content": "hello"}],
            system="",
            temperature=0.0,
            print_stream=False,
            meta=CallMeta("generator", "q1:cuda", 1, "cuda"),
        )
    finally:
        trace.set_sink(previous)
    assert result.text == "ok" and result.tokens_estimated
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert "stream_options" not in seen[1]
    calls = sink.of("llm.call")
    assert len(calls) == 2 and [call["ok"] for call in calls] == [False, True]
    assert [call["attempt"] for call in calls] == [1, 2]
    assert all(call["tokens_estimated"] for call in calls)
    trace.emit("job.end", job_key="q1:cuda", status="success")


def test_nvidia_stream_options_400_disables_usage_process_wide(monkeypatch, caplog) -> None:
    monkeypatch.setattr(llm_module, "_nvidia_stream_usage_supported", True)
    seen: list[dict[str, object]] = []
    sink = trace.MemorySink()
    previous = trace.set_sink(sink)

    def create(**kwargs: object):
        seen.append(kwargs)
        if "stream_options" in kwargs:
            raise _bad_request()
        return iter([Obj(choices=[Obj(delta=Obj(content="ok", reasoning_content=""))], usage=None)])

    try:
        for _ in range(2):
            client = NvidiaOpenAIClient.__new__(NvidiaOpenAIClient)
            client.settings = Settings(
                llm_provider="nvidia", nvidia_api_key="test-nim-key", model=""
            )
            client._stream_usage_supported = True
            client._client = Obj(chat=Obj(completions=Obj(create=create)))
            assert (
                client.stream_completion(
                    messages=USER_TURN, system="", temperature=0, print_stream=False
                ).text
                == "ok"
            )
    finally:
        trace.set_sink(previous)

    assert len(seen) == 3
    assert "stream_options" in seen[0]
    assert "stream_options" not in seen[1] and "stream_options" not in seen[2]
    assert [call["attempt"] for call in sink.of("llm.call")] == [1, 2, 1]
    assert sum("rejected stream_options" in message for message in caplog.messages) == 1


def test_nvidia_trace_uses_assigned_key_label(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_SFT_WORKER_KEY_LABEL", "nvidia#2")
    client = NvidiaOpenAIClient.__new__(NvidiaOpenAIClient)
    client.settings = Settings(llm_provider="nvidia", nvidia_api_key="test-nim-key", model="")
    client._stream_usage_supported = False
    client._client = Obj(
        chat=Obj(
            completions=Obj(
                create=lambda **_kwargs: iter(
                    [Obj(choices=[Obj(delta=Obj(content="ok", reasoning_content=""))], usage=None)]
                )
            )
        )
    )
    sink = trace.MemorySink()
    previous = trace.set_sink(sink)
    try:
        client.stream_completion(messages=USER_TURN, system="", temperature=0, print_stream=False)
    finally:
        trace.set_sink(previous)
    assert sink.of("llm.call")[0]["key_label"] == "nvidia#2"


def _bad_request():
    from openai import BadRequestError

    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError("stream_options is unsupported", response=response, body=None)


def test_openai_partial_stream_failure_is_metered() -> None:
    client = OpenAIChatClient.__new__(OpenAIChatClient)
    client.settings = _settings()
    client._stream_usage_supported = False

    def create(**_kwargs: object):
        yield Obj(
            choices=[Obj(delta=Obj(content="partial answer", reasoning_content=""))], usage=None
        )
        raise RuntimeError("broken stream")

    client._client = Obj(chat=Obj(completions=Obj(create=create)))
    sink = trace.MemorySink()
    previous = trace.set_sink(sink)
    try:
        try:
            client.stream_completion(
                messages=USER_TURN, system="", temperature=0, print_stream=False
            )
        except RuntimeError as exc:
            assert str(exc) == "broken stream"
        else:
            raise AssertionError("stream error was swallowed")
    finally:
        trace.set_sink(previous)
    call = sink.of("llm.call")[0]
    assert not call["ok"] and call["error_type"] == "RuntimeError"
    assert call["output_tokens"] > 0 and call["tokens_estimated"]


def test_openai_stream_cancellation_emits_call() -> None:
    client = OpenAIChatClient.__new__(OpenAIChatClient)
    client.settings = _settings()
    client._stream_usage_supported = False

    def create(**_kwargs: object):
        yield Obj(choices=[Obj(delta=Obj(content="partial", reasoning_content=""))], usage=None)
        raise KeyboardInterrupt

    client._client = Obj(chat=Obj(completions=Obj(create=create)))
    sink = trace.MemorySink()
    previous = trace.set_sink(sink)
    try:
        try:
            client.stream_completion(
                messages=USER_TURN, system="", temperature=0, print_stream=False
            )
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("cancellation was swallowed")
    finally:
        trace.set_sink(previous)
    call = sink.of("llm.call")[0]
    assert call["cancelled"] and not call["ok"] and call["output_tokens"] > 0


def test_retryable_status_and_authentication_status() -> None:
    retryable = Obj(status_code=529)
    unauthorized = Obj(status_code=401)
    assert is_retryable_llm_error(RuntimeError("temporarily overloaded"))
    assert is_retryable_llm_error(type("ServiceError", (Exception,), vars(retryable))("busy"))
    assert not is_retryable_llm_error(
        type("AuthError", (Exception,), vars(unauthorized))("temporarily unavailable")
    )


def test_empty_messages_fail_before_sdk_request() -> None:
    client = OpenAIChatClient.__new__(OpenAIChatClient)
    client.settings = _settings()
    client._stream_usage_supported = False
    client._client = Obj(
        chat=Obj(completions=Obj(create=lambda **_kwargs: pytest.fail("SDK request reached")))
    )
    with pytest.raises(ValueError, match="alternate user/assistant"):
        client.stream_completion(messages=[], system="", temperature=0, print_stream=False)
