"""Role-specific provider routing and credential limits without live API calls."""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from anthropic import Anthropic
from openai import OpenAI
from pydantic import ValidationError

import cuda_sft.llm as llm
import cuda_sft.graph as kernel_graph
from cuda_sft.agents.contracts import GenerateResult
from cuda_sft.config import ROLE_CONFIG_PREFIXES, Settings
from cuda_sft.runtime import limits


def test_all_roles_have_overrides_and_inherit_worker_defaults() -> None:
    settings = Settings(
        llm_provider="nvidia",
        nvidia_api_key="nv-key",
        model="global-model",
        critic_provider="openai",
        critic_model="critic-model",
        critic_api_base_url="https://proxy.example/v1/",
    )
    for prefix in ROLE_CONFIG_PREFIXES.values():
        for suffix in ("provider", "model", "api_base_url"):
            assert hasattr(settings, f"{prefix}_{suffix}")
    assert settings.for_role("generator").llm_provider == "nvidia"
    assert settings.for_role("generator").resolved_model == "global-model"
    critic = settings.for_role("critic")
    assert (critic.llm_provider, critic.resolved_model, critic.resolved_base_url) == (
        "openai", "critic-model", "https://proxy.example/v1"
    )
    assert Settings(llm_provider="openai").resolved_base_url == "https://www.poke2api.com/v1"
    assert Settings(llm_provider="nvidia").resolved_base_url == "https://integrate.api.nvidia.com/v1"
    assert Settings(llm_provider="openrouter", api_base_url="https://or.example/v1/").resolved_base_url == "https://or.example"
    assert Settings(llm_provider="openrouter", api_base_url="https://or.example/").resolved_base_url == "https://or.example"


@pytest.mark.parametrize(
    ("failure", "expected_role", "expected_diagnostic"),
    [
        ({"compile_error": "undefined symbol", "compile_ok": False}, "repair.compile", "undefined symbol"),
        (
            {
                "compile_ok": True,
                "refval_ok": False,
                "refval_error": "numeric mismatch: output differs",
                "refval_error_class": "numeric_mismatch",
            },
            "repair.numeric",
            "numeric mismatch",
        ),
        (
            {
                "compile_ok": True,
                "refval_ok": True,
                "critic_pass": False,
                "critic_must_fix": [],
                "critic_issues": [],
                "metadata": {
                    "critic": {
                        "status": "failed",
                        "failure": {"kind": "critic_rejected", "message": "wrong algorithm"},
                    }
                },
            },
            "repair.semantic",
            "wrong algorithm",
        ),
    ],
)
def test_kernel_repair_routes_to_failure_specific_role(
    failure: dict[str, Any], expected_role: str, expected_diagnostic: str
) -> None:
    settings = Settings(async_llm_enabled=False, refval_enabled=False)
    state = {
        "question_id": 5,
        "question": "Implement vector add",
        "dialect": "cuda",
        "candidate_idx": 1,
        "repair_idx": 0,
        "repair_cap": 1,
        "messages": [{"role": "user", "content": "original task"}],
        "system_prompt": "kernel",
        "code": "void add() {}",
        "last_gate": {"gate": "refval" if failure.get("refval_ok") is False else "compile"},
        **failure,
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kernel_graph, "get_settings", lambda: settings)
        repaired = kernel_graph.repair(state)
        generated_state = {**state, **repaired}
        captured: dict[str, Any] = {}

        def complete(**kwargs: Any) -> GenerateResult:
            captured.update(kwargs)
            return GenerateResult(text="updated source")

        mp.setattr(kernel_graph, "complete_chat", complete)
        kernel_graph.generate(generated_state)

    assert captured["meta"].role == expected_role
    assert expected_diagnostic in repaired["messages"][-1]["content"]


def test_role_provider_validation_and_missing_key() -> None:
    settings = Settings(
        llm_provider="nvidia", nvidia_api_key="nv-key", critic_provider="openai"
    )
    assert settings.missing_provider_secrets(["nvidia"], roles=["critic"]) == ["OPENAI_API_KEY"]
    assert settings.missing_provider_secrets(["nvidia"], roles=["generator"]) == []
    assert Settings(
        llm_provider="nvidia", generator_provider="openai", openai_api_key="oa-key"
    ).missing_provider_secrets(["nvidia"], roles=["generator"]) == []
    assert Settings(critic_provider="nv").for_role("critic").llm_provider == "nvidia"
    with pytest.raises(ValidationError):
        Settings(critic_provider="unsupported")


def test_kernel_preflight_includes_each_repair_route() -> None:
    settings = Settings(
        task_mode="kernel", llm_provider="nvidia", nvidia_api_key="nv-key",
        repair_numeric_provider="openai", repair_semantic_provider="openrouter",
        openrouter_api_key="",
    )
    roles = settings.active_llm_roles()
    assert {"repair.compile", "repair.numeric", "repair.semantic"} <= set(roles)
    assert settings.missing_provider_secrets(["nvidia"], roles=roles) == [
        "OPENAI_API_KEY", "OPENROUTER_API_KEY"
    ]


def test_worker_provider_pool_remains_default_for_unconfigured_roles() -> None:
    settings = Settings(
        llm_provider="nvidia", llm_providers="nvidia,openrouter",
        nvidia_api_key="nv-1", nvidia_api_key_2="nv-2",
        openrouter_api_key="or-key", critic_provider="openai",
    )
    slots = settings.build_worker_assignments(
        workers=1, workers_per_provider=1, providers=settings.provider_pool()
    )
    assert [slot.label for slot in slots] == ["nvidia#1", "nvidia#2", "openrouter"]
    for slot in slots:
        worker = settings.model_copy(update={"llm_provider": slot.provider})
        assert worker.for_role("generator").llm_provider == slot.provider
        assert worker.for_role("critic").llm_provider == "openai"


def test_role_clients_use_their_endpoint_and_cache_by_effective_route(monkeypatch) -> None:
    monkeypatch.setattr(limits, "_root", None)
    monkeypatch.setattr(llm, "Anthropic", lambda **kwargs: SimpleNamespace(**kwargs))
    settings = Settings(
        llm_provider="nvidia", nvidia_api_key="nv-key",
        openai_api_key="oa-key", openrouter_api_key="or-key",
        generator_provider="openai", generator_model="gen-model",
        generator_api_base_url="https://proxy.example",
        critic_provider="nvidia", critic_model="critic-model",
        critic_api_base_url="https://nim.example/v1/",
        cot_editor_provider="openrouter", cot_editor_api_base_url="https://or.example/v1",
    )
    llm.reset_llm_client()
    try:
        generator = llm.get_llm_client(settings, role="generator")
        critic = llm.get_llm_client(settings, role="critic")
        editor = llm.get_llm_client(settings, role="cot_editor")
        assert generator is llm.get_llm_client(settings, role="generator")
        assert generator is not critic
        assert (generator.settings.llm_provider, generator.settings.resolved_model) == (
            "openai", "gen-model"
        )
        assert str(generator._client.base_url) == "https://proxy.example/v1/"
        assert str(critic._client.base_url) == "https://nim.example/v1/"
        assert editor._client.base_url == "https://or.example"
        generator._client.close()
        critic._client.close()
    finally:
        llm.reset_llm_client()


def test_llm_limit_uses_actual_provider_and_key(monkeypatch, tmp_path) -> None:
    for name in ("_root", "_slot", "_llm_count", "_compile_count", "_deadline"):
        monkeypatch.setattr(limits, name, getattr(limits, name))
    limits.configure(
        tmp_path, slot_label="worker-default", llm_concurrency=1,
        compile_concurrency=1,
    )

    class Client:
        def stream_completion(self, **_kwargs):
            return "ok"

    nvidia = Settings(llm_provider="nvidia", nvidia_api_key="nv-key")
    openai = Settings(llm_provider="openai", openai_api_key="oa-key")
    for route in (nvidia, nvidia, openai):
        client = limits.limited_client(Client(), settings=route)
        assert client.stream_completion() == "ok"
    assert len(list(tmp_path.glob("llm_*.lock"))) == 2


def test_nvidia_stream_uses_resolved_agent_url() -> None:
    settings = Settings(
        llm_provider="nvidia", nvidia_api_key="nv-key",
        generator_api_base_url="https://nim.example/v1/",
    ).for_role("generator")
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text=(
                'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"nv",'
                '"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = llm.NvidiaOpenAIClient(settings)
        client._client.close()
        client._client = OpenAI(
            api_key=settings.resolved_api_key,
            base_url=settings.resolved_base_url,
            http_client=http_client,
        )
        result = client.stream_completion(
            messages=[{"role": "user", "content": "hello"}], system="",
            temperature=0.2, thinking_level="none", print_stream=False,
        )
    assert result.text == "ok"
    assert [str(request.url) for request in requests] == [
        "https://nim.example/v1/chat/completions"
    ]


def test_openrouter_stream_uses_resolved_agent_url() -> None:
    sdk_http = importlib.import_module(
        "httpx2" if int(anthropic.__version__.split(".")[0]) >= 1 else "httpx"
    )
    settings = Settings(
        llm_provider="openrouter", openrouter_api_key="or-key",
        generator_api_base_url="https://or.example/v1/",
    ).for_role("generator")
    requests: list[Any] = []

    def respond(request: Any) -> Any:
        requests.append(request)
        return sdk_http.Response(
            200, headers={"content-type": "text/event-stream"},
            text=(
                'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1",'
                '"type":"message","role":"assistant","content":[],"model":"or",'
                '"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,'
                '"output_tokens":0}}}\n\n'
                'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
                '"content_block":{"type":"text","text":""}}\n\n'
                'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
                '"delta":{"type":"text_delta","text":"ok"}}\n\n'
                'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n'
                'event: message_delta\ndata: {"type":"message_delta","delta":'
                '{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}\n\n'
                'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            ),
        )

    with sdk_http.Client(transport=sdk_http.MockTransport(respond)) as http_client:
        client = llm.AnthropicOpenRouterClient(settings)
        client._client.close()
        client._client = Anthropic(
            api_key=settings.resolved_api_key,
            base_url=settings.resolved_base_url,
            http_client=http_client,
        )
        result = client.stream_completion(
            messages=[{"role": "user", "content": "hello"}], system="",
            temperature=0.2, thinking_level="none", print_stream=False,
        )
    assert result.text == "ok"
    assert [str(request.url) for request in requests] == ["https://or.example/v1/messages"]
    assert json.loads(requests[0].content)["temperature"] == 0.2
