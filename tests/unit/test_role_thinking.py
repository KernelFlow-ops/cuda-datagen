"""Role model and thinking overrides across generation and extraction paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import dotenv_values

import cuda_sft.agents.generate as agent_generate
import cuda_sft.graph as kernel_graph
import cuda_sft.llm as llm
from cuda_sft.agents.contracts import GenerateResult
from cuda_sft.config import Settings
from cuda_sft.cot import CotAgent
from cuda_sft.knowledge.cot import KnowledgeCotAgent
from cuda_sft.llm import LLMCompletion
from cuda_sft.refval import runner


class CaptureClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def stream_completion(self, **kwargs) -> LLMCompletion:
        self.calls.append(kwargs)
        return LLMCompletion(text="draft")


def test_example_routes_the_five_configured_roles() -> None:
    example = Path(__file__).resolve().parents[2] / ".env.example"
    values = dotenv_values(example)
    assert values["LLM_PROVIDER"] == "openai"
    assert values["THINKING_LEVEL"] == "high"
    fields = ("llm_provider", "thinking_level", "knowledge_thinking_level")
    roles = {
        "generator": ("gpt-6-luna", "high"),
        "repair_compile": ("gpt-6-sol", "medium"),
        "cot_editor": ("gpt-6-luna", "high"),
        "refval_extract": ("gpt-6-luna", "high"),
        "knowledge_generator": ("gpt-6-luna", "high"),
    }
    fields += tuple(f"{prefix}_{suffix}" for prefix in roles for suffix in ("provider", "model", "thinking_level"))
    settings = Settings(**{field: values[field.upper()] for field in fields})
    for prefix, (model, level) in roles.items():
        role = "repair.compile" if prefix == "repair_compile" else prefix
        route = settings.for_role(role)
        assert (route.llm_provider, route.resolved_model, route.thinking_level) == (
            "openai", model, level,
        )
    for role in ("repair.numeric", "repair.semantic", "critic"):
        route = settings.for_role(role)
        assert (route.llm_provider, route.resolved_model, route.thinking_level) == (
            "openai", "gpt-6-luna", "high",
        )
    for role in ("knowledge_repair", "knowledge_judge"):
        assert settings.for_role(role).thinking_level == "medium"


@pytest.mark.parametrize("role", ["cot_editor", "refval_extract"])
@pytest.mark.parametrize(("configured", "expected"), [("", "high"), ("none", "none")])
def test_editor_and_refval_thinking_fallback(role: str, configured: str, expected: str) -> None:
    settings = Settings(
        thinking_level="high",
        **{f"{role}_thinking_level": configured},
    )
    assert settings.for_role(role).thinking_level == expected


@pytest.mark.parametrize("agent_kind", ["kernel", "knowledge"])
@pytest.mark.parametrize(("configured", "expected"), [("", "medium"), ("none", "none"), ("high", "high")])
def test_both_cot_agents_use_editor_level(
    agent_kind: str, configured: str, expected: str,
) -> None:
    settings = Settings(thinking_level="medium", cot_editor_thinking_level=configured)
    client = CaptureClient()
    if agent_kind == "kernel":
        agent = CotAgent(settings, client)
        state = {
            "question_id": 1, "question": "vector add", "dialect": "cuda",
            "selected": {"code": "__global__ void add() {}"},
        }
    else:
        agent = KnowledgeCotAgent(settings, client)
        state = {
            "question_id": 1, "question": "Explain a warp", "answer": "A thread group",
            "topic": "architecture", "track": "knowledge:architecture",
        }
    agent._run_agent(state, raw_reasoning="")
    assert client.calls[0]["thinking_level"] == expected
    assert client.calls[0]["meta"].role == "cot_editor"


@pytest.mark.parametrize(("configured", "expected"), [("", "medium"), ("none", "none"), ("high", "high")])
def test_refval_sync_and_prefetch_use_extract_level(
    monkeypatch, configured: str, expected: str,
) -> None:
    settings = Settings(thinking_level="medium", refval_extract_thinking_level=configured)
    client = CaptureClient()
    monkeypatch.setattr(llm, "get_llm_client", lambda _settings, *, role: client)
    assert runner._llm_complete("user", "system", settings) == "draft"
    assert client.calls[0]["thinking_level"] == expected

    queued: list[dict] = []

    def enqueue(**kwargs) -> bool:
        queued.append(kwargs)
        return True

    monkeypatch.setattr(agent_generate, "enqueue_speculative_repair", enqueue)
    request_id = runner.enqueue_speculative_extract(
        question="vector add", code="__global__ void add() {}",
        question_id=1, dialect="cuda", candidate=1, repair=0, settings=settings,
    )
    assert request_id
    assert queued[0]["llm_options"]["thinking_level"] == expected
    assert queued[0]["meta"].role == "refval_extract"


@pytest.mark.parametrize("dialect", ["cuda", "cutlass"])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize(("repair", "expected"), [(0, "high"), (1, "medium")])
def test_explicit_role_level_wins_over_dialect_and_fast_mode(
    monkeypatch, dialect: str, fast: bool, repair: int, expected: str,
) -> None:
    settings = Settings(
        async_llm_enabled=False,
        kernel_fast_mode=fast,
        generator_thinking_level="high",
        repair_compile_thinking_level="medium",
    )
    monkeypatch.setattr(kernel_graph, "get_settings", lambda: settings)
    captured: list[dict] = []

    def complete(**kwargs) -> GenerateResult:
        captured.append(kwargs)
        return GenerateResult(text="source", reasoning="", reasoning_source="empty")

    monkeypatch.setattr(kernel_graph, "complete_chat", complete)
    kernel_graph.generate({
        "question_id": 1, "dialect": dialect, "candidate_idx": 1,
        "repair_idx": repair, "messages": [{"role": "user", "content": "add"}],
        "system_prompt": "kernel",
    })
    options = captured[0]["llm_options"]
    assert options["thinking_level"] == expected
    assert captured[0]["meta"].role == ("repair.compile" if repair else "generator")
    if fast or dialect == "cutlass":
        assert options["max_output_tokens"] == 8192
    if dialect == "cutlass":
        assert options["reasoning_max_tokens"] == settings.cutlass_reasoning_max_tokens


@pytest.mark.parametrize(("dialect", "fast"), [("cuda", True), ("cutlass", False)])
def test_special_mode_level_still_applies_without_role_override(
    monkeypatch, dialect: str, fast: bool,
) -> None:
    settings = Settings(async_llm_enabled=False, kernel_fast_mode=fast)
    monkeypatch.setattr(kernel_graph, "get_settings", lambda: settings)
    captured: list[dict] = []

    def complete(**kwargs) -> GenerateResult:
        captured.append(kwargs)
        return GenerateResult(text="source", reasoning="", reasoning_source="empty")

    monkeypatch.setattr(kernel_graph, "complete_chat", complete)
    kernel_graph.generate({
        "question_id": 1, "dialect": dialect, "candidate_idx": 1, "repair_idx": 0,
        "messages": [{"role": "user", "content": "add"}], "system_prompt": "kernel",
    })
    assert captured[0]["llm_options"]["thinking_level"] == "low"
