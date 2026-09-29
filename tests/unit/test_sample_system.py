"""Training system text comes from an explicit mode and selected candidate."""

from types import SimpleNamespace

import pytest

from cuda_sft.core.sample import pick_reasoning, training_system
from cuda_sft.knowledge.prompt import (
    TRAINING_SYSTEM_PROMPT_EN as KNOWLEDGE_TRAINING_SYSTEM_PROMPT_EN,
    TRAINING_SYSTEM_PROMPT_ZH as KNOWLEDGE_TRAINING_SYSTEM_PROMPT_ZH,
)
from cuda_sft.prompt import TRAINING_SYSTEM_PROMPT_EN, TRAINING_SYSTEM_PROMPT_ZH


@pytest.mark.parametrize(
    ("mode", "question", "expected"),
    [
        ("fixed", "实现 CUDA 加法", TRAINING_SYSTEM_PROMPT_ZH),
        ("fixed", "Implement vector add", TRAINING_SYSTEM_PROMPT_EN),
        ("generation", "Implement vector add", "generation system"),
        ("none", "Implement vector add", None),
    ],
)
def test_training_system_modes(mode: str, question: str, expected: str | None) -> None:
    selected = {"gen_system": "generation system"}
    assert training_system(selected, question, SimpleNamespace(sft_system_mode=mode)) == expected


@pytest.mark.parametrize(
    ("mode", "question", "expected"),
    [
        ("fixed", "Explain CUDA occupancy", KNOWLEDGE_TRAINING_SYSTEM_PROMPT_EN),
        ("fixed", "请解释 CUDA 占用率", KNOWLEDGE_TRAINING_SYSTEM_PROMPT_ZH),
        ("generation", "Explain CUDA occupancy", "generation system"),
        ("none", "Explain CUDA occupancy", None),
    ],
)
def test_knowledge_training_system_modes(
    mode: str, question: str, expected: str | None
) -> None:
    selected = {"gen_system": "generation system"}
    system = training_system(
        selected, question, SimpleNamespace(sft_system_mode=mode), kind="knowledge"
    )
    assert system == expected
    if mode == "fixed":
        assert "compil" not in system.lower()


def test_fixed_system_never_includes_repair_protocol() -> None:
    selected = {"gen_system": "You are a compile-fix repairer"}
    system = training_system(selected, "Implement vector add", SimpleNamespace(sft_system_mode="fixed"))
    assert system is not None
    assert "repairer" not in system.lower()
    assert "compile-fix" not in system.lower()


@pytest.mark.parametrize(
    ("repairs", "policy", "expected"),
    [
        (0, "synthetic", ("first", "api", "polish")),
        (1, "synthetic", ("", "none", "synthetic")),
        (1, "drop_cot", ("", "none", "drop")),
        (1, "first_turn", ("first", "api", "polish")),
    ],
)
def test_pick_reasoning_uses_repaired_policy(repairs: int, policy: str, expected: tuple[str, str, str]) -> None:
    selected = {"repairs": repairs, "first_turn_reasoning": "first", "first_turn_reasoning_source": "api"}
    assert pick_reasoning(selected, SimpleNamespace(cot_repaired_policy=policy)) == expected
