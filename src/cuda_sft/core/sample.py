"""Render training prompts from the selected candidate only."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cuda_sft.prompt import TRAINING_SYSTEM_PROMPT_EN, TRAINING_SYSTEM_PROMPT_ZH
from cuda_sft.prompts.selection import looks_chinese


def training_system(
    selected: Mapping[str, Any], question: str, cfg: Any, *, kind: str = "kernel"
) -> str | None:
    mode = str(getattr(cfg, "sft_system_mode", "fixed") or "fixed")
    if mode == "none":
        return None
    if mode == "generation":
        return str(selected["gen_system"])
    if kind == "knowledge":
        from cuda_sft.knowledge.prompt import (
            TRAINING_SYSTEM_PROMPT_EN as KNOWLEDGE_TRAINING_SYSTEM_PROMPT_EN,
            TRAINING_SYSTEM_PROMPT_ZH as KNOWLEDGE_TRAINING_SYSTEM_PROMPT_ZH,
        )

        return (
            KNOWLEDGE_TRAINING_SYSTEM_PROMPT_ZH
            if looks_chinese(question)
            else KNOWLEDGE_TRAINING_SYSTEM_PROMPT_EN
        )
    return TRAINING_SYSTEM_PROMPT_ZH if looks_chinese(question) else TRAINING_SYSTEM_PROMPT_EN


def training_user(state: Mapping[str, Any], selected: Mapping[str, Any], cfg: Any) -> str:
    question = str(state.get("question") or "").strip()
    generated = str(selected.get("gen_user") or "").strip()
    if getattr(cfg, "sft_user_is_raw_question", True) and question:
        return question
    return generated or question


def pick_reasoning(selected: Mapping[str, Any], cfg: Any) -> tuple[str, str, str]:
    if int(selected.get("repairs") or 0) == 0:
        return (
            str(selected.get("first_turn_reasoning") or ""),
            str(selected.get("first_turn_reasoning_source") or "empty"),
            "polish",
        )
    policy = str(getattr(cfg, "cot_repaired_policy", "synthetic") or "synthetic")
    if policy == "drop_cot":
        return "", "none", "drop"
    if policy == "first_turn":
        return (
            str(selected.get("first_turn_reasoning") or ""),
            str(selected.get("first_turn_reasoning_source") or "empty"),
            "polish",
        )
    return "", "none", "synthetic"
