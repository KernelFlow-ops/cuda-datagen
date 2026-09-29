"""LangGraph state for one knowledge question (no compile fields)."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class KnowledgeGraphState(TypedDict, total=False):
    """Mutable state for the knowledge pipeline."""

    question_id: int
    question: str
    input_metadata: dict[str, Any]
    kind: str
    topic: str
    track: str
    system_prompt: str
    user_prompt: str
    gen_system: str
    gen_user: str
    gen_prompt_variant: dict[str, Any]
    messages: list[dict[str, str]]
    candidate_idx: int
    repair_idx: int
    temperature: float
    raw_response: str
    answer: str
    gate_ok: bool
    gate_reasons: list[str]
    judge_pass: bool
    judge_score: float
    judge_issues: list[str]
    judge_must_fix: list[str]
    judge_dimensions: dict[str, float]
    judge_error: str
    judge_unavailable: bool
    judge_skipped_llm: bool
    status: Literal["running", "success", "abandoned"]
    abandon_reason: str
    attempts: list[dict[str, Any]]
    gpu_name: str
    cuda_arch: str
    cuda_version: str
    metadata: dict[str, Any]
    provenance: dict[str, Any]
    raw_reasoning: str
    reasoning_source: str
    origin: str
    cot: str
    cot_source: str
    cot_error: str
