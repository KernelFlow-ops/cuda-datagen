"""LangGraph state schemas for one-question CUDA SFT generation."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class AttemptRecord(TypedDict, total=False):
    """One generate/compile attempt (candidate + repair round)."""

    candidate: int
    repair: int
    ok: bool
    used_rdc: bool
    error: str


class GraphState(TypedDict, total=False):
    """Mutable state passed between LangGraph nodes for a single question."""

    question_id: int
    question: str
    system_prompt: str
    user_prompt: str
    messages: list[dict[str, str]]
    candidate_idx: int
    repair_idx: int
    temperature: float
    raw_response: str
    code: str
    compile_ok: bool
    compile_error: str
    used_rdc: bool
    status: Literal["running", "success", "abandoned"]
    attempts: list[dict[str, Any]]
    gpu_name: str
    cuda_arch: str
    cuda_version: str
