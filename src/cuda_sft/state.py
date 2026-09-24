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
    kind: str
    dialect: str
    system_prompt: str
    user_prompt: str
    messages: list[dict[str, str]]
    candidate_idx: int
    repair_idx: int
    repair_cap: int
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
    winner_found: bool
    winner_candidate: int
    judge_score: int
    judge_issues: list[str]
    judge_suggestions: list[str]
    speculative_requests: list[str]
    skip_repair: bool
    metadata: dict[str, Any]
    raw_reasoning: str
    reasoning_source: str
    cot: str
    cot_source: str
    cot_error: str
    difficulty: str
    candidate_cap: int
    use_critic: bool
    critic_pass: bool
    critic_skipped: bool
    critic_must_fix: list[str]
    critic_issues: list[str]
    # Stable task/oracle contracts and audit fields.  ``total=False`` keeps
    # these optional for legacy graph states loaded from older progress files.
    task_spec: dict[str, Any]
    oracle_spec: dict[str, Any]
    quality_status: dict[str, Any]
    provenance: dict[str, Any]
    candidate_reports: list[dict[str, Any]]
    input_metadata: dict[str, Any]
    source: str
    refval_ok: bool
    refval_status: str
    refval_error: str
    refval_error_class: str
