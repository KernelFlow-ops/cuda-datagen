"""Typed payloads passed between generator, repairer, critic, and graph nodes.

Graph state remains a ``TypedDict`` bag for LangGraph. These dataclasses
document the subset each role actually consumes or produces so new nodes
do not keep widening ``GraphState`` without a contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerateResult:
    """Visible text plus optional teacher reasoning from one LLM turn.

    Attributes:
        text: Assistant-visible completion (think tags already split).
        reasoning: Captured chain-of-thought / thinking, possibly empty.
        reasoning_source: How reasoning was obtained (``empty`` if none).
        used_speculative: True when the text came from the async repair pool.
    """

    text: str
    reasoning: str = ""
    reasoning_source: str = "empty"
    used_speculative: bool = False


@dataclass
class RepairRequest:
    """Inputs the Repairer needs; always includes the original question.

    Attributes:
        question: Raw problem text (not the generation suffix).
        previous_code: Last extracted source or knowledge answer.
        compile_error: Compiler / gate message, already trimmed.
        cuda_arch: Target SM string for kernel repairs.
        question_id: 1-based jsonl id (variant selection).
        candidate_idx: 1-based candidate.
        repair_idx: 1-based repair round about to run.
        dialect: Kernel dialect or ``knowledge``.
        error_class: Optional classifier label (filled in M2).
    """

    question: str
    previous_code: str
    compile_error: str
    cuda_arch: str = ""
    question_id: int = 1
    candidate_idx: int = 1
    repair_idx: int = 1
    dialect: str = "cuda"
    error_class: str = ""


@dataclass
class CriticResult:
    """Semantic critic output (wired in M3; defined here to freeze the shape).

    Attributes:
        passed: True when the sample needs no extra repair.
        must_fix: Blocking issues.
        issues: Non-blocking nits.
        skipped: True when adaptive policy did not call the LLM.
        raw: Optional parsed JSON payload.
    """

    passed: bool = True
    must_fix: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    skipped: bool = True
    raw: dict[str, Any] = field(default_factory=dict)
