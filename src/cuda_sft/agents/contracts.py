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
        origin: Source of the completion (``live_api`` for provider SDK calls).
    """

    text: str
    reasoning: str = ""
    reasoning_source: str = "empty"
    used_speculative: bool = False
    origin: str = "unknown"


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
