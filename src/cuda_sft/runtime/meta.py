"""LLM call metadata shared by the gateway, fakes, and tracing (spec 04_specs/01 §1)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Role = Literal[
    "generator",
    "repair.compile",
    "repair.numeric",
    "repair.semantic",
    "critic",
    "cot_editor",
    "refval_extract",
    "knowledge_judge",
    "knowledge_generator",
    "knowledge_repair",
]

@dataclass(frozen=True)
class CallMeta:
    """Who is calling the LLM and why.

    Attributes:
        role: Logical role; drives model routing and fake lookup.
        job_key: ``q{question_id}:{track}`` until S3, then ADR-004 key.
        question_id: Legacy 1-based line id.
        track: Dialect name or ``knowledge:<topic>``.
        candidate: 1-based candidate index; 0 when not candidate-scoped.
        repair: Repair round; 0 for first turn.
        purpose: Short free-form tag (``vote_2``, ``extract_retry``...).
        attempt: Gateway-internal retry number (filled by the gateway).
        pin_model: Model alias to force (repair turns reuse the candidate's first-turn alias).
    """

    role: Role
    job_key: str
    question_id: int
    track: str
    candidate: int = 0
    repair: int = 0
    purpose: str = ""
    attempt: int = 1
    pin_model: str = ""

    def key(self) -> str:
        """Scenario lookup key ``c{candidate}r{repair}``."""
        return f"c{self.candidate}r{self.repair}"

    def with_attempt(self, attempt: int) -> CallMeta:
        """Return a copy with ``attempt`` replaced."""
        return replace(self, attempt=attempt)


def legacy_job_key(question_id: int, track: str) -> str:
    """S0-S2 job key format."""
    return f"q{int(question_id)}:{track}"
