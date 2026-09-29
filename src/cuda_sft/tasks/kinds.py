"""Task kinds and the CLI-level Job record.

Kernel jobs keep ``track`` equal to the dialect id (``cuda``, ``cutlass``, ...).
Knowledge jobs use ``knowledge:{topic}`` so progress keys stay disjoint from
``KNOWN_DIALECTS`` without adding a fake dialect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import unicodedata
from typing import Any, Literal, Mapping

Kind = Literal["kernel", "knowledge"]

KNOWN_TOPICS = (
    "architecture",
    "memory",
    "execution",
    "formula",
    "cute",
    "cutlass",
    "isa",
    "api",
    "worked_example",
    "general",
)


def normalize_question(text: str) -> str:
    """Normalize question text for stable identity across input files."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def question_hash(text: str) -> str:
    """SHA256 of the normalized question, independent of its source line."""
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def knowledge_track(topic: str) -> str:
    """Progress / dialect-slot key for one knowledge topic."""
    name = (topic or "general").strip().lower() or "general"
    if name.startswith("knowledge:"):
        return name
    return f"knowledge:{name}"


def topic_from_track(track: str) -> str:
    """Extract the topic id from a knowledge progress key."""
    raw = (track or "").strip().lower()
    if raw.startswith("knowledge:"):
        return raw.split(":", 1)[1] or "general"
    if raw in KNOWN_TOPICS:
        return raw
    return "general"


@dataclass(frozen=True)
class QuestionRow:
    """One jsonl question with optional extra fields (``task``, ``topic``, ...)."""

    question_id: int
    question: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClassifiedQuestion:
    """A question after task-kind resolution, before dialect expansion."""

    question_id: int
    question: str
    kind: Kind
    topic: str
    source: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Job:
    """One pipeline invocation: a question plus a track (dialect or topic).

    Attributes:
        question_id: 1-based jsonl line id.
        question: Problem text.
        kind: ``kernel`` (compile gate) or ``knowledge`` (rubric gate).
        track: Dialect name, or ``knowledge:{topic}``.
        source: How ``kind`` was chosen (``cli``, ``field``, ``heuristic``).
        extras: Original json object, not compared for equality.
    """

    question_id: int
    question: str
    kind: Kind
    track: str
    source: str = "cli"
    extras: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def topic(self) -> str:
        """Knowledge topic id; empty for kernel jobs."""
        if self.kind != "knowledge":
            return ""
        return topic_from_track(self.track)

    @property
    def dialect(self) -> str:
        """Value written to ``progress.jsonl`` as ``dialect``."""
        return self.track


def coerce_job(job: Job | tuple[Any, ...]) -> Job:
    """Accept a :class:`Job` or a legacy ``(id, question[, dialect])`` tuple."""
    if isinstance(job, Job):
        return job
    if len(job) >= 3:
        return Job(
            question_id=int(job[0]),
            question=str(job[1]),
            kind="kernel",
            track=str(job[2] or "cuda"),
            source="tuple",
        )
    return Job(
        question_id=int(job[0]),
        question=str(job[1]),
        kind="kernel",
        track="cuda",
        source="tuple",
    )


def progress_key(job: Job) -> tuple[int, str]:
    """``(question_id, track)`` used by ``load_done_keys``."""
    return job.question_id, job.track


def row_from_mapping(question_id: int, payload: Mapping[str, Any]) -> QuestionRow | None:
    """Build a row from a json object; skip if ``question`` is missing."""
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return None
    return QuestionRow(
        question_id=int(question_id),
        question=question,
        raw=dict(payload),
    )
