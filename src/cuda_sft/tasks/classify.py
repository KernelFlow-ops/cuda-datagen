"""Resolve kernel vs knowledge without mutating the kernel dialect registry."""

from __future__ import annotations

import re
from typing import Any, Mapping

from cuda_sft.tasks.kinds import (
    KNOWN_TOPICS,
    ClassifiedQuestion,
    Kind,
    QuestionRow,
    topic_from_track,
)

KERNEL_FIELD_ALIASES = {
    "kernel",
    "code",
    "impl",
    "implementation",
    "operator",
    "cuda",
}
KNOWLEDGE_FIELD_ALIASES = {
    "knowledge",
    "theory",
    "explain",
    "concept",
    "prose",
    "qa",
    "reason",
}

TOPIC_ALIASES = {
    "arch": "architecture",
    "sm": "architecture",
    "gpu": "architecture",
    "hardware": "architecture",
    "mem": "memory",
    "shared": "memory",
    "coalescing": "memory",
    "occ": "execution",
    "occupancy": "execution",
    "scheduler": "execution",
    "math": "formula",
    "derive": "formula",
    "derivation": "formula",
    "roofline": "formula",
    "cutlass_model": "cutlass",
    "pipeline": "cutlass",
    "layout": "cute",
    "tiler": "cute",
    "ptx": "isa",
    "sass": "isa",
    "runtime": "api",
    "driver": "api",
    "example": "worked_example",
    "worked": "worked_example",
}

_KERNEL_RE = re.compile(
    r"solution\.cu|include/solution_header|function signature|"
    r"your implementation should|\bimplement a\b|\bnvcc\b|__global__|"
    r"write your implementation|实现一个|编写.{0,12}kernel",
    re.IGNORECASE,
)
_KNOWLEDGE_RE = re.compile(
    r"\bexplain\b|\bwhy does\b|\bwhat is\b|\bderive\b|occupancy formula|"
    r"memory hierarchy|cute layout|roofline|请解释|请推导|为什么|什么是|"
    r"工作原理|公式推导",
    re.IGNORECASE,
)

_TOPIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "formula",
        re.compile(
            r"\bderive\b|推导|occupancy formula|roofline|arithmetic intensity|"
            r"公式|量纲|prove that",
            re.IGNORECASE,
        ),
    ),
    (
        "cute",
        re.compile(
            r"cute::|cute layout|layout algebra|\btiler\b|mma atom|copy atom|"
            r"\bcosize\b|\bshape\b.+\bstride\b",
            re.IGNORECASE,
        ),
    ),
    (
        "cutlass",
        re.compile(
            r"cutlass (?:pipeline|collective|mainloop|epilogue)|collective builder",
            re.IGNORECASE,
        ),
    ),
    (
        "isa",
        re.compile(r"\bptx\b|\bsass\b|instruction set", re.IGNORECASE),
    ),
    (
        "api",
        re.compile(
            r"cuda runtime|cuda driver api|stream semantics|cudaMalloc|"
            r"cuda streams?|cudaEvent|asynchronous execution|kernel launch",
            re.IGNORECASE,
        ),
    ),
    (
        "memory",
        re.compile(
            r"coalesc|bank conflict|shared memory bank|\bl2 cache\b|\btma\b|"
            r"访存合并|memory hierarchy|local memory|constant memory|"
            r"unified memory|pinned host|pageable|read-only cach|"
            r"texture (?:or read-only )?cach|structure of arrays|"
            r"array of structures|\bAoS\b|\bSoA\b|memory alignment|"
            r"global memory transactions",
            re.IGNORECASE,
        ),
    ),
    (
        "execution",
        re.compile(
            r"\boccupancy\b|latency hiding|warp scheduler|issue slot|"
            r"\bSIMT\b|branch divergence|warp execution|"
            r"threads per block|block dimensions|tiling in CUDA",
            re.IGNORECASE,
        ),
    ),
    (
        "architecture",
        re.compile(
            r"streaming multiprocessor|\bsm architecture\b|tensor core|"
            r"gpu architecture|\bhopper\b|\bampere\b|\bblackwell\b|warp 调度|"
            r"threadIdx|blockIdx|gridDim|thread organization|"
            r"threads, blocks, and grids",
            re.IGNORECASE,
        ),
    ),
)


def parse_task_field(raw: Mapping[str, Any] | None) -> Kind | None:
    """Read an explicit ``task`` / ``type`` / ``kind`` field, if valid."""
    if not raw:
        return None
    for key in ("task", "type", "kind"):
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        name = value.strip().lower()
        if name in KERNEL_FIELD_ALIASES:
            return "kernel"
        if name in KNOWLEDGE_FIELD_ALIASES:
            return "knowledge"
    return None


def normalize_topic(raw: str | None) -> str:
    """Map aliases to a canonical topic; unknown names become ``general``."""
    name = (raw or "").strip().lower().replace(" ", "_")
    if name.startswith("knowledge:"):
        name = topic_from_track(name)
    name = TOPIC_ALIASES.get(name, name)
    if name in KNOWN_TOPICS:
        return name
    return "general"


def infer_topic(question: str, raw: Mapping[str, Any] | None = None) -> str:
    """Topic from the json field, else keyword heuristics, else ``general``."""
    if raw:
        field = raw.get("topic")
        if isinstance(field, str) and field.strip():
            return normalize_topic(field)
    text = question or ""
    for topic, pattern in _TOPIC_PATTERNS:
        if pattern.search(text):
            return topic
    return "general"


def heuristic_kind(question: str) -> Kind:
    """Conservative text heuristic. Conflicts and unknowns become kernel."""
    text = question or ""
    kernel_hit = bool(_KERNEL_RE.search(text))
    knowledge_hit = bool(_KNOWLEDGE_RE.search(text))
    if kernel_hit:
        return "kernel"
    if knowledge_hit:
        return "knowledge"
    return "kernel"


def classify_row(row: QuestionRow, mode: str) -> ClassifiedQuestion:
    """Resolve kind/topic for one question under ``TASK_MODE``.

    ``kernel`` / ``knowledge`` force the whole file. ``auto`` uses the row
    field, then a heuristic that prefers kernel on conflict.
    """
    forced = (mode or "kernel").strip().lower()
    topic = infer_topic(row.question, row.raw)
    if forced == "kernel":
        return ClassifiedQuestion(
            question_id=row.question_id,
            question=row.question,
            kind="kernel",
            topic=topic,
            source="cli",
            raw=dict(row.raw),
        )
    if forced == "knowledge":
        return ClassifiedQuestion(
            question_id=row.question_id,
            question=row.question,
            kind="knowledge",
            topic=topic,
            source="cli",
            raw=dict(row.raw),
        )

    field_kind = parse_task_field(row.raw)
    if field_kind is not None:
        return ClassifiedQuestion(
            question_id=row.question_id,
            question=row.question,
            kind=field_kind,
            topic=topic if field_kind == "knowledge" else "general",
            source="field",
            raw=dict(row.raw),
        )
    kind = heuristic_kind(row.question)
    return ClassifiedQuestion(
        question_id=row.question_id,
        question=row.question,
        kind=kind,
        topic=topic if kind == "knowledge" else "general",
        source="heuristic",
        raw=dict(row.raw),
    )
