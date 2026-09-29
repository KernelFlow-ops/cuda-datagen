"""Extract prose answers and cheap structural features (no CUDA extractor)."""

from __future__ import annotations

import re

from cuda_sft.core.cot import clean_raw_reasoning as _clean_raw_reasoning
from cuda_sft.core.cot import clip_at_boundary
from cuda_sft.parse import (
    ANY_FENCE_RE,
    collapse_blank_lines,
    extract_thinking,
    split_visible_and_thinking,
    strip_thinking,
)

MIN_COT_CHARS = 120

_TRUNCATED_TAIL_RE = re.compile(
    r"(?:\.\.\.\s*\[truncated[^\]]*\]|\[truncated[^\]]*\]|\.\.\.|…)$",
    re.IGNORECASE,
)


def clean_raw_reasoning(text: str, max_chars: int) -> str:
    """Normalize teacher thinking before storing or sending to the CoT editor.

    Formulas and code are kept (knowledge answers may quote them).

    Args:
        text: Raw API reasoning, possibly tagged.
        max_chars: Budget; ``<=0`` disables. Keeps the head and the tail.
    """
    return _clean_raw_reasoning(text, max_chars, strip_code=False)

HEADING_RE = re.compile(r"(?m)^(#{1,4}\s+\S+|#{0,4}\s*(结论|推导|机制|适用边界|答案|Answer|Conclusion|Derivation|Caveats)\b)")
NUMBERED_RE = re.compile(r"(?m)^\s*(?:\d+[\.\)]\s+\S+|[-*]\s+\S+)")
EQUATION_RE = re.compile(
    r"(\$[^$]+\$)|(\\frac\b)|(\\mathrm\b)|(occupancy\s*=)|"
    r"([A-Za-z\\][A-Za-z0-9_\\{}]*\s*=\s*[^\n]{1,80})|[≈≤≥]",
)
STEP_RE = re.compile(
    r"(?i)(因此|所以|故|假设|其中|step\s*\d|therefore|where\b|thus\b|hence\b|推导)",
)


def extract_answer(text: str) -> str:
    """Visible assistant text with think tags removed. Keeps formulas and fences.

    Args:
        text: Raw model reply.
    """
    visible, _thinking = split_visible_and_thinking(text or "")
    return collapse_blank_lines(visible)


def answer_integrity_issues(answer: str) -> list[str]:
    """Catch visible signs of an unfinished response before quality review."""
    body = (answer or "").rstrip()
    if not body:
        return []
    issues: list[str] = []
    if _TRUNCATED_TAIL_RE.search(body) or body.endswith((",", ":", ";", "=", "->", "→")):
        issues.append("answer ends mid-thought or with a truncation marker")
    if len(re.findall(r"(?m)^[ \t]*```", body)) % 2:
        issues.append("answer has an unclosed Markdown fence")
    if body.count("$$") % 2 or body.count(r"\[") != body.count(r"\]"):
        issues.append("answer has an unclosed display equation")
    return issues


def fence_char_ratio(text: str) -> float:
    """Fraction of characters that live inside markdown fences.

    Args:
        text: Knowledge answer; high ratios fail the hard gate on theory topics.
    """
    raw = text or ""
    if not raw:
        return 0.0
    total = 0
    for match in ANY_FENCE_RE.finditer(raw):
        total += len(match.group(1) or "")
    return total / max(len(raw), 1)


def has_equation(text: str) -> bool:
    """True if the text looks like it contains an equality or LaTeX formula."""
    return bool(EQUATION_RE.search(text or ""))


def looks_structured(text: str) -> bool:
    """True if there is a heading, numbered list, or labeled conclusion."""
    raw = text or ""
    if HEADING_RE.search(raw):
        return True
    return len(NUMBERED_RE.findall(raw)) >= 2


def has_derivation_steps(text: str) -> bool:
    """True if the text has more than a single concluding sentence."""
    raw = text or ""
    if len(STEP_RE.findall(raw)) >= 2:
        return True
    return len(NUMBERED_RE.findall(raw)) >= 3


def sanitize_knowledge_cot(text: str, max_chars: int) -> str:
    """Drop think wrappers; keep formulas. Unlike kernel CoT, do not strip math."""
    raw = text or ""
    tagged = extract_thinking(raw)
    body = tagged if tagged.strip() else strip_thinking(raw)
    body = collapse_blank_lines(body)
    return clip_at_boundary(body, max_chars)
