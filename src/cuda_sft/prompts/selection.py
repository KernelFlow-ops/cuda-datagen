"""Resume-stable prompt-variant indexing and candidate temperatures.

The hash is deterministic so a resumed job picks the same system/suffix
as the original run. Do not introduce ``random`` here.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

CANDIDATE_TEMPERATURES = (0.2, 0.5, 0.8)

_CJK_RE = re.compile(r"[\u3400-\u9fff]")


def looks_chinese(text: str) -> bool:
    """Return True when ``text`` contains CJK ideographs.

    Args:
        text: Question or prompt body.

    Returns:
        True if at least one CJK code point is present.
    """
    return bool(_CJK_RE.search(text or ""))


def candidate_temperature(candidate_idx: int) -> float:
    """Temperature for candidate 1/2/3 (0.2 / 0.5 / 0.8).

    Args:
        candidate_idx: 1-based candidate number.
    """
    if candidate_idx <= 1:
        return CANDIDATE_TEMPERATURES[0]
    if candidate_idx >= len(CANDIDATE_TEMPERATURES):
        return CANDIDATE_TEMPERATURES[-1]
    return CANDIDATE_TEMPERATURES[candidate_idx - 1]


def stable_index(n: int, question_id: int, candidate_idx: int, salt: int) -> int:
    """Deterministic index in ``[0, n)`` (no randomness; resume-stable).

    Args:
        n: Pool size.
        question_id: 1-based jsonl line id.
        candidate_idx: 1-based candidate number.
        salt: Distinguishes system vs suffix vs repair pools.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError("empty prompt pool")
    return (int(question_id) * 31 + int(candidate_idx) * 17 + salt) % n


def language_matched_index(
    pool: Sequence[str],
    question: str,
    question_id: int,
    candidate_idx: int,
    salt: int,
) -> int:
    """Pick a pool index whose CJK-ness matches the question when possible.

    Args:
        pool: System or suffix prompt texts.
        question: Raw problem text.
        question_id: 1-based jsonl id.
        candidate_idx: 1-based candidate.
        salt: Distinguishes system vs suffix vs repair.

    Returns:
        Index into the original ``pool`` (not a compressed sub-list).
    """
    if not pool:
        raise ValueError("empty prompt pool")
    chinese = looks_chinese(question)
    matched = [i for i, text in enumerate(pool) if looks_chinese(text) == chinese]
    if not matched:
        return stable_index(len(pool), question_id, candidate_idx, salt)
    return matched[stable_index(len(matched), question_id, candidate_idx, salt)]
