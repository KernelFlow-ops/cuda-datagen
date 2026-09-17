"""Conservative invariant fact cards. Arch-specific numbers are not checked here."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_AMD_RE = re.compile(r"\b(amd|rdna|cdna|wavefront|rocm)\b", re.IGNORECASE)
# Capture warp-size *claims*. Do not use a loose CUDA…128 skip: "warp size is 32;
# 128-byte sector" and "32, not 64" must not count as asserting 64/128.
_WARP_SIZE_CLAIM = re.compile(
    r"warp\s*(?:size|width|宽度|大小)\s*(?:is|=|:|为|是)?\s*(16|32|64|128)\b",
    re.IGNORECASE,
)
_WARP_HAS_CLAIM = re.compile(
    r"(?:a\s+)?warps?\s+(?:has|contains|of|is|为|是)\s*(16|32|64|128)\s*"
    r"(?:threads?|lanes?|个?线程)",
    re.IGNORECASE,
)
_WARP_CN_CLAIM = re.compile(
    r"(?:一个)?warp.{0,8}(?:有|为|是)\s*(16|32|64|128)\s*(?:个)?(?:线程|threads?)",
    re.IGNORECASE,
)
_NEG_BEFORE = re.compile(
    r"(?:not|n't|isn't|aren't|不是|而非|非|vs\.?|versus|对比)\s*$",
    re.IGNORECASE,
)
_UNIT_AFTER = re.compile(
    r"^\s*-?\s*(?:bytes?|B\b|KB|KiB|MB|GiB|bits?)",
    re.IGNORECASE,
)
_MAX_THREADS_WRONG = re.compile(
    r"(?:max(?:imum)?|at most|上限).{0,40}(?:2048|4096|8192)\s*"
    r"threads?\s*(?:per\s*)?block",
    re.IGNORECASE,
)
_UNLIMITED_THREADS = re.compile(
    r"(?:no limit|unlimited|没有上限).{0,40}threads?\s*(?:per\s*)?block|"
    r"threads?\s*(?:per\s*)?block.{0,40}(?:no limit|unlimited|没有上限)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FactCard:
    """One deterministic contradiction check."""

    fact_id: str
    description: str
    check: Callable[[str], str | None]
    """Return a reason string when the answer contradicts the fact, else None."""


def _mentions_amd(text: str) -> bool:
    return bool(_AMD_RE.search(text or ""))


def _is_asserted_wrong_warp_size(raw: str, match: re.Match[str]) -> bool:
    """True when ``match`` claims warp width 16/64/128, not a nearby 128-byte/not-64."""
    number = match.group(1)
    if number == "32":
        return False
    prefix = raw[: match.start()]
    suffix = raw[match.end() :]
    if _NEG_BEFORE.search(prefix[-24:]):
        return False
    if _UNIT_AFTER.match(suffix):
        return False
    return True


def _warp_size_violation(text: str) -> str | None:
    raw = text or ""
    if _mentions_amd(raw):
        return None
    for pattern in (_WARP_SIZE_CLAIM, _WARP_HAS_CLAIM, _WARP_CN_CLAIM):
        for match in pattern.finditer(raw):
            if _is_asserted_wrong_warp_size(raw, match):
                return "NVIDIA CUDA warp size is 32 threads, not 16/64/128"
    return None


def _threads_per_block_violation(text: str) -> str | None:
    raw = text or ""
    if _MAX_THREADS_WRONG.search(raw):
        return "modern NVIDIA GPUs cap threads per block at 1024, not 2048+"
    if _UNLIMITED_THREADS.search(raw):
        return "threads per block is limited (1024 on modern NVIDIA GPUs)"
    return None


FACT_CARDS: tuple[FactCard, ...] = (
    FactCard(
        "warp_size_32",
        "NVIDIA CUDA warp size is 32",
        _warp_size_violation,
    ),
    FactCard(
        "threads_per_block_1024",
        "Maximum threads per block is 1024 on modern NVIDIA GPUs",
        _threads_per_block_violation,
    ),
)


def fact_violations(text: str) -> list[str]:
    """Return human-readable reasons for invariant contradictions."""
    hits: list[str] = []
    for card in FACT_CARDS:
        reason = card.check(text)
        if reason:
            hits.append(reason)
    return hits
