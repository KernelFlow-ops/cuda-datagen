"""Extract CUDA source from model replies (markdown fences / thinking tags)."""

from __future__ import annotations

import re

THINK_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    re.DOTALL | re.IGNORECASE,
)
FENCE_RE = re.compile(
    r"```(?:cuda|cu|cpp|c\+\+|cc|cxx|c|hpp)?[ \t]*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
CUDA_HINTS = (
    "__global__",
    "__device__",
    "__host__",
    "__shared__",
    "cuda_runtime.h",
    "cuda_fp16.h",
    "<<<",
    "threadIdx",
    "blockIdx",
    "blockDim",
    "gridDim",
)


def strip_thinking(text: str) -> str:
    """Remove leaked ``<think>`` / ``<thinking>`` / ``<reasoning>`` blocks.

    Args:
        text: Raw model output.

    Returns:
        Text with thinking tags stripped.
    """
    cleaned = THINK_BLOCK_RE.sub("", text or "")
    return cleaned.strip()


def looks_like_cuda(source: str) -> bool:
    """Return True if ``source`` looks like CUDA/C++ device code.

    Args:
        source: Candidate source text.
    """
    lowered = source.lower()
    return any(hint.lower() in lowered for hint in CUDA_HINTS)


def extract_cuda_source(text: str) -> str:
    """Extract CUDA source from a model reply.

    Prefer the last fenced block that looks like CUDA; otherwise the last
    fenced block; otherwise the full reply with thinking stripped.
    """
    cleaned = strip_thinking(text)
    if not cleaned:
        return ""

    fences = [block.strip() for block in FENCE_RE.findall(cleaned) if block.strip()]
    if fences:
        for block in reversed(fences):
            if looks_like_cuda(block):
                return _ensure_trailing_newline(block)
        return _ensure_trailing_newline(fences[-1])

    return _ensure_trailing_newline(cleaned)


def _ensure_trailing_newline(source: str) -> str:
    """Strip surrounding whitespace and ensure a trailing newline.

    Args:
        source: CUDA source fragment.

    Returns:
        Normalized source, or empty string if ``source`` is blank.
    """
    source = source.strip()
    if not source:
        return ""
    return source + "\n"
