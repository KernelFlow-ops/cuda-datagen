"""Extract CUDA source from model replies (markdown fences / thinking tags)."""

from __future__ import annotations

import re
from collections.abc import Callable

THINK_BLOCK_RE = re.compile(
    r"<(?:think|thinking|reasoning)>.*?</(?:think|thinking|reasoning)>",
    re.DOTALL | re.IGNORECASE,
)
THINK_INNER_RE = re.compile(
    r"<(?:think|thinking|reasoning)>(.*?)</(?:think|thinking|reasoning)>",
    re.DOTALL | re.IGNORECASE,
)
FENCE_RE = re.compile(
    r"```(?:cuda|cu|cpp|c\+\+|cc|cxx|c|hpp)?[ \t]*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)
ANY_FENCE_RE = re.compile(
    r"```[^\n]*\n(.*?)```",
    re.DOTALL,
)
LABELED_FENCE_RE = re.compile(
    r"```([^\n]*)\n(.*?)```",
    re.DOTALL,
)
LEADING_FENCE_LANG_RE = re.compile(
    r"^(?:cuda|cu|cpp|c\+\+|python|py|triton|tilelang|cutlass|cute|c|cc|cxx|hpp)\s*$",
    re.IGNORECASE,
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


def extract_thinking(text: str) -> str:
    """Return concatenated inner text of think/thinking/reasoning tags.

    Args:
        text: Raw model output that may contain thinking tags.

    Returns:
        Joined thinking bodies, or empty string if none are present.
    """
    if not text:
        return ""
    parts = [block.strip() for block in THINK_INNER_RE.findall(text) if block.strip()]
    return "\n\n".join(parts).strip()


def split_visible_and_thinking(text: str) -> tuple[str, str]:
    """Split a reply into visible text and tagged thinking.

    Args:
        text: Raw model output.

    Returns:
        ``(visible, thinking)``. Visible has think tags removed.
    """
    raw = text or ""
    return strip_thinking(raw), extract_thinking(raw)


def wrap_cot_assistant(cot: str, code: str) -> str:
    """Build an SFT assistant label: optional ``<think>`` block then CUDA source.

    Args:
        cot: Polished chain of thought (may be empty).
        code: Compile-passed CUDA source.

    Returns:
        Assistant content. If ``cot`` is blank, returns ``code`` only.
    """
    code_out = _ensure_trailing_newline(code) if (code or "").strip() else ""
    cot_clean = (cot or "").strip()
    if not cot_clean:
        return code_out
    return f"<think>\n{cot_clean}\n</think>\n{code_out}"


def unwrap_cot_assistant(assistant: str) -> tuple[str, str]:
    """Split an assistant label into ``(cot, code)``.

    Args:
        assistant: SFT assistant content, with or without a think block.

    Returns:
        Polished CoT (empty if none) and CUDA source (thinking stripped).
    """
    raw = assistant or ""
    cot = extract_thinking(raw)
    code = extract_cuda_source(raw)
    return cot, code


def collapse_blank_lines(text: str) -> str:
    """Collapse runs of blank lines and strip edges."""
    return re.sub(r"\n{3,}", "\n\n", (text or "").strip())


def strip_fences(text: str) -> str:
    """Remove markdown fences, keeping inner text for non-CUDA fences.

    CUDA/C++ fences are dropped entirely so CoT does not keep a second kernel.
    Other fences keep their body as prose.
    """
    if not text:
        return ""

    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        lang_match = re.match(r"```([^\n]*)\n", raw)
        lang = (lang_match.group(1) if lang_match else "").strip().lower()
        body = match.group(1) if match.lastindex else ""
        cuda_langs = {"cuda", "cu", "cpp", "c++", "cc", "cxx", "c", "hpp"}
        if lang in cuda_langs or looks_like_cuda(body):
            return "\n"
        return body

    return ANY_FENCE_RE.sub(_replace, text)


def strip_code_from_cot(text: str) -> str:
    """Drop fenced or inline CUDA kernels from a CoT draft.

    Args:
        text: Agent or raw-thinking text.

    Returns:
        Prose-only CoT, possibly empty.
    """
    cleaned = strip_fences(text or "")
    cleaned = collapse_blank_lines(cleaned)
    if "__global__" not in cleaned:
        return cleaned
    cut = cleaned.find("__global__")
    prefix = collapse_blank_lines(cleaned[:cut])
    return prefix


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
    return extract_fenced_source(
        text,
        fence_langs=("cuda", "cu", "cpp", "c++", "cc", "cxx", "c", "hpp"),
        looks_like=looks_like_cuda,
    )


def extract_fenced_source(
    text: str,
    *,
    fence_langs: tuple[str, ...] | None = None,
    looks_like: Callable[[str], bool] | None = None,
) -> str:
    """Extract a source block from markdown fences.

    Prefer the last fence whose body matches ``looks_like``, then the last
    fence whose language tag is in ``fence_langs``, then the last fence,
    then the full reply with thinking stripped.
    """
    cleaned = strip_thinking(text)
    if not cleaned:
        return ""

    langs = {item.strip().lower() for item in (fence_langs or ()) if item.strip()}
    labeled = [
        (str(lang or "").strip().lower(), body.strip())
        for lang, body in LABELED_FENCE_RE.findall(cleaned)
        if body.strip()
    ]
    if not labeled:
        return _ensure_trailing_newline(cleaned)

    if looks_like is not None:
        for _lang, body in reversed(labeled):
            if looks_like(body):
                return _ensure_trailing_newline(_strip_leading_fence_lang(body))
    if langs:
        for lang, body in reversed(labeled):
            token = lang.split()[0] if lang else ""
            if token in langs:
                return _ensure_trailing_newline(_strip_leading_fence_lang(body))
    return _ensure_trailing_newline(_strip_leading_fence_lang(labeled[-1][1]))


def _strip_leading_fence_lang(body: str) -> str:
    """Drop a first line that is only a markdown language tag (e.g. ``cuda``)."""
    lines = (body or "").splitlines()
    if lines and LEADING_FENCE_LANG_RE.match(lines[0].strip()):
        return "\n".join(lines[1:]).lstrip("\n")
    return body


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
