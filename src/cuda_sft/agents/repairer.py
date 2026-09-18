"""Compile/gate Repairer: dedicated system prompt, error class, original question.

Repair turns must re-attach the raw problem. Chat history can be truncated
by ``MAX_INPUT_TOKENS``, which is how earlier repairs drifted off-spec.
"""

from __future__ import annotations

import re

# compile-fix specialist; generator system prompt is intentionally not reused.
REPAIR_SYSTEM_CUDA = (
    "You are a compile-fix CUDA kernel repairer, not a new-algorithm designer. "
    "Emit one complete translation unit that nvcc -c can compile. "
    "Keep the original problem's algorithm and host entry. "
    "Do not emit a diff. Do not invent project headers such as include/helpers.h."
)
REPAIR_SYSTEM_PYTHON = (
    "You are a compile-fix GPU Python kernel repairer (Triton or TileLang). "
    "Emit one complete solution.py that the import/JIT gate accepts. "
    "Keep the original algorithm. Do not rewrite it as CUDA C++. "
    "Do not launch kernels at import time. Do not emit a diff."
)
REPAIR_SYSTEM_KNOWLEDGE = (
    "You are a CUDA/NVIDIA architecture tutor rewriting a failed knowledge answer. "
    "Rewrite the full answer, not a patch. Do not emit a compilable kernel."
)

_MISSING_HEADER = re.compile(
    r"no such file or directory|cannot find|file not found|fatal error:.*\.h",
    re.IGNORECASE,
)
_UNDECLARED = re.compile(
    r"was not declared|undeclared identifier|identifier .* is undefined|"
    r"nameerror|not defined",
    re.IGNORECASE,
)
_SYNTAX = re.compile(
    r"expected |syntax error|invalid syntax|missing (?:semicolon|;)",
    re.IGNORECASE,
)
_TEMPLATE = re.compile(
    r"template|cute::|cutlass::|substitution failure|no matching function",
    re.IGNORECASE,
)
_IMPORT_TIME = re.compile(
    r"import(?:[- ]time)?|timeout|jit|triton.*error|tilelang",
    re.IGNORECASE,
)


def repair_system_prompt(dialect: str) -> str:
    """System prompt for a repair round.

    Args:
        dialect: Kernel dialect id or ``knowledge``.
    """
    name = (dialect or "cuda").strip().lower()
    if name == "knowledge":
        return REPAIR_SYSTEM_KNOWLEDGE
    if name in {"triton", "tilelang"}:
        return REPAIR_SYSTEM_PYTHON
    return REPAIR_SYSTEM_CUDA


def classify_compile_error(
    error: str,
    *,
    dialect: str = "cuda",
    code: str = "",
) -> str:
    """Label a compiler/gate failure for the repair prompt.

    Args:
        error: Compiler or gate text (already trimmed).
        dialect: Kernel dialect or ``knowledge``.
        code: Last extracted source; used to detect dialect leaks.

    Returns:
        One of ``empty_source``, ``missing_header``, ``undeclared``, ``syntax``,
        ``template``, ``dialect_violation``, ``import_time``, ``other``.
    """
    text = error or ""
    lowered = text.lower()
    source = code or ""
    name = (dialect or "cuda").strip().lower()
    if "no " in lowered and "source extracted" in lowered:
        return "empty_source"
    if name == "triton" and "__global__" in source:
        return "dialect_violation"
    if name == "tilelang" and "__global__" in source and "tilelang" not in source.lower():
        return "dialect_violation"
    if name == "cutlass" and any(
        tok in source for tok in ("SM90", "cute::SM90", "tma_load", "cp.async.bulk")
    ):
        return "dialect_violation"
    if _MISSING_HEADER.search(text):
        return "missing_header"
    if _UNDECLARED.search(text):
        return "undeclared"
    if _SYNTAX.search(text):
        return "syntax"
    if name in {"cuda", "cutlass"} and _TEMPLATE.search(text):
        return "template"
    if name in {"triton", "tilelang"} and _IMPORT_TIME.search(text):
        return "import_time"
    return "other"


def wrap_repair_user(
    *,
    question: str,
    inner: str,
    error_class: str,
    dialect: str = "cuda",
) -> str:
    """Prefix a dialect repair body with the original problem and contract.

    Args:
        question: Raw jsonl question (not the generation suffix).
        inner: Dialect-specific repair body (error log + previous source).
        error_class: Output of :func:`classify_compile_error`.
        dialect: Kernel dialect or ``knowledge``.

    Returns:
        User message for the next generate turn.
    """
    problem = (question or "").strip() or "(empty problem)"
    return (
        "## Original problem (authoritative; do not drop requirements)\n"
        f"{problem}\n\n"
        f"## Repair contract\n"
        f"- dialect: {dialect}\n"
        f"- error_class: {error_class}\n"
        "- Rewrite the full artifact, not a diff.\n"
        "- Keep the algorithm / host responsibilities from the problem.\n"
        "- Do not invent missing project headers.\n\n"
        f"{inner.strip()}\n"
    )
