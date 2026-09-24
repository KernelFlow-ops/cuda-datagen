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

# Repair policy mirrors the difficulty topology.  A repair is substantially
# cheaper than opening another candidate, but unbounded retries can still
# starve the remaining candidates, so every class has a small fixed default.
_REPAIR_BUDGETS = {"simple": 1, "medium": 2, "hard": 3}


def repair_budget(
    difficulty: str,
    configured: int | object | None = None,
) -> int:
    """Return the per-candidate repair cap for a difficulty class.

    ``configured`` may be an integer cap or a Settings-like object with a
    ``max_repairs``/``knowledge_max_repairs`` attribute.  Keeping this helper
    independent of graph state makes it useful to synchronous and speculative
    repair callers alike. Unknown difficulty values use the medium policy.
    """
    name = (difficulty or "medium").strip().lower()
    policy = _REPAIR_BUDGETS.get(name, _REPAIR_BUDGETS["medium"])
    if configured is None:
        return policy
    if isinstance(configured, bool):
        # bool is an int subclass but is never a meaningful retry count.
        ceiling = int(configured)
    elif isinstance(configured, int):
        ceiling = configured
    else:
        ceiling = int(getattr(configured, "max_repairs", policy))
    return max(0, min(policy, ceiling))


# Explicit name for callers that prefer to distinguish this from a total
# process retry budget.
repair_budget_for_difficulty = repair_budget


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


_REFVAL_CLASSES = {
    "numeric_mismatch",
    "nan_inf",
    "crash",
    "timeout",
    "signature_mismatch",
    "reference_error",
}


def classify_refval_error(
    error: str,
    *,
    error_class: str = "",
) -> str:
    """Label a numeric-validation failure for the repair prompt.

    Args:
        error: Compressed evidence / harness text.
        error_class: Pre-labeled class from :class:`RefvalReport` if any.

    Returns:
        One of ``numeric_mismatch``, ``nan_inf``, ``crash``, ``timeout``,
        ``signature_mismatch``, ``reference_error``.
    """
    preset = (error_class or "").strip().lower()
    if preset in _REFVAL_CLASSES:
        return preset
    text = (error or "").lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "reference" in text:
        return "reference_error"
    if any(
        tok in text
        for tok in ("signature", "undeclared", "not declared", "no matching function")
    ):
        return "signature_mismatch"
    if "nan" in text or "inf mismatch" in text or "inf in" in text:
        return "nan_inf"
    if any(tok in text for tok in ("crash", "cuda error", "segfault", "aborted", "illegal")):
        return "crash"
    return "numeric_mismatch"


def wrap_repair_user(
    *,
    question: str,
    inner: str,
    error_class: str,
    dialect: str = "cuda",
    evidence: str = "",
    repair_idx: int | None = None,
    max_repairs: int | None = None,
) -> str:
    """Prefix a dialect repair body with the original problem and contract.

    Args:
        question: Raw jsonl question (not the generation suffix).
        inner: Dialect-specific repair body (error log + previous source).
        error_class: Output of :func:`classify_compile_error` or
            :func:`classify_refval_error`.
        dialect: Kernel dialect or ``knowledge``.
        evidence: Optional compressed numeric-mismatch block.
        repair_idx: Optional 1-based round number for an explicit budget note.
        max_repairs: Optional per-candidate repair cap.

    Returns:
        User message for the next generate turn.
    """
    problem = (question or "").strip() or "(empty problem)"
    extra = ""
    if (evidence or "").strip():
        extra = f"## Validation evidence\n{evidence.strip()}\n\n"
    budget = ""
    if repair_idx is not None or max_repairs is not None:
        round_text = str(int(repair_idx)) if repair_idx is not None else "?"
        cap_text = str(int(max_repairs)) if max_repairs is not None else "?"
        budget = f"- repair_round: {round_text}/{cap_text}\n"
    return (
        "## Original problem (authoritative; do not drop requirements)\n"
        f"{problem}\n\n"
        f"## Repair contract\n"
        f"- dialect: {dialect}\n"
        f"- error_class: {error_class}\n"
        f"{budget}"
        "- Rewrite the full artifact, not a diff.\n"
        "- Keep the algorithm / host responsibilities from the problem.\n"
        "- Do not invent missing project headers.\n\n"
        f"{extra}"
        f"{inner.strip()}\n"
    )
