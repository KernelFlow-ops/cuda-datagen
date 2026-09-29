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
REPAIR_SYSTEM_NUMERIC = (
    "You repair a GPU kernel that compiled but failed real numeric validation. "
    "Use the failed cases and values to fix indexing, math, bounds, or data movement. "
    "You may change the implementation algorithm, but preserve the original problem, "
    "host entry signature, and requested dialect. Emit a complete source file, not a diff."
)
REPAIR_SYSTEM_SEMANTIC = (
    "You repair a GPU kernel rejected by a semantic reviewer. "
    "Fix the reported host/API or algorithm mismatch against the original problem. "
    "You may change the implementation algorithm, but preserve the required host entry "
    "and requested dialect. Emit a complete source file, not a diff."
)

_MISSING_HEADER = re.compile(
    r"no such file or directory|cannot find|file not found|fatal error:.*\.h|"
    r"no module named",
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
# Python gates: a timeout means the module did real work (or hung) on import;
# lowering / missing-kernel messages come from the JIT structure checks.
_IMPORT_TIME = re.compile(r"timed out|timeout|import[- ]time|at import", re.IGNORECASE)
_JIT_COMPILE = re.compile(
    r"lowering failed|no @triton\.jit kernel|no tilelang prim_func|prim_?func|jitkernel|"
    r"unsupported tilelang|compilationerror|triton\.compiler",
    re.IGNORECASE,
)
_C_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_PY_COMMENT_OR_STRING = re.compile(
    r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'|#[^\n]*'
)

# One actionable line per error class, shown as ``- hint:`` in the repair
# contract. A label alone gave the model nothing to act on.
_REPAIR_HINTS = {
    "empty_source": (
        "No source was extracted; reply with the complete file inside one fenced code block."
    ),
    "missing_header": (
        "Do not include headers or import modules that are absent; define what you need "
        "in this file and rely only on the toolkit and libraries that are installed."
    ),
    "undeclared": (
        "Declare or define every identifier you use, include the toolkit header that "
        "provides it, and check spelling and scope."
    ),
    "syntax": "Fix the syntax at the reported lines first: braces, semicolons, launch syntax.",
    "template": (
        "Simplify template usage: spell out template arguments and avoid overloads or "
        "metaprogramming the compiler cannot resolve."
    ),
    "dialect_violation": "The source leaves the requested dialect; rewrite it in that dialect only.",
    "import_time": (
        "Keep module import side-effect free: define kernels and the host entry, but do "
        "not launch or run tests at import time."
    ),
    "jit_compile": (
        "Fix the kernel so it JIT-compiles: a decorated kernel must exist and its "
        "language API calls, constexpr arguments, dtypes and shapes must be valid."
    ),
    "numeric_mismatch": (
        "Recheck index math, strides, reduction order and identity, and data movement "
        "against the failing case."
    ),
    "nan_inf": (
        "Look for division by zero, uninitialized shared memory, a wrong reduction "
        "identity, or overflow in exp/log."
    ),
    "crash": (
        "Look for out-of-bounds indexing, an invalid launch configuration, or shared "
        "memory beyond the per-block limit."
    ),
    "timeout": (
        "Check loop termination, grid-stride bounds, and __syncthreads() inside "
        "divergent branches."
    ),
    "signature_mismatch": (
        "Match the host entry name, parameter order, and parameter types exactly."
    ),
    "semantic": (
        "Resolve each reported issue explicitly and keep the parts that already satisfy "
        "the problem."
    ),
}


def repair_hint(error_class: str) -> str:
    """Actionable one-line hint for ``error_class`` (empty when none applies)."""
    return _REPAIR_HINTS.get((error_class or "").strip().lower(), "")


def _code_only(source: str, dialect: str) -> str:
    """Drop comments (and Python strings/docstrings) before scanning for dialect leaks.

    C++ string literals stay: inline PTX such as ``cp.async.bulk`` lives in them.
    """
    if dialect in {"triton", "tilelang"}:
        return _PY_COMMENT_OR_STRING.sub(" ", source)
    return _C_COMMENT.sub(" ", source)


# Dialects whose generator system prompt carries rules a repair must keep.
# Plain CUDA prompts only add compile-only framing, which would contradict a
# numeric or semantic repair, so they are not carried over.
_DIALECTS_WITH_GENERATION_RULES = {"cutlass", "triton", "tilelang"}


def _fence(dialect: str) -> str:
    return "python" if dialect in {"triton", "tilelang"} else "cuda"


def _default_filename(dialect: str) -> str:
    return "solution.py" if dialect in {"triton", "tilelang"} else "solution.cu"


def repair_system_prompt(
    dialect: str, *, role: str = "repair.compile", generation_system: str = ""
) -> str:
    """System prompt for a repair round.

    Args:
        dialect: Kernel dialect id or ``knowledge``.
        role: ``repair.compile`` / ``repair.numeric`` / ``repair.semantic``.
        generation_system: The candidate's generator system prompt. For
            CUTLASS / Triton / TileLang its dialect rules are appended, since
            the repair turn replaces that prompt.
    """
    name = (dialect or "cuda").strip().lower()
    if name == "knowledge":
        return REPAIR_SYSTEM_KNOWLEDGE
    if role == "repair.numeric":
        base = REPAIR_SYSTEM_NUMERIC
    elif role == "repair.semantic":
        base = REPAIR_SYSTEM_SEMANTIC
    elif name in {"triton", "tilelang"}:
        base = REPAIR_SYSTEM_PYTHON
    else:
        base = REPAIR_SYSTEM_CUDA
    rules = (generation_system or "").strip()
    if not rules or name not in _DIALECTS_WITH_GENERATION_RULES:
        return base
    return f"{base}\n\nDialect rules from the generation brief (still binding):\n{rules}"


def quality_repair_body(
    *,
    dialect: str,
    code: str,
    diagnosis: str,
    role: str,
    cuda_arch: str = "",
    filename: str = "",
) -> str:
    """Build a full-source repair request without compile-only instructions.

    Ends with an explicit output contract: a repair turn may not carry the
    generation prompt, and a reply with several fences would otherwise let the
    extractor pick a partial snippet.
    """
    label = "Numeric validation" if role == "repair.numeric" else "Semantic review"
    fence = _fence(dialect)
    name = filename or _default_filename(dialect)
    target = f" targeting `{cuda_arch}`" if cuda_arch else ""
    extra = (
        "Do not launch kernels or run tests at import time."
        if fence == "python"
        else "No `main()`, no test harness."
    )
    return (
        f"## {label} failure\n{diagnosis.strip()}\n\n"
        f"## Previous source\n```{fence}\n{code.strip()}\n```\n\n"
        f"## Output\nReply with exactly one ```{fence} block containing the complete "
        f"`{name}`{target}. {extra} No prose outside the block.\n"
    )


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
        ``template``, ``dialect_violation``, ``import_time``, ``jit_compile``,
        ``other``.
    """
    text = error or ""
    lowered = text.lower()
    name = (dialect or "cuda").strip().lower()
    # Leak markers only count in code: a comment such as "no SM90 here" or a
    # docstring naming ``__global__`` must not relabel an unrelated error.
    source = _code_only(code or "", name)
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
    if name in {"triton", "tilelang"}:
        if _IMPORT_TIME.search(text):
            return "import_time"
        if _JIT_COMPILE.search(text):
            return "jit_compile"
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
    role: str = "repair.compile",
    filename: str = "",
    history: str = "",
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
        role: Repair role; compile repairs keep the implementation, numeric and
            semantic repairs may change it.
        filename: Artifact name for the output contract (dialect default).
        history: Optional summary of earlier failed rounds of this candidate,
            used when the chat history is not resent.

    Returns:
        User message for the next generate turn.
    """
    name = (dialect or "cuda").strip().lower()
    problem = (question or "").strip() or "(empty problem)"
    extra = ""
    if (evidence or "").strip():
        extra = f"## Validation evidence\n{evidence.strip()}\n\n"
    earlier = ""
    if (history or "").strip():
        earlier = f"## Earlier failed attempts (do not repeat them)\n{history.strip()}\n\n"
    budget = ""
    if repair_idx is not None or max_repairs is not None:
        round_text = str(int(repair_idx)) if repair_idx is not None else "?"
        cap_text = str(int(max_repairs)) if max_repairs is not None else "?"
        budget = f"- repair_round: {round_text}/{cap_text}\n"
    hint = repair_hint(error_class)
    hint_line = f"- hint: {hint}\n" if hint else ""
    if name == "knowledge":
        rules = (
            "- Rewrite the full answer, not a patch.\n"
            "- Keep every requirement of the original problem.\n"
        )
    else:
        if role in {"repair.numeric", "repair.semantic"}:
            scope = (
                "- The implementation may change; the problem's requirements, the host "
                "entry signature and the dialect must not.\n"
            )
        else:
            scope = (
                "- Make the smallest change that fixes the diagnostics; keep the algorithm "
                "and host responsibilities from the problem.\n"
            )
        fence = _fence(name)
        rules = (
            "- Rewrite the full artifact, not a diff.\n"
            f"{scope}"
            "- Do not invent missing project headers.\n"
            f"- Reply with exactly one ```{fence} block containing the full "
            f"`{filename or _default_filename(name)}`.\n"
        )
    return (
        "## Original problem (authoritative; do not drop requirements)\n"
        f"{problem}\n\n"
        f"## Repair contract\n"
        f"- dialect: {dialect}\n"
        f"- error_class: {error_class}\n"
        f"{hint_line}"
        f"{budget}"
        f"{rules}\n"
        f"{earlier}"
        f"{extra}"
        f"{inner.strip()}\n"
    )
