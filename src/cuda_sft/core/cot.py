"""Deterministic CoT checks against the selected source."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Phrases that narrate the repair loop. Kept narrow: generic words such as
# "harness", "retry" or "debug" also describe legitimate kernels (harnessing
# shared memory, an atomicCAS retry loop), so they only count with context.
REPAIR_LEAK_PATTERNS = [
    r"编译(错误|失败|报错)", r"报错(信息|提示|日志)", r"(根据|针对|按照)[^。\n]{0,8}报错",
    r"修复", r"上一(版|个版本|次)", r"之前的(尝试|版本|代码)",
    r"重新(生成|提交)", r"评测(框架|脚本)", r"nvcc\s*-c",
    r"\b(?:test|testing|evaluation|grading|benchmark)\s+harness\b",
    r"\bcompile(r)?\s+error", r"\berror message",
    r"\b(?:fix(?:ed|ing)?|repair(?:ed|ing)?)\s+(?:the|this|my|our|previous|earlier)\s+"
    r"(?:code|kernel|implementation|bug|error|issue|attempt|version)\b",
    r"\bprevious (attempt|version|code)", r"\bearlier attempt",
    r"\bdebugg(?:ed|ing)\s+(?:the|this|my|our)\b",
    r"\bretr(?:y|ied|ying)\s+(?:the\s+)?(?:compil\w*|generat\w*|submi\w*)",
]

BUILTIN_IDENTIFIERS = {
    "threadIdx", "blockIdx", "blockDim", "gridDim", "__syncthreads", "__shared__",
    "tl", "triton", "torch", "T", "float", "double", "int", "half", "bool",
    "size_t", "nullptr", "np", "cuda", "__global__", "__device__", "__host__",
    # CUDA intrinsics, vector types and runtime API a CoT may name in passing.
    "warpSize", "__syncwarp", "__shfl_sync", "__shfl_down_sync", "__shfl_up_sync",
    "__shfl_xor_sync", "__ldg", "__restrict__", "__forceinline__", "__launch_bounds__",
    "atomicAdd", "atomicSub", "atomicMax", "atomicMin", "atomicCAS", "atomicExch",
    "__half", "__half2", "half2", "__nv_bfloat16", "float2", "float4", "int2", "int4",
    "dim3", "cudaMalloc", "cudaFree", "cudaMemcpy", "cudaMemcpyAsync", "cudaMemset",
    "cudaStream_t", "cudaDeviceSynchronize", "cudaGetLastError", "cudaError_t",
    "const", "constexpr", "unsigned", "void", "long", "char", "short", "extern",
    "int32_t", "uint32_t", "int64_t", "uint64_t",
    "expf", "__expf", "logf", "sqrtf", "rsqrtf", "fmaxf", "fminf", "fabsf", "tanhf",
    "INFINITY", "FLT_MAX", "NAN",
    # Triton / TileLang / CuTe vocabulary.
    "jit", "autotune", "program_id", "num_programs", "arange", "load", "store", "cdiv",
    "num_warps", "num_stages", "prim_func", "Kernel", "Parallel", "Pipelined",
    "alloc_shared", "alloc_fragment", "cute", "cutlass", "Tensor", "Layout",
    "make_tensor", "make_layout", "make_shape", "make_stride", "local_tile",
    "local_partition",
}

_IDENTIFIER = re.compile(r"`([A-Za-z_][A-Za-z0-9_:]*)`")
_CONSTANT = re.compile(r"\b((?:BLOCK|TILE|THREADS|WARP)[A-Z_]*)\s*(?:=|为|是|:)\s*(\d+)")


def _mentions(name: str, text: str) -> bool:
    return bool(re.search(rf"\b{re.escape(name)}\b", text))


def cot_consistency_report(
    cot: str, code: str, question: str = ""
) -> tuple[list[str], list[str]]:
    """Split CoT problems into hard (blocking) and soft (informational) issues.

    Hard: repair-loop narration and constants that contradict the source.
    Soft: backticked names found in neither the source, the question, nor the
    builtin vocabulary; the CoT may legitimately contrast with an API the code
    does not use, so these never block a draft on their own.

    Args:
        cot: Candidate CoT text.
        code: Final selected source.
        question: Raw problem text; leak phrases it already contains (for
            example a "fix this kernel" task) are not treated as leaks.
    """
    hard = repair_leaks(cot, question)
    soft: list[str] = []
    for name, value in _CONSTANT.findall(cot):
        actual = re.findall(rf"\b{re.escape(name)}\b\s*(?:=|\s)\s*(\d+)", code)
        if actual and int(value) not in {int(item) for item in actual}:
            values = "/".join(dict.fromkeys(actual))
            hard.append(f"constant mismatch: {name} cot={value} code={values}")
    for name in _IDENTIFIER.findall(cot):
        if name in BUILTIN_IDENTIFIERS or _mentions(name, code) or _mentions(name, question):
            continue
        soft.append(f"unknown identifier: {name}")
    return list(dict.fromkeys(hard)), list(dict.fromkeys(soft))


def cot_consistency_issues(cot: str, code: str, question: str = "") -> list[str]:
    """Return the blocking issues (repair leaks, constant mismatches) of a CoT draft."""
    return cot_consistency_report(cot, code, question)[0]


def repair_leaks(cot: str, question: str = "") -> list[str]:
    """Repair-loop narration only (no code checks); used by the knowledge editor."""
    leaks: list[str] = []
    for pattern in REPAIR_LEAK_PATTERNS:
        if question and re.search(pattern, question, flags=re.IGNORECASE):
            continue
        match = re.search(pattern, cot, flags=re.IGNORECASE)
        if match:
            leaks.append(f"repair leak: {match.group(0)}")
    return list(dict.fromkeys(leaks))


# ---------------------------------------------------------------------------
# Text helpers shared by the kernel and knowledge CoT editors.
# ---------------------------------------------------------------------------

_REASONING_MARKER = "\n...[truncated reasoning]...\n"
_SENTENCE_END = re.compile(r"[。！？.!?](?=\s|$)")


def clip_middle(text: str, max_chars: int, *, head_ratio: float = 0.3) -> str:
    """Keep the head and (mostly) the tail of ``text`` within ``max_chars``.

    Teacher thinking settles its final decisions near the end, so a head-only
    cut dropped exactly the part that matches the final code.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    room = max(0, max_chars - len(_REASONING_MARKER))
    head = int(room * head_ratio)
    tail = room - head
    return text[:head].rstrip() + _REASONING_MARKER + (text[-tail:].lstrip() if tail else "")


def clip_at_boundary(text: str, max_chars: int) -> str:
    """Hard cap ``text``, backing off to a paragraph or sentence end when possible.

    The fallback only looks at the last 20% of the budget so a long CoT is not
    cut short by an early boundary.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    floor = int(max_chars * 0.8)
    # The latest boundary of any kind wins, so a short final section keeps its
    # heading instead of being dropped back to the previous paragraph.
    candidates = [cut.rfind("\n")]
    candidates.extend(match.end() for match in _SENTENCE_END.finditer(cut))
    best = max((index for index in candidates if index >= floor), default=-1)
    return cut[:best].rstrip() if best >= 0 else cut.rstrip()


def clean_raw_reasoning(text: str, max_chars: int, *, strip_code: bool = True) -> str:
    """Normalize teacher thinking before storing it or sending it to an editor.

    Args:
        text: Raw API reasoning, possibly tagged or fenced.
        max_chars: Budget; ``<=0`` disables truncation. Keeps head and tail.
        strip_code: Drop kernel code (kernel CoT). Knowledge keeps formulas.
    """
    from cuda_sft.parse import collapse_blank_lines, extract_thinking, strip_code_from_cot

    raw = text or ""
    tagged = extract_thinking(raw)
    body = tagged if tagged.strip() else raw
    if strip_code:
        body = strip_code_from_cot(body)
    body = collapse_blank_lines(body)
    return clip_middle(body, max_chars)


def draft_problems(
    reply: str, cleaned: str, *, max_chars: int, headings: int, min_chars: int = 120
) -> list[str]:
    """Concrete reasons a CoT draft is unusable, phrased as feedback for a retry.

    Empty when the draft is structurally valid. The budget and code notes are
    only added next to a blocking problem, since they explain it.

    Args:
        reply: Visible editor reply before sanitizing.
        cleaned: Sanitized draft that would be stored.
        max_chars: Character budget told to the editor.
        headings: Required number of numbered headings.
        min_chars: Shortest acceptable draft.
    """
    from cuda_sft.parse import collapse_blank_lines, missing_numbered_headings, strip_code_from_cot

    if not (reply or "").strip():
        return ["the reply had no visible text; write the CoT itself in the reply"]
    problems: list[str] = []
    if len(cleaned) < min_chars:
        problems.append(
            f"the draft is too short ({len(cleaned)} chars); write all {headings} sections"
        )
    missing = missing_numbered_headings(cleaned, headings)
    if missing:
        problems.append(
            "missing or out-of-order numbered headings: "
            + ", ".join(str(item) for item in missing)
            + f"; use the {headings} numbered headings exactly as given"
        )
    if not problems:
        return []
    if max_chars > 0 and len(reply) > max_chars:
        problems.append(f"the draft exceeded the {max_chars}-character budget; be more concise")
    if "```" in reply or strip_code_from_cot(reply) != collapse_blank_lines(reply):
        problems.append("do not include source code or code fences; describe the code in prose")
    return problems


def editor_max_output_tokens(settings: object, *, cap: int | None = None) -> int:
    """Completion budget for a CoT editor call, reasoning tokens included.

    Twice the character budget leaves room for thinking and for CJK text,
    while still bounding a runaway reply (the global default is 100k).
    """
    resolved = int(getattr(settings, "resolved_max_output_tokens", 0) or 0) or 100_000
    wanted = max(4096, 2 * int(getattr(settings, "cot_max_chars", 8000) or 8000))
    value = min(resolved, wanted)
    return min(value, int(cap)) if cap else value


class EditorBudgetExhausted(RuntimeError):
    """Raised when a CoT refine has used all of its LLM calls."""


class EditorBudget:
    """Per-sample LLM call budget shared by drafts, retries and fallbacks."""

    def __init__(self, calls: int) -> None:
        self.limit = max(1, int(calls))
        self.used = 0

    @property
    def remaining(self) -> int:
        return self.limit - self.used

    def take(self) -> int:
        """Consume one call and return its 1-based attempt number."""
        if self.used >= self.limit:
            raise EditorBudgetExhausted(f"CoT editor budget of {self.limit} calls exhausted")
        self.used += 1
        return self.used


def call_editor(
    client: object,
    *,
    system: str,
    user: str,
    temperature: float,
    thinking_level: str | None,
    max_output_tokens: int | None,
    meta: object | None,
    budget: EditorBudget,
    label: str = "CoT editor",
) -> str:
    """Call the editor model and return visible text only, within ``budget``.

    Transport errors are retried while the budget lasts; every attempt counts.
    The editor's own reasoning is never returned: it is meta-commentary about
    the edit, not a CoT for the student.
    """
    from cuda_sft.llm import LLMError, is_retryable_llm_error

    messages = [{"role": "user", "content": user}]
    while True:
        attempt = budget.take()
        call_meta = meta.with_attempt(attempt) if meta is not None else None  # type: ignore[attr-defined]
        try:
            stream_completion = getattr(client, "stream_completion", None)
            if callable(stream_completion):
                completion = stream_completion(
                    messages=messages,
                    system=system,
                    temperature=temperature,
                    print_stream=False,
                    thinking_level=thinking_level,
                    max_output_tokens=max_output_tokens,
                    meta=call_meta,
                )
                return completion.text or ""
            return client.stream_text(  # type: ignore[attr-defined]
                messages=messages,
                system=system,
                temperature=temperature,
                print_stream=False,
                meta=call_meta,
            )
        except Exception as exc:
            retry = budget.remaining > 0 and (
                isinstance(exc, LLMError) or is_retryable_llm_error(exc)
            )
            if not retry:
                raise
            logger.warning("%s retry after call %s/%s: %s", label, attempt, budget.limit, exc)
