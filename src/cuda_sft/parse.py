"""Extract CUDA source from model replies (markdown fences / thinking tags)."""

from __future__ import annotations

import ast
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


_NUMBERED_HEADING_RE = re.compile(r"(?m)^\s*(\d+)\.\s+\S")


def has_numbered_headings(text: str, count: int) -> bool:
    """Return True when headings ``1.`` .. ``count.`` all appear.

    Used to reject truncated CoT checklists (models often stop mid heading 6).

    Args:
        text: Polished CoT body.
        count: Required last heading number (6 for kernels, 5 for knowledge).
    """
    found = {int(match.group(1)) for match in _NUMBERED_HEADING_RE.finditer(text or "")}
    return all(index in found for index in range(1, int(count) + 1))


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


JSON_OBJECT_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)
_HOST_FN_RE = re.compile(
    r"(?:^|\n)(?P<prefix>[^\n]{0,80}?)"
    r"(?P<ret>void|int|float|double|bool|size_t|unsigned|long|uint32_t|int32_t|"
    r"int64_t|__half|nv_bfloat16)"
    r"(?:\s+long)?"
    r"\s+(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^;{]*)\)\s*\{",
    re.MULTILINE,
)
_GLOBAL_RE = re.compile(r"__global__")
_PTR_RE = re.compile(r"\*")
_SIZE_HINTS = (
    "n", "m", "k", "size", "numel", "len", "length", "count", "rows", "cols",
    "h", "w", "height", "width", "batch", "dim", "dim0", "dim1", "dim2",
    "lda", "ldb", "ldc",
)
_OUT_HINTS = ("out", "output", "dst", "dest", "result", "c", "y", "z", "values", "indices")


def parse_json_object(text: str) -> dict | None:
    """Best-effort parse of the first JSON object in a model reply."""
    import json

    raw = (text or "").strip()
    if not raw:
        return None
    visible = strip_thinking(raw)
    tagged = extract_thinking(raw)
    candidates: list[str] = []
    for body in (visible, tagged, raw):
        if not body:
            continue
        candidates.extend(JSON_OBJECT_FENCE_RE.findall(body))
        candidates.append(body)
        start = body.find("{")
        end = body.rfind("}")
        if start >= 0 and end > start:
            candidates.append(body[start : end + 1])
    for blob in candidates:
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def parse_refval_manifest(
    text: str,
    *,
    question_id: int,
    dialect: str,
    seed: int = 0,
    extracted_from: str = "llm",
):
    """Parse a RefManifest from an extractor reply. Returns None on failure."""
    from cuda_sft.refval.spec import KernelABI, RefManifest, seed_for

    payload = parse_json_object(text)
    if not payload:
        return None
    abi_raw = payload.get("abi") if isinstance(payload.get("abi"), dict) else payload
    try:
        abi = KernelABI.from_dict(abi_raw)
    except Exception:
        return None
    # Reject malformed contracts at the parser boundary.  Callers may safely
    # treat a non-None manifest as having passed the ABI's own self-checks.
    if abi.issues():
        return None
    source = str(payload.get("reference_source") or payload.get("reference") or "")
    fn_name = str(payload.get("reference_fn_name") or "reference")
    if not source.strip():
        return None
    qid = int(question_id)
    name = dialect or "cuda"
    return RefManifest(
        question_id=qid,
        dialect=name,
        abi=abi,
        reference_source=source,
        reference_fn_name=fn_name,
        seed=int(seed) or seed_for(qid, name),
        extracted_from=extracted_from,
        notes=str(payload.get("notes") or ""),
        task_spec=dict(payload.get("task_spec") or {})
        if isinstance(payload.get("task_spec"), dict)
        else {},
        oracle_spec=dict(payload.get("oracle_spec") or {})
        if isinstance(payload.get("oracle_spec"), dict)
        else {},
        provenance=dict(payload.get("provenance") or {})
        if isinstance(payload.get("provenance"), dict)
        else {},
        manifest_hash=str(payload.get("manifest_hash") or ""),
    )


def diagnose_refval_extract(text: str) -> list[str]:
    """Explain why an extractor reply cannot become a manifest.

    An empty list means the ABI self-check passed and ``reference_source`` is
    non-empty. Host-signature match against the kernel is a separate check.
    ``parse_refval_manifest`` returns None for these same failures, so callers
    must use this list as the retry prompt instead of treating None as "no
    reason".
    """
    payload = parse_json_object(text)
    if not payload:
        snippet = " ".join((text or "").split())[:180]
        if not snippet:
            return ["empty extractor response"]
        return [f"response is not a JSON object: {snippet}"]
    abi_raw = payload.get("abi") if isinstance(payload.get("abi"), dict) else payload
    try:
        from cuda_sft.refval.spec import KernelABI

        abi = KernelABI.from_dict(abi_raw)
    except Exception as exc:
        return [f"abi parse error: {exc}"]
    issues = list(abi.issues())
    source = str(payload.get("reference_source") or payload.get("reference") or "")
    if not source.strip():
        issues.append("empty reference_source")
    return issues


def abi_matches_source(abi, source: str) -> list[str]:
    """Return ABI/signature issues for CUDA or Python source.

    CUDA validation intentionally supports a small, explicit host-signature
    subset.  Python validation uses AST rather than regex so unsupported
    argument forms are reported instead of silently accepted.
    """
    text = source or ""
    issues: list[str] = []
    entry = getattr(abi, "entry", "") or ""
    if not entry:
        issues.append("empty entry name")
    elif entry not in text:
        issues.append(f"entry {entry!r} not found in source")
    elif re.search(rf"(?m)^\s*(?:async\s+)?def\s+{re.escape(entry)}\s*\(", text):
        issues.extend(_python_signature_issues(abi, text))
    else:
        issues.extend(_cuda_signature_issues(abi, text))
    for issue in (list(abi.issues()) if hasattr(abi, "issues") else []):
        if issue not in issues:
            issues.append(issue)
    return issues


_SUPPORTED_C_TYPES = {
    "void": "void",
    "float": "f32",
    "double": "f64",
    "__half": "f16",
    "half": "f16",
    "nv_bfloat16": "bf16",
    "bfloat16": "bf16",
    "int": "i32",
    "int32_t": "i32",
    "uint32_t": "u32",
    "unsigned int": "u32",
    "unsigned": "u32",
    "long long": "i64",
    "int64_t": "i64",
    "size_t": "i64",
    "signed char": "i8",
    "int8_t": "i8",
    "unsigned char": "u8",
    "uint8_t": "u8",
    "bool": "bool",
}


def _canonical_c_type(raw: str) -> str | None:
    text = re.sub(r"\b(const|volatile|__restrict__|restrict|__restrict)\b", "", raw or "")
    text = " ".join(text.replace("*", " ").split()).lower()
    return _SUPPORTED_C_TYPES.get(text)


def _parse_supported_c_param(raw: str) -> tuple[str, str, bool, str | None]:
    """Return ``(name, dtype, pointer, issue)`` for the supported CUDA subset.

    The parameter name is the last identifier. A greedy type pattern would
    swallow all but the final character of names like ``rows`` (``int row`` + ``s``).
    """
    text = (raw or "").strip()
    if not text or text == "void":
        return "", "", False, None
    if any(token in text for token in ("[", "]", "(", ")", "=", "...", "->", "&")):
        return "", "", False, f"unsupported CUDA parameter syntax {text!r}"
    pointer = "*" in text
    cleaned = re.sub(r"\b(const|volatile|__restrict__|restrict|__restrict)\b", " ", text)
    cleaned = " ".join(cleaned.replace("*", " ").split())
    tokens = cleaned.split()
    if len(tokens) < 2 or not re.fullmatch(r"[A-Za-z_]\w*", tokens[-1]):
        return "", "", pointer, f"unsupported CUDA parameter syntax {text!r}"
    name = tokens[-1]
    type_text = " ".join(tokens[:-1])
    dtype = _canonical_c_type(type_text)
    if dtype is None:
        return name, "", pointer, f"unsupported CUDA type {type_text!r}"
    return name, dtype, pointer, None


def _cuda_entry_pattern(entry: str) -> re.Pattern[str]:
    return re.compile(
        rf"(?m)^\s*(?:(?:extern\s+\"C\"\s+)|(?:__host__|static|inline)\s+)*"
        rf"(?P<ret>[A-Za-z_][\w]*(?:\s+[A-Za-z_][\w]*)*)\s+"
        rf"{re.escape(entry)}\s*\((?P<args>[^()]*)\)\s*(?:\{{|;)",
    )


class _SignatureView:
    """Minimal ``re.Match`` stand-in for a type-substituted template signature."""

    def __init__(self, ret: str, args: str) -> None:
        self._groups = {"ret": ret, "args": args}

    def group(self, name: str) -> str:
        return self._groups[name]


def _adjacent_template_param(source: str, match_start: int) -> str | None:
    """Return the type parameter when a template head sits directly above ``match_start``."""
    before = source[:match_start]
    heads = list(
        re.finditer(
            r"template\s*<\s*(?:typename|class)\s+([A-Za-z_]\w*)\s*>",
            before,
        )
    )
    if not heads:
        return None
    head = heads[-1]
    if before[head.end() :].strip():
        return None
    return head.group(1)


def _explicit_type_args(source: str, entry: str) -> list[str]:
    """Concrete types from ``template void entry<float>(...);`` lines."""
    return re.findall(
        rf"(?m)^\s*template\s+[A-Za-z_][\w\s\*]*\b{re.escape(entry)}\s*<\s*([A-Za-z_][\w:]*)\s*>",
        source or "",
    )


def _instantiated_signature_views(source: str, entry: str) -> list[_SignatureView]:
    """Substitute explicit instantiation types into a templated host signature.

    ``abi_matches_source`` only accepts a concrete C type subset. A host
    written as ``template <typename T> void entry(const T* ...)`` plus
    ``template void entry<float>(...)`` is a float ABI, not an unsupported
    type ``T``. Without this, a valid reference is dropped for a heuristic
    ABI and every such compile-pass ends in ``reference_error``.
    """
    concretes = list(dict.fromkeys(_explicit_type_args(source, entry)))
    if not concretes:
        return []
    views: list[_SignatureView] = []
    for match in _cuda_entry_pattern(entry).finditer(source or ""):
        tparam = _adjacent_template_param(source, match.start())
        if not tparam:
            continue
        ret = match.group("ret")
        args = match.group("args")
        if not re.search(rf"\b{re.escape(tparam)}\b", f"{ret} {args}"):
            continue
        for concrete in concretes:
            views.append(
                _SignatureView(
                    re.sub(rf"\b{re.escape(tparam)}\b", concrete, ret),
                    re.sub(rf"\b{re.escape(tparam)}\b", concrete, args),
                )
            )
    return views


def _cuda_overload_issues(abi: Any, match: re.Match[str]) -> list[str]:
    """Issues for one host overload. Empty means this overload matches ``abi``."""
    issues: list[str] = []
    ret = _canonical_c_type(match.group("ret"))
    expected_ret = str(getattr(abi, "returns", "void") or "void")
    if ret is None:
        issues.append(f"unsupported CUDA return type {match.group('ret')!r}")
    elif ret != expected_ret:
        issues.append(f"return type mismatch: source={ret} abi={expected_ret}")
    raw_args = _split_c_args(match.group("args"))
    if len(raw_args) == 1 and raw_args[0].strip() == "void":
        raw_args = []
    params = list(getattr(abi, "params", ()) or ())
    if len(raw_args) != len(params):
        issues.append(f"parameter count mismatch: source={len(raw_args)} abi={len(params)}")
    for index, (raw, expected) in enumerate(zip(raw_args, params)):
        name, dtype, pointer, issue = _parse_supported_c_param(raw)
        if issue:
            issues.append(f"parameter {index}: {issue}")
            continue
        if name != expected.name:
            issues.append(f"parameter {index} name/order mismatch: source={name!r} abi={expected.name!r}")
        if dtype != expected.dtype:
            issues.append(f"parameter {expected.name!r} type mismatch: source={dtype} abi={expected.dtype}")
        if pointer != bool(getattr(expected, "is_tensor", False)):
            expected_kind = "pointer" if expected.is_tensor else "scalar"
            issues.append(f"parameter {expected.name!r} kind mismatch: expected {expected_kind}")
    return issues


def _cuda_signature_issues(abi: Any, source: str) -> list[str]:
    entry = str(getattr(abi, "entry", "") or "")
    text = source or ""
    matches = list(_cuda_entry_pattern(entry).finditer(text))
    instantiated = _instantiated_signature_views(text, entry)
    if not matches and not instantiated:
        return [f"unsupported or missing CUDA signature for entry {entry!r}"]
    # Templates and concrete overloads share an entry name. Accept the ABI when
    # any overload matches; otherwise report the closest overload.
    candidates = [_cuda_overload_issues(abi, match) for match in [*matches, *instantiated]]
    for issues in candidates:
        if not issues:
            return []
    return min(candidates, key=len)


def _python_annotation(annotation: ast.AST | None) -> tuple[str, str | None]:
    if annotation is None:
        return "unknown", None
    if isinstance(annotation, ast.Name):
        text = annotation.id
    elif isinstance(annotation, ast.Attribute):
        parts: list[str] = []
        node: ast.AST = annotation
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        text = ".".join(reversed(parts))
    else:
        return "unsupported", None
    low = text.lower()
    if low in {"int", "builtins.int"}:
        return "scalar", "i32"
    if low in {"float", "builtins.float"}:
        return "scalar", "f32"
    if low in {"bool", "builtins.bool"}:
        return "scalar", "bool"
    if low.endswith("tensor") or low.endswith("ndarray") or low in {"array", "buffer"}:
        return "tensor", None
    return "unsupported", None


def _python_signature_issues(abi: Any, source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"Python syntax error: {exc}"]
    entry = str(getattr(abi, "entry", "") or "")
    fn = next(
        (node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry),
        None,
    )
    if fn is None:
        return [f"missing Python function {entry!r}"]
    args = fn.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        return ["unsupported Python signature: varargs/keyword-only/positional-only parameters"]
    actual = list(args.args)
    expected = list(getattr(abi, "params", ()) or ())
    issues: list[str] = []
    if len(actual) != len(expected):
        issues.append(f"parameter count mismatch: source={len(actual)} abi={len(expected)}")
    for index, (node, param) in enumerate(zip(actual, expected)):
        if node.arg != param.name:
            issues.append(f"parameter {index} name/order mismatch: source={node.arg!r} abi={param.name!r}")
        kind, dtype = _python_annotation(node.annotation)
        if kind == "unsupported":
            issues.append(f"unsupported Python annotation for parameter {param.name!r}")
        elif kind != "unknown":
            if param.is_tensor and kind != "tensor":
                issues.append(f"parameter {param.name!r} expected tensor annotation")
            if not param.is_tensor and kind == "tensor":
                issues.append(f"parameter {param.name!r} expected scalar annotation")
            if dtype is not None and dtype != param.dtype:
                issues.append(f"parameter {param.name!r} type mismatch: source={dtype} abi={param.dtype}")
    return_anno_kind, return_dtype = _python_annotation(fn.returns)
    expected_ret = str(getattr(abi, "returns", "void") or "void")
    if fn.returns is not None and return_anno_kind == "unsupported":
        issues.append("unsupported Python return annotation")
    if return_dtype is not None and expected_ret != "void" and return_dtype != expected_ret:
        issues.append(f"return type mismatch: source={return_dtype} abi={expected_ret}")
    return issues


def _split_c_args(blob: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for char in blob or "":
        if char in "<([":
            depth += 1
        elif char in ">)]":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            item = "".join(buf).strip()
            if item:
                parts.append(item)
            buf = []
            continue
        buf.append(char)
    item = "".join(buf).strip()
    if item:
        parts.append(item)
    return parts


def _parse_c_param(raw: str):
    from cuda_sft.refval.spec import KernelParam, normalize_dtype

    text = re.sub(r"\b__restrict__\b|\brestrict\b|\b__restrict\b", "", raw or "")
    text = text.strip()
    if not text or text == "void":
        return None
    is_ptr = bool(_PTR_RE.search(text))
    is_const = bool(re.search(r"\bconst\b", text))
    lowered = text.lower()
    if "float2" in lowered or "double2" in lowered or "int2" in lowered:
        # Vector types are not in the canonical dtype set; keep as f32/i32 buffer.
        pass
    cleaned = text.replace("*", " ").replace("&", " ")
    cleaned = re.sub(r"\b(const|volatile|unsigned|signed)\b", " ", cleaned)
    tokens = [tok for tok in cleaned.split() if tok]
    if not tokens:
        return None
    name = tokens[-1]
    type_tok = " ".join(tokens[:-1]).strip() or "int"
    if "size_t" in lowered or "long long" in lowered or "int64" in lowered:
        dtype = "i64"
    elif "__half" in lowered or "half" in lowered.split():
        dtype = "f16"
    elif "nv_bfloat16" in lowered or "bfloat16" in lowered:
        dtype = "bf16"
    else:
        dtype = normalize_dtype(type_tok if type_tok else "i32")
    lname = name.lower()
    if not is_ptr:
        kind = "size" if any(h == lname or lname.endswith(h) for h in _SIZE_HINTS) else "scalar"
        if lname in _SIZE_HINTS or lname.endswith("_n") or lname.endswith("_size"):
            kind = "size"
        return KernelParam(name=name, kind=kind, dtype=dtype if kind != "size" else "i32", rank=0)
    kind = "input" if is_const else "output"
    if not is_const and not any(h in lname for h in _OUT_HINTS):
        # Non-const pointer with a typical input name stays inout-capable but default output
        # is safer for C, A, dst; for a/b/x keep as input if const was forgotten.
        if lname in {"a", "b", "x", "src", "input", "left", "right"}:
            kind = "input"
    rank = 1
    shape_from: tuple[str, ...] = ()
    return KernelParam(
        name=name,
        kind=kind,
        dtype=dtype,
        rank=rank,
        shape_from=shape_from,
        memory="device",
    )


def heuristic_cuda_abi(source: str):
    """Best-effort host ABI from a CUDA translation unit (no LLM).

    Prefers a non-``__global__`` function whose body contains a kernel launch.
    Returns None when nothing plausible is found.
    """
    from cuda_sft.refval.spec import KernelABI, KernelParam

    text = source or ""
    if not text.strip():
        return None
    matches = list(_HOST_FN_RE.finditer(text))
    candidates = []
    for match in matches:
        prefix = match.group("prefix") or ""
        name = match.group("name")
        if name == "main":
            continue
        if "__global__" in prefix or "__device__" in prefix:
            continue
        start = match.end()
        depth = 1
        idx = start
        while idx < len(text) and depth:
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
            idx += 1
        body = text[start:idx]
        score = 0
        if "<<<" in body:
            score += 5
        if "__global__" not in prefix:
            score += 1
        params = []
        for raw in _split_c_args(match.group("args") or ""):
            parsed = _parse_c_param(raw)
            if parsed is not None:
                params.append(parsed)
        if not params:
            continue
        candidates.append((score, match.start(), name, tuple(params), match.group("ret")))
    if not candidates:
        return None
    candidates.sort(key=lambda row: (-row[0], -row[1]))
    _score, _pos, name, params, ret = candidates[0]
    size_names = [p.name for p in params if p.kind == "size"]
    patched: list[KernelParam] = []
    for param in params:
        if param.is_tensor and not param.shape_from and size_names:
            patched.append(
                KernelParam(
                    name=param.name,
                    kind=param.kind,
                    dtype=param.dtype,
                    layout=param.layout,
                    rank=param.rank,
                    shape_from=tuple(size_names[: max(1, param.rank)]),
                    memory=param.memory,
                    optional=param.optional,
                    alias_of=param.alias_of,
                )
            )
        else:
            patched.append(param)
    returns = "void"
    if ret and ret.strip() not in {"void"}:
        from cuda_sft.refval.spec import normalize_dtype

        returns = normalize_dtype(ret)
    entry_name = name
    sort_outputs = any(
        "union" in token.lower() or "hist" in token.lower()
        for param in patched
        for token in (param.name, entry_name)
    )
    return KernelABI(
        entry=name,
        params=tuple(patched),
        dtype=next((p.dtype for p in patched if p.is_tensor), "f32"),
        returns=returns,
        sort_outputs=bool(sort_outputs),
    )
