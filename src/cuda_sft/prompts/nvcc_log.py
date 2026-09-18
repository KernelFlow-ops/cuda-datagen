"""Trim compiler logs so repair prompts stay under ``REPAIR_ERROR_MAX_CHARS``.

Short logs pass through unchanged. Long nvcc/gcc dumps are deduplicated
by diagnostic message. Python/TVM traces without ``file(line): error:``
markers keep a tail instead of an empty summary.
"""

from __future__ import annotations

import re

_NVCC_DIAG_RE = re.compile(
    r"^(?P<loc>\S.*?)\((?P<line>\d+)\):\s+"
    r"(?P<kind>error|warning|fatal error)\b"
    r"(?:\s+#\S+)?:\s+(?P<msg>.*)$",
    re.IGNORECASE,
)
# Host-compiler / cudafe stub lines: ``file:line:col: error: ...``
_GCC_DIAG_RE = re.compile(
    r"^(?P<loc>\S.*?):(?P<line>\d+)(?::(?P<col>\d+))?:\s+"
    r"(?P<kind>error|warning|fatal error)\s*:\s+(?P<msg>.*)$",
    re.IGNORECASE,
)
_NVCC_FATAL_RE = re.compile(
    r"^(?:(?P<loc>\S.*?):\s+)?fatal error:\s+(?P<msg>.*)$",
    re.IGNORECASE,
)
_TEMPLATE_CHUNK_RE = re.compile(r"<[^>]{24,}>")
_CARET_LINE_RE = re.compile(r"^[\s^~]+$")
_NOISE_SNIPPETS = (
    "the warnings can be suppressed",
    "remark: the warnings can be suppressed",
)


def _is_noise_line(line: str) -> bool:
    """Return True for caret pointers, empty lines, and nvcc remark boilerplate."""
    stripped = line.strip()
    if not stripped:
        return True
    if _CARET_LINE_RE.match(stripped):
        return True
    lowered = stripped.lower()
    return any(snippet in lowered for snippet in _NOISE_SNIPPETS)


def _normalize_diag_message(message: str) -> str:
    """Collapse whitespace and long template argument lists for dedup keys."""
    collapsed = _TEMPLATE_CHUNK_RE.sub("<>", message)
    return re.sub(r"\s+", " ", collapsed).strip().lower()


def format_nvcc_for_prompt(output: str, max_chars: int = 6000) -> str:
    """Prepare nvcc output for the repair prompt.

    If ``output`` is at most ``max_chars``, return it unchanged. Otherwise build
    a deduplicated summary of error/fatal lines (warnings only if space remains).

    Args:
        output: Full compiler stdout/stderr.
        max_chars: Threshold from ``REPAIR_ERROR_MAX_CHARS``.

    Returns:
        Full log or a summary that fits in ``max_chars``.
    """
    text = (output or "").strip()
    if not text:
        return "(empty compiler output)"
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    errors: list[str] = []
    warnings: list[str] = []
    seen_error: set[str] = set()
    seen_warning: set[str] = set()
    error_count = 0
    warning_count = 0

    for raw in text.splitlines():
        if _is_noise_line(raw):
            continue
        match = _NVCC_DIAG_RE.match(raw.strip())
        kind = ""
        msg = ""
        loc = ""
        line_no = ""
        if match:
            kind = match.group("kind").lower()
            msg = match.group("msg")
            loc = match.group("loc")
            line_no = match.group("line")
        else:
            gcc = _GCC_DIAG_RE.match(raw.strip())
            if gcc:
                kind = gcc.group("kind").lower()
                msg = gcc.group("msg")
                loc = gcc.group("loc")
                line_no = gcc.group("line")
            else:
                fatal = _NVCC_FATAL_RE.match(raw.strip())
                if fatal:
                    kind = "fatal error"
                    msg = fatal.group("msg")
                    loc = fatal.group("loc") or ""
        if not kind:
            continue
        key = f"{kind}|{_normalize_diag_message(msg)}"
        display = raw.strip()
        if loc and line_no and loc not in display:
            display = f"{loc}({line_no}): {kind}: {msg.strip()}"
        if kind in {"error", "fatal error"}:
            error_count += 1
            if key not in seen_error:
                seen_error.add(key)
                errors.append(display)
        else:
            warning_count += 1
            if key not in seen_warning:
                seen_warning.add(key)
                warnings.append(display)

    header = (
        f"{error_count} errors ({len(errors)} unique), "
        f"{warning_count} warnings ({len(warnings)} unique); "
        f"log truncated from {len(text)} chars"
    )
    if error_count == 0 and warning_count == 0:
        # Python/TVM dumps and cudafe stub text often lack nvcc ``file(line): error:``
        # markers; keep a tail so repairs are not fed an empty summary.
        if len(text) <= max_chars:
            return text
        keep = max(256, max_chars - 40)
        return f"...[truncated {len(text) - keep} chars]...\n{text[-keep:]}"
    parts = [header, ""]
    if errors:
        parts.append("errors:")
        parts.extend(errors)
        parts.append("")
    if warnings:
        parts.append("warnings:")
        parts.extend(warnings)

    summary = "\n".join(parts).strip()
    if len(summary) <= max_chars:
        return summary

    # Prefer unique errors over warnings if still too long.
    kept: list[str] = [header, "", "errors:"]
    used = len("\n".join(kept))
    for item in errors:
        extra = len(item) + 1
        if used + extra > max_chars:
            break
        kept.append(item)
        used += extra
    if used < max_chars - 20 and warnings:
        kept.append("")
        kept.append("warnings:")
        used = len("\n".join(kept))
        for item in warnings:
            extra = len(item) + 1
            if used + extra > max_chars:
                break
            kept.append(item)
            used += extra
    result = "\n".join(kept).strip()
    if len(result) > max_chars:
        return result[:max_chars]
    return result


def truncate_compile_error(error: str, max_chars: int = 6000) -> str:
    """Backward-compatible alias of :func:`format_nvcc_for_prompt`.

    Args:
        error: Compiler stdout/stderr.
        max_chars: Same cap as :func:`format_nvcc_for_prompt`.
    """
    return format_nvcc_for_prompt(error, max_chars=max_chars)
