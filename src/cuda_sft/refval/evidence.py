"""Compress a failed refval report into a repair-prompt evidence block."""

from __future__ import annotations

from cuda_sft.refval.spec import CaseResult, RefvalReport

_DEFAULT_MAX_CHARS = 2000
_MAX_MISMATCH_ROWS = 6


def _fmt_mismatch(row: dict) -> str:
    idx = row.get("index")
    tensor = row.get("tensor")
    prefix = f"{tensor}[{idx}]" if tensor else f"[{idx}]"
    got = row.get("got")
    exp = row.get("exp")
    abs_e = row.get("abs")
    rel_e = row.get("rel")
    extra = ""
    if abs_e is not None:
        extra = f" abs={abs_e:.4g} rel={rel_e:.4g}" if rel_e is not None else f" abs={abs_e:.4g}"
    return f"{prefix} got={got} exp={exp}{extra}"


def _fmt_case(item: CaseResult) -> str:
    shape = "x".join(str(d) for d in item.shape) if item.shape else "-"
    lines = [
        f"case={item.name} status={item.status} shape={shape} seed={item.seed}",
        f"  max_abs={item.max_abs:.6g} max_rel={item.max_rel:.6g} "
        f"mismatch={item.n_mismatch}/{item.n_compared}",
    ]
    if item.error:
        lines.append(f"  error={item.error}")
    for row in (item.mismatches or [])[:_MAX_MISMATCH_ROWS]:
        lines.append(f"  {_fmt_mismatch(row)}")
    return "\n".join(lines)


def compress_evidence(report: RefvalReport, *, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Build a short repair hint from the first failing cases.

    Includes case name, shape, seed, max_abs/max_rel, and the first few
    mismatch rows. Truncates to ``max_chars``.
    """
    header = [
        f"refval status={report.status} error_class={report.error_class or '-'}",
        f"dialect={report.dialect} cases_run={report.cases_run} "
        f"failed_case={report.failed_case or '-'}",
    ]
    if report.manifest_summary:
        entry = report.manifest_summary.get("entry") or "?"
        params = report.manifest_summary.get("params") or []
        header.append(f"abi entry={entry} params={params}")
    if report.reason:
        header.append(f"reason={report.reason}")
    body: list[str] = []
    failed = [item for item in report.results if not item.ok]
    if not failed and report.results:
        failed = report.results[:1]
    for item in failed[:3]:
        body.append(_fmt_case(item))
    text = "\n".join(header + body).strip()
    limit = max(200, int(max_chars))
    if len(text) > limit:
        text = text[: limit - 24].rstrip() + "\n...[truncated evidence]..."
    return text
