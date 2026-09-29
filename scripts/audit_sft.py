#!/usr/bin/env python3
"""Audit historical SFT rows for known training-data defects.

Usage:
    python scripts/audit_sft.py data/sft.jsonl --out runs/audit_v0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from cuda_sft.parse import extract_fenced_source, extract_thinking
from cuda_sft.tasks.kinds import question_hash

REPAIR_SYSTEM_MARKERS = ("repairer", "compile-fix", "rewriting a failed", "fix the previous")
PROTOCOL_SYSTEM_MARKERS = ("nvcc -c", "do not explain", "compile-only gate", "downstream checker")

# Matches 04_specs/10_prompt_packs.md section 7 until T1.3 supplies core.cot.
_FALLBACK_REPAIR_PATTERNS = (
    r"编译(错误|失败|报错)",
    r"报错",
    r"修复",
    r"上一(版|个版本|次)",
    r"之前的(尝试|版本|代码)",
    r"重新(生成|提交)",
    r"评测(框架|脚本)",
    r"nvcc\s*-c",
    r"harness",
    r"\bcompile(r)?\s+error",
    r"\berror message",
    r"\bfix(ed|ing)?\b",
    r"\bprevious (attempt|version|code)",
    r"\bearlier attempt",
    r"\bdebug(ging)?\b",
    r"\brepair",
    r"\bretry\b",
)
_BUILTIN_IDENTIFIERS = {
    "threadIdx",
    "blockIdx",
    "blockDim",
    "gridDim",
    "__syncthreads",
    "__shared__",
    "tl",
    "triton",
    "torch",
    "T",
    "float",
    "int",
    "half",
}
_IDENTIFIER_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_:]*)`")
_CONSTANT_RE = re.compile(r"\b((?:BLOCK|TILE|THREADS|WARP)[A-Z_]*)\s*[=为是:]\s*(\d+)")

try:
    from cuda_sft.core.cot import REPAIR_LEAK_PATTERNS, cot_consistency_issues
except ImportError:
    REPAIR_LEAK_PATTERNS = _FALLBACK_REPAIR_PATTERNS
    cot_consistency_issues = None

_LEAK_RE = [re.compile(pattern, re.IGNORECASE) for pattern in REPAIR_LEAK_PATTERNS]


def _message(row: dict[str, Any], role: str) -> str:
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and message.get("role") == role:
                content = message.get("content")
                return content if isinstance(content, str) else ""
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    fallbacks = {
        "system": (row.get("system"), metadata.get("system")),
        "user": (row.get("question"), row.get("user_prompt")),
        "assistant": (row.get("assistant"), row.get("code"), row.get("output")),
    }
    return next((value for value in fallbacks[role] if isinstance(value, str)), "")


def _fallback_entity_issues(think: str, code: str) -> list[str]:
    issues: list[str] = []
    for name in sorted(set(_IDENTIFIER_RE.findall(think)) - _BUILTIN_IDENTIFIERS):
        if not re.search(rf"(?<![\w:]){re.escape(name)}(?![\w:])", code):
            issues.append(f"unknown identifier: {name}")
    for name, value in _CONSTANT_RE.findall(think):
        match = re.search(rf"\b{re.escape(name)}\s*(?:=|:)\s*(\d+)\b", code)
        if match is None:
            match = re.search(rf"#\s*define\s+{re.escape(name)}\s+(\d+)\b", code)
        if match is not None and int(match.group(1)) != int(value):
            issues.append(f"constant mismatch: {name} cot={value} code={match.group(1)}")
    return issues


def flags_for(row: dict[str, Any]) -> list[str]:
    """Return independent defect labels for one old or new SFT row."""
    flags: list[str] = []
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    system = _message(row, "system").lower()
    assistant = _message(row, "assistant")
    think = extract_thinking(assistant)
    code = extract_fenced_source(assistant)

    if any(marker in system for marker in REPAIR_SYSTEM_MARKERS):
        flags.append("repair_system_leak")
    if any(marker in system for marker in PROTOCOL_SYSTEM_MARKERS):
        flags.append("protocol_system_leak")
    if any(pattern.search(think) for pattern in _LEAK_RE):
        flags.append("cot_repair_leak")
    if think:
        issues = (
            cot_consistency_issues(think, code)
            if cot_consistency_issues is not None
            else _fallback_entity_issues(think, code)
        )
        if any(not issue.startswith("repair leak:") for issue in issues):
            flags.append("cot_entity_mismatch")

    pool = metadata.get("candidate_pool")
    if isinstance(pool, dict):
        selected, count = pool.get("selected_candidate"), pool.get("count")
        if selected is not None or count is not None:
            candidate = metadata.get("candidate")
            mismatch = (
                type(selected) is not int
                or type(count) is not int
                or not 1 <= selected <= count
                or (candidate is not None and (type(candidate) is not int or candidate != selected))
            )
            if mismatch:
                flags.append("pool_mismatch_risk")
    task = str(metadata.get("task") or row.get("task") or "").strip().lower()
    track = str(metadata.get("track") or metadata.get("dialect") or "").strip().lower()
    knowledge = task == "knowledge" or metadata.get("language") == "prose" or track.startswith("knowledge:")
    if not knowledge:
        refval = metadata.get("refval")
        if not isinstance(refval, dict) or refval.get("status") != "pass":
            flags.append("no_refval_evidence")
    return flags


def _regen_record(row: dict[str, Any] | None, flags: list[str], line_number: int) -> dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}
    original = _message(row, "user") if isinstance(row, dict) else ""
    if isinstance(row, dict) and isinstance(row.get("question"), str):
        original = row["question"]
    recorded_hash = metadata.get("question_hash")
    return {
        "id": row.get("id") if isinstance(row, dict) else None,
        "dialect": metadata.get("dialect") or metadata.get("track"),
        "question_hash": recorded_hash
        if isinstance(recorded_hash, str) and recorded_hash
        else question_hash(original)
        if original
        else None,
        "flags": flags,
        "source_line": line_number,
    }


def audit(source: Path, out: Path) -> dict[str, Any]:
    """Read the input unchanged and atomically publish audit artifacts."""
    names = ("needs_regen.jsonl", "clean_sft.jsonl", "audit_report.json")
    if source.resolve() in {(out / name).resolve() for name in names}:
        raise ValueError("audit output would replace the input file")
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    total = flagged = 0
    with tempfile.TemporaryDirectory(prefix=".audit_sft.", dir=out) as temporary:
        staging = Path(temporary)
        with (
            source.open("r", encoding="utf-8") as input_file,
            (staging / "needs_regen.jsonl").open("w", encoding="utf-8") as regen,
            (staging / "clean_sft.jsonl").open("w", encoding="utf-8") as clean,
        ):
            for line_number, line in enumerate(input_file, 1):
                if not line.strip():
                    continue
                total += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row, flags = None, ["invalid_json"]
                else:
                    flags = flags_for(row) if isinstance(row, dict) else ["invalid_row"]
                counts.update(flags)
                if flags:
                    flagged += 1
                    regen.write(
                        json.dumps(_regen_record(row, flags, line_number), ensure_ascii=False)
                        + "\n"
                    )
                else:
                    clean.write(line if line.endswith("\n") else line + "\n")
        report = {
            "input": str(source),
            "total": total,
            "flagged": flagged,
            "clean": total - flagged,
            "flag_counts": dict(counts),
            "flag_ratio": {name: round(count / total, 4) for name, count in counts.items()}
            if total
            else {},
        }
        (staging / "audit_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for name in names:
            os.replace(staging / name, out / name)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sft", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.sft, args.out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
