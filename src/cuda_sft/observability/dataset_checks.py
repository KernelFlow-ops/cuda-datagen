"""Validate SFT JSONL against the applicable L6 dataset gates."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from cuda_sft.core.cot import REPAIR_LEAK_PATTERNS, cot_consistency_issues
from cuda_sft.knowledge.rubrics import DIMENSIONS
from cuda_sft.parse import extract_thinking, strip_thinking

CHECKS = tuple(f"H{i}" for i in range(1, 14))
REPAIR_SYSTEM = ("repairer", "compile-fix", "rewriting a failed", "fix the previous")
PROTOCOL_SYSTEM = ("nvcc -c", "do not explain", "compile-only gate", "downstream checker")
FENCE = re.compile(r"\A```[^\n`]*\n(.*?)\n?```\Z", re.DOTALL)
THINK = re.compile(r"\A<think>(.*?)</think>(.*)\Z", re.DOTALL)
PROSE_LINE = re.compile(
    r"^\s*(?:here (?:is|are)|this (?:code|kernel)|the (?:code|kernel)|"
    r"explanation:|note:|以下是|代码如下|说明[:：]|上述代码)",
    re.IGNORECASE,
)
LEAK_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in REPAIR_LEAK_PATTERNS)


def _version(metadata: dict[str, Any]) -> int:
    match = re.fullmatch(r"cuda-sft-(\d+)", str(metadata.get("dataset_version") or ""))
    return int(match.group(1)) if match else 0


def _kernel(metadata: dict[str, Any]) -> bool:
    return metadata.get("task") != "knowledge" and metadata.get("language") != "prose" and not str(metadata.get("track") or "").startswith("knowledge:")


def _assistant_parts(content: str, language: str) -> tuple[str, str, str | None]:
    body = content.strip()
    think = ""
    if body.startswith("<think>"):
        match = THINK.fullmatch(body)
        if match is None:
            return "", "", "think segment is malformed or repeated"
        think, body = match.group(1).strip(), match.group(2).strip()
    if re.search(r"</?(?:think|thinking|reasoning)>\s*", body, re.IGNORECASE):
        return think, "", "unexpected thinking tag in source"
    if "```" in body:
        match = FENCE.fullmatch(body)
        if match is None:
            return think, "", "expected exactly one fenced source block and no prose"
        if "```" in match.group(1):
            return think, "", "expected exactly one fenced source block"
        body = match.group(1).strip()
    if not body:
        return think, "", "source is empty"
    if any(PROSE_LINE.match(line) for line in body.splitlines()):
        return think, "", "assistant contains explanation outside source code"
    if language == "python":
        try:
            tree = ast.parse(body)
        except SyntaxError:
            return think, "", "Python source does not parse"
        if not tree.body or all(isinstance(node, ast.Expr) for node in tree.body):
            return think, "", "assistant contains prose instead of Python source"
    elif not any(token in body for token in (";", "{", "#include", "__global__", "<<<")):
        return think, "", "assistant does not look like source code"
    return think, body + "\n", None


def _required(metadata: dict[str, Any], *, kernel: bool, assistant: str = "") -> list[str]:
    missing: list[str] = []

    def need(obj: Any, key: str, typ: type, path: str) -> Any:
        value = obj.get(key) if isinstance(obj, dict) else None
        if value is None or not isinstance(value, typ) or (typ is int and isinstance(value, bool)):
            missing.append(path)
        return value

    for key in ("dataset_version", "sample_key", "question_hash", "track", "dialect", "language", "system_mode", "user_mode", "release_tier"):
        need(metadata, key, str, key)
    need(metadata, "source_line", int, "source_line")
    for key in ("candidate", "repairs"):
        need(metadata, key, int, key)
    generation = need(metadata, "generation", dict, "generation")
    generation_fields: tuple[tuple[str, type], ...] = (("system", str), ("user", str), ("prompt_variant", dict))
    for key, typ in generation_fields:
        need(generation, key, typ, f"generation.{key}")
    cot = need(metadata, "cot", dict, "cot")
    for key, typ in (("source", str), ("policy", str), ("chars", int)):
        need(cot, key, typ, f"cot.{key}")
    if isinstance(cot, dict) and "consistency_issues" in cot:
        need(cot, "consistency_issues", list, "cot.consistency_issues")
    provenance = need(metadata, "provenance", dict, "provenance")
    for key, typ in (("model", str), ("provider", str), ("prompt_packs", dict), ("run_id", str), ("created_at", str)):
        need(provenance, key, typ, f"provenance.{key}")
    if kernel:
        need(metadata, "selected_code_sha256", str, "selected_code_sha256")
        need(metadata, "refval", dict, "refval")
        critic = need(metadata, "critic", dict, "critic")
        need(critic, "status", str, "critic.status")
        need(metadata, "judge", dict, "judge")
        pool = need(metadata, "candidate_pool", dict, "candidate_pool")
        pool_fields: tuple[tuple[str, type], ...] = (("count", int), ("selected_candidate", int), ("eligible_count", int), ("reports", list))
        for key, typ in pool_fields:
            need(pool, key, typ, f"candidate_pool.{key}")
    else:
        topic = need(metadata, "topic", str, "topic")
        review = need(metadata, "knowledge_judge", dict, "knowledge_judge")
        if isinstance(review, dict):
            for key, expected in (("pass", True), ("unavailable", False), ("skipped_llm", False), ("hard_gate_failed", False)):
                if review.get(key) is not expected:
                    missing.append(f"knowledge_judge.{key}")
            if review.get("judge_error", "") != "":
                missing.append("knowledge_judge.judge_error")
            if review.get("must_fix") != []:
                missing.append("knowledge_judge.must_fix")
            if "topic" in review and review["topic"] != topic:
                missing.append("knowledge_judge.topic")
            body = strip_thinking(assistant)
            digest = hashlib.sha256(body.encode("utf-8")).hexdigest() if body else ""
            if not digest or review.get("answer_sha256") != digest:
                missing.append("knowledge_judge.answer_sha256")
            dimensions = need(review, "dimensions", dict, "knowledge_judge.dimensions")
            if isinstance(dimensions, dict):
                for key in DIMENSIONS:
                    score = dimensions.get(key)
                    if not isinstance(score, (int, float)) or isinstance(score, bool) or not 1 <= score <= 10:
                        missing.append(f"knowledge_judge.dimensions.{key}")
            overall = review.get("overall")
            if not isinstance(overall, (int, float)) or isinstance(overall, bool) or not 1 <= overall <= 10:
                missing.append("knowledge_judge.overall")
    return missing


def _quantile(values: list[int], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = fraction * (len(ordered) - 1)
    lo = int(pos)
    return round(ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo), 2)


def check_dataset(source: Path) -> dict[str, Any]:
    """Read a JSONL dataset and return per-rule pass/fail/n/a counts."""
    rules: dict[str, dict[str, int]] = {key: {"pass": 0, "fail": 0, "n/a": 0} for key in CHECKS}
    failures: list[dict[str, Any]] = []
    by_track: Counter[str] = Counter()
    by_tier: Counter[str] = Counter()
    cot_lengths: list[int] = []
    repaired = total = failed_rows = 0

    with source.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            total += 1
            row_failed = False

            def record(rule: str, error: str | None = None, *, _line_no: int = line_no) -> None:
                nonlocal row_failed
                state = "fail" if error else "pass"
                rules[rule][state] += 1
                if error:
                    row_failed = True
                    failures.append({"line": _line_no, "check": rule, "message": error})

            def na(rule: str) -> None:
                rules[rule]["n/a"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                record("H1", f"invalid JSON: {exc.msg}")
                for rule in CHECKS[1:]:
                    na(rule)
                failed_rows += 1
                continue
            messages = row.get("messages") if isinstance(row, dict) else None
            metadata = row.get("metadata") if isinstance(row, dict) else None
            roles = [m.get("role") for m in messages] if isinstance(messages, list) and all(isinstance(m, dict) for m in messages) else []
            if not isinstance(messages, list) or roles not in (["user", "assistant"], ["system", "user", "assistant"]) or not isinstance(metadata, dict) or not all(isinstance(m.get("content"), str) for m in messages):
                record("H1", "expected metadata and [system,] user, assistant messages with text content")
                for rule in CHECKS[1:]:
                    na(rule)
                failed_rows += 1
                continue
            assert isinstance(messages, list)
            record("H1")
            system = messages[0]["content"] if roles[0] == "system" else ""
            assistant = messages[-1]["content"]
            system_lower = system.lower()
            record("H2", "repair system text found" if any(s in system_lower for s in REPAIR_SYSTEM) else None)
            if metadata.get("system_mode") == "fixed":
                record("H3", "generation protocol found in fixed system" if any(s in system_lower for s in PROTOCOL_SYSTEM) else None)
            else:
                na("H3")

            kernel = _kernel(metadata)
            think = extract_thinking(assistant)
            code = ""
            if kernel:
                think, code, error = _assistant_parts(assistant, str(metadata.get("language") or ""))
                record("H4", error)
            else:
                na("H4")
            if _version(metadata) >= 2 and kernel:
                if code:
                    got = hashlib.sha256(code.encode("utf-8")).hexdigest()
                    record("H5", "selected_code_sha256 differs from assistant source" if got != metadata.get("selected_code_sha256") else None)
                else:
                    na("H5")
            else:
                na("H5")
            record("H6", "repair narrative found in think" if any(p.search(think) for p in LEAK_PATTERNS) else None)
            cot = metadata.get("cot")
            if kernel and isinstance(cot, dict) and "consistency_issues" in cot and code:
                issues = cot_consistency_issues(think, code) if think else []
                record("H7", "; ".join(issues) if issues else None)
            else:
                na("H7")
            if kernel and metadata.get("release_tier") in {"strict", "strict_perf"}:
                refval = metadata.get("refval")
                refval = refval if isinstance(refval, dict) else {}
                cases = refval.get("cases_run")
                good = refval.get("status") == "pass" and isinstance(cases, int) and not isinstance(cases, bool) and cases >= 3 and bool(refval.get("manifest_hash"))
                record("H8", "strict tier lacks passing refval with >=3 cases and manifest hash" if not good else None)
            else:
                na("H8")
            for rule in ("H9", "H10", "H11", "H12"):
                na(rule)
            if _version(metadata) >= 2:
                missing = _required(metadata, kernel=kernel, assistant=assistant)
                record("H13", f"missing or invalid fields: {', '.join(missing)}" if missing else None)
            else:
                na("H13")

            by_track[str(metadata.get("track") or metadata.get("dialect") or "unknown")] += 1
            by_tier[str(metadata.get("release_tier") or "unknown")] += 1
            cot_lengths.append(len(think))
            repaired += int(isinstance(metadata.get("repairs"), int) and metadata["repairs"] > 0)
            failed_rows += int(row_failed)

    if total == 0:
        rules["H1"]["fail"] += 1
        failures.append({"line": 0, "check": "H1", "message": "dataset has no nonblank rows"})
        failed_rows = 1
    return {
        "source": str(source),
        "rows": total,
        "failed_rows": failed_rows,
        "ok": failed_rows == 0,
        "checks": rules,
        "failures": failures,
        "metrics": {
            "by_track": dict(by_track),
            "by_tier": dict(by_tier),
            "repaired_ratio": round(repaired / total, 4) if total else None,
            "cot_chars_p5": _quantile(cot_lengths, 0.05),
            "cot_chars_p50": _quantile(cot_lengths, 0.5),
            "cot_chars_p95": _quantile(cot_lengths, 0.95),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reasoning-dir", type=Path)
    parser.add_argument("--state-db", type=Path)
    args = parser.parse_args(argv)
    if args.source.resolve() == args.out.resolve():
        parser.error("--out must not replace the source dataset")
    try:
        report = check_dataset(args.source)
    except OSError as exc:
        print(f"dataset check failed: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"L6: {report['rows']} rows, {report['failed_rows']} failed; {args.out}")
    return 0 if report["ok"] else 1
