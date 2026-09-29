"""Build a deterministic stratified kernel question sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from cuda_sft.agents.difficulty import kernel_difficulty, knowledge_difficulty
from cuda_sft.graph import _operation_family
from cuda_sft.tasks.classify import infer_topic
from cuda_sft.tasks.kinds import question_hash


def build(source: Path, output: Path, *, limit: int = 120, task: str = "kernel") -> dict[str, Any]:
    if limit < 0:
        raise ValueError("limit must be non-negative")
    if task not in {"kernel", "knowledge"}:
        raise ValueError("task must be kernel or knowledge")
    if output.suffix != ".jsonl":
        output = output / "baseline_v1.jsonl"
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for source_line, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            continue
        family = _operation_family(question) if task == "kernel" else infer_topic(question, row)
        difficulty = kernel_difficulty(question) if task == "kernel" else knowledge_difficulty(family)
        enriched = {
            **row,
            "source_line": source_line,
            "question_hash": question_hash(question),
            ("family" if task == "kernel" else "topic"): family,
            "difficulty": difficulty,
        }
        if task == "knowledge":
            enriched["task"] = "knowledge"
        strata[(family, difficulty)].append(enriched)
    for rows in strata.values():
        rows.sort(key=lambda row: (row["question_hash"], row["source_line"]))

    selected: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    keys = sorted(strata)
    depth = 0
    while len(selected) < limit:
        added = False
        for key in keys:
            if depth < len(strata[key]):
                added = True
                candidate = strata[key][depth]
                if candidate["question_hash"] not in seen_hashes and len(selected) < limit:
                    selected.append(candidate)
                    seen_hashes.add(candidate["question_hash"])
        if not added:
            break
        depth += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    stats = {
        "source": str(source),
        "source_rows": sum(len(rows) for rows in strata.values()),
        "selected_rows": len(selected),
        "source_strata": {
            f"{family}/{difficulty}": len(rows)
            for (family, difficulty), rows in sorted(strata.items())
        },
        "strata": {
            f"{family}/{difficulty}": count
            for (family, difficulty), count in sorted(
                Counter((r["family" if task == "kernel" else "topic"], r["difficulty"]) for r in selected).items()
            )
        },
    }
    stats["task"] = task
    stats["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    stats["input_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, nargs="?", default=Path("question.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("benchmarks/baseline_v1.jsonl"))
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--task", choices=("kernel", "knowledge"), default="kernel")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.out, limit=args.limit, task=args.task), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
