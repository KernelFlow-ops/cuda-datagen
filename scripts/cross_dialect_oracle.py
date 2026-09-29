#!/usr/bin/env python3
"""Real GPU cross-dialect oracle benchmark.

This is an executable benchmark, not a unit-test wrapper. It compiles/imports
canonical CUDA/CUTLASS/Triton sources, launches them on the local GPU, compares
outputs against the independent canonical reference, and records unavailable
backends explicitly.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from cuda_sft.config import Settings
from cuda_sft.dialects.agent import get_spec
from cuda_sft.refval.cases import case_plan_hash
from cuda_sft.refval.cross_dialect import (
    canonical_plans,
    canonical_tasks,
    group_hash,
    make_manifest,
    task_source,
)
from cuda_sft.refval.runner import run_refval
from cuda_sft.refval.spec import RefvalReport, stable_hash


def mutation_caught(report: RefvalReport) -> bool:
    """Count a mutant only when a numeric comparison detected the error."""
    return (
        report.status == "fail"
        and report.error_class in {"numeric_mismatch", "nan_inf"}
        and report.cases_run > 0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real GPU cross-dialect numeric oracle")
    parser.add_argument("--tasks", default="elementwise_add,scale,row_sum")
    parser.add_argument("--dialects", default="cuda,cutlass,triton,tilelang")
    parser.add_argument("--cases", default="smoke", choices=("smoke", "standard", "full"))
    parser.add_argument(
        "--mutants", action="store_true", help="also run deliberate wrong implementations"
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = Settings(
        refval_enabled=True,
        refval_cases=args.cases,
        refval_timeout_sec=180,
        refval_strict=True,
        refval_cache=False,
        async_llm_enabled=False,
        workers=1,
        work_dir=str(args.work_dir or tempfile.mkdtemp(prefix="cross-dialect-oracle-")),
    )
    registry = canonical_tasks()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    dialects = [item.strip().lower() for item in args.dialects.split(",") if item.strip()]
    rows: list[dict[str, object]] = []

    for task_id in tasks:
        task = registry.get(task_id)
        if task is None:
            rows.append(
                {"task_id": task_id, "status": "unavailable", "reason": "unknown canonical task"}
            )
            continue
        for dialect in dialects:
            spec = get_spec(dialect)
            available, reason = spec.available(settings)
            row_base = {
                "task_id": task_id,
                "dialect": dialect,
                "group_hash": group_hash(task),
                "cases": args.cases,
            }
            if not available:
                rows.append({**row_base, "status": "unavailable", "reason": reason})
                continue
            source = task_source(task, dialect)
            if not source.strip():
                rows.append(
                    {
                        **row_base,
                        "status": "unavailable",
                        "reason": "no canonical adapter source for dialect",
                    }
                )
                continue
            qid = int(stable_hash({"task": task_id})[:8], 16) & 0x7FFFFFFF
            manifest = make_manifest(task, question_id=qid, dialect=dialect)
            plans = canonical_plans(task, question_id=manifest.question_id, suite=args.cases)
            started = time.monotonic()
            report = run_refval(
                question=task_id,
                code=source,
                question_id=manifest.question_id,
                dialect=dialect,
                dialect_spec=spec.refval_spec(settings),
                settings=settings,
                workdir=Path(settings.work_dir) / f"{task_id}-{dialect}-good",
                manifest=manifest,
                task_spec=manifest.semantic_contract,
                oracle_spec=manifest.backend_contract,
                case_plans=plans,
            )
            row = {
                **row_base,
                "variant": "good",
                "status": report.status,
                "error_class": report.error_class,
                "cases_run": report.cases_run,
                "failed_case": report.failed_case,
                "cases_hash": report.cases_hash or case_plan_hash(plans),
                "manifest_hash": report.manifest_hash,
                "elapsed_sec": round(report.elapsed_sec or (time.monotonic() - started), 3),
            }
            rows.append(row)

            if args.mutants and dialect in {"cuda", "cutlass", "triton", "tilelang"}:
                mutant = task_source(task, dialect, mutant=True)
                mutant_report = run_refval(
                    question=task_id,
                    code=mutant,
                    # Mutants consume the identical frozen case suite; only
                    # the source and artifact directory differ.
                    question_id=manifest.question_id,
                    dialect=dialect,
                    dialect_spec=spec.refval_spec(settings),
                    settings=settings,
                    workdir=Path(settings.work_dir) / f"{task_id}-{dialect}-mutant",
                    manifest=manifest,
                    task_spec=manifest.semantic_contract,
                    oracle_spec=manifest.backend_contract,
                    case_plans=plans,
                )
                rows.append(
                    {
                        **row_base,
                        "variant": "mutant",
                        "status": mutant_report.status,
                        "error_class": mutant_report.error_class,
                        "cases_run": mutant_report.cases_run,
                        "failed_case": mutant_report.failed_case,
                        "cases_hash": mutant_report.cases_hash or case_plan_hash(plans),
                    "mutation_caught": mutation_caught(mutant_report),
                        "elapsed_sec": round(mutant_report.elapsed_sec, 3),
                    }
                )

    summary = {
        "tasks": tasks,
        "dialects": dialects,
        "cases": args.cases,
        "rows": rows,
        "good_pass": sum(
            1 for r in rows if r.get("variant") == "good" and r.get("status") == "pass"
        ),
        "mutants_caught": sum(
            1 for r in rows if r.get("variant") == "mutant" and r.get("mutation_caught")
        ),
        "unavailable": sum(1 for r in rows if r.get("status") == "unavailable"),
        "work_dir": settings.work_dir,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0 if summary["good_pass"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
