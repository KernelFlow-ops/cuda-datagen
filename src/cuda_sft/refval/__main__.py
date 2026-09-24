"""``python -m cuda_sft.refval`` — offline batch validation of ``work/**/solution.*``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cuda_sft.config import PROJECT_ROOT, get_settings
from cuda_sft.refval.runner import run_offline
from cuda_sft.store import iter_questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline numeric validation of existing kernels")
    parser.add_argument("--work", type=Path, default=None, help="work root (default: settings.work_path)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input", type=Path, default=None, help="question.jsonl for LLM extract context")
    args = parser.parse_args(argv)
    settings = get_settings()
    work = args.work or settings.work_path
    questions_path = args.input or Path(settings.questions_path)
    if not questions_path.is_absolute():
        questions_path = PROJECT_ROOT / questions_path
    qmap = {}
    if questions_path.exists():
        qmap = {qid: text for qid, text in iter_questions(questions_path)}
    reports = run_offline(work_root=work, settings=settings, limit=args.limit, questions=qmap)
    n = len(reports)
    n_pass = sum(1 for r in reports if r.status == "pass")
    n_fail = sum(1 for r in reports if r.status == "fail")
    n_skip = sum(1 for r in reports if r.status == "skip")
    n_ref = sum(1 for r in reports if r.status == "reference_error")
    n_sig = sum(1 for r in reports if r.error_class == "signature_mismatch")
    print(
        f"refval-offline n={n} pass={n_pass} fail={n_fail} skip={n_skip} "
        f"reference_error={n_ref} signature_mismatch={n_sig}"
    )
    if n and (n_ref + n_sig) / n > 0.20:
        print(
            "WARNING: reference_error+signature_mismatch > 20%; retune extract prompt",
            file=sys.stderr,
        )
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
