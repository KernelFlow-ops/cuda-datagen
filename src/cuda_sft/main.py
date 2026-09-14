from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm

from cuda_sft.compile import smoke_compile
from cuda_sft.config import PROJECT_ROOT, get_settings
from cuda_sft.graph import build_graph, recursion_limit, set_print_stream
from cuda_sft.parse import extract_cuda_source
from cuda_sft.formats import export_training_files
from cuda_sft.store import init_store, iter_questions, load_done_ids

logger = logging.getLogger("cuda_sft")


def _parse_ids(raw: str | None) -> set[int] | None:
    if not raw:
        return None
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ids.add(int(part))
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CUDA operator SFT data with LangGraph + OpenRouter"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="question jsonl path (default: ./question.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=None, help="max questions to process")
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip the first N questions in the jsonl file",
    )
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="comma-separated question ids (1-based jsonl line numbers)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="reprocess questions even if they are already in progress.jsonl",
    )
    parser.add_argument(
        "--dry-compile",
        action="store_true",
        help="compile a built-in smoke kernel and exit (no API calls)",
    )
    parser.add_argument(
        "--export-sft",
        action="store_true",
        help="rebuild sft_ms_swift.jsonl and sft_openrlhf.jsonl from data/sft.jsonl (no API)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="do not print streamed model tokens",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="output directory for sft/abandoned/progress jsonl",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="logging level (default: INFO)",
    )
    return parser


def _configure_logging(level: str, log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _select_questions(
    path: Path,
    *,
    offset: int,
    limit: int | None,
    ids: set[int] | None,
    done: set[int],
) -> list[tuple[int, str]]:
    rows = list(iter_questions(path))
    if offset:
        rows = rows[offset:]
    if ids is not None:
        rows = [row for row in rows if row[0] in ids]
    if done:
        rows = [row for row in rows if row[0] not in done]
    if limit is not None:
        rows = rows[:limit]
    return rows


def run_dry_compile() -> int:
    settings = get_settings()
    print(
        f"nvcc={settings.nvcc_bin} arch={settings.resolved_cuda_arch} "
        f"gpu={settings.resolved_gpu_name} cuda={settings.resolved_cuda_version}"
    )
    parsed = extract_cuda_source("```cuda\n__global__ void k() {}\n```")
    if "__global__" not in parsed:
        print("parse smoke failed", file=sys.stderr)
        return 1
    result = smoke_compile(settings)
    if result.ok:
        print("dry-compile PASS")
        return 0
    print("dry-compile FAIL", file=sys.stderr)
    print(result.output, file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    set_print_stream(not args.quiet)

    if args.dry_compile:
        _configure_logging(args.log_level)
        return run_dry_compile()

    if args.export_sft:
        _configure_logging(args.log_level)
        data_dir = Path(args.data_dir) if args.data_dir else settings.data_path
        src = Path(args.input) if args.input else data_dir / "sft.jsonl"
        if not src.is_absolute():
            src = PROJECT_ROOT / src if not src.exists() else src
        if not src.exists():
            print(f"sft archive not found: {src}", file=sys.stderr)
            return 2
        n, swift_path, orl_path = export_training_files(src, data_dir)
        print(f"exported {n} samples")
        print(f"ms-swift   {swift_path}")
        print(f"openrlhf   {orl_path}")
        return 0 if n else 1

    if not settings.openrouter_api_key.strip():
        print(
            "OPENROUTER_API_KEY is empty. Put your key in .env and retry.",
            file=sys.stderr,
        )
        return 2

    questions_path = Path(args.input) if args.input else Path(settings.questions_path)
    if not questions_path.is_absolute():
        questions_path = PROJECT_ROOT / questions_path
    if not questions_path.exists():
        print(f"questions file not found: {questions_path}", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir) if args.data_dir else settings.data_path
    _configure_logging(args.log_level, data_dir / "run.log")
    store = init_store(data_dir)
    done: set[int] = set() if args.overwrite else load_done_ids(store.progress_path)
    ids = _parse_ids(args.ids)
    jobs = _select_questions(
        questions_path,
        offset=args.offset,
        limit=args.limit,
        ids=ids,
        done=done,
    )

    print(
        f"model={settings.model} thinking={settings.thinking_level} "
        f"arch={settings.resolved_cuda_arch} gpu={settings.resolved_gpu_name}"
    )
    print(
        f"candidates={settings.max_candidates} repairs={settings.max_repairs} "
        f"queued={len(jobs)} skipped_done={len(done)}"
    )
    if not jobs:
        print("nothing to do")
        return 0

    app = build_graph()
    limit = recursion_limit()
    success = 0
    abandoned = 0
    failed = 0

    for index, (qid, question) in enumerate(
        tqdm(jobs, desc="questions", file=sys.stderr, unit="q"), start=1
    ):
        print(f"\n===== [{index}/{len(jobs)}] question {qid} =====", flush=True)
        final = None
        for attempt in range(1, 4):
            try:
                final = app.invoke(
                    {"question_id": qid, "question": question, "status": "running"},
                    {"recursion_limit": limit},
                )
                break
            except KeyboardInterrupt:
                print("\ninterrupted", file=sys.stderr)
                return 130
            except Exception as exc:
                if attempt >= 3:
                    logger.exception("question %s crashed after %s attempts", qid, attempt)
                    failed += 1
                    break
                wait = 8 * attempt
                logger.warning(
                    "question %s crashed (attempt %s/3): %s; retrying in %ss",
                    qid,
                    attempt,
                    exc,
                    wait,
                )
                time.sleep(wait)
        if final is None:
            continue

        status = (final or {}).get("status")
        if status == "success":
            success += 1
        elif status == "abandoned":
            abandoned += 1
        else:
            failed += 1
            logger.error("question %s ended with unexpected status %s", qid, status)

    print(
        f"\ndone success={success} abandoned={abandoned} crashed={failed} "
        f"sft={store.sft_path} abandoned_log={store.abandoned_path}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
