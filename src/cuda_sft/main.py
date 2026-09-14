"""CLI: generate CUDA SFT data, export training jsonl, or dry-compile."""

from __future__ import annotations

import argparse
import fcntl
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

from cuda_sft.compile import smoke_compile
from cuda_sft.config import PROJECT_ROOT, get_settings
from cuda_sft.graph import build_graph, recursion_limit, set_print_stream
from cuda_sft.parse import extract_cuda_source
from cuda_sft.formats import export_training_files
from cuda_sft.store import init_store, iter_questions, load_done_ids

logger = logging.getLogger("cuda_sft")


def _parse_ids(raw: str | None) -> set[int] | None:
    """Parse ``--ids 1,2,10`` into a set of ints, or None if unset."""
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
    """Build the ``run.py`` argument parser."""
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
        "--workers",
        type=int,
        default=None,
        help="parallel worker processes (default: WORKERS in .env, else 1)",
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


class _WorkerFilter(logging.Filter):
    """Attach ``record.worker`` so the log format can show ``[wN]``."""

    def __init__(self, worker_id: int) -> None:
        """Store the worker index used in log lines.

        Args:
            worker_id: Process index written into each log record.
        """
        super().__init__()
        self.worker_id = worker_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Set ``record.worker`` and always keep the record."""
        record.worker = self.worker_id
        return True


class _LockedFileHandler(logging.Handler):
    """Process-safe append to a single run.log."""

    def __init__(self, path: Path) -> None:
        """Write to ``path`` using exclusive flock per emit.

        Args:
            path: Shared ``data/run.log``.
        """
        super().__init__()
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        """Format and append one log line under ``LOCK_EX``."""
        try:
            line = self.format(record) + "\n"
            with self.path.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    handle.write(line)
                    handle.flush()
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            self.handleError(record)


def _reset_run_logs(data_dir: Path) -> Path:
    """Delete previous ``run*.log`` files and create an empty ``run.log``.

    Args:
        data_dir: Output directory.

    Returns:
        Path to the new ``run.log``.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for old in data_dir.glob("run*.log"):
        try:
            old.unlink()
        except OSError:
            pass
    log_file = data_dir / "run.log"
    log_file.write_text("", encoding="utf-8")
    return log_file


def _configure_logging(
    level: str,
    log_file: Path | None = None,
    *,
    worker_id: int = 0,
) -> None:
    """Configure stderr + optional locked ``run.log``; quiet noisy libraries.

    Args:
        level: Root log level name (e.g. ``INFO``).
        log_file: If set, also write to this file.
        worker_id: Value shown as ``[wN]`` in each line.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        handlers.append(_LockedFileHandler(log_file))
    worker_filter = _WorkerFilter(worker_id)
    fmt = "%(asctime)s %(levelname)s [w%(worker)s] %(name)s: %(message)s"
    for handler in handlers:
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        handler.addFilter(worker_filter)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=handlers,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "openai", "anthropic", "langgraph"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _select_questions(
    path: Path,
    *,
    offset: int,
    limit: int | None,
    ids: set[int] | None,
    done: set[int],
) -> list[tuple[int, str]]:
    """Apply offset, id filter, resume skip, and limit to the question list.

    Args:
        path: ``question.jsonl``.
        offset: Skip this many rows from the start of the file.
        limit: Max remaining questions to keep.
        ids: If set, only these 1-based ids.
        done: Ids already success/abandoned (skipped unless ``--overwrite``).
    """
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
    """Compile a tiny kernel to verify nvcc; return process exit code."""
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
    """CLI entry: dry-compile, export SFT formats, or run generation.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success, non-zero on failure / missing config.
    """
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

    if not settings.resolved_api_key:
        if settings.llm_provider == "nvidia":
            print("NVIDIA_API_KEY is empty. Put the nvapi- key in .env and retry.", file=sys.stderr)
        else:
            print("OPENROUTER_API_KEY is empty. Put your key in .env and retry.", file=sys.stderr)
        return 2

    questions_path = Path(args.input) if args.input else Path(settings.questions_path)
    if not questions_path.is_absolute():
        questions_path = PROJECT_ROOT / questions_path
    if not questions_path.exists():
        print(f"questions file not found: {questions_path}", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir) if args.data_dir else settings.data_path
    log_file = _reset_run_logs(data_dir)
    _configure_logging(args.log_level, log_file, worker_id=0)
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
        f"provider={settings.llm_provider} model={settings.resolved_model} "
        f"thinking={settings.thinking_level} base={settings.resolved_base_url} "
        f"arch={settings.resolved_cuda_arch} gpu={settings.resolved_gpu_name}"
    )
    workers = args.workers if args.workers is not None else settings.workers
    workers = max(1, int(workers))
    if workers > 1:
        set_print_stream(False)

    print(
        f"candidates={settings.max_candidates} repairs={settings.max_repairs} "
        f"workers={workers} queued={len(jobs)} skipped_done={len(done)}"
    )
    if not jobs:
        print("nothing to do")
        return 0

    if workers == 1:
        success, abandoned, failed = run_job_list(
            jobs, data_dir=data_dir, worker_id=0, show_progress=True
        )
    else:
        success, abandoned, failed = run_multiprocess(
            jobs, data_dir=data_dir, workers=workers, log_level=args.log_level
        )

    print(
        f"\ndone success={success} abandoned={abandoned} crashed={failed} "
        f"sft={store.sft_path} abandoned_log={store.abandoned_path}"
    )
    return 0 if failed == 0 else 1


def run_job_list(
    jobs: list[tuple[int, str]],
    *,
    data_dir: Path,
    worker_id: int = 0,
    show_progress: bool = True,
) -> tuple[int, int, int]:
    """Run the LangGraph pipeline on ``jobs`` in this process.

    Args:
        jobs: ``(question_id, question)`` pairs.
        data_dir: Output directory for jsonl / logs.
        worker_id: Index used in log prefixes.
        show_progress: If True, wrap the loop in tqdm.

    Returns:
        ``(success, abandoned, crashed)`` counts.
    """
    init_store(data_dir)
    app = build_graph()
    limit = recursion_limit()
    success = 0
    abandoned = 0
    failed = 0
    prefix = f"w{worker_id}"
    iterable: Any = jobs
    if show_progress:
        iterable = tqdm(jobs, desc=f"questions[{prefix}]", file=sys.stderr, unit="q")

    for index, (qid, question) in enumerate(iterable, start=1):
        print(f"\n===== [{prefix} {index}/{len(jobs)}] question {qid} =====", flush=True)
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
                raise
            except Exception as exc:
                if attempt >= 3:
                    logger.exception(
                        "[%s] question %s crashed after %s attempts", prefix, qid, attempt
                    )
                    failed += 1
                    break
                wait = 8 * attempt
                logger.warning(
                    "[%s] question %s crashed (attempt %s/3): %s; retrying in %ss",
                    prefix,
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
            logger.error(
                "[%s] question %s ended with unexpected status %s", prefix, qid, status
            )
    return success, abandoned, failed


def _mp_entry(payload: dict[str, Any]) -> dict[str, Any]:
    """Spawn-worker entry: configure logging then :func:`run_job_list`.

    Args:
        payload: ``worker_id``, ``data_dir``, ``jobs``, ``log_level``.
    """
    worker_id = int(payload["worker_id"])
    data_dir = Path(payload["data_dir"])
    jobs: list[tuple[int, str]] = payload["jobs"]
    log_level = str(payload["log_level"])
    time.sleep(0.6 * worker_id)
    _configure_logging(log_level, data_dir / "run.log", worker_id=worker_id)
    set_print_stream(False)
    success, abandoned, failed = run_job_list(
        jobs, data_dir=data_dir, worker_id=worker_id, show_progress=True
    )
    return {
        "worker_id": worker_id,
        "success": success,
        "abandoned": abandoned,
        "failed": failed,
        "n": len(jobs),
    }


def run_multiprocess(
    jobs: list[tuple[int, str]],
    *,
    data_dir: Path,
    workers: int,
    log_level: str,
) -> tuple[int, int, int]:
    """Shard ``jobs`` round-robin across ``workers`` spawn processes.

    Args:
        jobs: Full remaining question list.
        data_dir: Shared output directory (jsonl writes are flocked).
        workers: Process count.
        log_level: Passed to each child.

    Returns:
        Aggregated ``(success, abandoned, crashed)``.
    """
    shards: list[list[tuple[int, str]]] = [[] for _ in range(workers)]
    for i, job in enumerate(jobs):
        shards[i % workers].append(job)
    src = str(PROJECT_ROOT / "src")
    pythonpath = os.environ.get("PYTHONPATH", "")
    if src not in pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = src + (os.pathsep + pythonpath if pythonpath else "")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        payloads = [
            {
                "worker_id": i,
                "data_dir": str(data_dir),
                "jobs": shard,
                "log_level": log_level,
            }
            for i, shard in enumerate(shards)
            if shard
        ]
        try:
            results = pool.map(_mp_entry, payloads)
        except KeyboardInterrupt:
            pool.terminate()
            print("\ninterrupted", file=sys.stderr)
            raise
    success = sum(r["success"] for r in results)
    abandoned = sum(r["abandoned"] for r in results)
    failed = sum(r["failed"] for r in results)
    for r in results:
        print(
            f"worker {r['worker_id']}: n={r['n']} success={r['success']} "
            f"abandoned={r['abandoned']} crashed={r['failed']}",
            flush=True,
        )
    return success, abandoned, failed


if __name__ == "__main__":
    raise SystemExit(main())
