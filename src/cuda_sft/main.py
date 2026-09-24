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

from cuda_sft.config import PROJECT_ROOT, WorkerSlot, get_settings
from cuda_sft.dialects.agent import get_dialect_agent
from cuda_sft.graph import build_graph, recursion_limit
from cuda_sft.llm import is_retryable_llm_error
from cuda_sft.formats import export_training_files
from cuda_sft.pipeline.common import set_print_stream
from cuda_sft.store import init_store, iter_question_rows, load_done_keys
from cuda_sft.tasks.kinds import Job, QuestionRow, coerce_job, progress_key
from cuda_sft.tasks.router import expand_pipeline_jobs

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
        "--setup",
        action="store_true",
        help="detect kernel dialect environments and install anything missing (same as bash scripts/setup_env.sh)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="with --setup (or alone): only detect, do not install",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        help="with --setup: skip dialect smoke compiles",
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
        "--providers",
        type=str,
        default=None,
        help="comma-separated providers, e.g. nvidia,openrouter",
    )
    parser.add_argument(
        "--workers-per-provider",
        type=int,
        default=None,
        help="workers per provider slot (NVIDIA: each API key is a slot; overrides WORKERS)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="output directory for sft/abandoned/progress jsonl",
    )
    parser.add_argument(
        "--dialects",
        type=str,
        default=None,
        help="comma-separated kernel dialects: cuda,cutlass,triton,tilelang (aliases: cute)",
    )
    parser.add_argument(
        "--kernel-mode",
        type=str,
        default=None,
        help="single (one dialect) or all (every listed dialect per question)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="kernel (default, compile gate), knowledge (rubric gate), or auto",
    )
    parser.add_argument(
        "--no-refval",
        action="store_true",
        help="disable CPU-reference GPU numeric validation (compile gate only)",
    )
    parser.add_argument(
        "--refval-offline",
        action="store_true",
        help="batch-validate existing work/**/solution.cu|.py (no generation)",
    )
    parser.add_argument(
        "--refval-limit",
        type=int,
        default=None,
        help="with --refval-offline: max kernels to validate",
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
    """Open ``run.log`` for this process; keep prior sessions when resuming.

    A resume (same ``data_dir``, existing progress) must not wipe compile
    diagnostics. Start a new file only when none exists.

    Args:
        data_dir: Output directory.

    Returns:
        Path to ``run.log``.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    log_file = data_dir / "run.log"
    if log_file.exists() and log_file.stat().st_size > 0:
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n===== session {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n"
            )
    else:
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


def _select_question_rows(
    path: Path,
    *,
    offset: int,
    limit: int | None,
    ids: set[int] | None,
    done: set[int],
) -> list[QuestionRow]:
    """Apply offset, id filter, resume skip, and limit to the question list.

    Args:
        path: ``question.jsonl``.
        offset: Skip this many rows from the start of the file.
        limit: Max remaining questions to keep.
        ids: If set, only these 1-based ids.
        done: Ids already success/abandoned (skipped unless ``--overwrite``).
    """
    rows = [
        QuestionRow(question_id=qid, question=question, raw=raw)
        for qid, question, raw in iter_question_rows(path)
    ]
    if offset:
        rows = rows[offset:]
    if ids is not None:
        rows = [row for row in rows if row.question_id in ids]
    if done:
        rows = [row for row in rows if row.question_id not in done]
    if limit is not None:
        rows = rows[:limit]
    return rows


def _apply_kernel_cli(args: argparse.Namespace) -> None:
    """Copy ``--dialects`` / ``--kernel-mode`` / ``--task`` into env and drop cache."""
    changed = False
    if args.dialects:
        os.environ["KERNEL_DIALECTS"] = args.dialects
        changed = True
    if args.kernel_mode:
        os.environ["KERNEL_MODE"] = args.kernel_mode
        changed = True
    if getattr(args, "task", None):
        os.environ["TASK_MODE"] = args.task
        changed = True
    if getattr(args, "no_refval", False):
        os.environ["REFVAL_ENABLED"] = "false"
        changed = True
    if changed:
        get_settings.cache_clear()


def _set_all_print_stream(enabled: bool) -> None:
    """Toggle token streaming for kernel and knowledge generate nodes."""
    set_print_stream(enabled)


def run_dry_compile() -> int:
    """Smoke-compile every available kernel dialect and return process exit code."""
    settings = get_settings()
    agent = get_dialect_agent()
    print(
        f"nvcc={settings.nvcc_bin} arch={settings.resolved_cuda_arch} "
        f"gpu={settings.resolved_gpu_name} cuda={settings.resolved_cuda_version} "
        f"dialects={','.join(agent.requested_names(settings))}"
    )
    failed = 0
    for spec in agent.resolve(settings):
        workdir = settings.work_path / f"_smoke_{spec.name}"
        result = spec.smoke(settings, workdir)
        status = "PASS" if result.ok else "FAIL"
        print(f"dry-compile {spec.name}: {status}")
        if not result.ok:
            failed += 1
            print(result.output, file=sys.stderr)
    if failed:
        print(f"dry-compile FAIL ({failed} dialect(s))", file=sys.stderr)
        return 1
    print("dry-compile PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry: dry-compile, export SFT formats, or run generation.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        0 on success, non-zero on failure / missing config.
    """
    args = build_parser().parse_args(argv)
    _apply_kernel_cli(args)
    settings = get_settings()
    _set_all_print_stream(not args.quiet)

    if args.setup or args.check:
        from cuda_sft.dialects.agent import parse_dialect_list
        from cuda_sft.setup_env import run_setup

        _configure_logging(args.log_level)
        names = parse_dialect_list(args.dialects) if args.dialects else None
        return run_setup(
            dialects=names,
            check_only=not args.setup or args.check,
            no_smoke=args.no_smoke,
        )

    if args.dry_compile:
        _configure_logging(args.log_level)
        return run_dry_compile()

    if getattr(args, "refval_offline", False):
        from cuda_sft.refval.runner import run_offline
        from cuda_sft.store import iter_questions

        _configure_logging(args.log_level)
        questions_path = Path(args.input) if args.input else Path(settings.questions_path)
        if not questions_path.is_absolute():
            questions_path = PROJECT_ROOT / questions_path
        qmap = {}
        if questions_path.exists():
            qmap = {qid: text for qid, text in iter_questions(questions_path)}
        reports = run_offline(
            work_root=settings.work_path,
            settings=settings,
            limit=getattr(args, "refval_limit", None),
            questions=qmap,
        )
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

    if args.providers:
        object.__setattr__(settings, "llm_providers", args.providers)
    providers = settings.provider_pool()
    missing = settings.missing_provider_secrets(providers)
    if missing:
        joined = " / ".join(missing)
        print(f"{joined} is empty. Put the key(s) in .env and retry.", file=sys.stderr)
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
    ids = _parse_ids(args.ids)
    questions = _select_question_rows(
        questions_path,
        offset=args.offset,
        limit=args.limit,
        ids=ids,
        done=set(),
    )
    all_jobs = expand_pipeline_jobs(questions, settings)
    if not args.overwrite:
        done_keys = load_done_keys(store.progress_path)
        jobs = [job for job in all_jobs if progress_key(job) not in done_keys]
    else:
        jobs = all_jobs
    skipped_done = 0 if args.overwrite else len(all_jobs) - len(jobs)

    wpp = (
        args.workers_per_provider
        if args.workers_per_provider is not None
        else settings.workers_per_provider
    )
    workers_arg = args.workers if args.workers is not None else settings.workers
    assignments = settings.build_worker_assignments(
        workers=max(1, int(workers_arg)),
        workers_per_provider=int(wpp) if wpp else 0,
        providers=providers,
    )
    workers = max(1, len(assignments))
    if workers > 1:
        _set_all_print_stream(False)

    nvidia_n = len(settings.nvidia_api_keys()) if "nvidia" in providers else 0
    labels = [slot.label for slot in assignments]
    dialect_label = "-"
    if settings.task_mode != "knowledge":
        try:
            dialect_label = ",".join(
                spec.name for spec in get_dialect_agent().resolve(settings)
            )
        except RuntimeError:
            dialect_label = "none"
    print(
        f"providers={','.join(providers)} nvidia_keys={nvidia_n} "
        f"assignments={labels} "
        f"thinking={settings.thinking_level} "
        f"cot={settings.cot_enabled}/{settings.cot_agent_enabled} "
        f"task_mode={settings.task_mode} "
        f"dialects={dialect_label} "
        f"kernel_mode={settings.kernel_mode} "
        f"refval={settings.refval_enabled}/{settings.refval_cases} "
        f"max_in={settings.max_input_tokens} max_out={settings.resolved_max_output_tokens} "
        f"arch={settings.resolved_cuda_arch} gpu={settings.resolved_gpu_name}"
    )
    print(
        f"candidates={settings.max_candidates} repairs={settings.max_repairs} "
        f"workers={workers} queued={len(jobs)} skipped_done={skipped_done}"
    )
    if not jobs:
        print("nothing to do")
        return 0

    if workers == 1:
        _apply_worker_slot(assignments[0] if assignments else None)
        success, abandoned, failed = run_job_list(
            jobs, data_dir=data_dir, worker_id=0, show_progress=True
        )
    else:
        success, abandoned, failed = run_multiprocess(
            jobs,
            data_dir=data_dir,
            workers=workers,
            log_level=args.log_level,
            assignments=assignments,
        )

    print(
        f"\ndone success={success} abandoned={abandoned} crashed={failed} "
        f"sft={store.sft_path} abandoned_log={store.abandoned_path}"
    )
    return 0 if failed == 0 else 1


def run_job_list(
    jobs: list[Job] | list[tuple[int, str, str]] | list[tuple[int, str]],
    *,
    data_dir: Path,
    worker_id: int = 0,
    show_progress: bool = True,
) -> tuple[int, int, int]:
    """Run the matching LangGraph pipeline on ``jobs`` in this process.

    Args:
        jobs: :class:`Job` records or legacy ``(id, question[, dialect])`` tuples.
        data_dir: Output directory for jsonl / logs.
        worker_id: Index used in log prefixes.
        show_progress: If True, wrap the loop in tqdm.

    Returns:
        ``(success, abandoned, crashed)`` counts.
    """
    init_store(data_dir)
    kernel_app = None
    knowledge_app = None
    kernel_limit = recursion_limit()
    knowledge_limit = 80
    success = 0
    abandoned = 0
    failed = 0
    prefix = f"w{worker_id}"
    iterable: Any = jobs
    if show_progress:
        iterable = tqdm(jobs, desc=f"questions[{prefix}]", file=sys.stderr, unit="q")

    for index, raw_job in enumerate(iterable, start=1):
        job = coerce_job(raw_job)
        qid = job.question_id
        # Preserve the dialect group context even though LangGraph executes
        # one backend job at a time.  The cross-dialect oracle and downstream
        # reports use these fields to distinguish an unavailable backend from
        # a backend that was never requested.
        settings_for_group = get_settings()
        requested_dialects = get_dialect_agent().requested_names(settings_for_group)
        available_dialects: list[str] = []
        for requested in requested_dialects:
            try:
                ok, _reason = get_dialect_agent().spec(requested).available(settings_for_group)
            except Exception:
                ok = False
            if ok:
                available_dialects.append(requested)
        group_metadata = {
            "requested_dialects": requested_dialects,
            "available_dialects": available_dialects,
            "group_id": f"q{qid}:{job.question_id}",
        }
        if job.kind == "knowledge":
            if knowledge_app is None:
                from cuda_sft.knowledge.graph import (
                    build_knowledge_graph,
                    knowledge_recursion_limit,
                )

                knowledge_app = build_knowledge_graph()
                knowledge_limit = knowledge_recursion_limit()
            app = knowledge_app
            limit = knowledge_limit
            init_state: dict[str, Any] = {
                "question_id": qid,
                "question": job.question,
                "kind": "knowledge",
                "topic": job.topic,
                "track": job.track,
                "input_metadata": {**dict(job.extras), **group_metadata},
                "source": job.source,
                "status": "running",
            }
            label = f"kind=knowledge topic={job.topic}"
        else:
            if kernel_app is None:
                kernel_app = build_graph()
            app = kernel_app
            limit = kernel_limit
            init_state = {
                "question_id": qid,
                "question": job.question,
                "dialect": job.track,
                "kind": "kernel",
                "input_metadata": {**dict(job.extras), **group_metadata},
                "source": job.source,
                "status": "running",
            }
            label = f"kind=kernel dialect={job.track}"
        print(
            f"\n===== [{prefix} {index}/{len(jobs)}] question {qid} {label} =====",
            flush=True,
        )
        final = None
        for attempt in range(1, 4):
            try:
                final = app.invoke(init_state, {"recursion_limit": limit})
                break
            except KeyboardInterrupt:
                print("\ninterrupted", file=sys.stderr)
                raise
            except Exception as exc:
                retryable = is_retryable_llm_error(exc)
                if attempt >= 3 or not retryable:
                    log = logger.exception if retryable else logger.error
                    log(
                        "[%s] question %s %s crashed after %s attempts "
                        "(retryable=%s): %s",
                        prefix,
                        qid,
                        label,
                        attempt,
                        retryable,
                        exc,
                    )
                    failed += 1
                    break
                wait = 8 * attempt
                logger.warning(
                    "[%s] question %s %s crashed (attempt %s/3): %s; retrying in %ss",
                    prefix,
                    qid,
                    label,
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


def _apply_worker_slot(slot: WorkerSlot | None) -> None:
    """Pin this process to one provider / NVIDIA key and drop cached clients.

    Extra NVIDIA keys are cleared so the child only uses the assigned key.

    Args:
        slot: Worker assignment; ignored if None or missing a provider name.
    """
    if slot is None or not slot.provider:
        return
    os.environ["LLM_PROVIDER"] = slot.provider
    if slot.provider == "nvidia" and slot.api_key:
        os.environ["NVIDIA_API_KEY"] = slot.api_key
        os.environ["NVIDIA_API_KEY_2"] = ""
        os.environ["NVIDIA_API_KEY_3"] = ""
    elif slot.provider == "openrouter" and slot.api_key:
        os.environ["OPENROUTER_API_KEY"] = slot.api_key
    from cuda_sft.config import get_settings as _gs
    from cuda_sft.llm import reset_llm_client

    _gs.cache_clear()
    reset_llm_client()


def _mp_entry(payload: dict[str, Any]) -> dict[str, Any]:
    """Spawn-worker entry: configure logging then :func:`run_job_list`.

    Args:
        payload: ``worker_id``, ``data_dir``, ``jobs``, ``log_level``,
            plus optional ``provider`` / ``api_key`` / ``label``.
    """
    worker_id = int(payload["worker_id"])
    data_dir = Path(payload["data_dir"])
    jobs: list[tuple[int, str]] = payload["jobs"]
    log_level = str(payload["log_level"])
    provider = str(payload.get("provider") or "").strip().lower()
    label = str(payload.get("label") or provider or "-")
    slot = WorkerSlot(
        provider=provider,
        api_key=str(payload.get("api_key") or ""),
        label=label,
    )
    _apply_worker_slot(slot if provider else None)
    time.sleep(0.6 * worker_id)
    _configure_logging(log_level, data_dir / "run.log", worker_id=worker_id)
    _set_all_print_stream(False)
    if provider:
        from cuda_sft.config import get_settings as _gs2

        cfg = _gs2()
        logger.info(
            "worker %s provider=%s model=%s",
            worker_id,
            label,
            cfg.resolved_model,
        )
    success, abandoned, failed = run_job_list(
        jobs, data_dir=data_dir, worker_id=worker_id, show_progress=True
    )
    return {
        "worker_id": worker_id,
        "provider": label,
        "success": success,
        "abandoned": abandoned,
        "failed": failed,
        "n": len(jobs),
    }


def run_multiprocess(
    jobs: list[Job] | list[tuple[int, str, str]] | list[tuple[int, str]],
    *,
    data_dir: Path,
    workers: int,
    log_level: str,
    assignments: list[WorkerSlot] | None = None,
) -> tuple[int, int, int]:
    """Shard ``jobs`` round-robin across ``workers`` spawn processes.

    Args:
        jobs: Full remaining question list.
        data_dir: Shared output directory (jsonl writes are flocked).
        workers: Process count.
        log_level: Passed to each child.
        assignments: Provider + NVIDIA key per worker (same length as shards).

    Returns:
        Aggregated ``(success, abandoned, crashed)``.
    """
    shards: list[list[Any]] = [[] for _ in range(workers)]
    for i, job in enumerate(jobs):
        shards[i % workers].append(job)
    src = str(PROJECT_ROOT / "src")
    pythonpath = os.environ.get("PYTHONPATH", "")
    if src not in pythonpath.split(os.pathsep):
        os.environ["PYTHONPATH"] = src + (os.pathsep + pythonpath if pythonpath else "")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        payloads = []
        for i, shard in enumerate(shards):
            if not shard:
                continue
            slot = assignments[i] if assignments and i < len(assignments) else None
            payloads.append(
                {
                    "worker_id": i,
                    "data_dir": str(data_dir),
                    "jobs": shard,
                    "log_level": log_level,
                    "provider": slot.provider if slot else "",
                    "api_key": slot.api_key if slot else "",
                    "label": slot.label if slot else "",
                }
            )
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
            f"worker {r['worker_id']} provider={r.get('provider') or '-'}: "
            f"n={r['n']} success={r['success']} "
            f"abandoned={r['abandoned']} crashed={r['failed']}",
            flush=True,
        )
    return success, abandoned, failed


if __name__ == "__main__":
    raise SystemExit(main())
