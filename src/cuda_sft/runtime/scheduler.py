"""Bounded, supervised generation jobs with candidate-level kernel parallelism."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import signal
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cuda_sft.agents.difficulty import plan_topology
from cuda_sft.config import WorkerSlot, get_settings
from cuda_sft.core.types import merge_into_pool
from cuda_sft.runtime import limits, trace
from cuda_sft.store import get_store
from cuda_sft.tasks.kinds import Job, coerce_job, question_hash

logger = logging.getLogger(__name__)


def _compute_kernel(init: dict[str, Any]) -> dict[str, Any]:
    from cuda_sft.graph import (
        build_candidate_graph,
        cot,
        recursion_limit,
        select_best,
    )

    settings = get_settings()
    topology = plan_topology(
        question=str(init["question"]), kind="kernel", settings=settings
    )
    cap = 1 if settings.kernel_fast_mode else topology.max_candidates
    graph = build_candidate_graph()

    def run_candidate(index: int) -> dict[str, Any]:
        trace.push_context(job_key=f"q{init['question_id']}:{init['dialect']}", candidate=index)
        try:
            return dict(
                graph.invoke(
                    {**init, "candidate_idx": index},
                    {"recursion_limit": recursion_limit()},
                )
            )
        finally:
            trace.pop_context()

    states_by_index: dict[int, dict[str, Any]] = {}
    candidate_errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(cap, max(2, settings.llm_concurrency))) as pool:
        futures = {
            pool.submit(run_candidate, index): index for index in range(1, cap + 1)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                states_by_index[index] = future.result()
            except Exception as exc:
                logger.warning("candidate %s failed: %s", index, type(exc).__name__)
                candidate_errors.append({"candidate": index, "error_type": type(exc).__name__})
                trace.emit("candidate.error", candidate=index, error_type=type(exc).__name__)

    states = [states_by_index[index] for index in sorted(states_by_index)]
    candidate_errors.sort(key=lambda item: item["candidate"])
    if not states:
        return {
            **init,
            "candidate_reports": [],
            "candidate_errors": candidate_errors,
            "attempts": [],
            "status": "abandoned",
            "abandon_reason": "candidate_exception",
        }

    reports: list[Any] = []
    attempts: list[Any] = []
    for state in states:
        for report in state.get("candidate_reports") or []:
            reports = merge_into_pool(reports, report)
        attempts.extend(state.get("attempts") or [])
    base = {
        **states[0],
        "candidate_reports": reports,
        "candidate_errors": candidate_errors,
        "attempts": attempts,
    }
    metadata = dict(base.get("metadata") or {})
    pool = dict(metadata.get("candidate_pool") or {})
    pool["errors"] = candidate_errors
    metadata["candidate_pool"] = pool
    base["metadata"] = metadata
    base.update(select_best(base))
    if base.get("winner_found"):
        base.update(cot(base))
        base["status"] = "success"
    else:
        base["status"] = "abandoned"
    return base


def _compute_knowledge(init: dict[str, Any]) -> dict[str, Any]:
    from cuda_sft.knowledge.graph import (
        build_knowledge_compute_graph,
        knowledge_recursion_limit,
    )

    return dict(
        build_knowledge_compute_graph().invoke(
            init, {"recursion_limit": knowledge_recursion_limit()}
        )
    )


def solve_job_for_tests(init: dict[str, Any]) -> dict[str, Any]:
    """Run the compute path with injected test dependencies, without publishing."""
    return _compute_knowledge(init) if init.get("kind") == "knowledge" else _compute_kernel(init)


def _job_process(payload: dict[str, Any], connection: Any) -> None:
    """Compute a job in a killable process; never write formal samples here."""
    started: float | None = None
    trace_active = False
    job_end_emitted = False
    try:
        os.setsid()
        from cuda_sft.main import _apply_worker_slot, _configure_logging, _install_cassette
        from cuda_sft.pipeline.common import set_print_stream
        slot = WorkerSlot(*payload["slot"])
        _apply_worker_slot(slot)
        settings = get_settings()
        data_dir = Path(payload["data_dir"])
        if settings.trace_enabled:
            trace_dir = Path(settings.trace_dir or "trace")
            trace.ensure_file_sink(trace_dir if trace_dir.is_absolute() else data_dir / trace_dir)
        else:
            trace.disable_file_sink()
        _install_cassette(settings)
        _configure_logging(payload["log_level"], data_dir / "run.log", worker_id=payload["slot_index"])
        set_print_stream(False)
        limits.configure(
            Path(payload["limits_root"]),
            slot_label=slot.label,
            llm_concurrency=settings.llm_concurrency,
            compile_concurrency=settings.compile_concurrency,
            deadline=payload["deadline"],
        )
        job: Job = payload["job"]
        qid = job.question_id
        requested: list[str] = []
        available: list[str] = []
        if job.kind == "kernel":
            from cuda_sft.dialects.agent import get_dialect_agent

            agent = get_dialect_agent()
            requested = agent.requested_names(settings)
            for name in requested:
                try:
                    ok, _reason = agent.spec(name).available(settings)
                except Exception:
                    ok = False
                if ok:
                    available.append(name)
        metadata = {
            **dict(job.extras),
            "requested_dialects": requested,
            "available_dialects": available,
            "group_id": f"q{qid}:{question_hash(job.question)[:12]}",
        }
        init: dict[str, Any] = {
            "question_id": qid,
            "question": job.question,
            "kind": job.kind,
            "input_metadata": metadata,
            "source": job.source,
            "status": "running",
        }
        started = time.monotonic()
        topology = plan_topology(
            question=job.question, kind=job.kind, topic=job.topic, settings=settings
        )
        trace.push_context(job_key=f"q{qid}:{job.track}")
        trace_active = True
        trace.emit(
            "job.start",
            question_hash=question_hash(job.question),
            source_line=int(job.extras.get("source_line", qid)),
            track=job.track,
            difficulty=topology.difficulty,
            candidate_cap=topology.max_candidates,
            repair_cap=topology.max_repairs,
        )
        if job.kind == "knowledge":
            init.update(topic=job.topic, track=job.track)
            final = _compute_knowledge(init)
        else:
            init["dialect"] = job.track
            final = _compute_kernel(init)
        provenance = dict(final.get("provenance") or {})
        provenance.setdefault("provider", settings.llm_provider)
        provenance.setdefault("model", settings.resolved_model)
        final["provenance"] = provenance
        release_tier = str(
            (final.get("quality_status") or {}).get("release_tier")
            or final.get("release_tier")
            or ("strict" if job.kind == "knowledge" and final.get("status") == "success" else "quarantine")
        )
        trace.emit(
            "job.end",
            stage="compute",
            status=str(final.get("status") or "crashed"),
            abandon_reason=str(final.get("abandon_reason") or ""),
            selected_candidate=int(final.get("winner_candidate") or 0),
            release_tier=release_tier,
            elapsed_s=time.monotonic() - started,
        )
        job_end_emitted = True
        trace.pop_context()
        trace_active = False
        connection.send({"state": final, "model": provenance["model"]})
    except BaseException as exc:
        if trace_active and started is not None and not job_end_emitted:
            trace.emit(
                "job.end", stage="compute", status="crashed", abandon_reason="pipeline_exception",
                selected_candidate=0, release_tier="quarantine",
                elapsed_s=time.monotonic() - started,
            )
            trace.pop_context()
        with suppress(BrokenPipeError, OSError):
            connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        trace.get_sink().flush()
        connection.close()


@dataclass
class _Running:
    job: Job
    slot_index: int
    process: Any
    connection: Any
    deadline: float | None
    started: float


def _stop_process(process: Any) -> None:
    if not process.is_alive():
        process.join(timeout=0)
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.terminate()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def _commit(job: Job, payload: dict[str, Any], data_dir: Path) -> str:
    from cuda_sft.compile import finalize_question_work
    from cuda_sft.dialects.agent import get_dialect_agent
    from cuda_sft.knowledge.graph import _finalize_answer
    from cuda_sft.store import get_store

    state = dict(payload["state"])
    model = str(payload["model"])
    store = get_store()
    settings = get_settings()
    status = str(state.get("status") or "")
    if status == "success":
        saved = store.write_success(state, model_name=model)
        if not saved:
            status = "abandoned"
            state["abandon_reason"] = "strict_quality_gate"
    elif status == "abandoned":
        store.write_abandoned(state)
    else:
        raise RuntimeError(f"unexpected compute status {status!r}")
    if job.kind == "knowledge":
        _finalize_answer(state, success=status == "success")
    else:
        spec = get_dialect_agent().spec(job.track)
        nest = get_dialect_agent().nest_workdir(job.track, settings)
        finalize_question_work(
            settings,
            job.question_id,
            code=str(state.get("code") or ""),
            success=status == "success",
            dialect=job.track,
            filename=spec.source_filename,
            nest_dialect=nest,
            candidate_idx=int(state.get("candidate_idx") or 1),
            repair_idx=int(state.get("repair_idx") or 0),
        )
    return status


def _timeout_state(job: Job) -> dict[str, Any]:
    state: dict[str, Any] = {
        "question_id": job.question_id,
        "question": job.question,
        "kind": job.kind,
        "status": "abandoned",
        "abandon_reason": "kernel_deadline_exceeded",
    }
    if job.kind == "knowledge":
        state.update(track=job.track, topic=job.topic)
    else:
        state["dialect"] = job.track
    return state


def run_supervised(
    jobs: list[Job] | list[Any],
    *,
    data_dir: Path,
    assignments: list[WorkerSlot],
    log_level: str,
) -> tuple[int, int, int]:
    """Run jobs with two in flight per provider slot by default."""
    settings = get_settings()
    slots = assignments or [
        WorkerSlot(settings.llm_provider, settings.resolved_api_key, settings.llm_provider)
    ]
    pending: list[deque[Job]] = [deque() for _ in slots]
    for index, raw in enumerate(jobs):
        pending[index % len(slots)].append(coerce_job(raw))
    limits_root = data_dir / ".stage_limits" / f"{os.getpid()}_{uuid.uuid4().hex[:12]}"
    ctx = mp.get_context("spawn")
    active: list[_Running] = []
    counts = {"success": 0, "abandoned": 0, "failed": 0}
    if settings.trace_enabled:
        trace_dir = Path(settings.trace_dir or "trace")
        trace.ensure_file_sink(trace_dir if trace_dir.is_absolute() else data_dir / trace_dir)
    try:
        while any(pending) or active:
            for slot_index, queue in enumerate(pending):
                while queue and sum(item.slot_index == slot_index for item in active) < settings.max_inflight_jobs:
                    job = queue.popleft()
                    deadline = (
                        time.monotonic() + settings.kernel_deadline_sec
                        if job.kind == "kernel" and settings.kernel_fast_mode
                        else None
                    )
                    read_end, write_end = ctx.Pipe(duplex=False)
                    slot = slots[slot_index]
                    payload = {
                        "job": job,
                        "slot": tuple(slot),
                        "slot_index": slot_index,
                        "data_dir": str(data_dir),
                        "limits_root": str(limits_root),
                        "log_level": log_level,
                        "deadline": deadline,
                    }
                    proc = ctx.Process(target=_job_process, args=(payload, write_end))
                    proc.start()
                    write_end.close()
                    active.append(_Running(job, slot_index, proc, read_end, deadline, time.monotonic()))
            for item in list(active):
                if item.connection.poll():
                    try:
                        result = item.connection.recv()
                    except EOFError:
                        result = {"error": "job process closed without a result"}
                    if item.deadline is not None and time.monotonic() >= item.deadline:
                        _stop_process(item.process)
                        result = {"timeout": True}
                    else:
                        item.process.join(timeout=1)
                        if item.process.is_alive():
                            _stop_process(item.process)
                    if "state" in result:
                        try:
                            outcome = _commit(item.job, result, data_dir)
                        except Exception:
                            logger.exception("Q%s %s commit failed", item.job.question_id, item.job.track)
                            outcome = "failed"
                    elif result.get("timeout"):
                        get_store().write_abandoned(_timeout_state(item.job))
                        outcome = "abandoned"
                    else:
                        logger.error("Q%s %s failed: %s", item.job.question_id, item.job.track, result.get("error"))
                        outcome = "failed"
                    counts[outcome] += 1
                    final = result.get("state") if isinstance(result.get("state"), dict) else {}
                    quality = final.get("quality_status") if isinstance(final.get("quality_status"), dict) else {}
                    trace.emit(
                        "job.commit", job_key=f"q{item.job.question_id}:{item.job.track}",
                        status=outcome,
                        release_tier=str(quality.get("release_tier") or "strict") if outcome == "success" else "quarantine",
                        elapsed_s=time.monotonic() - item.started,
                    )
                    print(f"[Q{item.job.question_id} {item.job.track}] {outcome}", flush=True)
                    item.connection.close()
                    active.remove(item)
                elif item.deadline is not None and time.monotonic() >= item.deadline:
                    _stop_process(item.process)
                    get_store().write_abandoned(_timeout_state(item.job))
                    counts["abandoned"] += 1
                    trace.emit(
                        "job.commit", job_key=f"q{item.job.question_id}:{item.job.track}",
                        status="abandoned", release_tier="quarantine",
                        elapsed_s=time.monotonic() - item.started,
                    )
                    item.connection.close()
                    active.remove(item)
                elif not item.process.is_alive():
                    item.process.join(timeout=0)
                    logger.error("Q%s %s process exited without result (%s)", item.job.question_id, item.job.track, item.process.exitcode)
                    counts["failed"] += 1
                    trace.emit(
                        "job.commit", job_key=f"q{item.job.question_id}:{item.job.track}",
                        status="failed", release_tier="quarantine",
                        elapsed_s=time.monotonic() - item.started,
                    )
                    item.connection.close()
                    active.remove(item)
            if active:
                time.sleep(0.05)
    finally:
        for item in active:
            _stop_process(item.process)
            item.connection.close()
        trace.get_sink().flush()
    return counts["success"], counts["abandoned"], counts["failed"]
