"""Offline concurrency and supervision checks for the generation scheduler."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from cuda_sft.config import Settings, WorkerSlot
from cuda_sft.runtime import deps, scheduler, trace
from cuda_sft.tasks.kinds import Job
from cuda_sft.testing.fakes import FakeCompiler, FakeLLM

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def graph_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from cuda_sft import config

    env = {
        "DATA_DIR": str(tmp_path / "data"),
        "WORK_DIR": str(tmp_path / "work"),
        "CUDA_HOME": str(tmp_path / "cuda"),
        "CUDA_ARCH": "sm_86",
        "GPU_NAME": "FakeGPU",
        "DIFFICULTY_AWARE": "false",
        "KERNEL_FAST_MODE": "false",
        "MAX_CANDIDATES": "2",
        "MAX_REPAIRS": "0",
        "LLM_CONCURRENCY": "2",
        "ASYNC_LLM_ENABLED": "false",
        "REFVAL_ENABLED": "false",
        "JUDGE_ENABLED": "false",
        "KERNEL_LLM_CRITIC": "off",
        "COT_ENABLED": "false",
        "TRACE_ENABLED": "false",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(config, "detect_cuda_version", lambda *_a, **_k: "12.4")
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def _init(qid: int = 1) -> dict[str, object]:
    return {
        "question_id": qid,
        "question": "Implement float vector addition as a CUDA kernel.",
        "kind": "kernel",
        "dialect": "cuda",
        "source": "test",
        "status": "running",
    }


def test_candidate_generation_overlaps_other_candidate_compile(graph_settings) -> None:
    rendezvous = threading.Barrier(2, timeout=5)
    stages: dict[str, float] = {}
    compiler = FakeCompiler()

    class GatedLLM(FakeLLM):
        def complete(self, role, *, messages, system, meta, token=None):
            if role == "generator" and meta.key() == "c2r0":
                stages["c2_generate_start"] = time.monotonic()
                rendezvous.wait()
                stages["c2_generate_end"] = time.monotonic()
            return super().complete(
                role, messages=messages, system=system, meta=meta, token=token
            )

    llm = GatedLLM(
        script={"generator": {
            "c1r0": {"code": "kernels/add_ok.cu"},
            "c2r0": {"code": "kernels/add_ok.cu"},
        }},
        fixtures_root=FIXTURES,
    )

    def compile_fn(dialect, code, workdir, settings):
        if workdir.parts[-2] == "c1":
            stages["c1_compile_start"] = time.monotonic()
            rendezvous.wait()
            stages["c1_compile_end"] = time.monotonic()
        return compiler(dialect, code, workdir, settings)

    with deps.use(deps.Deps(llm_factory=llm.factory, compile_fn=compile_fn)):
        final = scheduler.solve_job_for_tests(_init())

    assert final["status"] == "abandoned"
    assert final["abandon_reason"] == "all_candidates_failed"
    assert max(stages["c1_compile_start"], stages["c2_generate_start"]) < min(
        stages["c1_compile_end"], stages["c2_generate_end"]
    )
    assert {report["candidate"] for report in final["candidate_reports"]} == {1, 2}
    assert len(compiler.calls) == 2


def test_candidate_threads_preserve_trace_job_key(graph_settings, monkeypatch) -> None:
    from cuda_sft import graph

    class Graph:
        def invoke(self, _state, _config):
            trace.emit("node.start", node="generate")
            return {"candidate_reports": [], "attempts": []}

    monkeypatch.setattr(graph, "build_candidate_graph", lambda: Graph())
    monkeypatch.setattr(graph, "select_best", lambda _state: {"winner_found": False})
    sink = trace.MemorySink()
    trace.set_sink(sink)
    try:
        scheduler.solve_job_for_tests(_init())
        events = sink.of("node.start")
        assert len(events) == 2
        assert {(event["job_key"], event["candidate"]) for event in events} == {
            ("q1:cuda", 1), ("q1:cuda", 2)
        }
    finally:
        trace.set_sink(None)


def test_candidate_exception_preserves_other_winner(graph_settings, monkeypatch) -> None:
    from cuda_sft import graph
    from cuda_sft.core.types import build_snapshot
    from tests.unit.test_snapshot import _state

    class Graph:
        def invoke(self, state, _config):
            if state["candidate_idx"] == 1:
                raise RuntimeError("candidate failed")
            candidate_state = _state()
            candidate_state["candidate_idx"] = 2
            candidate_state["code"] = "winner"
            return {
                **state,
                "candidate_reports": [build_snapshot(candidate_state, banked_reason="final")],
                "attempts": [{"candidate": 2}],
                "metadata": {},
                "quality_status": {},
            }

    monkeypatch.setattr(graph, "build_candidate_graph", lambda: Graph())
    monkeypatch.setattr(graph, "cot", lambda _state: {})
    final = scheduler.solve_job_for_tests(_init())
    assert final["status"] == "success"
    assert final["winner_candidate"] == 2
    assert final["code"] == "winner"
    assert final["candidate_errors"] == [{"candidate": 1, "error_type": "RuntimeError"}]


def test_all_candidate_exceptions_are_attributed(graph_settings, monkeypatch) -> None:
    from cuda_sft import graph

    class Graph:
        def invoke(self, _state, _config):
            raise RuntimeError("candidate failed")

    monkeypatch.setattr(graph, "build_candidate_graph", lambda: Graph())
    final = scheduler.solve_job_for_tests(_init())
    assert final["status"] == "abandoned"
    assert final["abandon_reason"] == "candidate_exception"
    assert [item["candidate"] for item in final["candidate_errors"]] == [1, 2]


def test_supervised_job_writes_trace(graph_settings, monkeypatch, tmp_path: Path) -> None:
    from cuda_sft import config, main

    monkeypatch.setenv("TRACE_ENABLED", "true")
    monkeypatch.setenv("TRACE_DIR", "audit-trace")
    config.get_settings.cache_clear()
    monkeypatch.setattr(scheduler.os, "setsid", lambda: None)
    monkeypatch.setattr(main, "_apply_worker_slot", lambda _slot: None)
    monkeypatch.setattr(main, "_install_cassette", lambda _settings: None)
    monkeypatch.setattr(main, "_configure_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler.limits, "configure", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        scheduler, "_compute_knowledge", lambda _init: {"status": "success", "provenance": {}}
    )
    connection = Mock()
    trace.set_sink(None)
    try:
        scheduler._job_process(
            {
                "slot": ("openrouter", "test-key", "test-slot"),
                "data_dir": str(tmp_path),
                "log_level": "ERROR",
                "slot_index": 0,
                "limits_root": str(tmp_path / "limits"),
                "deadline": None,
                "job": Job(1, "Explain CUDA memory hierarchy.", "knowledge", "knowledge:general"),
            },
            connection,
        )
        path = next((tmp_path / "audit-trace").glob("trace-*.jsonl"))
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert [event["event"] for event in events] == ["job.start", "job.end"]
        assert all(event["job_key"] == "q1:knowledge:general" for event in events)
        assert events[-1]["status"] == "success"
        assert events[-1]["release_tier"] == "strict"
        connection.send.assert_called_once()
    finally:
        trace.set_sink(None)


def test_repair_starts_after_compile_failure_diagnostic(graph_settings, monkeypatch) -> None:
    monkeypatch.setenv("MAX_CANDIDATES", "1")
    monkeypatch.setenv("MAX_REPAIRS", "1")
    from cuda_sft.config import get_settings

    get_settings.cache_clear()
    compiler = FakeCompiler()
    timestamps: dict[str, float] = {}

    class ObservedLLM(FakeLLM):
        def complete(self, role, *, messages, system, meta, token=None):
            if role == "repair.compile":
                timestamps["repair_start"] = time.monotonic()
                assert "expected a" in messages[-1]["content"]
            return super().complete(
                role, messages=messages, system=system, meta=meta, token=token
            )

    llm = ObservedLLM(
        script={
            "generator": {"c1r0": {"code": "kernels/add_syntax_error.cu"}},
            "repair.compile": {"c1r1": {"code": "kernels/add_ok.cu"}},
        },
        fixtures_root=FIXTURES,
    )

    def compile_fn(dialect, code, workdir, settings):
        result = compiler(dialect, code, workdir, settings)
        if workdir.parts[-1] == "r0":
            assert not result.ok
            timestamps["failed_compile_end"] = time.monotonic()
        return result

    with deps.use(deps.Deps(llm_factory=llm.factory, compile_fn=compile_fn)):
        final = scheduler.solve_job_for_tests(_init())

    assert final["status"] == "abandoned"
    assert final["abandon_reason"] == "all_candidates_failed"
    assert timestamps["repair_start"] >= timestamps["failed_compile_end"]
    assert [call.key for call in llm.calls_for("repair.compile")] == ["c1r1"]


def test_supervisor_overlaps_generation_and_compile_across_jobs(
    graph_settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ctx = mp.get_context("fork")
    rendezvous = ctx.Barrier(2, timeout=10)
    events = ctx.Queue()
    monkeypatch.setattr(scheduler.mp, "get_context", lambda _method: ctx)
    monkeypatch.setattr(
        scheduler, "_commit", lambda _job, payload, _data_dir: payload["state"]["status"]
    )
    monkeypatch.setenv("MAX_CANDIDATES", "1")
    from cuda_sft.config import get_settings

    get_settings.cache_clear()

    class CrossJobLLM(FakeLLM):
        def complete(self, role, *, messages, system, meta, token=None):
            if role == "generator" and meta.question_id == 2:
                events.put(("generate_start", time.monotonic()))
                rendezvous.wait()
                events.put(("generate_end", time.monotonic()))
            return super().complete(
                role, messages=messages, system=system, meta=meta, token=token
            )

    llm = CrossJobLLM(
        script={"generator": {"c1r0": {"code": "kernels/add_ok.cu"}}},
        fixtures_root=FIXTURES,
    )
    compiler = FakeCompiler()

    def compile_fn(dialect, code, workdir, settings):
        if "q1" in workdir.parts:
            events.put(("compile_start", time.monotonic()))
            rendezvous.wait()
            events.put(("compile_end", time.monotonic()))
        return compiler(dialect, code, workdir, settings)

    jobs = [
        Job(question_id=qid, question="vector add", kind="kernel", track="cuda")
        for qid in (1, 2)
    ]
    with deps.use(deps.Deps(llm_factory=llm.factory, compile_fn=compile_fn)):
        counts = scheduler.run_supervised(
            jobs,
            data_dir=tmp_path,
            assignments=[WorkerSlot("openrouter", "test-key", "test-slot")],
            log_level="ERROR",
        )

    assert counts == (0, 2, 0)
    stages = dict(events.get(timeout=1) for _ in range(4))
    assert max(stages["compile_start"], stages["generate_start"]) < min(
        stages["compile_end"], stages["generate_end"]
    )


def test_supervisor_kills_timed_out_job_without_late_commit(monkeypatch, tmp_path: Path) -> None:
    ctx = mp.get_context("fork")
    monkeypatch.setattr(scheduler.mp, "get_context", lambda _method: ctx)
    monkeypatch.setattr(
        scheduler,
        "get_settings",
        lambda: Settings(max_inflight_jobs=2, kernel_fast_mode=True, kernel_deadline_sec=1),
    )
    committed = Mock(side_effect=AssertionError("late result committed"))
    monkeypatch.setattr(scheduler, "_commit", committed)
    abandoned: list[dict[str, object]] = []
    store = Mock(write_abandoned=lambda state: abandoned.append(state))
    monkeypatch.setattr(scheduler, "get_store", lambda: store)

    def slow_worker(_payload, connection):
        os.setsid()
        time.sleep(3)
        connection.send({"state": {"status": "success"}, "model": "test"})
        connection.close()

    monkeypatch.setattr(scheduler, "_job_process", slow_worker)
    counts = scheduler.run_supervised(
        [Job(question_id=3, question="vector add", kind="kernel", track="cuda")],
        data_dir=tmp_path,
        assignments=[WorkerSlot("openrouter", "test-key", "test-slot")],
        log_level="ERROR",
    )

    assert counts == (0, 1, 0)
    committed.assert_not_called()
    assert len(abandoned) == 1
    assert abandoned[0]["abandon_reason"] == "kernel_deadline_exceeded"
