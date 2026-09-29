"""Stable question identity and grouped work."""

import multiprocessing as mp
from pathlib import Path

from cuda_sft import main
from cuda_sft.agents.difficulty import plan_topology
from cuda_sft.config import get_settings
from cuda_sft.graph import _task_contract
from cuda_sft.runtime import trace
from cuda_sft.tasks.kinds import Job, normalize_question, question_hash


def test_question_hash_normalizes_unicode_and_whitespace() -> None:
    assert normalize_question("  A\n B\tC  ") == "A B C"
    assert question_hash("e\u0301  add") == question_hash("\u00e9 add")
    assert question_hash("add") != question_hash("sub")


def test_task_contract_uses_normalized_question_hash() -> None:
    question = "  e\u0301  add "
    contract = _task_contract({"question": question}, dialect="cuda", settings=get_settings())
    assert contract["question_hash"] == question_hash(question)


def test_group_id_distinguishes_questions_with_same_source_line(
    monkeypatch, tmp_path: Path
) -> None:
    states = []

    class Graph:
        def invoke(self, state, _config):
            states.append(state)
            return {"status": "success"}

    class Dialects:
        def requested_names(self, _settings):
            return []

    monkeypatch.setattr(main, "build_graph", lambda: Graph())
    monkeypatch.setattr(main, "get_dialect_agent", lambda: Dialects())
    sink = trace.MemorySink()
    trace.set_sink(sink)
    jobs = [
        Job(1, "add", "kernel", "cuda", extras={"source_line": 89}),
        Job(1, "subtract", "kernel", "cuda"),
    ]
    assert main.run_job_list(jobs, data_dir=tmp_path, show_progress=False) == (2, 0, 0)
    groups = [state["input_metadata"]["group_id"] for state in states]
    assert groups == [f"q1:{question_hash(job.question)[:12]}" for job in jobs]
    assert groups[0] != groups[1]
    starts = sink.of("job.start")
    assert [start["source_line"] for start in starts] == [89, 1]
    for start, job in zip(starts, jobs, strict=True):
        topology = plan_topology(question=job.question, kind=job.kind, settings=get_settings())
        assert start["question_hash"] == question_hash(job.question)
        assert start["difficulty"] == topology.difficulty
        assert start["candidate_cap"] == topology.max_candidates
        assert start["repair_cap"] == topology.max_repairs


def test_worker_payload_sets_actual_parallelism(monkeypatch, tmp_path: Path) -> None:
    observed = []
    monkeypatch.setattr(main, "_apply_worker_slot", lambda _slot: None)
    monkeypatch.setattr(main, "_configure_logging", lambda *_a, **_kw: None)
    monkeypatch.setattr(main, "_set_all_print_stream", lambda _enabled: None)
    monkeypatch.setattr(main.deps, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        main,
        "run_job_list",
        lambda *_a, **_kw: observed.append(get_settings().workers) or (0, 0, 0),
    )
    main._mp_entry(
        {
            "worker_id": 0,
            "data_dir": str(tmp_path),
            "jobs": [],
            "log_level": "INFO",
            "workers_total": 4,
        }
    )
    assert observed == [4]


def test_child_process_sees_cli_workers(tmp_path: Path) -> None:
    payload = {
        "worker_id": 0,
        "data_dir": str(tmp_path),
        "jobs": [],
        "log_level": "ERROR",
        "workers_total": 3,
    }
    with mp.get_context("spawn").Pool(1) as pool:
        result = pool.apply(main._mp_entry, (payload,))
    assert result["workers_total"] == 3
