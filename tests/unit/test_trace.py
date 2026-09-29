"""Structured node events and concurrent JSONL writes."""

import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from cuda_sft.runtime import trace


def _run_fake_job(monkeypatch, data_dir) -> None:
    from cuda_sft import main
    from cuda_sft.tasks.kinds import Job

    class Dialects:
        def requested_names(self, _settings):
            return []

    class Graph:
        def invoke(self, _state, _config):
            return {"status": "success"}

    monkeypatch.setattr(main, "get_dialect_agent", lambda: Dialects())
    monkeypatch.setattr(main, "build_graph", lambda: Graph())
    main.run_job_list([Job(1, "add", "kernel", "cuda")], data_dir=data_dir, show_progress=False)


def _read_job_events(directory) -> list[dict]:
    path = next(directory.glob("trace-*.jsonl"))
    return [
        event
        for line in path.read_text(encoding="utf-8").splitlines()
        if (event := json.loads(line)).get("event") in {"job.start", "job.end"}
    ]


def test_trace_directory_resolves_below_data_dir(monkeypatch, tmp_path) -> None:
    from cuda_sft import config, main

    class Dialects:
        def requested_names(self, _settings):
            return []

    monkeypatch.setattr(main, "get_dialect_agent", lambda: Dialects())
    monkeypatch.setenv("TRACE_ENABLED", "true")
    monkeypatch.setenv("TRACE_DIR", "session-trace")
    config.get_settings.cache_clear()
    trace.set_sink(None)
    main.run_job_list([], data_dir=tmp_path / "enabled", show_progress=False)
    assert (tmp_path / "enabled" / "session-trace").is_dir()
    assert not (tmp_path / "session-trace").exists()


def test_trace_disabled_does_not_create_file(monkeypatch, tmp_path) -> None:
    from cuda_sft import config, main

    class Dialects:
        def requested_names(self, _settings):
            return []

    monkeypatch.setattr(main, "get_dialect_agent", lambda: Dialects())
    monkeypatch.setenv("TRACE_ENABLED", "false")
    config.get_settings.cache_clear()
    trace.set_sink(None)
    main.run_job_list([], data_dir=tmp_path / "disabled", show_progress=False)
    assert not (tmp_path / "disabled" / "trace").exists()


def test_disabling_trace_stops_previous_file_sink(monkeypatch, tmp_path) -> None:
    from cuda_sft import config

    trace.set_sink(None)
    monkeypatch.setenv("TRACE_ENABLED", "true")
    config.get_settings.cache_clear()
    _run_fake_job(monkeypatch, tmp_path / "first")

    monkeypatch.setenv("TRACE_ENABLED", "false")
    config.get_settings.cache_clear()
    _run_fake_job(monkeypatch, tmp_path / "second")
    assert len(_read_job_events(tmp_path / "first" / "trace")) == 2
    assert not (tmp_path / "second" / "trace").exists()


def test_enabled_trace_switches_to_new_data_dir(monkeypatch, tmp_path) -> None:
    from cuda_sft import config

    trace.set_sink(None)
    monkeypatch.setenv("TRACE_ENABLED", "true")
    config.get_settings.cache_clear()
    _run_fake_job(monkeypatch, tmp_path / "first")
    _run_fake_job(monkeypatch, tmp_path / "second")
    trace.get_sink().flush()
    assert len(_read_job_events(tmp_path / "first" / "trace")) == 2
    assert len(_read_job_events(tmp_path / "second" / "trace")) == 2


def test_traced_node_emits_paired_events() -> None:
    sink = trace.MemorySink()
    trace.set_sink(sink)
    wrapped = trace.traced("extract")(lambda state: {"code": state["code"]})
    assert wrapped({"question_id": 3, "dialect": "cuda", "code": "x"}) == {"code": "x"}
    assert [event["event"] for event in sink.events] == ["node.start", "node.end"]
    assert {event["job_key"] for event in sink.events} == {"q3:cuda"}
    assert sink.events[0]["span_id"] == sink.events[1]["span_id"]
    assert trace.current_span() is None


def test_sink_change_and_new_job_clear_old_usage() -> None:
    first = trace.MemorySink()
    trace.set_sink(first)
    trace.emit("llm.call", job_key="q1:cuda", input_tokens=11, output_tokens=4)
    second = trace.MemorySink()
    trace.set_sink(second)
    trace.emit("job.end", job_key="q1:cuda", status="success")
    assert second.of("job.end")[0]["input_tokens"] == 0

    trace.emit("llm.call", job_key="q1:cuda", input_tokens=7, output_tokens=2)
    trace.emit("job.start", job_key="q1:cuda")
    trace.emit("job.end", job_key="q1:cuda", status="success")
    assert second.of("job.end")[-1]["llm_calls"] == 0


def test_traced_node_emits_error_and_reraises() -> None:
    sink = trace.MemorySink()
    trace.set_sink(sink)

    @trace.traced("broken")
    def broken(_state):
        raise ValueError("bad")

    with pytest.raises(ValueError, match="bad"):
        broken({"question_id": 1})
    assert [event["event"] for event in sink.events] == ["node.start", "node.error"]


def test_file_sink_writes_all_concurrent_events(tmp_path) -> None:
    sink = trace.FileSink(tmp_path, max_buffer=1000)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: sink.write({"event": "unit", "i": i}), range(1000)))
    sink.flush()
    lines = next(tmp_path.glob("trace-*.jsonl")).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1000
    assert {json.loads(line)["i"] for line in lines} == set(range(1000))


def test_file_sink_flushes_after_age_without_more_writes(tmp_path) -> None:
    sink = trace.FileSink(tmp_path, max_buffer=50, max_age_s=0.05)
    sink.write({"event": "one"})
    deadline = time.monotonic() + 1
    while not list(tmp_path.glob("trace-*.jsonl")) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list(tmp_path.glob("trace-*.jsonl"))
