"""Cassette keys reject prompt drift while preserving recorded output."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from cuda_sft.llm import LLMCompletion
from cuda_sft.observability.report import compute
from cuda_sft.runtime import trace
from cuda_sft.runtime.meta import CallMeta
from cuda_sft.testing.cassette import CassetteMiss, RecordingClient, ReplayClient


class Client:
    def stream_completion(self, **_kwargs):
        return LLMCompletion(
            "code", "thinking", "api", {"input_tokens": 4, "output_tokens": 2}, True
        )


def test_record_and_replay_exact_request(tmp_path) -> None:
    meta = CallMeta("generator", "q1:cuda", 1, "cuda", candidate=1)
    request = {
        "messages": [{"role": "user", "content": "add"}],
        "system": "kernel",
        "temperature": 0.2,
        "meta": meta,
    }
    recorded = RecordingClient(Client(), tmp_path).stream_completion(**request)
    assert ReplayClient(tmp_path).stream_completion(**request) == replace(recorded, origin="replay")
    with pytest.raises(CassetteMiss, match="closest") as exc:
        ReplayClient(tmp_path).stream_completion(
            **{**request, "messages": [{"role": "user", "content": "add!"}]}
        )
    assert "generator" in str(exc.value)
    assert "request_hash" in str(exc.value)


def test_concurrent_identical_records_are_idempotent(tmp_path) -> None:
    client = RecordingClient(Client(), tmp_path)
    request = {
        "messages": [{"role": "user", "content": "same"}],
        "system": "kernel",
        "temperature": 0.2,
    }
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: client.stream_completion(**request), range(24)))
    assert all(result == results[0] for result in results)
    assert len((tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert ReplayClient(tmp_path).stream_completion(**request) == replace(results[0], origin="replay")
    run_meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert set(run_meta) == {"model", "date", "git_commit"}


def test_conflicting_record_does_not_replace_first_response(tmp_path) -> None:
    class ChangedClient:
        def stream_completion(self, **_kwargs):
            return LLMCompletion("different")

    request = {"messages": [], "system": "kernel", "temperature": 0.2}
    first = RecordingClient(Client(), tmp_path).stream_completion(**request)
    with pytest.raises(ValueError, match="conflicting cassette response"):
        RecordingClient(ChangedClient(), tmp_path).stream_completion(**request)
    assert ReplayClient(tmp_path).stream_completion(**request) == replace(first, origin="replay")
    assert len((tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_replay_emits_usage_and_job_totals(tmp_path) -> None:
    meta = CallMeta("generator", "q99:cuda", 99, "cuda", candidate=2, purpose="first")
    request = {"messages": [], "system": "kernel", "temperature": 0.2, "meta": meta}
    RecordingClient(Client(), tmp_path).stream_completion(**request)
    sink = trace.MemorySink()
    previous = trace.set_sink(sink)
    try:
        replayed = ReplayClient(tmp_path).stream_completion(**request)
        trace.emit("job.end", job_key=meta.job_key, status="success")
    finally:
        trace.set_sink(previous)
    call = sink.of("llm.call")[0]
    assert call["role"] == "generator" and call["purpose"] == "first"
    assert call["candidate"] == 2 and call["ok"]
    assert call["input_tokens"] == replayed.usage["input_tokens"]
    assert call["output_tokens"] == replayed.usage["output_tokens"]
    assert call["tokens_estimated"]
    job = sink.of("job.end")[0]
    assert (job["llm_calls"], job["input_tokens"], job["output_tokens"]) == (1, 4, 2)
    metrics = compute(sink.events)
    assert metrics["llm_calls_per_success"] == 1
    assert metrics["input_tokens_per_success"] == 4
    assert metrics["output_tokens_per_success"] == 2


def test_replay_never_inherits_recorded_live_origin(tmp_path) -> None:
    class LiveClient:
        def stream_completion(self, **_kwargs):
            return LLMCompletion("code", origin="live_api")

    request = {"messages": [], "system": "kernel", "temperature": 0.2}
    recorded = RecordingClient(LiveClient(), tmp_path).stream_completion(**request)
    replayed = ReplayClient(tmp_path).stream_completion(**request)
    assert recorded.origin == "live_api"
    assert replayed.origin == "replay"


def test_replay_miss_emits_failed_call(tmp_path) -> None:
    meta = CallMeta("generator", "q98:cuda", 98, "cuda")
    sink = trace.MemorySink()
    previous = trace.set_sink(sink)
    try:
        with pytest.raises(CassetteMiss):
            ReplayClient(tmp_path).stream_completion(
                messages=[], system="kernel", temperature=0, meta=meta
            )
        trace.emit("job.end", job_key=meta.job_key, status="crashed")
    finally:
        trace.set_sink(previous)
    call = sink.of("llm.call")[0]
    assert not call["ok"] and call["error_type"] == "CassetteMiss"
    assert call["input_tokens"] > 0 and call["output_tokens"] == 0


def test_existing_response_repairs_missing_index(tmp_path) -> None:
    request = {"messages": [], "system": "kernel", "temperature": 0.2}
    client = RecordingClient(Client(), tmp_path)
    client.stream_completion(**request)
    (tmp_path / "index.jsonl").unlink()
    client.stream_completion(**request)
    rows = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["role"] == "generator"
