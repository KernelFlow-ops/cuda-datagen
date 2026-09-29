"""Structured trace events (spec 04_specs/04_trace_schema.md). T0.5 completes TODOs."""

from __future__ import annotations

import atexit
import datetime as _dt
import fcntl
import functools
import json
import os
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

RUN_ID = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(3)
SPEC_VERSION = 1


@dataclass(frozen=True)
class Span:
    id: str
    parent_id: str
    name: str
    started: float


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


class TraceSink:
    """Interface: receives fully-formed event dicts."""

    def write(self, event: dict[str, Any]) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def flush(self) -> None:
        return None


class MemorySink(TraceSink):
    """Collects events in memory (tests)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []

    def write(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.events.append(event)

    def of(self, name: str) -> list[dict[str, Any]]:
        with self._lock:
            return [e for e in self.events if e.get("event") == name]


class FileSink(TraceSink):
    """Buffered JSONL writer: flush every 50 events or 2 seconds, under flock."""

    def __init__(self, directory: Path, *, max_buffer: int = 50, max_age_s: float = 2.0) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._buf: list[str] = []
        self._last = time.monotonic()
        self._max_buffer = max_buffer
        self._max_age = max_age_s
        self._timer: threading.Timer | None = None
        atexit.register(self.flush)

    def _path(self) -> Path:
        day = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d")
        return self.directory / f"trace-{day}.jsonl"

    def write(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            if not self._buf:
                self._timer = threading.Timer(self._max_age, self.flush)
                self._timer.daemon = True
                self._timer.start()
            self._buf.append(line)
            due = (
                len(self._buf) >= self._max_buffer or time.monotonic() - self._last >= self._max_age
            )
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            timer, self._timer = self._timer, None
            if timer is not None:
                timer.cancel()
            lines, self._buf = self._buf, []
            self._last = time.monotonic()
        if not lines:
            return
        with open(self._path(), "a", encoding="utf-8") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.write("\n".join(lines) + "\n")
                fh.flush()
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class _NullSink(TraceSink):
    def write(self, event: dict[str, Any]) -> None:
        return None


_sink_lock = threading.Lock()
_sink: TraceSink = _NullSink()
_local = threading.local()
_usage_lock = threading.Lock()
_job_usage: dict[str, dict[str, int]] = {}


def set_sink(sink: TraceSink | None) -> TraceSink:
    """Install a sink process-wide; returns the previous one."""
    global _sink
    with _usage_lock:
        _job_usage.clear()
    with _sink_lock:
        previous = _sink
        _sink = sink or _NullSink()
    if isinstance(previous, FileSink) and previous is not sink:
        previous.flush()
    return previous


def get_sink() -> TraceSink:
    with _sink_lock:
        return _sink


def ensure_file_sink(directory: Path) -> TraceSink:
    """Use a file sink when tracing is enabled and no sink was explicitly installed."""
    global _sink
    old: FileSink | None = None
    with _sink_lock:
        if isinstance(_sink, FileSink) and _sink.directory != Path(directory):
            old = _sink
            _sink = FileSink(directory)
        elif isinstance(_sink, _NullSink):
            _sink = FileSink(directory)
        active = _sink
    if old is not None:
        old.flush()
    return active


def disable_file_sink() -> None:
    """Turn off a previously installed file sink without replacing test sinks."""
    global _sink
    with _sink_lock:
        old = _sink if isinstance(_sink, FileSink) else None
        if old is not None:
            _sink = _NullSink()
    if old is not None:
        old.flush()


def _ctx() -> dict[str, Any]:
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    return stack[-1] if stack else {}


def push_context(**fields: Any) -> None:
    """Set job/candidate context for the current thread (job_key, candidate, repair)."""
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    merged = {**(stack[-1] if stack else {}), **fields}
    stack.append(merged)


def pop_context() -> None:
    stack = getattr(_local, "stack", None)
    if stack:
        stack.pop()


def current_span() -> Span | None:
    """Return the active node span for this thread, if any."""
    span = _ctx().get("span")
    return span if isinstance(span, Span) else None


def emit(event: str, **fields: Any) -> dict[str, Any]:
    """Emit one event with the common fields filled from thread context."""
    ctx = _ctx()
    record: dict[str, Any] = {
        "v": SPEC_VERSION,
        "ts": _now_iso(),
        "run_id": RUN_ID,
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
        "event": event,
        "job_key": ctx.get("job_key", ""),
        "candidate": int(ctx.get("candidate", 0) or 0),
        "repair": int(ctx.get("repair", 0) or 0),
        "span_id": secrets.token_hex(8),
        "parent_span_id": ctx.get("span_id", ""),
    }
    record.update(fields)
    job_key = str(record.get("job_key") or "")
    if job_key and event == "job.start":
        with _usage_lock:
            _job_usage.pop(job_key, None)
    elif job_key and event == "llm.call":
        with _usage_lock:
            totals = _job_usage.setdefault(
                job_key, {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
            totals["llm_calls"] += 1
            totals["input_tokens"] += int(record.get("input_tokens") or 0)
            totals["output_tokens"] += int(record.get("output_tokens") or 0)
    elif job_key and event == "job.end":
        with _usage_lock:
            totals = _job_usage.pop(
                job_key, {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0}
            )
        for key, value in totals.items():
            record.setdefault(key, value)
        record.setdefault("cost_usd", None)
    get_sink().write(record)
    return record


def _outcome(update: Any) -> dict[str, Any]:
    if not isinstance(update, dict):
        return {"route": None}
    gate = update.get("last_gate") or {}
    return {
        "route": None,
        "gate": gate.get("gate"),
        "passed": gate.get("passed"),
        "owner": gate.get("owner"),
        "error_class": gate.get("error_class"),
    }


def traced(node_name: str) -> Callable[[F], F]:
    """Decorate a LangGraph node: node.start / node.end / node.error events.

    The first positional argument is the state dict; job context is read from it
    (question_id + dialect/track, candidate_idx, repair_idx). S3 adds a CancelToken
    check from ``config["configurable"]["cancel_token"]`` (TODO T3.4).
    """

    def deco(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            state = args[0] if args else {}
            track = state.get("dialect") or state.get("track") or ""
            parent_id = str(_ctx().get("span_id") or "")
            span = Span(secrets.token_hex(8), parent_id, node_name, time.monotonic())
            push_context(
                job_key=f"q{state.get('question_id', 0)}:{track}",
                candidate=state.get("candidate_idx", 0),
                repair=state.get("repair_idx", 0),
                span_id=span.id,
                span=span,
            )
            try:
                emit("node.start", node=node_name, span_id=span.id, parent_span_id=span.parent_id)
                try:
                    update = fn(*args, **kwargs)
                except Exception as exc:
                    emit(
                        "node.error",
                        node=node_name,
                        span_id=span.id,
                        parent_span_id=span.parent_id,
                        error_type=type(exc).__name__,
                        error=str(exc)[:2000],
                    )
                    raise
                emit(
                    "node.end",
                    node=node_name,
                    span_id=span.id,
                    parent_span_id=span.parent_id,
                    elapsed_s=round(time.monotonic() - span.started, 4),
                    outcome=_outcome(update),
                )
                return update
            finally:
                pop_context()

        return wrapper  # type: ignore[return-value]

    return deco
