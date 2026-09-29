"""Process-shared stage limits for one generation run.

Each lease is an advisory file lock. The OS releases it when a timed-out job
process exits, including when that process is killed by the supervisor.
"""

from __future__ import annotations

import fcntl
import hashlib
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from cuda_sft.runtime import trace
from cuda_sft.runtime.meta import CallMeta

_root: Path | None = None
_slot = ""
_llm_count = 2
_compile_count = 1
_deadline: float | None = None
_config_lock = threading.Lock()


def configure(
    root: Path,
    *,
    slot_label: str,
    llm_concurrency: int,
    compile_concurrency: int,
    deadline: float | None = None,
) -> None:
    """Activate limits in a job process; ``root`` is unique to this run."""
    global _root, _slot, _llm_count, _compile_count, _deadline
    with _config_lock:
        _root = Path(root)
        _slot = hashlib.sha256(slot_label.encode("utf-8")).hexdigest()[:16]
        _llm_count = max(1, int(llm_concurrency))
        _compile_count = max(1, int(compile_concurrency))
        _deadline = deadline


@contextmanager
def stage_lock(kind: str, *, slot_label: str | None = None) -> Iterator[None]:
    """Hold one available API or compiler slot until the stage finishes."""
    root = _root
    if root is None:
        yield
        return
    if kind not in {"llm", "compile"}:
        raise ValueError(f"unknown stage kind: {kind}")
    slot = (
        hashlib.sha256(slot_label.encode("utf-8")).hexdigest()[:16]
        if slot_label is not None else _slot
    )
    prefix = f"llm_{slot}" if kind == "llm" else "compile"
    count = _llm_count if kind == "llm" else _compile_count
    root.mkdir(parents=True, exist_ok=True)
    waiting_since = time.monotonic()
    while True:
        if _deadline is not None and time.monotonic() >= _deadline:
            raise TimeoutError(f"job deadline while waiting for {kind} slot")
        for index in range(count):
            handle = (root / f"{prefix}_{index}.lock").open("a+b")
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            try:
                wait_s = time.monotonic() - waiting_since
                if wait_s >= 0.025:
                    trace.emit("stage.wait", stage=kind, elapsed_s=wait_s)
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                handle.close()
            return
        time.sleep(0.05)


class _LimitedClient:
    """Keep every model role under the same per-provider lease budget."""

    def __init__(self, client: Any, slot_label: str | None) -> None:
        self._client = client
        self._slot_label = slot_label

    def stream_completion(self, *, meta: CallMeta | None = None, **kwargs: Any) -> Any:
        with stage_lock("llm", slot_label=self._slot_label):
            return self._client.stream_completion(meta=meta, **kwargs)

    def stream_text(self, *, meta: CallMeta | None = None, **kwargs: Any) -> str:
        with stage_lock("llm", slot_label=self._slot_label):
            return self._client.stream_text(meta=meta, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def limited_client(client: Any, *, settings: Any = None) -> Any:
    """Wrap a client only when a generation supervisor configured limits."""
    if _root is None or isinstance(client, _LimitedClient):
        return client
    route = settings or getattr(client, "settings", None)
    slot_label = (
        f"{route.llm_provider}:{route.resolved_api_key}"
        if route is not None else None
    )
    return _LimitedClient(client, slot_label)
