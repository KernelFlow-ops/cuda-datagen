"""Process-wide dependency injection for tests (T0.2).

Production code asks :func:`current` for overrides at a handful of seams:

* ``llm.get_llm_client(settings, role=...)``   -> ``llm_factory(role)``
* ``graph.compile_node``                        -> ``compile_fn(dialect, code, workdir, settings)``
* ``graph.validate``                            -> ``refval_fn(**same kwargs as run_refval)``
* every ``time.sleep`` in graph/main code       -> :func:`sleep`

The registry is a process global guarded by a lock (NOT a ContextVar) so that
worker threads spawned by the scheduler see the same overrides.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Deps:
    """Injectable collaborators; ``None`` means "use the real implementation"."""

    llm_factory: Callable[[str], Any] | None = None
    compile_fn: Callable[[str, str, Path, Any], Any] | None = None
    refval_fn: Callable[..., Any] | None = None
    sleep_fn: Callable[[float], None] | None = None


_EMPTY = Deps()
_lock = threading.Lock()
_current: Deps = _EMPTY


def current() -> Deps:
    """Return the active overrides (never ``None``)."""
    with _lock:
        return _current


def install(deps: Deps) -> Deps:
    """Install ``deps`` process-wide and return the previous value."""
    global _current
    with _lock:
        previous = _current
        _current = deps
        return previous


def reset() -> None:
    """Drop all overrides."""
    install(_EMPTY)


@contextmanager
def use(deps: Deps) -> Iterator[Deps]:
    """Temporarily install ``deps``; restores the previous value on exit."""
    previous = install(deps)
    try:
        yield deps
    finally:
        install(previous)


def sleep(seconds: float) -> None:
    """Sleep via the injected ``sleep_fn`` (tests record instead of waiting)."""
    fn = current().sleep_fn
    if fn is not None:
        fn(float(seconds))
        return
    if seconds > 0:
        time.sleep(seconds)
