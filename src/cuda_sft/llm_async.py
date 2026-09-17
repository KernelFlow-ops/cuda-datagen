"""Asynchronous LLM call pooling for speculative generation during compile."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

from cuda_sft.llm import LLMCompletion

logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """A speculative LLM request in flight.

    Attributes:
        request_id: Unique identifier (e.g., "c1_r2" for candidate 1, repair 2).
        future: Future that will hold the LLM completion (text + reasoning).
        started_at: Timestamp when the request was initiated.
    """

    request_id: str
    future: Future[LLMCompletion]
    started_at: float


class AsyncLLMPool:
    """Thread-based async pool for speculative LLM calls.

    Since LangGraph nodes run synchronously, we use a ThreadPoolExecutor
    to launch LLM calls in the background. Compile nodes can enqueue
    speculative repair requests, and repair nodes can check if the response
    is ready.
    """

    def __init__(self, max_workers: int = 2) -> None:
        """Initialize the pool.

        Args:
            max_workers: Maximum number of concurrent LLM calls.
        """
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="llm_async",
        )
        self.pending: dict[str, PendingRequest] = {}
        self._lock = threading.Lock()

    def enqueue(
        self,
        request_id: str,
        llm_client: Any,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        thinking_level: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_max_tokens: int | None = None,
    ) -> None:
        """Start a speculative LLM call in the background.

        Args:
            request_id: Unique identifier for this request.
            llm_client: LLM client with stream_text method.
            messages: Chat history.
            system: System prompt.
            temperature: Sampling temperature.
            thinking_level: Optional per-call thinking override.
            max_output_tokens: Optional per-call max_tokens override.
            reasoning_max_tokens: Optional reasoning token cap.
        """
        with self._lock:
            if request_id in self.pending:
                logger.warning("Request %s already pending; skipping duplicate", request_id)
                return

        def _call() -> LLMCompletion:
            """Wrapper for executor."""
            try:
                logger.info("Async LLM call started: %s", request_id)
                stream_completion = getattr(llm_client, "stream_completion", None)
                if callable(stream_completion):
                    completion = stream_completion(
                        messages=messages,
                        system=system,
                        temperature=temperature,
                        print_stream=False,
                        thinking_level=thinking_level,
                        max_output_tokens=max_output_tokens,
                        reasoning_max_tokens=reasoning_max_tokens,
                    )
                else:
                    text = llm_client.stream_text(
                        messages=messages,
                        system=system,
                        temperature=temperature,
                        print_stream=False,
                    )
                    completion = LLMCompletion(
                        text=text, reasoning="", reasoning_source="empty"
                    )
                logger.info(
                    "Async LLM call completed: %s (%d chars, reasoning=%d)",
                    request_id,
                    len(completion.text or ""),
                    len(completion.reasoning or ""),
                )
                return completion
            except Exception:
                logger.exception("Async LLM call failed: %s", request_id)
                raise

        future = self.executor.submit(_call)
        req = PendingRequest(
            request_id=request_id,
            future=future,
            started_at=time.time(),
        )

        with self._lock:
            self.pending[request_id] = req

    def is_pending(self, request_id: str) -> bool:
        """Return True if ``request_id`` is still tracked (running or finished)."""
        with self._lock:
            return request_id in self.pending

    def try_get(self, request_id: str, timeout_sec: float = 0) -> LLMCompletion | None:
        """Attempt to retrieve a completed LLM response.

        A timeout leaves the request in the pool so the caller can wait
        again instead of launching a duplicate LLM call. Python 3.10's
        ``concurrent.futures.TimeoutError`` is not the builtin
        ``TimeoutError``, so both are treated as "still running".

        Args:
            request_id: Unique identifier.
            timeout_sec: How long to wait. 0 = non-blocking check.

        Returns:
            LLM completion if ready, else None.
        """
        with self._lock:
            req = self.pending.get(request_id)

        if req is None:
            return None

        try:
            wait = timeout_sec if timeout_sec > 0 else 0
            if wait <= 0 and not req.future.done():
                return None
            text = req.future.result(timeout=wait)
        except (TimeoutError, FuturesTimeoutError):
            return None
        except Exception as exc:
            if not req.future.done():
                return None
            logger.warning("Failed to retrieve async result for %s: %s", request_id, exc)
            with self._lock:
                self.pending.pop(request_id, None)
            return None

        with self._lock:
            self.pending.pop(request_id, None)
        return text

    def cancel(self, request_id: str) -> None:
        """Cancel a pending request (e.g., when another candidate wins).

        Args:
            request_id: Request to cancel.
        """
        with self._lock:
            req = self.pending.pop(request_id, None)

        if req is not None:
            req.future.cancel()
            logger.info("Cancelled async request: %s", request_id)

    def cancel_all(self) -> None:
        """Cancel all pending requests."""
        with self._lock:
            ids = list(self.pending.keys())

        for request_id in ids:
            self.cancel(request_id)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for all pending requests to complete.
        """
        self.executor.shutdown(wait=wait)


# Global pool instance (one per worker process)
_pool: AsyncLLMPool | None = None
_pool_lock = threading.Lock()


def get_async_pool(max_workers: int = 2) -> AsyncLLMPool:
    """Get or create the global async LLM pool.

    Args:
        max_workers: Maximum concurrent LLM calls.

    Returns:
        Singleton pool instance.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = AsyncLLMPool(max_workers=max_workers)
    return _pool


def reset_async_pool() -> None:
    """Reset the global pool (used in tests or multiprocess forks)."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.shutdown(wait=False)
            _pool = None
