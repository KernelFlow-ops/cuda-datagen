"""LangGraph helpers shared by the kernel and knowledge pipelines.

Print-stream state lives here (not as a per-graph global) so ``--quiet``
and ``workers > 1`` toggle both graphs with one call. Import
:func:`print_stream_enabled` instead of copying a ``PRINT_STREAM`` name:
a ``from … import PRINT_STREAM`` binding would not see later updates.
"""

from __future__ import annotations

from cuda_sft.llm import is_retryable_llm_error

_PRINT_STREAM = True


def set_print_stream(enabled: bool) -> None:
    """Enable or disable printing streamed model tokens to stdout.

    Args:
        enabled: False under ``--quiet`` or when ``workers > 1``.
    """
    global _PRINT_STREAM
    _PRINT_STREAM = bool(enabled)


def print_stream_enabled() -> bool:
    """Return whether generate nodes should echo streamed tokens."""
    return _PRINT_STREAM


def retry_policy():
    """RetryPolicy for transient LLM failures on the generate node.

    Returns:
        A LangGraph ``RetryPolicy`` (imported lazily so tests can load
        this module without langgraph installed).
    """
    from langgraph.types import RetryPolicy

    return RetryPolicy(
        max_attempts=5,
        initial_interval=4.0,
        backoff_factor=2.0,
        max_interval=60.0,
        retry_on=is_retryable_llm_error,
    )


def graph_recursion_limit(
    *,
    max_candidates: int,
    max_repairs: int,
    extra_per_candidate: int = 5,
) -> int:
    """LangGraph superstep cap covering candidates × (gen + repairs + extras).

    Args:
        max_candidates: 1-based candidate budget.
        max_repairs: Repair rounds per candidate.
        extra_per_candidate: Nodes after the generate/extract/gate loop
            (judge, cot, save, …). Kernel uses 5; knowledge uses 6.

    Returns:
        At least 80, matching the historical kernel/knowledge formulas.
    """
    per_candidate = (int(max_repairs) + 1) * 3 + int(max_repairs) + int(extra_per_candidate)
    return max(80, 10 + int(max_candidates) * per_candidate)
