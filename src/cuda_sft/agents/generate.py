"""Shared generate-node LLM call for kernel and knowledge graphs.

The async pool supports refval prefetch while compilation runs. Repair calls
follow their diagnostic directly through :func:`complete_chat`.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

from cuda_sft.agents.contracts import GenerateResult
from cuda_sft.config import get_settings
from cuda_sft.llm import LLMClient, LLMCompletion, get_llm_client
from cuda_sft.llm_async import get_async_pool
from cuda_sft.pipeline.common import print_stream_enabled
from cuda_sft.runtime.meta import CallMeta

logger = logging.getLogger(__name__)


def complete_chat(
    *,
    messages: list[dict[str, str]],
    system: str,
    temperature: float,
    request_id: str | None = None,
    llm_options: Mapping[str, Any] | None = None,
    client: LLMClient | None = None,
    log_header: str = "",
    meta: CallMeta | None = None,
) -> GenerateResult:
    """Stream one assistant turn, optionally waiting on a speculative repair.

    Args:
        messages: Chat turns already containing the latest user message.
        system: System prompt for this candidate.
        temperature: Sampling temperature.
        request_id: Speculative pool key; ``None`` skips the cache.
        llm_options: Extra kwargs for ``stream_completion`` (thinking, max tokens).
        client: Injected LLM client (tests); default is the process singleton.
        log_header: Optional ``[Q…]`` prefix copied from the graph node.

    Returns:
        Visible text, reasoning, and whether a speculative hit was used.
    """
    settings = get_settings()
    llm = client or get_llm_client(role=meta.role if meta else "generator")
    options = dict(llm_options or {})
    completion: LLMCompletion | None = None
    used_speculative = False
    header = log_header or "generate"

    if request_id and settings.async_llm_enabled:
        pool = get_async_pool(settings.async_llm_max_workers)
        if pool.is_pending(request_id):
            completion = pool.get(request_id, timeout_sec=settings.llm_timeout_sec)
            if completion is not None:
                used_speculative = True
                logger.info("%s using speculative LLM response (saved ~10-30s)", header)
                print(f"\n{header} using cached response...", flush=True)

    if completion is None:
        logger.info("%s calling model", header)
        print(f"\n{header} generating...", flush=True)
        stream_completion = getattr(llm, "stream_completion", None)
        if callable(stream_completion):
            completion = stream_completion(
                messages=messages,
                system=system,
                temperature=temperature,
                print_stream=print_stream_enabled(),
                meta=meta,
                **options,
            )
        else:
            text = llm.stream_text(
                messages=messages,
                system=system,
                temperature=temperature,
                print_stream=print_stream_enabled(),
                meta=meta,
            )
            completion = LLMCompletion(
                text=text, reasoning="", reasoning_source="empty"
            )

    return GenerateResult(
        text=completion.text,
        reasoning=completion.reasoning if settings.cot_enabled else "",
        reasoning_source=(
            completion.reasoning_source if settings.cot_enabled else "empty"
        ),
        used_speculative=used_speculative,
        origin=getattr(completion, "origin", "unknown"),
    )


def assistant_state_update(
    state: Mapping[str, Any],
    result: GenerateResult,
    *,
    log_header: str = "",
) -> dict[str, Any]:
    """Append the assistant turn and return a LangGraph partial update.

    Args:
        state: Current graph state; must contain ``messages``.
        result: Output of :func:`complete_chat`.
        log_header: Optional ``[Q…]`` prefix for the reasoning-capture log.

    Returns:
        Keys ``raw_response``, ``messages``, ``raw_reasoning``, ``reasoning_source``.
    """
    text = result.text
    if result.reasoning:
        prefix = log_header or "generate"
        logger.info(
            "%s captured reasoning (%s chars, source=%s)",
            prefix,
            len(result.reasoning),
            result.reasoning_source,
        )
    assistant_content = text if text.strip() else "(empty response)"
    messages = list(state.get("messages") or [])
    messages.append({"role": "assistant", "content": assistant_content})
    return {
        "raw_response": text,
        "messages": messages,
        "raw_reasoning": result.reasoning,
        "reasoning_source": result.reasoning_source,
        "origin": result.origin,
    }


def enqueue_speculative_repair(
    *,
    request_id: str,
    messages: list[dict[str, str]],
    system: str,
    temperature: float,
    llm_options: Mapping[str, Any] | None = None,
    client: LLMClient | None = None,
    meta: CallMeta | None = None,
) -> bool:
    """Queue a request for stages which can actually overlap work.

    Args:
        request_id: Pool key, typically ``q{id}_{dialect}_c{c}_r{r}``.
        messages: History including the not-yet-sent repair user turn.
        system: Same system prompt as the live candidate.
        temperature: Same temperature as the live candidate.
        llm_options: Dialect/knowledge sampling overrides.
        client: Injected LLM client (tests).

    Returns:
        True when the request was queued; False on a swallowed enqueue error.
    """
    settings = get_settings()
    if not settings.async_llm_enabled:
        return False
    options = dict(llm_options or {})
    try:
        pool = get_async_pool(settings.async_llm_max_workers)
        llm = client or get_llm_client(role=meta.role if meta else "repair.compile")
        pool.enqueue(
            request_id=request_id,
            llm_client=llm,
            messages=messages,
            system=system,
            temperature=temperature,
            meta=meta,
            **options,
        )
        logger.info("Started speculative repair request: %s", request_id)
        return True
    except Exception:
        logger.exception("Failed to enqueue speculative repair request: %s", request_id)
        return False


def cancel_speculative(request_ids: list[str]) -> None:
    """Drop pending speculative repairs after a compile-passing winner.

    Args:
        request_ids: Keys previously returned by :func:`enqueue_speculative_repair`.
    """
    if not request_ids:
        return
    settings = get_settings()
    if not settings.async_llm_enabled:
        return
    pool = get_async_pool(settings.async_llm_max_workers)
    for request_id in request_ids:
        pool.cancel(request_id)
    logger.info("Cancelled %d speculative requests (winner found)", len(request_ids))
