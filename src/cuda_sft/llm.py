"""Streaming LLM clients: OpenRouter (Anthropic Messages) and NVIDIA NIM (OpenAI)."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
import threading
import time
from collections.abc import Iterable
from concurrent.futures import CancelledError as FutureCancelledError
from dataclasses import dataclass, replace
from typing import Any, Protocol

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:  # OpenAI/NVIDIA-only installs must still import this module.
    anthropic = None  # type: ignore[assignment]
    Anthropic = None  # type: ignore[assignment,misc]

from cuda_sft.config import Settings, get_settings
from cuda_sft.parse import extract_thinking, strip_thinking
from cuda_sft.runtime import deps, trace
from cuda_sft.runtime.meta import CallMeta

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
UPSTREAM_RETRY_HINTS = (
    "overloaded",
    "unavailable",
    "temporarily",
    "upstream",
    "rate limit",
    "timeout",
    "try again",
    "capacity",
    "incomplete chunked",
    "peer closed",
    "connection reset",
    "broken pipe",
)
_AFFORD_TOKENS_RE = re.compile(r"can only afford (\d+)", re.IGNORECASE)
# OpenRouter reserves max_tokens against remaining credits; 100k often 402s.
OPENROUTER_MAX_TOKENS_CAP = 32768
OPENROUTER_MIN_TOKENS = 256
_nvidia_stream_usage_supported = True
_nvidia_stream_usage_lock = threading.Lock()


class LLMError(Exception):
    """Retryable LLM / transport failure."""


@dataclass(frozen=True)
class LLMCompletion:
    """One streamed completion, with visible text split from reasoning.

    Attributes:
        text: Visible assistant text (think tags stripped when they were extracted).
        reasoning: Concatenated chain-of-thought / thinking.
        reasoning_source: How reasoning was obtained.
        origin: Whether the response came from a live provider SDK call.
    """

    text: str
    reasoning: str = ""
    reasoning_source: str = "empty"
    usage: dict[str, int] | None = None
    tokens_estimated: bool = False
    origin: str = "unknown"


class LLMClient(Protocol):
    """Minimal interface used by the generate node."""

    def stream_completion(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        print_stream: bool = True,
        meta: CallMeta | None = None,
    ) -> LLMCompletion:
        """Stream a completion and return visible text plus reasoning."""
        ...

    def stream_text(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        print_stream: bool = True,
        meta: CallMeta | None = None,
    ) -> str:
        """Stream a completion and return visible assistant text (no thinking).

        Args:
            messages: OpenAI/Anthropic-style chat turns (user/assistant).
            system: System prompt (Anthropic ``system=`` or OpenAI system message).
            temperature: Sampling temperature.
            print_stream: If True, print content tokens to stdout.
        """
        ...


def _error_text(exc: BaseException) -> str:
    """Flatten exception + optional ``body``/``message`` for retry matching."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    message = getattr(exc, "message", None)
    if message:
        parts.append(str(message))
    return " ".join(parts).lower()


def _openai_retryable_types() -> tuple[type[BaseException], ...]:
    """Return OpenAI transport exception classes if the package is installed."""
    try:
        import openai
    except ImportError:
        return ()
    types: list[type[BaseException]] = []
    for name in (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    ):
        cls = getattr(openai, name, None)
        if isinstance(cls, type):
            types.append(cls)
    return tuple(types)


def _anthropic_error_types(*names: str) -> tuple[type[BaseException], ...]:
    """Return Anthropic SDK exception classes when the optional SDK is installed."""
    if anthropic is None:
        return ()
    types: list[type[BaseException]] = []
    for name in names:
        cls = getattr(anthropic, name, None)
        if isinstance(cls, type):
            types.append(cls)
    return tuple(types)


def affordable_max_tokens(exc: BaseException, current: int) -> int | None:
    """Return a smaller ``max_tokens`` when OpenRouter 402 blames the budget.

    OpenRouter rejects the request if ``max_tokens`` exceeds remaining credits,
    even when the actual completion would be much shorter.

    Args:
        exc: HTTP error from the Anthropic/OpenRouter SDK.
        current: ``max_tokens`` used on the failed attempt.

    Returns:
        Next budget to try, or ``None`` if this is not a credit/max_tokens 402
        or the budget cannot be reduced further.
    """
    status = getattr(exc, "status_code", None)
    text = _error_text(exc)
    billing = (
        status == 402
        or "billing_error" in text
        or "can only afford" in text
        or ("payment_required" in text and "credits" in text)
    )
    if not billing:
        return None
    match = _AFFORD_TOKENS_RE.search(text)
    if match:
        nxt = min(int(current) - 1, int(match.group(1)) - 1)
        return nxt if nxt >= OPENROUTER_MIN_TOKENS else None
    nxt = min(int(current) // 2, OPENROUTER_MAX_TOKENS_CAP)
    return nxt if nxt >= OPENROUTER_MIN_TOKENS and nxt < int(current) else None


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Return True for rate limits, 5xx, overload, and connection timeouts.

    Used by LangGraph ``RetryPolicy`` on the generate node.
    """
    if isinstance(exc, LLMError):
        return True
    status = getattr(exc, "status_code", None)
    if status in {401, 403}:
        return False
    if status in RETRYABLE_STATUS:
        return True
    if _anthropic_error_types(
        "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"
    ) and isinstance(
        exc,
        _anthropic_error_types(
            "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"
        ),
    ):
        return True
    if _anthropic_error_types("APIStatusError") and isinstance(
        exc, _anthropic_error_types("APIStatusError")
    ):
        status = getattr(exc, "status_code", None)
        if status in RETRYABLE_STATUS:
            return True
        return any(hint in _error_text(exc) for hint in UPSTREAM_RETRY_HINTS)
    openai_types = _openai_retryable_types()
    if openai_types and isinstance(exc, openai_types):
        return True
    try:
        import openai
    except ImportError:
        openai = None
    if openai is not None and isinstance(exc, openai.APIStatusError):
        status = getattr(exc, "status_code", None)
        if status in RETRYABLE_STATUS:
            return True
        return any(hint in _error_text(exc) for hint in UPSTREAM_RETRY_HINTS)
    return any(hint in _error_text(exc) for hint in UPSTREAM_RETRY_HINTS)


def _block_type(block: Any) -> str:
    """Return a content-block type string from an object or dict."""
    if isinstance(block, dict):
        return str(block.get("type") or "")
    return str(getattr(block, "type", None) or "")


def _block_field(block: Any, *names: str) -> str:
    """Read the first non-empty string field from a content block."""
    for name in names:
        if isinstance(block, dict):
            value = block.get(name)
        else:
            value = getattr(block, name, None)
        if value:
            return str(value)
    return ""


def _text_from_content_blocks(content: Iterable[Any]) -> str:
    """Join Anthropic ``type=text`` content blocks into one string."""
    parts: list[str] = []
    for block in content:
        if _block_type(block) == "text":
            parts.append(_block_field(block, "text"))
    return "".join(parts)


def _thinking_from_content_blocks(content: Iterable[Any]) -> str:
    """Join Anthropic thinking / reasoning content blocks."""
    parts: list[str] = []
    for block in content:
        block_type = _block_type(block)
        if block_type in {"thinking", "redacted_thinking"}:
            parts.append(_block_field(block, "thinking", "text"))
        elif block_type in {"reasoning", "reasoning.text"}:
            parts.append(_block_field(block, "text", "thinking", "reasoning"))
    return "".join(parts)


def _extra_mapping(obj: Any) -> dict[str, Any]:
    """Best-effort extra-field dict from an SDK model or mapping."""
    if isinstance(obj, dict):
        return obj
    for attr in ("model_extra", "__pydantic_extra__", "extra"):
        extra = getattr(obj, attr, None)
        if isinstance(extra, dict):
            return extra
    return {}


def _attr_text(obj: Any, *names: str) -> str:
    """Read the first non-empty string attribute / extra field / dict key."""
    extra = _extra_mapping(obj)
    for name in names:
        value = getattr(obj, name, None)
        if value:
            return str(value)
        if extra.get(name):
            return str(extra[name])
        if isinstance(obj, dict) and obj.get(name):
            return str(obj[name])
    return ""


def reasoning_from_openrouter_message(message: Any) -> str:
    """Collect OpenRouter ``reasoning`` / ``reasoning_details`` from a message."""
    text = _attr_text(message, "reasoning", "reasoning_content")
    details = getattr(message, "reasoning_details", None)
    extra = _extra_mapping(message)
    if details is None:
        details = extra.get("reasoning_details")
    if isinstance(message, dict) and details is None:
        details = message.get("reasoning_details")
    if not details:
        return text
    parts: list[str] = []
    for item in details:
        chunk = _attr_text(item, "text", "reasoning", "thinking")
        if chunk:
            parts.append(chunk)
    detail_text = "".join(parts)
    if detail_text.strip() and (not text.strip() or len(detail_text) > len(text)):
        return detail_text
    return text


def assemble_completion(
    *,
    visible_text: str,
    reasoning_candidates: list[tuple[str, str]],
) -> LLMCompletion:
    """Pick the first non-empty reasoning candidate, then think-tag fallback.

    Args:
        visible_text: Streamed or final visible assistant text.
        reasoning_candidates: ``(text, source)`` pairs in priority order.

    Returns:
        Completion with think tags stripped from ``text`` when they were used
        as the reasoning source (or when tags were present alongside API fields).
    """
    tagged = extract_thinking(visible_text)
    visible = strip_thinking(visible_text) if tagged else (visible_text or "")
    for candidate, source in reasoning_candidates:
        if candidate and str(candidate).strip():
            return LLMCompletion(
                text=visible or (visible_text or ""),
                reasoning=str(candidate).strip(),
                reasoning_source=source,
            )
    if tagged.strip():
        return LLMCompletion(
            text=visible,
            reasoning=tagged.strip(),
            reasoning_source="think_tags",
        )
    return LLMCompletion(
        text=visible or (visible_text or ""),
        reasoning="",
        reasoning_source="empty",
    )


def _partial_completion(text: list[str], reasoning: list[str], source: str) -> LLMCompletion | None:
    if not text and not reasoning:
        return None
    return assemble_completion(
        visible_text="".join(text),
        reasoning_candidates=[("".join(reasoning), source)],
    )


def _stream_event_delta(event: Any) -> Any:
    """Return the delta payload of an Anthropic stream event, if any."""
    if event is None:
        return None
    if isinstance(event, dict):
        return event.get("delta")
    return getattr(event, "delta", None)


def _stream_event_type(event: Any) -> str:
    """Return the type of an Anthropic stream event."""
    if isinstance(event, dict):
        return str(event.get("type") or "")
    return str(getattr(event, "type", None) or "")


def approx_tokens(text: str) -> int:
    """Rough token count without a tokenizer (ASCII ~4 chars, other ~1 char)."""
    if not text:
        return 0
    ascii_n = 0
    other = 0
    for char in text:
        if ord(char) < 128:
            ascii_n += 1
        else:
            other += 1
    return (ascii_n + 3) // 4 + other


def _usage_value(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _read_usage(usage: Any) -> dict[str, int] | None:
    if usage is None:
        return None
    input_tokens = _usage_value(usage, "input_tokens")
    output_tokens = _usage_value(usage, "output_tokens")
    if input_tokens is None:
        input_tokens = _usage_value(usage, "prompt_tokens")
    if output_tokens is None:
        output_tokens = _usage_value(usage, "completion_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    details = _usage_value(usage, "completion_tokens_details")
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "reasoning_tokens": int(_usage_value(details, "reasoning_tokens") or 0),
    }


def _trace_key_label(provider: str) -> str:
    label = os.environ.get("CUDA_SFT_WORKER_KEY_LABEL", "").strip()
    if label == provider:
        return f"{provider}#1"
    if re.fullmatch(rf"{re.escape(provider)}#[1-9][0-9]*", label):
        return label
    return f"{provider}#1"


def _emit_usage(
    *, meta: CallMeta | None, provider: str, model: str, started: float,
    first_token: float | None, system: str, messages: list[dict[str, str]],
    completion: LLMCompletion | None, attempt: int = 1, error: BaseException | None = None,
) -> LLMCompletion | None:
    """Record one explicit client request; S0 cannot observe SDK-internal retries."""
    usage = completion.usage if completion is not None else None
    estimated = usage is None
    if estimated:
        usage = {
            "input_tokens": approx_tokens(system) + _message_tokens(messages),
            "output_tokens": approx_tokens((completion.text + completion.reasoning) if completion else ""),
            "reasoning_tokens": approx_tokens(completion.reasoning) if completion else 0,
        }
    cancelled = isinstance(error, (asyncio.CancelledError, FutureCancelledError, KeyboardInterrupt, GeneratorExit))
    trace.emit(
        "llm.call", job_key=meta.job_key if meta else "", candidate=meta.candidate if meta else 0,
        repair=meta.repair if meta else 0, role=meta.role if meta else "generator",
        purpose=meta.purpose if meta else "", provider=provider, model=model,
        key_label=_trace_key_label(provider), attempt=attempt,
        caller_attempt=meta.attempt if meta else 1, n_messages=len(messages),
        elapsed_s=round(time.monotonic() - started, 4),
        ttft_s=round(first_token - started, 4) if first_token is not None else None,
        input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
        reasoning_tokens=usage["reasoning_tokens"], tokens_estimated=estimated,
        ok=error is None, error_type=type(error).__name__ if error else "",
        retryable=is_retryable_llm_error(error) if error else False,
        cancelled=cancelled, cost_usd=None,
    )
    return replace(completion, usage=usage, tokens_estimated=estimated) if completion is not None else None


def _message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(approx_tokens(m.get("content") or "") + 4 for m in messages)


def _clip_text_tail(text: str, max_tokens: int) -> str:
    """Keep the tail of ``text`` so estimated tokens stay within ``max_tokens``."""
    if max_tokens <= 0:
        return ""
    if approx_tokens(text) <= max_tokens:
        return text
    # Binary-search a suffix; CUDA repair prompts keep latest code/errors at the end.
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        suffix = text[len(text) - mid :] if mid else ""
        if approx_tokens(suffix) <= max_tokens:
            best = suffix
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _clip_text_middle(text: str, max_tokens: int) -> str:
    """Fit the latest user turn, retaining roughly 30% head and 70% tail."""
    if max_tokens <= 0:
        return ""
    if approx_tokens(text) <= max_tokens:
        return text
    marker = "\n...[truncated]...\n"
    if approx_tokens(marker) > max_tokens:
        return _clip_text_tail(text, max_tokens)

    best = marker
    lo, hi = 0, len(text) - 1
    while lo <= hi:
        retained = (lo + hi) // 2
        head_len = min(retained - 1, max(1, round(retained * 0.3))) if retained >= 2 else 0
        tail_len = retained - head_len
        candidate = text[:head_len] + marker + text[-tail_len:] if tail_len else text[:head_len] + marker
        if approx_tokens(candidate) <= max_tokens:
            best = candidate
            lo = retained + 1
        else:
            hi = retained - 1
    return best


def fit_to_input_budget(
    system: str,
    messages: list[dict[str, str]],
    max_input_tokens: int,
) -> tuple[str, list[dict[str, str]]]:
    """Keep the first and latest user turns while fitting the input budget.

    Args:
        system: System prompt.
        messages: User/assistant turns (copied, not mutated).
        max_input_tokens: Budget from ``MAX_INPUT_TOKENS``. ``<=0`` disables clipping.

    Returns:
        Possibly truncated ``(system, messages)``.
    """
    msgs = [{"role": m.get("role"), "content": m.get("content") or ""} for m in messages]

    def validate_turns() -> None:
        if not msgs or any(
            message["role"] != ("user" if index % 2 == 0 else "assistant")
            for index, message in enumerate(msgs)
        ) or msgs[-1]["role"] != "user":
            raise ValueError("messages must alternate user/assistant and end with user")

    validate_turns()
    if max_input_tokens <= 0:
        return system, messages
    sys_text = system or ""

    def total() -> int:
        return approx_tokens(sys_text) + 4 + _message_tokens(msgs)

    if total() > max_input_tokens:
        logger.warning(
            "input ~%s tokens exceeds MAX_INPUT_TOKENS=%s; clipping",
            total(),
            max_input_tokens,
        )
        while len(msgs) > 3 and total() > max_input_tokens:
            del msgs[1:3]

        if total() > max_input_tokens:
            sys_budget = max(0, max_input_tokens - 4 - _message_tokens(msgs))
            sys_text = _clip_text_tail(sys_text, sys_budget)

        if total() > max_input_tokens:
            room = max(
                0,
                max_input_tokens - 4 - approx_tokens(sys_text) - _message_tokens(msgs[:-1]) - 4,
            )
            msgs[-1]["content"] = _clip_text_middle(msgs[-1]["content"], room)

    validate_turns()
    return sys_text, msgs


def _thinking_enabled(settings: Settings, level: str | None = None) -> bool:
    """True unless the thinking level is none/off/false/0."""
    thinking = (level if level is not None else settings.thinking_level).strip().lower()
    return thinking not in {"", "none", "off", "false", "0"}


class AnthropicOpenRouterClient:
    """OpenRouter via Anthropic Messages streaming API."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create the Anthropic SDK client.

        Args:
            settings: App settings; defaults to :func:`get_settings`.

        Raises:
            LLMError: If ``OPENROUTER_API_KEY`` is empty.
        """
        self.settings = settings or get_settings()
        if not self.settings.openrouter_api_key.strip():
            raise LLMError(
                "OPENROUTER_API_KEY is empty. Put the key in .env before generating."
            )
        if Anthropic is None:
            raise LLMError(
                "anthropic package is required for LLM_PROVIDER=openrouter; "
                "run pip install -e . or use LLM_PROVIDER=nvidia"
            )
        self._client = Anthropic(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.resolved_base_url,
            timeout=self.settings.llm_timeout_sec,
            default_headers={
                "HTTP-Referer": self.settings.http_referer,
                "X-Title": self.settings.x_title,
            },
        )

    def stream_text(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        print_stream: bool = True,
        meta: CallMeta | None = None,
    ) -> str:
        """Stream an Anthropic Messages completion; return visible text."""
        return self.stream_completion(
            messages=messages,
            system=system,
            temperature=temperature,
            print_stream=print_stream,
            meta=meta,
        ).text

    def stream_completion(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        print_stream: bool = True,
        thinking_level: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_max_tokens: int | None = None,
        meta: CallMeta | None = None,
    ) -> LLMCompletion:
        """Stream an Anthropic Messages completion with reasoning captured.

        Args:
            messages: User/assistant turns (no system role).
            system: System prompt.
            temperature: Sampling temperature.
            print_stream: Echo content tokens to stdout.
            thinking_level: Override ``THINKING_LEVEL`` for this call.
            max_output_tokens: Override completion token limit.
            reasoning_max_tokens: OpenRouter ``reasoning.max_tokens`` budget.

        Returns:
            Visible text plus reasoning (thinking blocks / OpenRouter field).

        Raises:
            LLMError: Retryable HTTP/transport failures.
        """
        extra_body: dict[str, Any] = {"temperature": temperature}
        thinking = (thinking_level or self.settings.thinking_level).strip()
        if _thinking_enabled(self.settings, thinking):
            reasoning: dict[str, Any] = {"effort": thinking, "exclude": False}
            budget = int(reasoning_max_tokens or 0)
            if budget > 0:
                reasoning["max_tokens"] = budget
            extra_body["reasoning"] = reasoning

        system, messages = fit_to_input_budget(
            system, messages, self.settings.max_input_tokens
        )
        requested = int(max_output_tokens or self.settings.resolved_max_output_tokens)
        max_tokens = min(requested, OPENROUTER_MAX_TOKENS_CAP)
        logger.debug("LLM call provider=openrouter role=%s", meta.role if meta else "generator")
        last_exc: BaseException | None = None
        for _attempt in range(4):
            started = time.monotonic()
            first_token: float | None = None
            attempt_meta = (
                replace(meta, purpose=f"{meta.purpose}:afford")
                if meta is not None and _attempt > 0
                else meta
            )
            request: dict[str, Any] = {
                "model": self.settings.resolved_model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": messages,
                "extra_body": extra_body,
            }

            streamed_text: list[str] = []
            streamed_reasoning: list[str] = []
            try:
                with self._client.messages.stream(**request) as stream:
                    for event in stream:
                        if _stream_event_type(event) != "content_block_delta":
                            continue
                        delta = _stream_event_delta(event)
                        if delta is None:
                            continue
                        delta_type = _block_type(delta)
                        if delta_type == "thinking_delta":
                            chunk = _attr_text(delta, "thinking", "text")
                            if chunk:
                                first_token = first_token or time.monotonic()
                                streamed_reasoning.append(chunk)
                        elif delta_type == "text_delta":
                            chunk = _attr_text(delta, "text")
                            if not chunk:
                                continue
                            first_token = first_token or time.monotonic()
                            streamed_text.append(chunk)
                            if print_stream:
                                print(chunk, end="", file=sys.stdout, flush=True)
                    final = stream.get_final_message()
            except _anthropic_error_types(
                "APIStatusError", "APIConnectionError", "APITimeoutError"
            ) as exc:
                last_exc = exc
                _emit_usage(
                    meta=attempt_meta, provider="openrouter", model=self.settings.resolved_model,
                    started=started, first_token=first_token, system=system,
                    messages=messages,
                    completion=_partial_completion(
                        streamed_text, streamed_reasoning, "anthropic_thinking"
                    ),
                    attempt=_attempt + 1, error=exc,
                )
                nxt = affordable_max_tokens(exc, max_tokens)
                if nxt is not None:
                    logger.warning(
                        "OpenRouter HTTP %s max_tokens=%s; retrying with %s",
                        getattr(exc, "status_code", None),
                        max_tokens,
                        nxt,
                    )
                    max_tokens = nxt
                    continue
                status = getattr(exc, "status_code", None)
                body = getattr(exc, "body", None)
                message = f"OpenRouter HTTP {status}: {exc}"
                if body:
                    message = f"{message} body={body}"
                if is_retryable_llm_error(exc):
                    raise LLMError(message) from exc
                raise
            except BaseException as exc:
                _emit_usage(
                    meta=attempt_meta, provider="openrouter", model=self.settings.resolved_model,
                    started=started, first_token=first_token, system=system,
                    messages=messages,
                    completion=_partial_completion(
                        streamed_text, streamed_reasoning, "anthropic_thinking"
                    ),
                    attempt=_attempt + 1, error=exc,
                )
                raise

            if print_stream and streamed_text:
                print(file=sys.stdout, flush=True)

            content = getattr(final, "content", None) or []
            final_text = _text_from_content_blocks(content)
            text = final_text or "".join(streamed_text)
            block_reasoning = _thinking_from_content_blocks(content)
            field_reasoning = reasoning_from_openrouter_message(final)
            completion = assemble_completion(
                visible_text=text,
                reasoning_candidates=[
                    (block_reasoning, "anthropic_thinking"),
                    (field_reasoning, "openrouter_reasoning"),
                    ("".join(streamed_reasoning), "anthropic_thinking"),
                ],
            )
            completion = replace(
                completion,
                usage=_read_usage(getattr(final, "usage", None)),
                origin="live_api",
            )
            completion = _emit_usage(
                meta=attempt_meta, provider="openrouter", model=self.settings.resolved_model,
                started=started, first_token=first_token, system=system,
                messages=messages, completion=completion, attempt=_attempt + 1,
            )
            assert completion is not None
            if not completion.text.strip():
                logger.warning("LLM returned empty text content (thinking-only or blank).")
            return completion

        assert last_exc is not None
        raise last_exc


class OpenAIChatClient:
    """OpenAI-compatible streaming Chat Completions client."""

    provider = "openai"

    def __init__(self, settings: Settings | None = None) -> None:
        """Create an OpenAI-compatible client for the selected provider.

        Args:
            settings: App settings; defaults to :func:`get_settings`.

        Raises:
            LLMError: Missing provider key or missing ``openai`` package.
        """
        self.settings = settings or get_settings()
        api_key = self.settings.resolved_api_key
        if not api_key:
            key_name = "NVIDIA_API_KEY" if self.provider == "nvidia" else "OPENAI_API_KEY"
            raise LLMError(f"{key_name} is empty. Set it in .env before generating.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError(f"openai package is required for LLM_PROVIDER={self.provider}") from exc
        self._client = OpenAI(
            base_url=self.settings.resolved_base_url,
            api_key=api_key,
            timeout=self.settings.llm_timeout_sec,
        )
        self._stream_usage_supported = True

    def stream_text(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        print_stream: bool = True,
        meta: CallMeta | None = None,
    ) -> str:
        """Stream Chat Completions; return visible text (not reasoning)."""
        return self.stream_completion(
            messages=messages,
            system=system,
            temperature=temperature,
            print_stream=print_stream,
            meta=meta,
        ).text

    def stream_completion(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        print_stream: bool = True,
        thinking_level: str | None = None,
        max_output_tokens: int | None = None,
        reasoning_max_tokens: int | None = None,
        meta: CallMeta | None = None,
        _attempt: int = 1,
    ) -> LLMCompletion:
        """Stream Chat Completions; collect content and reasoning deltas.

        Args:
            messages: User/assistant turns.
            system: Prepended as a system message if non-empty.
            temperature: Sampling temperature.
            print_stream: Echo content tokens to stdout.
            thinking_level: Override ``THINKING_LEVEL`` for this call.
            max_output_tokens: Override completion ``max_tokens``.
            reasoning_max_tokens: Optional provider reasoning token cap.

        Returns:
            Visible text plus ``reasoning_content`` / think-tag fallback.

        Raises:
            LLMError: Retryable HTTP/transport failures.
        """
        global _nvidia_stream_usage_supported
        import openai

        system, messages = fit_to_input_budget(
            system, messages, self.settings.max_input_tokens
        )
        oa_messages: list[dict[str, str]] = []
        if system.strip():
            oa_messages.append({"role": "system", "content": system})
        oa_messages.extend(messages)

        thinking = (thinking_level or self.settings.thinking_level).strip().lower()
        logger.debug("LLM call provider=%s role=%s", self.provider, meta.role if meta else "generator")
        out_tokens = int(max_output_tokens or self.settings.resolved_max_output_tokens)
        streamed_text: list[str] = []
        streamed_reasoning: list[str] = []
        started = time.monotonic()
        first_token: float | None = None
        usage: dict[str, int] | None = None
        try:
            request: dict[str, Any] = {
                "model": self.settings.resolved_model,
                "messages": oa_messages,
                "stream": True,
            }
            if self.provider == "nvidia":
                extra_body: dict[str, Any] = {
                    "chat_template_kwargs": {
                        "enable_thinking": _thinking_enabled(self.settings, thinking),
                    }
                }
                if _thinking_enabled(self.settings, thinking) and thinking in {"low", "medium", "high"}:
                    extra_body["chat_template_kwargs"]["reasoning_effort"] = thinking
                request.update(
                    temperature=temperature,
                    top_p=self.settings.top_p,
                    max_tokens=out_tokens,
                    extra_body=extra_body,
                )
            else:
                request["max_completion_tokens"] = out_tokens
                if thinking in {"low", "medium", "high"}:
                    request["reasoning_effort"] = thinking
            if self.provider == "nvidia":
                with _nvidia_stream_usage_lock:
                    include_usage = _nvidia_stream_usage_supported
            else:
                include_usage = self._stream_usage_supported
            if include_usage:
                request["stream_options"] = {"include_usage": True}
            completion = self._client.chat.completions.create(**request)
            for chunk in completion:
                usage = _read_usage(getattr(chunk, "usage", None)) or usage
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reason = _attr_text(delta, "reasoning_content", "reasoning")
                if reason:
                    first_token = first_token or time.monotonic()
                    streamed_reasoning.append(reason)
                content = _attr_text(delta, "content")
                if content:
                    first_token = first_token or time.monotonic()
                    streamed_text.append(content)
                    if print_stream:
                        print(content, end="", file=sys.stdout, flush=True)
        except (
            openai.APIStatusError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.APIError,
        ) as exc:
            _emit_usage(
                meta=meta, provider=self.provider, model=self.settings.resolved_model,
                started=started, first_token=first_token, system=system,
                messages=messages,
                completion=_partial_completion(
                    streamed_text, streamed_reasoning, f"{self.provider}_delta"
                ),
                attempt=_attempt, error=exc,
            )
            if getattr(exc, "status_code", None) == 400 and "stream_options" in str(exc).lower() and include_usage:
                if self.provider == "nvidia":
                    with _nvidia_stream_usage_lock:
                        warn = _nvidia_stream_usage_supported
                        _nvidia_stream_usage_supported = False
                else:
                    warn = self._stream_usage_supported
                    self._stream_usage_supported = False
                if warn:
                    logger.warning(
                        "%s endpoint rejected stream_options; falling back to estimated usage",
                        self.provider,
                    )
                return self.stream_completion(
                    messages=messages, system=system, temperature=temperature,
                    print_stream=print_stream, thinking_level=thinking_level,
                    max_output_tokens=max_output_tokens,
                    reasoning_max_tokens=reasoning_max_tokens, meta=meta,
                    _attempt=_attempt + 1,
                )
            status = getattr(exc, "status_code", None)
            message = f"{self.provider} HTTP {status}: {exc}"
            if is_retryable_llm_error(exc):
                raise LLMError(message) from exc
            raise
        except BaseException as exc:
            _emit_usage(
                meta=meta, provider=self.provider, model=self.settings.resolved_model,
                started=started, first_token=first_token, system=system,
                messages=messages,
                completion=_partial_completion(
                    streamed_text, streamed_reasoning, f"{self.provider}_delta"
                ),
                attempt=_attempt, error=exc,
            )
            raise

        if print_stream and streamed_text:
            print(file=sys.stdout, flush=True)

        text = "".join(streamed_text)
        result = assemble_completion(
            visible_text=text,
            reasoning_candidates=[
                ("".join(streamed_reasoning), f"{self.provider}_delta"),
            ],
        )
        result = replace(result, usage=usage, origin="live_api")
        emitted = _emit_usage(
            meta=meta, provider=self.provider, model=self.settings.resolved_model,
            started=started, first_token=first_token, system=system,
            messages=messages, completion=result, attempt=_attempt,
        )
        assert emitted is not None
        result = emitted
        if not result.text.strip():
            logger.warning("LLM returned empty text content (thinking-only or blank).")
        return result


class NvidiaOpenAIClient(OpenAIChatClient):
    """NVIDIA NIM variant with its provider-specific request options."""

    provider = "nvidia"


_clients: dict[tuple[Any, ...], LLMClient] = {}
_clients_lock = threading.Lock()


def get_llm_client(settings: Settings | None = None, *, role: str = "generator") -> LLMClient:
    """Return a cached client for the role's effective provider and endpoint."""
    cfg = (settings or get_settings()).for_role(role)
    factory = deps.current().llm_factory
    if factory is not None:
        from cuda_sft.runtime.limits import limited_client

        return limited_client(factory(role), settings=cfg)
    key = (
        role, cfg.llm_provider, cfg.resolved_api_key, cfg.resolved_base_url,
        cfg.resolved_model, cfg.llm_timeout_sec, cfg.thinking_level,
        cfg.max_input_tokens, cfg.resolved_max_output_tokens, cfg.top_p,
        cfg.http_referer, cfg.x_title,
    )
    with _clients_lock:
        client = _clients.get(key)
        if client is None:
            if cfg.llm_provider == "openrouter":
                client = AnthropicOpenRouterClient(cfg)
            elif cfg.llm_provider == "nvidia":
                client = NvidiaOpenAIClient(cfg)
            else:
                client = OpenAIChatClient(cfg)
            _clients[key] = client
    from cuda_sft.runtime.limits import limited_client

    return limited_client(client, settings=cfg)


def reset_llm_client() -> None:
    """Drop cached role clients (tests / provider switch in-process)."""
    with _clients_lock:
        _clients.clear()
