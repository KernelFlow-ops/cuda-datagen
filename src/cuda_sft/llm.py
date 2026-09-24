"""Streaming LLM clients: OpenRouter (Anthropic Messages) and NVIDIA NIM (OpenAI)."""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

try:
    import anthropic
    from anthropic import Anthropic
except ImportError:  # OpenAI/NVIDIA-only installs must still import this module.
    anthropic = None  # type: ignore[assignment]
    Anthropic = None  # type: ignore[assignment,misc]

from cuda_sft.config import Settings, get_settings
from cuda_sft.parse import extract_thinking, strip_thinking

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


class LLMError(Exception):
    """Retryable LLM / transport failure."""


@dataclass(frozen=True)
class LLMCompletion:
    """One streamed completion, with visible text split from reasoning.

    Attributes:
        text: Visible assistant text (think tags stripped when they were extracted).
        reasoning: Concatenated chain-of-thought / thinking.
        reasoning_source: How reasoning was obtained.
    """

    text: str
    reasoning: str = ""
    reasoning_source: str = "empty"


class LLMClient(Protocol):
    """Minimal interface used by the generate node."""

    def stream_completion(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        print_stream: bool = True,
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


def _message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(approx_tokens(m.get("content") or "") + 4 for m in messages)


def _clip_text_tail(text: str, max_tokens: int) -> str:
    """Keep the tail of ``text`` so estimated tokens stay within ``max_tokens``."""
    if max_tokens <= 0 or approx_tokens(text) <= max_tokens:
        return text
    # Binary-search a suffix; CUDA repair prompts keep latest code/errors at the end.
    lo, hi = 0, len(text)
    best = text[- min(len(text), max(1, max_tokens)) :]
    while lo <= hi:
        mid = (lo + hi) // 2
        suffix = text[len(text) - mid :] if mid else ""
        if approx_tokens(suffix) <= max_tokens:
            best = suffix
            lo = mid + 1
        else:
            hi = mid - 1
    if best and not text.endswith(best):
        return "\n...[truncated input]...\n" + best
    return best


def fit_to_input_budget(
    system: str,
    messages: list[dict[str, str]],
    max_input_tokens: int,
) -> tuple[str, list[dict[str, str]]]:
    """Drop oldest turns, then trim system / last message to fit ``max_input_tokens``.

    Args:
        system: System prompt.
        messages: User/assistant turns (copied, not mutated).
        max_input_tokens: Budget from ``MAX_INPUT_TOKENS``. ``<=0`` disables clipping.

    Returns:
        Possibly truncated ``(system, messages)``.
    """
    if max_input_tokens <= 0:
        return system, messages
    msgs = [{"role": m.get("role", "user"), "content": m.get("content") or ""} for m in messages]
    sys_text = system or ""

    def total() -> int:
        return approx_tokens(sys_text) + 4 + _message_tokens(msgs)

    if total() <= max_input_tokens:
        return sys_text, msgs

    logger.warning(
        "input ~%s tokens exceeds MAX_INPUT_TOKENS=%s; clipping",
        total(),
        max_input_tokens,
    )
    while len(msgs) > 1 and total() > max_input_tokens:
        msgs.pop(0)

    if total() <= max_input_tokens:
        return sys_text, msgs

    last_min = min(256, max_input_tokens // 4)
    sys_budget = max(0, max_input_tokens - _message_tokens(msgs) - 8)
    if approx_tokens(sys_text) > sys_budget:
        sys_text = _clip_text_tail(sys_text, sys_budget)

    if total() <= max_input_tokens:
        return sys_text, msgs

    if msgs:
        last = dict(msgs[-1])
        room = max(last_min, max_input_tokens - approx_tokens(sys_text) - 8)
        last["content"] = _clip_text_tail(last.get("content") or "", room)
        msgs[-1] = last
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
                "install requirements.txt or use LLM_PROVIDER=nvidia"
            )
        self._client = Anthropic(
            api_key=self.settings.openrouter_api_key,
            base_url=self.settings.openrouter_base_url,
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
    ) -> str:
        """Stream an Anthropic Messages completion; return visible text."""
        return self.stream_completion(
            messages=messages,
            system=system,
            temperature=temperature,
            print_stream=print_stream,
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
    ) -> LLMCompletion:
        """Stream an Anthropic Messages completion with reasoning captured.

        Args:
            messages: User/assistant turns (no system role).
            system: System prompt.
            temperature: Sampling temperature.
            print_stream: Echo content tokens to stdout.
            thinking_level: Override ``THINKING_LEVEL`` for this call.
            max_output_tokens: Override completion ``max_tokens``.
            reasoning_max_tokens: OpenRouter ``reasoning.max_tokens`` budget.

        Returns:
            Visible text plus reasoning (thinking blocks / OpenRouter field).

        Raises:
            LLMError: Retryable HTTP/transport failures.
        """
        extra_body: dict[str, Any] = {}
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
        last_exc: BaseException | None = None
        for _attempt in range(4):
            request: dict[str, Any] = {
                "model": self.settings.resolved_model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system,
                "messages": messages,
            }
            if extra_body:
                request["extra_body"] = extra_body

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
                                streamed_reasoning.append(chunk)
                        elif delta_type == "text_delta":
                            chunk = _attr_text(delta, "text")
                            if not chunk:
                                continue
                            streamed_text.append(chunk)
                            if print_stream:
                                print(chunk, end="", file=sys.stdout, flush=True)
                    final = stream.get_final_message()
            except _anthropic_error_types(
                "APIStatusError", "APIConnectionError", "APITimeoutError"
            ) as exc:
                last_exc = exc
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
            if not completion.text.strip():
                logger.warning("LLM returned empty text content (thinking-only or blank).")
            return completion

        assert last_exc is not None
        raise last_exc


class NvidiaOpenAIClient:
    """NVIDIA NIM via OpenAI-compatible Chat Completions."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Create the OpenAI-compatible NVIDIA NIM client.

        Args:
            settings: App settings; defaults to :func:`get_settings`.

        Raises:
            LLMError: Missing ``NVIDIA_API_KEY`` or missing ``openai`` package.
        """
        self.settings = settings or get_settings()
        api_key = self.settings.resolved_api_key
        if not api_key:
            raise LLMError(
                "NVIDIA_API_KEY is empty. Put the nvapi- key in .env before generating."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("openai package is required for LLM_PROVIDER=nvidia") from exc
        self._client = OpenAI(
            base_url=self.settings.nvidia_base_url,
            api_key=api_key,
            timeout=self.settings.llm_timeout_sec,
        )

    def stream_text(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        print_stream: bool = True,
    ) -> str:
        """Stream Chat Completions; return visible text (not reasoning)."""
        return self.stream_completion(
            messages=messages,
            system=system,
            temperature=temperature,
            print_stream=print_stream,
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
    ) -> LLMCompletion:
        """Stream Chat Completions; collect content and reasoning deltas.

        Args:
            messages: User/assistant turns.
            system: Prepended as a system message if non-empty.
            temperature: Sampling temperature.
            print_stream: Echo content tokens to stdout.
            thinking_level: Override ``THINKING_LEVEL`` for this call.
            max_output_tokens: Override completion ``max_tokens``.
            reasoning_max_tokens: Optional NVIDIA ``max_thinking_tokens``.

        Returns:
            Visible text plus ``reasoning_content`` / think-tag fallback.

        Raises:
            LLMError: Retryable HTTP/transport failures.
        """
        import openai

        system, messages = fit_to_input_budget(
            system, messages, self.settings.max_input_tokens
        )
        oa_messages: list[dict[str, str]] = []
        if system.strip():
            oa_messages.append({"role": "system", "content": system})
        oa_messages.extend(messages)

        thinking = (thinking_level or self.settings.thinking_level).strip().lower()
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": _thinking_enabled(self.settings, thinking),
            }
        }
        if _thinking_enabled(self.settings, thinking) and thinking in {"low", "medium", "high"}:
            extra_body["chat_template_kwargs"]["reasoning_effort"] = thinking

        out_tokens = int(max_output_tokens or self.settings.resolved_max_output_tokens)
        streamed_text: list[str] = []
        streamed_reasoning: list[str] = []
        try:
            completion = self._client.chat.completions.create(
                model=self.settings.resolved_model,
                messages=oa_messages,
                temperature=temperature,
                top_p=self.settings.top_p,
                max_tokens=out_tokens,
                extra_body=extra_body,
                stream=True,
            )
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reason = _attr_text(delta, "reasoning_content", "reasoning")
                if reason:
                    streamed_reasoning.append(reason)
                content = _attr_text(delta, "content")
                if content:
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
            status = getattr(exc, "status_code", None)
            message = f"NVIDIA NIM HTTP {status}: {exc}"
            if is_retryable_llm_error(exc):
                raise LLMError(message) from exc
            raise

        if print_stream and streamed_text:
            print(file=sys.stdout, flush=True)

        text = "".join(streamed_text)
        result = assemble_completion(
            visible_text=text,
            reasoning_candidates=[
                ("".join(streamed_reasoning), "nvidia_delta"),
            ],
        )
        if not result.text.strip():
            logger.warning("LLM returned empty text content (thinking-only or blank).")
        return result


_client: LLMClient | None = None


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Return a process-wide client for ``LLM_PROVIDER`` (created on first use).

    Args:
        settings: Used only when constructing the first client.
    """
    global _client
    if _client is None:
        cfg = settings or get_settings()
        if cfg.llm_provider == "nvidia":
            _client = NvidiaOpenAIClient(cfg)
        else:
            _client = AnthropicOpenRouterClient(cfg)
    return _client


def reset_llm_client() -> None:
    """Drop the cached client (tests / provider switch in-process)."""
    global _client
    _client = None
