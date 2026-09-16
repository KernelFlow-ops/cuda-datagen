"""Streaming LLM clients: OpenRouter (Anthropic Messages) and NVIDIA NIM (OpenAI)."""

from __future__ import annotations

import logging
import sys
from typing import Any, Iterable, Protocol

import anthropic
from anthropic import Anthropic

from cuda_sft.config import Settings, get_settings

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
)


class LLMError(Exception):
    """Retryable LLM / transport failure."""


class LLMClient(Protocol):
    """Minimal interface used by the generate node."""

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


def is_retryable_llm_error(exc: BaseException) -> bool:
    """Return True for rate limits, 5xx, overload, and connection timeouts.

    Used by LangGraph ``RetryPolicy`` on the generate node.
    """
    if isinstance(exc, LLMError):
        return True
    if isinstance(
        exc,
        (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.RateLimitError,
            anthropic.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, anthropic.APIStatusError):
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


def _text_from_content_blocks(content: Iterable[Any]) -> str:
    """Join Anthropic ``type=text`` content blocks into one string."""
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", "") or "")
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


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


def _thinking_enabled(settings: Settings) -> bool:
    """True unless ``THINKING_LEVEL`` is none/off/false/0."""
    thinking = settings.thinking_level.strip().lower()
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
        """Stream an Anthropic Messages completion; return text (not thinking).

        Args:
            messages: User/assistant turns (no system role).
            system: System prompt.
            temperature: Sampling temperature.
            print_stream: Echo content tokens to stdout.

        Returns:
            Assistant visible text.

        Raises:
            LLMError: Retryable HTTP/transport failures.
        """
        extra_body: dict[str, Any] = {}
        thinking = self.settings.thinking_level.strip()
        if _thinking_enabled(self.settings):
            extra_body["reasoning"] = {"effort": thinking}

        system, messages = fit_to_input_budget(
            system, messages, self.settings.max_input_tokens
        )
        request: dict[str, Any] = {
            "model": self.settings.resolved_model,
            "max_tokens": self.settings.resolved_max_output_tokens,
            "temperature": temperature,
            "system": system,
            "messages": messages,
        }
        if extra_body:
            request["extra_body"] = extra_body

        streamed_text: list[str] = []
        try:
            with self._client.messages.stream(**request) as stream:
                for chunk in stream.text_stream:
                    if not chunk:
                        continue
                    streamed_text.append(chunk)
                    if print_stream:
                        print(chunk, end="", file=sys.stdout, flush=True)
                final = stream.get_final_message()
        except (anthropic.APIStatusError, anthropic.APIConnectionError, anthropic.APITimeoutError) as exc:
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

        final_text = _text_from_content_blocks(getattr(final, "content", []) or [])
        text = final_text or "".join(streamed_text)
        if not text.strip():
            logger.warning("LLM returned empty text content (thinking-only or blank).")
        return text


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
        """Stream Chat Completions; collect ``delta.content`` only (not reasoning).

        Args:
            messages: User/assistant turns.
            system: Prepended as a system message if non-empty.
            temperature: Sampling temperature.
            print_stream: Echo content tokens to stdout.

        Returns:
            Assistant visible text.

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

        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": _thinking_enabled(self.settings),
            }
        }
        thinking = self.settings.thinking_level.strip().lower()
        if _thinking_enabled(self.settings) and thinking in {"low", "medium", "high"}:
            extra_body["chat_template_kwargs"]["reasoning_effort"] = thinking

        streamed_text: list[str] = []
        try:
            completion = self._client.chat.completions.create(
                model=self.settings.resolved_model,
                messages=oa_messages,
                temperature=temperature,
                top_p=self.settings.top_p,
                max_tokens=self.settings.resolved_max_output_tokens,
                extra_body=extra_body,
                stream=True,
            )
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
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
        if not text.strip():
            logger.warning("LLM returned empty text content (thinking-only or blank).")
        return text


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
