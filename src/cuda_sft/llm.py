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

        request: dict[str, Any] = {
            "model": self.settings.resolved_model,
            "max_tokens": self.settings.max_tokens,
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
        if not self.settings.nvidia_api_key.strip():
            raise LLMError(
                "NVIDIA_API_KEY is empty. Put the nvapi- key in .env before generating."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMError("openai package is required for LLM_PROVIDER=nvidia") from exc
        self._client = OpenAI(
            base_url=self.settings.nvidia_base_url,
            api_key=self.settings.nvidia_api_key,
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
                max_tokens=self.settings.max_tokens,
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
