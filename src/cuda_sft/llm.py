from __future__ import annotations

import logging
import sys
from typing import Any, Iterable

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


def _error_text(exc: BaseException) -> str:
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    message = getattr(exc, "message", None)
    if message:
        parts.append(str(message))
    return " ".join(parts).lower()


def is_retryable_llm_error(exc: BaseException) -> bool:
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
    return any(hint in _error_text(exc) for hint in UPSTREAM_RETRY_HINTS)


def _text_from_content_blocks(content: Iterable[Any]) -> str:
    parts: list[str] = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(getattr(block, "text", "") or "")
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "".join(parts)


class AnthropicOpenRouterClient:
    def __init__(self, settings: Settings | None = None) -> None:
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
        extra_body: dict[str, Any] = {}
        thinking = self.settings.thinking_level.strip()
        if thinking and thinking not in {"none", "off", "false"}:
            extra_body["reasoning"] = {"effort": thinking}

        request: dict[str, Any] = {
            "model": self.settings.model,
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


_client: AnthropicOpenRouterClient | None = None


def get_llm_client(settings: Settings | None = None) -> AnthropicOpenRouterClient:
    global _client
    if _client is None:
        _client = AnthropicOpenRouterClient(settings)
    return _client


def reset_llm_client() -> None:
    global _client
    _client = None
