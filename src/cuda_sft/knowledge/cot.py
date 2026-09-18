"""Knowledge CoT editor. Does not reuse the kernel CotAgent (code-faithful)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cuda_sft.config import Settings, get_settings
from cuda_sft.knowledge.parse import (
    MIN_COT_CHARS,
    clean_raw_reasoning,
    sanitize_knowledge_cot,
)
from cuda_sft.parse import has_numbered_headings
from cuda_sft.knowledge.prompt import COT_SYSTEM, build_cot_user
from cuda_sft.knowledge.state import KnowledgeGraphState
from cuda_sft.llm import LLMClient, LLMError, get_llm_client, is_retryable_llm_error

logger = logging.getLogger(__name__)

AGENT_ATTEMPTS = 3


@dataclass
class KnowledgeCotResult:
    """Polished CoT for one quality-gated knowledge sample."""

    cot: str
    source: str
    raw_reasoning: str
    error: str = ""


def _clip(text: str, max_chars: int) -> str:
    """Hard-truncate ``text`` when ``max_chars`` is positive.

    Args:
        text: CoT body.
        max_chars: ``<=0`` disables truncation.
    """
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


class KnowledgeCotAgent:
    """LLM editor that turns teacher thinking into a knowledge CoT."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = llm_client

    def _client_or_default(self) -> LLMClient:
        return self._client or get_llm_client(self.settings)

    def refine(self, state: KnowledgeGraphState) -> KnowledgeCotResult:
        """Never raises on LLM failure; falls back per kernel CoT env flags."""
        settings = self.settings
        raw = clean_raw_reasoning(
            str(state.get("raw_reasoning") or ""),
            settings.cot_raw_max_chars,
        )
        if not settings.cot_agent_enabled:
            if raw.strip():
                return KnowledgeCotResult(
                    cot=_clip(raw, settings.cot_max_chars),
                    source="raw",
                    raw_reasoning=raw,
                )
            return self._fallback(state, raw, error="empty teacher thinking", mode=settings.cot_on_empty)

        try:
            polished = self._run_agent(state, raw_reasoning=raw)
        except Exception as exc:
            logger.warning("knowledge CoT agent failed: %s", exc)
            return self._fallback(
                state, raw, error=str(exc), mode=settings.cot_on_agent_fail
            )

        cleaned = sanitize_knowledge_cot(polished, settings.cot_max_chars)
        if len(cleaned) < MIN_COT_CHARS or not has_numbered_headings(cleaned, 5):
            try:
                polished = self._run_agent(state, raw_reasoning=raw)
                retry = sanitize_knowledge_cot(polished, settings.cot_max_chars)
            except Exception as exc:
                logger.warning("knowledge CoT heading retry failed: %s", exc)
                retry = ""
            if len(retry) >= MIN_COT_CHARS and has_numbered_headings(retry, 5):
                return KnowledgeCotResult(cot=retry, source="agent", raw_reasoning=raw)
            return self._fallback(
                state,
                raw,
                error="agent output too short or missing numbered headings",
                mode=settings.cot_on_agent_fail,
            )
        return KnowledgeCotResult(cot=cleaned, source="agent", raw_reasoning=raw)

    def _fallback(
        self,
        state: KnowledgeGraphState,
        raw: str,
        *,
        error: str,
        mode: str,
    ) -> KnowledgeCotResult:
        chosen = (mode or "raw").strip().lower()
        if chosen == "raw":
            if raw.strip():
                return KnowledgeCotResult(
                    cot=_clip(raw, self.settings.cot_max_chars),
                    source="raw",
                    raw_reasoning=raw,
                    error=error,
                )
            chosen = (
                self.settings.cot_on_empty
                if self.settings.cot_on_empty != "raw"
                else "empty"
            )
        if chosen == "synthetic":
            try:
                polished = self._run_agent(state, raw_reasoning="")
                cleaned = sanitize_knowledge_cot(polished, self.settings.cot_max_chars)
            except Exception as exc:
                logger.warning("knowledge CoT synthetic rewrite failed: %s", exc)
                return KnowledgeCotResult(
                    cot="",
                    source="empty",
                    raw_reasoning=raw,
                    error=f"{error}; synthetic:{exc}",
                )
            if len(cleaned) < MIN_COT_CHARS:
                return KnowledgeCotResult(
                    cot="",
                    source="empty",
                    raw_reasoning=raw,
                    error=f"{error}; synthetic too short",
                )
            return KnowledgeCotResult(
                cot=cleaned, source="synthetic", raw_reasoning=raw, error=error
            )
        return KnowledgeCotResult(cot="", source="empty", raw_reasoning=raw, error=error)

    def _run_agent(self, state: KnowledgeGraphState, *, raw_reasoning: str) -> str:
        settings = self.settings
        user = build_cot_user(
            question=str(state.get("question") or ""),
            answer=str(state.get("answer") or ""),
            raw_reasoning=raw_reasoning,
            topic=str(state.get("topic") or "general"),
            repair_idx=int(state.get("repair_idx") or 0),
            gate="; ".join(state.get("gate_reasons") or []),
            issues=list(state.get("judge_issues") or []),
            max_chars=settings.cot_max_chars,
        )
        client = self._client_or_default()
        last_exc: BaseException | None = None
        for attempt in range(1, AGENT_ATTEMPTS + 1):
            try:
                stream_completion = getattr(client, "stream_completion", None)
                base = {
                    "messages": [{"role": "user", "content": user}],
                    "system": COT_SYSTEM,
                    "temperature": float(settings.cot_temperature),
                    "print_stream": False,
                }
                if callable(stream_completion):
                    completion = stream_completion(
                        **base,
                        thinking_level="none",
                        max_output_tokens=min(
                            max(256, settings.cot_max_chars),
                            settings.knowledge_max_output_tokens,
                        ),
                    )
                    text = completion.text or ""
                    if not text.strip():
                        text = completion.reasoning or ""
                    return text
                return client.stream_text(**base)
            except Exception as exc:
                last_exc = exc
                retry = attempt < AGENT_ATTEMPTS and (
                    isinstance(exc, LLMError) or is_retryable_llm_error(exc)
                )
                if retry:
                    logger.warning(
                        "knowledge CoT retry %s/%s: %s", attempt, AGENT_ATTEMPTS, exc
                    )
                    continue
                raise
        assert last_exc is not None
        raise last_exc
