"""CoT Agent: polish teacher reasoning into SFT chain-of-thought."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cuda_sft.config import Settings, get_settings
from cuda_sft.llm import LLMClient, LLMError, get_llm_client, is_retryable_llm_error
from cuda_sft.parse import (
    collapse_blank_lines,
    extract_thinking,
    has_numbered_headings,
    strip_code_from_cot,
)
from cuda_sft.prompt import (
    COT_SKELETON_PY_ZH,
    COT_SKELETON_ZH,
    build_cot_user_prompt,
    cot_system_for,
)
from cuda_sft.prompts.selection import looks_chinese
from cuda_sft.state import GraphState

logger = logging.getLogger(__name__)

MIN_COT_CHARS = 120
AGENT_ATTEMPTS = 3


@dataclass
class CotResult:
    """Polished CoT for one compile-passing sample.

    Attributes:
        cot: Text written into the SFT ``<think>`` block (may be empty).
        source: ``agent``, ``raw``, ``synthetic``, or ``empty``.
        raw_reasoning: Cleaned teacher thinking used as input.
        error: Failure reason when falling back.
    """

    cot: str
    source: str
    raw_reasoning: str
    error: str = ""


def clean_raw_reasoning(text: str, max_chars: int) -> str:
    """Normalize teacher thinking before storing or sending to the agent.

    Args:
        text: Raw API reasoning, possibly tagged or fenced.
        max_chars: Truncate to this many characters; ``<=0`` disables.

    Returns:
        Prose-only thinking, possibly empty.
    """
    raw = text or ""
    tagged = extract_thinking(raw)
    body = tagged if tagged.strip() else raw
    body = strip_code_from_cot(body)
    body = collapse_blank_lines(body)
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars].rstrip() + "\n...[truncated reasoning]..."
    return body


def sanitize_cot_output(text: str, max_chars: int) -> str:
    """Drop think wrappers and code fences from an agent reply.

    Args:
        text: Agent visible text.
        max_chars: Hard cap on the polished CoT.

    Returns:
        Clean CoT prose, possibly empty.
    """
    raw = text or ""
    tagged = extract_thinking(raw)
    body = tagged if tagged.strip() else raw
    body = strip_code_from_cot(body)
    body = collapse_blank_lines(body)
    if max_chars > 0 and len(body) > max_chars:
        body = body[:max_chars].rstrip()
    return body


def _clip(text: str, max_chars: int) -> str:
    """Hard-truncate ``text`` when ``max_chars`` is positive."""
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars].rstrip()
    return text


class CotAgent:
    """LLM editor that turns teacher thinking into pedagogical CoT."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        """Bind settings and an optional injected LLM client (tests)."""
        self.settings = settings or get_settings()
        self._client = llm_client

    def _client_or_default(self) -> LLMClient:
        return self._client or get_llm_client(self.settings)

    def refine(self, state: GraphState) -> CotResult:
        """Produce CoT for a compile-passing graph state.

        Never raises on LLM failure; falls back per ``COT_ON_AGENT_FAIL``.

        Args:
            state: Winning state with ``code`` and optional ``raw_reasoning``.
        """
        settings = self.settings
        raw = clean_raw_reasoning(
            str(state.get("raw_reasoning") or ""),
            settings.cot_raw_max_chars,
        )
        if not settings.cot_agent_enabled:
            if raw.strip():
                return CotResult(cot=_clip(raw, settings.cot_max_chars), source="raw", raw_reasoning=raw)
            return self._fallback(state, raw, error="empty teacher thinking", mode=settings.cot_on_empty)

        try:
            polished = self._run_agent(state, raw_reasoning=raw)
        except Exception as exc:
            logger.warning("CoT agent failed: %s", exc)
            return self._fallback(
                state, raw, error=str(exc), mode=settings.cot_on_agent_fail
            )

        cleaned = sanitize_cot_output(polished, settings.cot_max_chars)
        if len(cleaned) < MIN_COT_CHARS or not has_numbered_headings(cleaned, 6):
            try:
                polished = self._run_agent(state, raw_reasoning=raw)
                retry = sanitize_cot_output(polished, settings.cot_max_chars)
            except Exception as exc:
                logger.warning("CoT heading retry failed: %s", exc)
                retry = ""
            if len(retry) >= MIN_COT_CHARS and has_numbered_headings(retry, 6):
                return CotResult(cot=retry, source="agent", raw_reasoning=raw)
            return self._fallback(
                state,
                raw,
                error="agent output too short, code-only, or missing numbered headings",
                mode=settings.cot_on_agent_fail,
            )
        return CotResult(cot=cleaned, source="agent", raw_reasoning=raw)

    def _fallback(
        self,
        state: GraphState,
        raw: str,
        *,
        error: str,
        mode: str,
    ) -> CotResult:
        """Apply ``raw`` / ``synthetic`` / ``empty`` after an agent miss."""
        chosen = (mode or "raw").strip().lower()
        if chosen == "raw":
            if raw.strip():
                return CotResult(
                    cot=_clip(raw, self.settings.cot_max_chars),
                    source="raw",
                    raw_reasoning=raw,
                    error=error,
                )
            chosen = self.settings.cot_on_empty if self.settings.cot_on_empty != "raw" else "empty"
        if chosen == "synthetic":
            try:
                polished = self._run_agent(state, raw_reasoning="")
                cleaned = sanitize_cot_output(polished, self.settings.cot_max_chars)
            except Exception as exc:
                logger.warning("CoT synthetic rewrite failed: %s", exc)
                return CotResult(cot="", source="empty", raw_reasoning=raw, error=f"{error}; synthetic:{exc}")
            if len(cleaned) < MIN_COT_CHARS:
                return CotResult(
                    cot="",
                    source="empty",
                    raw_reasoning=raw,
                    error=f"{error}; synthetic too short",
                )
            return CotResult(
                cot=cleaned, source="synthetic", raw_reasoning=raw, error=error
            )
        return CotResult(cot="", source="empty", raw_reasoning=raw, error=error)

    def _run_agent(self, state: GraphState, *, raw_reasoning: str) -> str:
        """Call the editor model; retry retryable LLM errors a few times."""
        settings = self.settings
        dialect = str(state.get("dialect") or "cuda")
        language = "python" if dialect in {"triton", "tilelang"} else "cuda-cpp"
        fence = "python" if language == "python" else "cuda"
        skeleton = ""
        try:
            from cuda_sft.dialects.agent import get_spec

            skeleton = get_spec(dialect).cot_skeleton()
        except Exception:
            skeleton = ""
        if looks_chinese(str(state.get("question") or "")):
            skeleton = COT_SKELETON_PY_ZH if language == "python" else COT_SKELETON_ZH
        user = build_cot_user_prompt(
            question=str(state.get("question") or ""),
            code=str(state.get("code") or ""),
            raw_reasoning=raw_reasoning,
            repair_idx=int(state.get("repair_idx") or 0),
            error_summary=str(state.get("compile_error") or ""),
            judge_issues=list(state.get("judge_issues") or []),
            judge_suggestions=list(state.get("judge_suggestions") or []),
            max_chars=settings.cot_max_chars,
            dialect=dialect,
            language=language,
            skeleton=skeleton,
            fence=fence,
        )
        client = self._client_or_default()
        last_exc: BaseException | None = None
        for attempt in range(1, AGENT_ATTEMPTS + 1):
            try:
                stream_completion = getattr(client, "stream_completion", None)
                if callable(stream_completion):
                    completion = stream_completion(
                        messages=[{"role": "user", "content": user}],
                        system=cot_system_for(dialect=dialect),
                        temperature=float(settings.cot_temperature),
                        print_stream=False,
                        thinking_level="none",
                    )
                    return completion.text or ""
                return client.stream_text(
                    messages=[{"role": "user", "content": user}],
                    system=cot_system_for(dialect=dialect),
                    temperature=float(settings.cot_temperature),
                    print_stream=False,
                )
            except Exception as exc:
                last_exc = exc
                retry = attempt < AGENT_ATTEMPTS and (
                    isinstance(exc, LLMError) or is_retryable_llm_error(exc)
                )
                if retry:
                    logger.warning(
                        "CoT agent retry %s/%s: %s", attempt, AGENT_ATTEMPTS, exc
                    )
                    continue
                raise
        assert last_exc is not None
        raise last_exc
