"""Knowledge CoT editor. Does not reuse the kernel CotAgent (code-faithful)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from cuda_sft.config import Settings, get_settings
from cuda_sft.core.cot import (
    EditorBudget,
    EditorBudgetExhausted,
    call_editor,
    clip_at_boundary,
    draft_problems,
    editor_max_output_tokens,
    repair_leaks,
)
from cuda_sft.knowledge.parse import (
    MIN_COT_CHARS,
    clean_raw_reasoning,
    sanitize_knowledge_cot,
)
from cuda_sft.knowledge.prompt import COT_SYSTEM, build_cot_user
from cuda_sft.knowledge.state import KnowledgeGraphState
from cuda_sft.llm import LLMClient, get_llm_client
from cuda_sft.parse import has_numbered_headings
from cuda_sft.runtime.meta import CallMeta, legacy_job_key

logger = logging.getLogger(__name__)

HEADING_COUNT = 5
# Default call budget when ``_run_agent`` is used without a refine budget.
AGENT_ATTEMPTS = 3


@dataclass
class KnowledgeCotResult:
    """Polished CoT for one quality-gated knowledge sample."""

    cot: str
    source: str
    raw_reasoning: str
    error: str = ""


def _clip(text: str, max_chars: int) -> str:
    """Cap ``text`` at a paragraph/sentence boundary when ``max_chars`` is positive.

    Args:
        text: CoT body.
        max_chars: ``<=0`` disables truncation.
    """
    return clip_at_boundary(text, max_chars)


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
        return self._client or get_llm_client(self.settings, role="cot_editor")

    def refine(self, state: KnowledgeGraphState) -> KnowledgeCotResult:
        """Never raises on LLM failure; falls back per kernel CoT env flags.

        Repaired answers follow ``COT_REPAIRED_POLICY``: ``drop_cot`` keeps no
        CoT; otherwise the editor reconstructs one without the repair-turn
        thinking (knowledge keeps no first-turn thinking to polish).
        """
        settings = self.settings
        question = str(state.get("question") or "")
        repaired = int(state.get("repair_idx") or 0) > 0
        if repaired and getattr(settings, "cot_repaired_policy", "synthetic") == "drop_cot":
            return KnowledgeCotResult(
                cot="", source="empty", raw_reasoning="", error="dropped_by_policy"
            )
        raw = clean_raw_reasoning(
            str(state.get("raw_reasoning") or ""),
            settings.cot_raw_max_chars,
        )
        if repaired:
            raw = ""
        if not settings.cot_agent_enabled:
            if raw.strip():
                return KnowledgeCotResult(
                    cot=_clip(raw, settings.cot_max_chars),
                    source="raw",
                    raw_reasoning=raw,
                )
            return KnowledgeCotResult(
                cot="",
                source="empty",
                raw_reasoning=raw,
                error="empty teacher thinking",
            )

        budget = EditorBudget(getattr(settings, "cot_max_calls", AGENT_ATTEMPTS))
        fail_mode = settings.cot_on_agent_fail
        reserve = 1 if raw.strip() and self._fallback_may_synthesize(fail_mode) else 0
        reserve = min(reserve, budget.limit - 1)
        feedback: list[str] = []
        problems: list[str] = []
        leaks: list[str] = []
        while budget.remaining > reserve:
            try:
                reply = self._run_agent(state, raw_reasoning=raw, feedback=feedback, budget=budget)
            except EditorBudgetExhausted:
                break
            except Exception as exc:
                logger.warning("knowledge CoT agent failed: %s", exc)
                return self._fallback(
                    state, raw, budget, error=str(exc), mode=fail_mode, llm_failed=True
                )
            cleaned = sanitize_knowledge_cot(reply, settings.cot_max_chars)
            problems = draft_problems(
                reply,
                cleaned,
                max_chars=settings.cot_max_chars,
                headings=HEADING_COUNT,
                min_chars=MIN_COT_CHARS,
            )
            if problems:
                logger.info("knowledge CoT draft rejected, retrying with feedback: %s", problems)
                feedback = problems
                continue
            leaks = self._leaks(cleaned, question)
            if not leaks:
                return KnowledgeCotResult(
                    cot=cleaned,
                    source="agent" if raw.strip() else "synthetic",
                    raw_reasoning=raw,
                )
            logger.info("knowledge CoT draft rejected, retrying with feedback: %s", leaks)
            feedback = leaks
        if problems:
            error = "agent output invalid: " + "; ".join(problems)
        elif leaks:
            error = "CoT repair-narration check failed: " + "; ".join(leaks)
        else:
            error = "CoT editor call budget exhausted"
        return self._fallback(state, raw, budget, error=error, mode=fail_mode)

    def _fallback_may_synthesize(self, mode: str) -> bool:
        chosen = (mode or "raw").strip().lower()
        if chosen == "synthetic":
            return True
        return chosen == "raw" and self.settings.cot_on_empty == "synthetic"

    def _leaks(self, cot: str, question: str) -> list[str]:
        if not getattr(self.settings, "cot_consistency_check", True):
            return []
        return repair_leaks(cot, question)

    def _valid(self, cot: str, question: str) -> bool:
        return (
            len(cot) >= MIN_COT_CHARS
            and has_numbered_headings(cot, HEADING_COUNT)
            and not self._leaks(cot, question)
        )

    def _fallback(
        self,
        state: KnowledgeGraphState,
        raw: str,
        budget: EditorBudget | None,
        *,
        error: str,
        mode: str,
        llm_failed: bool = False,
    ) -> KnowledgeCotResult:
        """Raw teacher thinking must look like an edited CoT; otherwise escalate."""
        question = str(state.get("question") or "")
        chosen = (mode or "raw").strip().lower()
        if chosen == "raw":
            if raw.strip():
                clipped = _clip(raw, self.settings.cot_max_chars)
                if self._valid(clipped, question):
                    return KnowledgeCotResult(
                        cot=clipped, source="raw", raw_reasoning=raw, error=error
                    )
                error = f"{error}; raw reasoning unstructured"
            chosen = (
                self.settings.cot_on_empty
                if self.settings.cot_on_empty != "raw"
                else "empty"
            )
        if chosen == "synthetic" and not llm_failed:
            try:
                reply = self._run_agent(state, raw_reasoning="", budget=budget)
                cleaned = sanitize_knowledge_cot(reply, self.settings.cot_max_chars)
            except Exception as exc:
                logger.warning("knowledge CoT synthetic rewrite failed: %s", exc)
                return KnowledgeCotResult(
                    cot="",
                    source="empty",
                    raw_reasoning=raw,
                    error=f"{error}; synthetic:{exc}",
                )
            if not self._valid(cleaned, question):
                return KnowledgeCotResult(
                    cot="",
                    source="empty",
                    raw_reasoning=raw,
                    error=f"{error}; synthetic invalid",
                )
            return KnowledgeCotResult(
                cot=cleaned, source="synthetic", raw_reasoning=raw, error=error
            )
        return KnowledgeCotResult(cot="", source="empty", raw_reasoning=raw, error=error)

    def _run_agent(
        self,
        state: KnowledgeGraphState,
        *,
        raw_reasoning: str,
        feedback: list[str] | None = None,
        budget: EditorBudget | None = None,
    ) -> str:
        """Call the editor once; visible text only (its own reasoning is not a CoT)."""
        settings = self.settings
        user = build_cot_user(
            question=str(state.get("question") or ""),
            answer=str(state.get("answer") or ""),
            raw_reasoning=raw_reasoning,
            topic=str(state.get("topic") or "general"),
            max_chars=settings.cot_max_chars,
        )
        if feedback:
            user += "\n\nThe previous draft violated these rules:\n" + "\n".join(
                f"- {issue}" for issue in feedback
            )
        qid = int(state.get("question_id") or 0)
        track = str(state.get("track") or "knowledge:general")
        meta = CallMeta(
            role="cot_editor",
            job_key=legacy_job_key(qid, track),
            question_id=qid,
            track=track,
            candidate=int(state.get("candidate_idx") or 1),
            repair=int(state.get("repair_idx") or 0),
            purpose="synthetic" if not raw_reasoning else "polish",
        )
        return call_editor(
            self._client_or_default(),
            system=COT_SYSTEM,
            user=user,
            temperature=float(settings.cot_temperature),
            thinking_level=settings.for_role("cot_editor").thinking_level,
            max_output_tokens=editor_max_output_tokens(
                settings, cap=settings.knowledge_max_output_tokens
            ),
            meta=meta,
            budget=budget or EditorBudget(AGENT_ATTEMPTS),
            label="knowledge CoT",
        )
