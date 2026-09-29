"""CoT Agent: polish teacher reasoning into SFT chain-of-thought."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cuda_sft.config import Settings, get_settings
from cuda_sft.core.cot import (
    EditorBudget,
    EditorBudgetExhausted,
    call_editor,
    clip_at_boundary,
    cot_consistency_report,
    draft_problems,
    editor_max_output_tokens,
)
from cuda_sft.core.cot import clean_raw_reasoning as _clean_raw_reasoning
from cuda_sft.llm import LLMClient, get_llm_client
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
from cuda_sft.runtime.meta import CallMeta, legacy_job_key
from cuda_sft.state import GraphState

logger = logging.getLogger(__name__)

MIN_COT_CHARS = 120
HEADING_COUNT = 6
# Default call budget when ``_run_agent`` is used without a refine budget.
AGENT_ATTEMPTS = 3


@dataclass
class CotResult:
    """Polished CoT for one compile-passing sample.

    Attributes:
        cot: Text written into the SFT ``<think>`` block (may be empty).
        source: ``agent``, ``raw``, ``synthetic``, or ``empty``.
        raw_reasoning: Cleaned teacher thinking used as input.
        error: Failure reason when falling back.
        consistency_issues: Blocking issues that forced a fallback.
        soft_issues: Non-blocking notes (e.g. names absent from the code).
    """

    cot: str
    source: str
    raw_reasoning: str
    error: str = ""
    consistency_issues: list[str] = field(default_factory=list)
    soft_issues: list[str] = field(default_factory=list)


def clean_raw_reasoning(text: str, max_chars: int) -> str:
    """Normalize teacher thinking before storing or sending to the agent.

    Args:
        text: Raw API reasoning, possibly tagged or fenced.
        max_chars: Budget; ``<=0`` disables. Keeps the head and the tail.

    Returns:
        Prose-only thinking, possibly empty.
    """
    return _clean_raw_reasoning(text, max_chars, strip_code=True)


def sanitize_cot_output(text: str, max_chars: int) -> str:
    """Drop think wrappers and code from an agent reply.

    Args:
        text: Agent visible text.
        max_chars: Cap on the polished CoT; cut at a paragraph/sentence end.

    Returns:
        Clean CoT prose, possibly empty.
    """
    raw = text or ""
    tagged = extract_thinking(raw)
    body = tagged if tagged.strip() else raw
    body = strip_code_from_cot(body)
    body = collapse_blank_lines(body)
    return clip_at_boundary(body, max_chars)


def _clip(text: str, max_chars: int) -> str:
    """Cap ``text`` at a paragraph/sentence boundary when ``max_chars`` is positive."""
    return clip_at_boundary(text, max_chars)


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
        return self._client or get_llm_client(self.settings, role="cot_editor")

    def refine(self, state: GraphState) -> CotResult:
        """Produce CoT for the selected, compile-passing candidate.

        Never raises on LLM failure; falls back per ``COT_ON_AGENT_FAIL``.
        Every editor call (drafts, feedback retries, the synthetic fallback and
        transport retries) comes out of one ``COT_MAX_CALLS`` budget.

        Args:
            state: Winning state with ``selected`` and ``cot_mode``.
        """
        selected = state.get("selected")
        if not isinstance(selected, dict) or not selected:
            raise RuntimeError("CotAgent.refine requires state['selected']")
        cot_mode = str(state.get("cot_mode") or "polish")
        if cot_mode not in {"drop", "synthetic", "polish"}:
            raise ValueError(f"unknown cot_mode: {cot_mode}")
        if cot_mode == "drop":
            return CotResult(cot="", source="empty", raw_reasoning="", error="dropped_by_policy")

        settings = self.settings
        question = str(state.get("question") or "")
        code = str(selected.get("code") or "")
        raw = clean_raw_reasoning(
            "" if cot_mode == "synthetic" else str(state.get("raw_reasoning") or ""),
            settings.cot_raw_max_chars,
        )
        if not settings.cot_agent_enabled:
            if cot_mode == "synthetic":
                return CotResult(cot="", source="empty", raw_reasoning="", error="agent_disabled")
            if raw.strip():
                # Raw teacher thinking is the configured output here, so it is
                # not held to the editor's heading structure.
                return self._raw_result(raw, code, error="", question=question)
            return self._fallback(
                state, raw, None, error="empty teacher thinking", mode=settings.cot_on_empty
            )

        budget = EditorBudget(getattr(settings, "cot_max_calls", AGENT_ATTEMPTS))
        fail_mode = "empty" if cot_mode == "synthetic" else settings.cot_on_agent_fail
        # Keep one call for the synthetic fallback when a polish miss can reach it.
        reserve = 1 if cot_mode == "polish" and self._fallback_may_synthesize(fail_mode) else 0
        reserve = min(reserve, budget.limit - 1)
        feedback: list[str] = []
        problems: list[str] = []
        hard: list[str] = []
        while budget.remaining > reserve:
            try:
                reply = self._run_agent(state, raw_reasoning=raw, feedback=feedback, budget=budget)
            except EditorBudgetExhausted:
                break
            except Exception as exc:
                logger.warning("CoT agent failed: %s", exc)
                # The editor is unavailable: a synthetic rewrite would hit the
                # same outage, so only raw/empty remain.
                return self._fallback(
                    state, raw, budget, error=str(exc), mode=fail_mode,
                    consistency_issues=hard, llm_failed=True,
                )
            cleaned = sanitize_cot_output(reply, settings.cot_max_chars)
            problems = draft_problems(
                reply,
                cleaned,
                max_chars=settings.cot_max_chars,
                headings=HEADING_COUNT,
                min_chars=MIN_COT_CHARS,
            )
            if problems:
                logger.info("CoT draft rejected, retrying with feedback: %s", problems)
                feedback, hard = problems, []
                continue
            hard, soft = self._consistency_report(cleaned, code, question)
            if not hard:
                return CotResult(
                    cot=cleaned,
                    # Polish with empty teacher thinking is a reconstruction from code.
                    source="synthetic" if cot_mode == "synthetic" or not raw.strip() else "agent",
                    raw_reasoning=raw,
                    soft_issues=soft,
                )
            logger.info("CoT draft inconsistent, retrying with feedback: %s", hard)
            feedback = [*hard, *soft]
        if problems:
            error = "agent output invalid: " + "; ".join(problems)
        elif hard:
            error = "CoT consistency check failed"
        else:
            error = "CoT editor call budget exhausted"
        return self._fallback(
            state, raw, budget, error=error, mode=fail_mode, consistency_issues=hard
        )

    def _fallback_may_synthesize(self, mode: str) -> bool:
        chosen = (mode or "raw").strip().lower()
        if chosen == "synthetic":
            return True
        return chosen == "raw" and self.settings.cot_on_empty == "synthetic"

    @staticmethod
    def _valid_draft(cot: str) -> bool:
        return len(cot) >= MIN_COT_CHARS and has_numbered_headings(cot, HEADING_COUNT)

    def _consistency_report(
        self, cot: str, code: str, question: str = ""
    ) -> tuple[list[str], list[str]]:
        """Return ``(hard, soft)`` issues; both empty when checking is disabled."""
        if not getattr(self.settings, "cot_consistency_check", True):
            return [], []
        return cot_consistency_report(cot, code, question)

    def _consistency_issues(self, cot: str, code: str, question: str = "") -> list[str]:
        return self._consistency_report(cot, code, question)[0]

    def _raw_result(
        self,
        raw: str,
        code: str,
        *,
        error: str,
        issues: list[str] | None = None,
        question: str = "",
        require_structure: bool = False,
    ) -> CotResult:
        """Use cleaned teacher thinking as the CoT, or reject it with a reason.

        After an editor miss the raw text must have the same heading structure
        as an edited CoT; otherwise unstructured monologue would be mixed into
        the training set next to edited samples.
        """
        clipped = _clip(raw, self.settings.cot_max_chars)
        raw_issues = self._consistency_issues(clipped, code, question)
        all_issues = list(dict.fromkeys([*(issues or []), *raw_issues]))
        if require_structure and not self._valid_draft(clipped):
            return CotResult(
                cot="",
                source="empty",
                raw_reasoning=raw,
                error=f"{error}; raw reasoning unstructured".strip("; "),
                consistency_issues=all_issues,
            )
        if raw_issues:
            return CotResult(
                cot="",
                source="empty",
                raw_reasoning=raw,
                error=f"{error}; raw reasoning inconsistent".strip("; "),
                consistency_issues=all_issues,
            )
        return CotResult(
            cot=clipped,
            source="raw",
            raw_reasoning=raw,
            error=error,
            consistency_issues=all_issues,
        )

    def _fallback(
        self,
        state: GraphState,
        raw: str,
        budget: EditorBudget | None,
        *,
        error: str,
        mode: str,
        consistency_issues: list[str] | None = None,
        llm_failed: bool = False,
    ) -> CotResult:
        """Apply ``raw`` / ``synthetic`` / ``empty`` after an agent miss.

        A rejected raw fallback escalates to ``COT_ON_EMPTY`` (typically
        ``synthetic``) instead of ending with no CoT, unless the editor itself
        was unavailable.
        """
        chosen = (mode or "raw").strip().lower()
        code = str((state.get("selected") or {}).get("code") or "")
        question = str(state.get("question") or "")
        issues = list(consistency_issues or [])
        if chosen == "raw":
            if raw.strip():
                result = self._raw_result(
                    raw,
                    code,
                    error=error,
                    issues=issues,
                    question=question,
                    require_structure=self.settings.cot_agent_enabled,
                )
                if result.cot or llm_failed:
                    return result
                error, issues = result.error, list(result.consistency_issues)
            chosen = self.settings.cot_on_empty if self.settings.cot_on_empty != "raw" else "empty"
        if chosen == "synthetic" and not llm_failed:
            return self._synthetic(state, budget, error=error, issues=issues)
        return CotResult(
            cot="",
            source="empty",
            raw_reasoning=raw,
            error=error,
            consistency_issues=issues,
        )

    def _synthetic(
        self,
        state: GraphState,
        budget: EditorBudget | None,
        *,
        error: str,
        issues: list[str],
    ) -> CotResult:
        """One reconstruction from problem + code, without teacher thinking."""
        if not self.settings.cot_agent_enabled:
            return CotResult(
                cot="", source="empty", raw_reasoning="", error=f"{error}; agent_disabled",
                consistency_issues=issues,
            )
        code = str((state.get("selected") or {}).get("code") or "")
        question = str(state.get("question") or "")
        try:
            reply = self._run_agent(state, raw_reasoning="", budget=budget)
            cleaned = sanitize_cot_output(reply, self.settings.cot_max_chars)
        except Exception as exc:
            logger.warning("CoT synthetic rewrite failed: %s", exc)
            return CotResult(
                cot="", source="empty", raw_reasoning="", error=f"{error}; synthetic:{exc}",
                consistency_issues=issues,
            )
        synthetic_issues, soft = self._consistency_report(cleaned, code, question)
        if not self._valid_draft(cleaned) or synthetic_issues:
            return CotResult(
                cot="",
                source="empty",
                raw_reasoning="",
                error=f"{error}; synthetic invalid",
                consistency_issues=list(dict.fromkeys([*issues, *synthetic_issues])),
            )
        return CotResult(
            cot=cleaned,
            source="synthetic",
            raw_reasoning="",
            error=error,
            consistency_issues=issues,
            soft_issues=soft,
        )

    def _skeleton(self, dialect: str, question: str) -> str:
        """Dialect headings in the problem's language."""
        lang = "zh" if looks_chinese(question) else "en"
        try:
            from cuda_sft.dialects.agent import get_spec

            return get_spec(dialect).cot_skeleton(lang)
        except Exception:
            if lang != "zh":
                return ""
            return COT_SKELETON_PY_ZH if dialect in {"triton", "tilelang"} else COT_SKELETON_ZH

    def _run_agent(
        self,
        state: GraphState,
        *,
        raw_reasoning: str,
        feedback: list[str] | None = None,
        budget: EditorBudget | None = None,
    ) -> str:
        """Call the editor model once (transport retries come out of ``budget``)."""
        settings = self.settings
        dialect = str(state.get("dialect") or "cuda")
        question = str(state.get("question") or "")
        language = "python" if dialect in {"triton", "tilelang"} else "cuda-cpp"
        fence = "python" if language == "python" else "cuda"
        selected = state.get("selected") or {}
        code = str(selected["code"]) if "code" in selected else str(state.get("code") or "")
        user = build_cot_user_prompt(
            question=question,
            code=code,
            raw_reasoning=raw_reasoning,
            max_chars=settings.cot_max_chars,
            dialect=dialect,
            language=language,
            skeleton=self._skeleton(dialect, question),
            fence=fence,
        )
        if feedback:
            user += "\n\nThe previous draft violated these rules:\n" + "\n".join(
                f"- {issue}" for issue in feedback
            )
        qid = int(state.get("question_id") or 0)
        meta = CallMeta(
            role="cot_editor",
            job_key=legacy_job_key(qid, dialect),
            question_id=qid,
            track=dialect,
            candidate=int(selected.get("candidate", state.get("candidate_idx") or 1)),
            repair=int(selected.get("repairs", state.get("repair_idx") or 0)),
            purpose="synthetic" if not raw_reasoning else "polish",
        )
        return call_editor(
            self._client_or_default(),
            system=cot_system_for(dialect=dialect, question=question),
            user=user,
            temperature=float(settings.cot_temperature),
            thinking_level=settings.for_role("cot_editor").thinking_level,
            max_output_tokens=editor_max_output_tokens(settings),
            meta=meta,
            budget=budget or EditorBudget(AGENT_ATTEMPTS),
            label="CoT agent",
        )
