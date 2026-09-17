"""Hard gates plus optional LLM-as-judge for knowledge answers."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from cuda_sft.config import Settings, get_settings
from cuda_sft.knowledge.facts import fact_violations
from cuda_sft.knowledge.parse import (
    fence_char_ratio,
    has_derivation_steps,
    has_equation,
    looks_structured,
)
from cuda_sft.knowledge.prompt import JUDGE_SYSTEM, build_judge_user
from cuda_sft.knowledge.rubrics import (
    DIMENSIONS,
    clamp_score,
    overall_score,
    passes_threshold,
)
from cuda_sft.llm import LLMClient, LLMError, get_llm_client, is_retryable_llm_error
from cuda_sft.parse import extract_thinking, strip_thinking

logger = logging.getLogger(__name__)

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)
CODE_RATIO_MAX = 0.40
JUDGE_MAX_OUTPUT_TOKENS = 4096
WORKED_EXAMPLE_TOPICS = {"worked_example"}
DIMENSION_ALIASES = {
    "factual": ("factual", "factual_correctness", "facts", "accuracy"),
    "completeness": ("completeness", "coverage", "complete"),
    "derivation": ("derivation", "derivation_quality", "reasoning", "math"),
    "terminology": ("terminology", "terms", "vocab"),
    "structure": ("structure", "organization", "format"),
    "grounding": ("grounding", "assumptions", "caveats", "scope"),
}


@dataclass
class KnowledgeJudgeResult:
    """Quality report for one knowledge answer.

    Attributes:
        passed: True if the sample may be saved.
        overall: Weighted 1-10 score (0 if judge skipped after a hard-gate fail).
        dimensions: Per-axis 1-10 scores.
        must_fix: Blocking issues.
        issues: Non-blocking nits (hard-gate reasons land here too).
        hard_gate_failed: True when a deterministic gate fired.
        hard_gate_reasons: Gate messages.
        judge_error: LLM/JSON failure; never save when this is set after retries.
        judge_unavailable: True when the LLM judge could not produce JSON.
        skipped_llm: True when ``KNOWLEDGE_JUDGE_ENABLED`` is false.
    """

    passed: bool
    overall: float = 0.0
    dimensions: dict[str, float] = field(default_factory=dict)
    must_fix: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    hard_gate_failed: bool = False
    hard_gate_reasons: list[str] = field(default_factory=list)
    judge_error: str = ""
    judge_unavailable: bool = False
    skipped_llm: bool = False


def hard_gate(
    answer: str,
    *,
    topic: str,
    question: str = "",
    min_chars: int = 400,
    require_structure: bool = True,
) -> list[str]:
    """Return reasons the answer must not be saved yet. Empty means the gate passed."""
    reasons: list[str] = []
    text = answer or ""
    topic_name = (topic or "general").strip().lower()
    if len(text.strip()) < int(min_chars):
        reasons.append(f"answer shorter than {min_chars} characters")
    ratio = fence_char_ratio(text)
    if ratio > CODE_RATIO_MAX and topic_name not in WORKED_EXAMPLE_TOPICS:
        reasons.append(
            f"markdown-fence ratio {ratio:.2f} exceeds {CODE_RATIO_MAX:.2f} on a theory topic"
        )
    if require_structure and text.strip() and not looks_structured(text):
        reasons.append("missing headings or listed structure")
    if topic_name == "formula" and text.strip() and not has_equation(text):
        reasons.append("formula topic requires at least one equation")
    derivation_asked = topic_name == "formula" or bool(
        re.search(r"推导|derive|why does|为什么", question or "", re.IGNORECASE)
    )
    if derivation_asked and text.strip() and not has_derivation_steps(text):
        reasons.append("derivation/why question lacks step-by-step reasoning")
    reasons.extend(fact_violations(text))
    return reasons


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` / ``]`` so slightly-invalid JSON still loads."""
    prev = text
    for _ in range(8):
        nxt = re.sub(r",\s*([}\]])", r"\1", prev)
        if nxt == prev:
            return nxt
        prev = nxt
    return prev


def _balanced_json_objects(text: str) -> list[str]:
    """Return brace-balanced ``{...}`` slices, last object first (final JSON after CoT)."""
    found: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        escape = False
        end = None
        for j in range(i, n):
            ch = text[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end is None:
            break
        found.append(text[i : end + 1])
        i = end + 1
    found.reverse()
    return found


def _normalize_judge_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Accept ``dimensions`` or ``scores``; return a dict with ``dimensions``."""
    if "dimensions" not in payload and isinstance(payload.get("scores"), dict):
        payload = dict(payload)
        payload["dimensions"] = payload["scores"]
    if "dimensions" not in payload and "scores" not in payload:
        return None
    return payload


def _loads_judge_dict(blob: str) -> dict[str, Any] | None:
    for item in (blob, _strip_trailing_commas(blob)):
        try:
            payload = json.loads(item)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return _normalize_judge_payload(payload)
    return None


def parse_judge_json(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model reply (raw, fenced, CoT+JSON, think tags)."""
    raw = (text or "").strip()
    if not raw:
        return None
    visible = strip_thinking(raw)
    tagged = extract_thinking(raw)
    parts = [visible, tagged, raw]
    candidates: list[str] = []
    seen: set[str] = set()
    for part in parts:
        body = (part or "").strip()
        if not body or body in seen:
            continue
        seen.add(body)
        candidates.append(body)
        for fenced in JSON_FENCE_RE.findall(body):
            inner = (fenced or "").strip()
            if inner and inner not in seen:
                seen.add(inner)
                candidates.append(inner)
        for obj in _balanced_json_objects(body):
            if obj not in seen:
                seen.add(obj)
                candidates.append(obj)
    for item in candidates:
        payload = _loads_judge_dict(item)
        if payload is not None:
            return payload
    return None


def _dimensions_from_payload(payload: dict[str, Any]) -> dict[str, float] | None:
    dims = payload.get("dimensions")
    if not isinstance(dims, dict):
        dims = payload.get("scores")
    if not isinstance(dims, dict):
        return None
    lowered = {str(key).strip().lower(): value for key, value in dims.items()}
    out: dict[str, float] = {}
    for key in DIMENSIONS:
        value = None
        for alias in DIMENSION_ALIASES.get(key, (key,)):
            if alias in lowered:
                value = lowered[alias]
                break
        if value is None:
            if key == "derivation":
                out[key] = 10.0
                continue
            return None
        out[key] = clamp_score(value)
    return out


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


class KnowledgeJudge:
    """Hard gate first; optional LLM JSON grader with internal retry."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = llm_client

    def _client_or_default(self) -> LLMClient:
        return self._client or get_llm_client(self.settings)

    def judge(
        self,
        *,
        question: str,
        answer: str,
        topic: str,
    ) -> KnowledgeJudgeResult:
        """Run gates then the LLM judge. Never raises on LLM failure."""
        settings = self.settings
        reasons = hard_gate(
            answer,
            topic=topic,
            question=question,
            min_chars=settings.knowledge_min_answer_chars,
            require_structure=settings.knowledge_require_structure,
        )
        if reasons:
            return KnowledgeJudgeResult(
                passed=False,
                overall=0.0,
                must_fix=list(reasons),
                issues=list(reasons),
                hard_gate_failed=True,
                hard_gate_reasons=list(reasons),
            )

        if not settings.knowledge_judge_enabled:
            return KnowledgeJudgeResult(
                passed=True,
                overall=0.0,
                skipped_llm=True,
                issues=["llm judge disabled; hard gate passed"],
            )

        attempts = 2 if settings.knowledge_on_judge_fail == "retry" else 1
        last_error = "judge returned no JSON"
        payload: dict[str, Any] | None = None
        for attempt in range(1, attempts + 1):
            try:
                text = self._run_llm(question=question, answer=answer, topic=topic)
            except Exception as exc:
                last_error = str(exc)
                logger.warning("knowledge judge LLM attempt %s/%s: %s", attempt, attempts, exc)
                retry = attempt < attempts and (
                    isinstance(exc, LLMError) or is_retryable_llm_error(exc)
                )
                if retry:
                    continue
                break
            payload = parse_judge_json(text)
            if payload is not None:
                break
            snippet = (text or "").replace("\n", " ")[:400]
            last_error = "judge output was not valid JSON with dimensions"
            logger.warning(
                "knowledge judge JSON parse failed (attempt %s/%s) snippet=%r",
                attempt,
                attempts,
                snippet,
            )

        if payload is None:
            return KnowledgeJudgeResult(
                passed=False,
                judge_error=last_error,
                judge_unavailable=True,
            )

        dimensions = _dimensions_from_payload(payload)
        if dimensions is None:
            return KnowledgeJudgeResult(
                passed=False,
                judge_error="judge JSON missing required dimension keys",
                judge_unavailable=True,
            )

        must_fix = _string_list(payload.get("must_fix"))
        issues = _string_list(payload.get("issues"))
        overall = overall_score(dimensions, topic)
        passed = passes_threshold(
            overall=overall,
            dimensions=dimensions,
            must_fix=must_fix,
            min_score=settings.knowledge_min_score,
            factual_min=settings.knowledge_factual_min,
        )
        return KnowledgeJudgeResult(
            passed=passed,
            overall=overall,
            dimensions=dimensions,
            must_fix=must_fix,
            issues=issues,
        )

    def _run_llm(self, *, question: str, answer: str, topic: str) -> str:
        client = self._client_or_default()
        user = build_judge_user(question=question, answer=answer, topic=topic)
        stream_completion = getattr(client, "stream_completion", None)
        base = {
            "messages": [{"role": "user", "content": user}],
            "system": JUDGE_SYSTEM,
            "temperature": 0.2,
            "print_stream": False,
        }
        if callable(stream_completion):
            completion = stream_completion(
                **base,
                max_output_tokens=min(
                    JUDGE_MAX_OUTPUT_TOKENS,
                    int(self.settings.knowledge_max_output_tokens),
                ),
                thinking_level="none",
            )
            visible = completion.text or ""
            reasoning = completion.reasoning or ""
            if parse_judge_json(visible) is not None:
                return visible
            if parse_judge_json(reasoning) is not None:
                return reasoning
            return "\n".join(part for part in (visible, reasoning) if part.strip())
        return client.stream_text(**base)
