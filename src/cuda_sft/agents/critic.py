"""Adaptive semantic critic for compile-passing kernels.

The compiler is the hard gate. This critic only looks for signature /
algorithm / bounds mistakes that still compile. Adaptive mode skips the
LLM when the heuristic judge is clean (score >= 8 and no issues).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from cuda_sft.agents.contracts import CriticResult
from cuda_sft.config import Settings, get_settings
from cuda_sft.judge import JudgeResult
from cuda_sft.llm import LLMClient, get_llm_client
from cuda_sft.parse import extract_thinking, strip_thinking
from cuda_sft.runtime.meta import CallMeta

logger = logging.getLogger(__name__)

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL | re.IGNORECASE)

CRITIC_SYSTEM = """You are a CUDA kernel semantic checker, not a style linter.
The source already compiled. Return ONE JSON object and nothing else:
  "pass": boolean
  "must_fix": array of blocking semantic errors (empty if none)
  "issues": array of non-blocking nits
must_fix only for: host/API mismatch vs the question, wrong algorithm,
missing bounds that will miscompute, or the wrong kernel dialect.
Do not demand micro-optimizations. If the code is a plausible solution, pass=true.
"""

CRITIC_USER = """## Question
{question}

## Dialect
{dialect}

## Heuristic notes
score={score}; issues={issues}; suggestions={suggestions}

## GPU reference validation
{refval_block}

## Source
```
{code}
```

JSON only."""


def _annotate_result(
    result: CriticResult,
    *,
    status: str,
    quality_score: int | None = None,
    failure: dict[str, str] | None = None,
) -> CriticResult:
    """Attach stable, optional metadata without widening GraphState.

    ``CriticResult`` is a shared contract used by older callers, so its
    constructor remains backwards compatible.  The attributes below are
    deliberately plain JSON values; :func:`critic_result_to_dict` is the
    canonical persistence boundary for stores and benchmark reports.
    """
    result.status = status  # type: ignore[attr-defined]
    result.quality_score = quality_score  # type: ignore[attr-defined]
    result.failure = failure  # type: ignore[attr-defined]
    return result


def critic_result_to_dict(result: CriticResult) -> dict[str, Any]:
    """Return a JSON-safe critic record, including verification status.

    A critic that was requested but could not produce a valid answer is
    ``unverified``.  This distinction is important for downstream quality
    analysis: ``passed=False`` alone cannot tell a semantic rejection from an
    unavailable reviewer.
    """
    status = str(
        getattr(result, "status", "skipped" if result.skipped else
                ("verified" if result.passed else "failed"))
    )
    failure = getattr(result, "failure", None)
    raw = result.raw if isinstance(result.raw, dict) else {}
    # Payload values originate in json.loads, but copying through a JSON
    # round-trip protects callers that construct CriticResult manually.
    try:
        raw = json.loads(json.dumps(raw, ensure_ascii=False))
    except (TypeError, ValueError):
        raw = {"serialization_error": "critic raw payload was not JSON-safe"}
    return {
        "passed": bool(result.passed),
        "status": status,
        "skipped": bool(result.skipped),
        "must_fix": [str(item) for item in result.must_fix],
        "issues": [str(item) for item in result.issues],
        "quality_score": getattr(result, "quality_score", None),
        "failure": failure if isinstance(failure, dict) else None,
        "raw": raw,
    }


def should_run_critic(
    settings: Settings,
    heuristic: JudgeResult,
    *,
    use_critic: bool = True,
    refval_status: str = "",
    refval_error_class: str = "",
) -> bool:
    """Adaptive trigger: off / always / low heuristic score or issues.

    Numeric-validation failures force the critic so a compile-passing but
    wrong kernel cannot skip it via score>=8 with empty issues. When
    refval already passed, the critic must not re-do numeric judgment.

    Args:
        settings: Pipeline settings (``KERNEL_LLM_CRITIC``).
        heuristic: Rule-based judge result already computed.
        use_critic: False when difficulty topology skipped the critic.
        refval_status: ``pass`` / ``fail`` / ``skip`` / ``reference_error``.
        refval_error_class: Class from the numeric gate, if any.
    """
    mode = (getattr(settings, "kernel_llm_critic", "adaptive") or "adaptive").strip().lower()
    if mode in {"off", "false", "0"}:
        return False
    if (refval_status or "").strip().lower() == "fail" or (
        refval_error_class or ""
    ).strip().lower() in {"numeric_mismatch", "nan_inf"}:
        return True
    if not use_critic:
        return False
    if mode == "always":
        return True
    if heuristic.quality_score < 8:
        return True
    return bool(heuristic.issues)


def parse_critic_json(text: str, reasoning: str = "") -> dict[str, Any] | None:
    """Parse the decision from either visible output or reasoning channels."""
    bodies: list[str] = []
    for raw in (text or "", reasoning or ""):
        raw = raw.strip()
        if raw:
            bodies.extend((strip_thinking(raw), extract_thinking(raw), raw))
    if text and reasoning:
        bodies.append(f"{text}\n{reasoning}")
    seen: set[str] = set()
    for body in bodies:
        if body in seen:
            continue
        seen.add(body)
        if not body:
            continue
        for blob in JSON_FENCE_RE.findall(body) + [body]:
            try:
                payload = json.loads(blob)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and ("pass" in payload or "must_fix" in payload):
                return payload
    return None


class KernelCritic:
    """Optional LLM critic used after a compile-passing heuristic judge."""

    def __init__(
        self,
        settings: Settings | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = llm_client

    def evaluate(
        self,
        *,
        question: str,
        code: str,
        dialect: str,
        heuristic: JudgeResult,
        use_critic: bool = True,
        refval_status: str = "",
        refval_error_class: str = "",
        refval_summary: str = "",
        meta: CallMeta | None = None,
    ) -> CriticResult:
        """Return a critic decision. Never raises on LLM failure.

        Args:
            question: Raw problem text.
            code: Compile-passing source.
            dialect: Kernel dialect id.
            heuristic: Rule-based scores/issues.
            use_critic: Topology flag from :mod:`cuda_sft.agents.difficulty`.
            refval_status: Numeric-gate status (``pass`` skips numeric nits).
            refval_error_class: Numeric-gate class, if any.
            refval_summary: Compact evidence already computed on GPU.
        """
        if not should_run_critic(
            self.settings,
            heuristic,
            use_critic=use_critic,
            refval_status=refval_status,
            refval_error_class=refval_error_class,
        ):
            return _annotate_result(
                CriticResult(passed=True, skipped=True),
                status="skipped",
                quality_score=int(heuristic.quality_score),
            )
        if (refval_status or "").strip().lower() == "pass":
            refval_block = (
                "status=pass. Numeric correctness was already checked against a "
                "CPU reference on GPU. Do NOT re-judge element values, NaNs, or "
                "tolerances. Only flag host/API mismatch vs the question, the "
                "wrong algorithm, or the wrong dialect."
            )
        elif refval_summary:
            refval_block = (
                f"status={refval_status or 'unknown'} class={refval_error_class or '-'}\n"
                f"{refval_summary}\n"
                "Treat the GPU report as authoritative for numbers; do not redo "
                "the arithmetic. Focus on why the kernel disagrees with the question."
            )
        else:
            refval_block = (
                f"status={refval_status or 'skip'} class={refval_error_class or '-'}. "
                "No GPU numeric report; do not invent a numerical verdict."
            )
        user = CRITIC_USER.format(
            question=(question or "").strip() or "(empty)",
            dialect=dialect or "cuda",
            score=heuristic.quality_score,
            issues="; ".join(heuristic.issues) or "(none)",
            suggestions="; ".join(heuristic.suggestions) or "(none)",
            refval_block=refval_block,
            code=(code or "").strip() or "(empty)",
        )
        client = self._client
        retries = max(0, int(getattr(self.settings, "critic_retry_on_error", 1)))
        failure_kind = "invalid_response"
        failure_message = "critic did not return a valid JSON decision"
        failure_issues = ["critic_invalid_response"]
        failure_raw: dict[str, Any] = {}
        for attempt in range(1, retries + 2):
            request_meta = meta.with_attempt(attempt) if meta else None
            try:
                client = client or get_llm_client(self.settings, role="critic")
                stream_completion = getattr(client, "stream_completion", None)
                if callable(stream_completion):
                    completion = stream_completion(
                        messages=[{"role": "user", "content": user}],
                        system=CRITIC_SYSTEM,
                        temperature=0.0,
                        print_stream=False,
                        thinking_level="low",
                        max_output_tokens=4096,
                        meta=request_meta,
                    )
                    text = completion.text or ""
                    reasoning = completion.reasoning or ""
                else:
                    text = client.stream_text(
                        messages=[{"role": "user", "content": user}],
                        system=CRITIC_SYSTEM,
                        temperature=0.0,
                        print_stream=False,
                        meta=request_meta,
                    )
                    reasoning = ""
                payload = parse_critic_json(text, reasoning)
            except Exception as exc:
                failure_message = str(exc).strip() or exc.__class__.__name__
                failure_kind = "critic_error"
                failure_issues = [f"critic_error:{failure_message}"]
                failure_raw = {}
                logger.warning("kernel critic attempt %s/%s failed: %s", attempt, retries + 1, exc)
                continue

            if payload is None:
                failure_kind = "invalid_response"
                failure_message = "critic did not return a JSON object with pass/must_fix"
                failure_issues = ["critic_invalid_response"]
                failure_raw = {}
                continue

            must_fix_raw = payload.get("must_fix")
            issues_raw = payload.get("issues")
            if (must_fix_raw is not None and not isinstance(must_fix_raw, list)) or (
                issues_raw is not None and not isinstance(issues_raw, list)
            ):
                failure_kind = "invalid_response"
                failure_message = "critic must_fix and issues must be arrays"
                failure_issues = ["critic_invalid_response"]
                failure_raw = payload
                continue
            must_fix = [
                item.strip()
                for item in must_fix_raw or []
                if isinstance(item, str) and item.strip()
            ]
            issues = [
                item.strip()
                for item in issues_raw or []
                if isinstance(item, str) and item.strip()
            ]
            pass_value = payload.get("pass")
            if must_fix:
                passed = False
                status = "failed"
                failure = {"kind": "must_fix", "message": "critic reported blocking issues"}
            elif isinstance(pass_value, bool):
                passed = pass_value
                status = "verified" if passed else "failed"
                failure = None if passed else {
                    "kind": "critic_rejected",
                    "message": "critic returned pass=false",
                }
            else:
                failure_kind = "invalid_response"
                failure_message = "critic pass must be a boolean"
                failure_issues = [*issues, "critic_invalid_response"]
                failure_raw = payload
                continue
            return _annotate_result(
                CriticResult(
                    passed=passed,
                    must_fix=must_fix,
                    issues=issues,
                    skipped=False,
                    raw=payload,
                ),
                status=status,
                quality_score=int(heuristic.quality_score),
                failure=failure,
            )

        return _annotate_result(
            CriticResult(
                passed=False,
                skipped=False,
                issues=failure_issues,
                raw=failure_raw,
            ),
            status="unverified",
            quality_score=int(heuristic.quality_score),
            failure={"kind": failure_kind, "message": failure_message},
        )
