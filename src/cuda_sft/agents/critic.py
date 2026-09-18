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

## Source
```
{code}
```

JSON only."""


def should_run_critic(
    settings: Settings,
    heuristic: JudgeResult,
    *,
    use_critic: bool = True,
) -> bool:
    """Adaptive trigger: off / always / low heuristic score or issues.

    Args:
        settings: Pipeline settings (``KERNEL_LLM_CRITIC``).
        heuristic: Rule-based judge result already computed.
        use_critic: False when difficulty topology skipped the critic.
    """
    mode = (getattr(settings, "kernel_llm_critic", "adaptive") or "adaptive").strip().lower()
    if mode in {"off", "false", "0"}:
        return False
    if not use_critic:
        return False
    if mode == "always":
        return True
    if heuristic.quality_score < 8:
        return True
    return bool(heuristic.issues)


def parse_critic_json(text: str) -> dict[str, Any] | None:
    """Parse a critic JSON object from a model reply."""
    raw = (text or "").strip()
    if not raw:
        return None
    visible = strip_thinking(raw)
    tagged = extract_thinking(raw)
    for body in (visible, tagged, raw):
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
    ) -> CriticResult:
        """Return a critic decision. Never raises on LLM failure.

        Args:
            question: Raw problem text.
            code: Compile-passing source.
            dialect: Kernel dialect id.
            heuristic: Rule-based scores/issues.
            use_critic: Topology flag from :mod:`cuda_sft.agents.difficulty`.
        """
        if not should_run_critic(self.settings, heuristic, use_critic=use_critic):
            return CriticResult(passed=True, skipped=True)
        client = self._client or get_llm_client(self.settings)
        user = CRITIC_USER.format(
            question=(question or "").strip() or "(empty)",
            dialect=dialect or "cuda",
            score=heuristic.quality_score,
            issues="; ".join(heuristic.issues) or "(none)",
            suggestions="; ".join(heuristic.suggestions) or "(none)",
            code=(code or "").strip() or "(empty)",
        )
        try:
            stream_completion = getattr(client, "stream_completion", None)
            if callable(stream_completion):
                completion = stream_completion(
                    messages=[{"role": "user", "content": user}],
                    system=CRITIC_SYSTEM,
                    temperature=0.0,
                    print_stream=False,
                    thinking_level="none",
                    max_output_tokens=1024,
                )
                text = completion.text or completion.reasoning or ""
            else:
                text = client.stream_text(
                    messages=[{"role": "user", "content": user}],
                    system=CRITIC_SYSTEM,
                    temperature=0.0,
                    print_stream=False,
                )
        except Exception as exc:
            logger.warning("kernel critic failed: %s", exc)
            return CriticResult(passed=True, skipped=False, issues=[f"critic_error:{exc}"])
        payload = parse_critic_json(text) or {}
        must_fix = [
            item.strip()
            for item in payload.get("must_fix") or []
            if isinstance(item, str) and item.strip()
        ]
        issues = [
            item.strip()
            for item in payload.get("issues") or []
            if isinstance(item, str) and item.strip()
        ]
        passed = bool(payload.get("pass", not must_fix)) and not must_fix
        return CriticResult(
            passed=passed,
            must_fix=must_fix,
            issues=issues,
            skipped=False,
            raw=payload if isinstance(payload, dict) else {},
        )
