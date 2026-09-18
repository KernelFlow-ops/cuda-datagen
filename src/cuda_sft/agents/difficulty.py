"""Rule-based difficulty → candidate budget and whether to spend an LLM critic.

Inspired by difficulty-aware topology selection: easy elementwise kernels
should not pay a 3-candidate + critic tax. This is a regex plan, not a
learned router, so it stays deterministic and resume-stable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cuda_sft.config import Settings, get_settings

_HARD_KERNEL = re.compile(
    r"gemm|matmul|conv(?:olution)?|softmax|attention|scan|reduc(?:e|tion)|"
    r"transpose|sort|fft|wmma|tensor[ -]?core|cute|cutlass|histogram|"
    r"prefix|warp.?shuffle",
    re.IGNORECASE,
)
_SIMPLE_KERNEL = re.compile(
    r"\b(add|scale|relu|copy|fill|axpy|element[- ]?wise|vector add|saxpy)\b",
    re.IGNORECASE,
)
_HARD_TOPICS = {"formula", "cute", "cutlass", "isa"}


@dataclass(frozen=True)
class TopologyPlan:
    """Per-question candidate/critic budget.

    Attributes:
        difficulty: ``simple``, ``medium``, or ``hard``.
        max_candidates: Cap for this question (never above Settings.max_*).
        use_critic: Whether adaptive critic is allowed to run.
    """

    difficulty: str
    max_candidates: int
    use_critic: bool


def kernel_difficulty(question: str) -> str:
    """Classify a kernel question as simple / medium / hard.

    Args:
        question: Raw problem text.
    """
    text = question or ""
    if _HARD_KERNEL.search(text):
        return "hard"
    if _SIMPLE_KERNEL.search(text):
        return "simple"
    return "medium"


def knowledge_difficulty(topic: str) -> str:
    """Classify a knowledge topic. Formula/CuTe/ISA are treated as hard.

    Args:
        topic: Knowledge topic id.
    """
    name = (topic or "general").strip().lower()
    if name in _HARD_TOPICS:
        return "hard"
    if name in {"general", "worked_example"}:
        return "simple"
    return "medium"


def plan_topology(
    *,
    question: str,
    kind: str,
    topic: str = "",
    settings: Settings | None = None,
) -> TopologyPlan:
    """Return the candidate cap and critic eligibility for one job.

    Args:
        question: Raw problem text.
        kind: ``kernel`` or ``knowledge``.
        topic: Knowledge topic (ignored for kernels).
        settings: Pipeline settings; ``DIFFICULTY_AWARE=false`` returns full budget.
    """
    cfg = settings or get_settings()
    if kind == "knowledge":
        difficulty = knowledge_difficulty(topic)
        full = int(cfg.knowledge_max_candidates)
    else:
        difficulty = kernel_difficulty(question)
        full = int(cfg.max_candidates)
    if not getattr(cfg, "difficulty_aware", True):
        return TopologyPlan(difficulty=difficulty, max_candidates=full, use_critic=True)
    # Keep the full candidate budget. Compile-gated dialects (especially
    # TileLang) still fail easy problems; shrinking candidates drops recall.
    # The cost saving is skipping the LLM critic on simple questions.
    if difficulty == "simple":
        return TopologyPlan(difficulty=difficulty, max_candidates=full, use_critic=False)
    return TopologyPlan(difficulty=difficulty, max_candidates=full, use_critic=True)
