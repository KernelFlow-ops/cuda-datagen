"""Rule-based difficulty → candidate/repair budgets and critic policy.

Simple, medium, and hard jobs receive 2, 3, and 4 candidate slots. Repair
rounds scale with the same ordering (1, 2, and 3). Settings remain a hard
ceiling, so operators can lower cost without changing the classifier.
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

# Policy values are kept separate from Settings defaults. This makes the
# topology auditable and keeps a benchmark stable when global limits change.
_CANDIDATE_BUDGETS = {"simple": 2, "medium": 3, "hard": 4}
_REPAIR_BUDGETS = {"simple": 1, "medium": 2, "hard": 3}


@dataclass(frozen=True)
class TopologyPlan:
    """Per-question candidate/repair/critic budget.

    Attributes:
        difficulty: ``simple``, ``medium``, or ``hard``.
        max_candidates: Cap for this question (never above Settings.max_*).
        max_repairs: Repair rounds per candidate (never above Settings.max_*).
        use_critic: Whether adaptive critic is allowed to run.
    """

    difficulty: str
    max_candidates: int
    use_critic: bool
    # Default keeps the three-field constructor source-compatible for callers
    # written before per-difficulty repair budgets were introduced.
    max_repairs: int = 0

    @property
    def repair_budget(self) -> int:
        """Compatibility alias for callers that call this value a budget."""
        return self.max_repairs


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
        repair_full = int(cfg.knowledge_max_repairs)
    else:
        difficulty = kernel_difficulty(question)
        full = int(cfg.max_candidates)
        repair_full = int(cfg.max_repairs)
    if not getattr(cfg, "difficulty_aware", True):
        return TopologyPlan(
            difficulty=difficulty,
            max_candidates=full,
            max_repairs=repair_full,
            use_critic=True,
        )
    # Apply configured limits after policy limits: lowering MAX_* always lowers
    # spend, while a sufficiently high ceiling exposes the 2/3/4 and 1/2/3
    # difficulty ladder.
    return TopologyPlan(
        difficulty=difficulty,
        max_candidates=max(1, min(full, _CANDIDATE_BUDGETS[difficulty])),
        max_repairs=max(0, min(repair_full, _REPAIR_BUDGETS[difficulty])),
        # Simple jobs avoid an extra LLM call when the heuristic judge is clean.
        use_critic=difficulty != "simple",
    )
