"""Weighted scoring dimensions for knowledge answers."""

from __future__ import annotations

from typing import Mapping

DIMENSIONS = (
    "factual",
    "completeness",
    "derivation",
    "terminology",
    "structure",
    "grounding",
)

DEFAULT_WEIGHTS: dict[str, float] = {
    "factual": 0.30,
    "completeness": 0.20,
    "derivation": 0.20,
    "terminology": 0.10,
    "structure": 0.10,
    "grounding": 0.10,
}

DERIVATION_TOPICS = {"formula"}


def weights_for(topic: str) -> dict[str, float]:
    """Zero the derivation axis for non-formula topics, then renormalize.

    Args:
        topic: Knowledge topic id (``formula`` keeps the derivation weight).
    """
    weights = dict(DEFAULT_WEIGHTS)
    if (topic or "").strip().lower() not in DERIVATION_TOPICS:
        weights["derivation"] = 0.0
    total = sum(weights.values()) or 1.0
    return {key: value / total for key, value in weights.items()}


def clamp_score(value: object) -> float:
    """Force a judge score into ``[1, 10]``. Invalid values become 1."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1.0
    if number != number:  # NaN
        return 1.0
    return max(1.0, min(10.0, number))


def overall_score(dimensions: Mapping[str, object], topic: str) -> float:
    """Weighted mean in ``[1, 10]``."""
    weights = weights_for(topic)
    total = 0.0
    for key, weight in weights.items():
        total += weight * clamp_score(dimensions.get(key, 1))
    return round(total, 4)


def passes_threshold(
    *,
    overall: float,
    dimensions: Mapping[str, object],
    must_fix: list[str],
    min_score: float,
    factual_min: float,
) -> bool:
    """Apply the save/abandon rule (independent of the model's ``pass`` field)."""
    if must_fix:
        return False
    if overall < float(min_score):
        return False
    factual = clamp_score(dimensions.get("factual", 1))
    return factual >= float(factual_min)
