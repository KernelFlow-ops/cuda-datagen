"""Hard-gate eligibility and lexicographic candidate ordering."""

from __future__ import annotations

from typing import Any

from cuda_sft.core.types import CandidateSnapshot

MIN_CASES = 3
CRITIC_RANK = {"verified": 3, "skipped": 2, "unverified": 1, "failed": 0}


def critic_rejected(report: dict[str, Any]) -> bool:
    return (
        str(report.get("status") or "").strip().lower() == "failed"
        or report.get("passed") is False
        or bool(report.get("must_fix"))
    )


def eligible(s: CandidateSnapshot, cfg: Any) -> bool:
    if s["compile"] != "pass" or s["contract"] == "fail" or s["oracle_blocked"]:
        return False
    rv = str(s["refval"].get("status") or "skip")
    if getattr(cfg, "refval_strict", True) and rv != "pass":
        return False
    if not getattr(cfg, "refval_strict", True) and rv == "fail":
        return False
    return not (
        critic_rejected(s["critic"])
        and getattr(cfg, "kernel_critic_blocks_save", False)
    )


def rank_key(s: CandidateSnapshot) -> tuple[int, int, int, int, float, int, int]:
    rv = s["refval"]
    perf = s.get("perf") or {}
    return (
        1 if rv.get("status") == "pass" else 0,
        1 if int(rv.get("cases_run") or 0) >= MIN_CASES else 0,
        CRITIC_RANK.get(str(s["critic"].get("status") or "skipped"), 1),
        -int(s["repairs"]),
        float(perf.get("speedup") or 0.0),
        int(s["judge"].get("quality_score") or 0),
        -int(s["candidate"]),
    )
