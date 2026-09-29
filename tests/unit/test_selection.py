"""Selection never trades a validated kernel for an unverified score."""

from types import SimpleNamespace

from cuda_sft.core.selection import eligible, rank_key
from cuda_sft.core.types import build_snapshot
from cuda_sft.graph import select_best
from tests.unit.test_snapshot import _state


def _snap(
    *,
    refval: str = "pass",
    score: int = 7,
    repairs: int = 0,
    critic: str = "verified",
    banked_reason: str = "final",
) -> dict:
    state = _state(score=score, repairs=repairs)
    state["refval_status"] = refval
    state["metadata"]["refval"]["status"] = refval
    state["metadata"]["critic"]["status"] = critic
    return build_snapshot(state, banked_reason=banked_reason)


def test_refval_pass_beats_higher_judge() -> None:
    assert rank_key(_snap(score=5)) > rank_key(_snap(refval="skip", score=10))


def test_selection_prefers_fewer_repairs_at_equal_score() -> None:
    first = _state(code="slow", repairs=2, score=8)
    second = _state(code="clean", repairs=0, score=8)
    second["candidate_idx"] = 2
    second["candidate_ctx"]["candidate"] = 2
    reports = [
        build_snapshot(first, banked_reason="final"),
        build_snapshot(second, banked_reason="final"),
    ]
    selected = select_best({"candidate_reports": reports, "metadata": {}, "quality_status": {}})
    assert selected["code"] == "clean"


def test_unverified_critic_not_vetoed() -> None:
    cfg = SimpleNamespace(refval_strict=True, kernel_critic_blocks_save=True)
    assert eligible(_snap(critic="unverified"), cfg)


def test_critic_block_setting_controls_failed_candidates() -> None:
    cfg = SimpleNamespace(refval_strict=True, kernel_critic_blocks_save=True)
    assert not eligible(_snap(critic="failed"), cfg)
    assert not eligible(_snap(critic="failed", banked_reason="pre_semantic_repair"), cfg)
    cfg.kernel_critic_blocks_save = False
    assert eligible(_snap(critic="failed"), cfg)
    assert eligible(_snap(critic="failed", banked_reason="pre_semantic_repair"), cfg)
    assert rank_key(_snap(critic="verified")) > rank_key(
        _snap(critic="failed", banked_reason="pre_semantic_repair")
    )


def test_oracle_blocked_not_eligible() -> None:
    cfg = SimpleNamespace(refval_strict=True, kernel_critic_blocks_save=False)
    snap = _snap()
    snap["oracle_blocked"] = True
    assert not eligible(snap, cfg)


def test_inconsistent_critic_fields_still_veto_when_enabled() -> None:
    snap = _snap()
    snap["critic"]["passed"] = False
    cfg = SimpleNamespace(refval_strict=True, kernel_critic_blocks_save=True)
    assert not eligible(snap, cfg)
