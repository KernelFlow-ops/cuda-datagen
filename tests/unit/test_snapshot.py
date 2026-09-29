"""Candidate snapshots preserve the exact version selected for training."""

import hashlib

from cuda_sft.core.types import build_snapshot, merge_into_pool, snapshot_for_metadata


def _state(*, code: str = "kernel", repairs: int = 0, score: int = 7) -> dict:
    return {
        "candidate_idx": 1,
        "repair_idx": repairs,
        "code": code,
        "compile_ok": True,
        "compile_error": "",
        "refval_status": "pass",
        "metadata": {
            "refval": {"status": "pass", "cases_run": 12, "manifest_hash": "abc"},
            "critic": {"status": "verified", "passed": True},
            "judge": {"quality_score": score, "issues": [], "suggestions": []},
        },
        "candidate_ctx": {
            "candidate": 1,
            "gen_system": "generation system",
            "gen_user": "generation user",
            "prompt_variant": {"temperature": 0.2},
            "first_turn_reasoning": "first",
            "first_turn_reasoning_source": "api",
            "final_turn_reasoning": "final",
            "final_turn_reasoning_source": "api",
            "origin": "live_api",
        },
    }


def test_build_snapshot_fields_complete() -> None:
    snap = build_snapshot(_state(), banked_reason="final")
    assert snap["code_sha256"] == hashlib.sha256(b"kernel").hexdigest()
    assert snap["gen_system"] == "generation system"
    assert snap["first_turn_reasoning"] == "first"
    assert snap["final_turn_reasoning"] == "final"
    assert snap["origin"] == "live_api"
    assert snap["refval"]["cases_run"] == 12
    assert snap["banked_reason"] == "final"
    assert snap["version"] == 1


def test_contract_pass_requires_independent_oracle() -> None:
    state = _state()
    state["quality_status"] = {"contract": "pass"}
    state["metadata"]["refval"]["verification_tier"] = "model_consistency"
    assert build_snapshot(state, banked_reason="final")["contract"] == "skip"
    state["metadata"]["refval"]["verification_tier"] = "independent"
    assert build_snapshot(state, banked_reason="final")["contract"] == "pass"
    state["quality_status"]["contract"] = "fail"
    assert build_snapshot(state, banked_reason="final")["contract"] == "fail"


def test_merge_only_upgrades_and_versions() -> None:
    first = build_snapshot(_state(score=7), banked_reason="final")
    worse = build_snapshot(_state(code="broken", score=2), banked_reason="repair_exhausted")
    worse["compile"] = "fail"
    worse["refval"] = {"status": "skip", "cases_run": 0}
    assert merge_into_pool([first], worse) == [first]

    better = build_snapshot(_state(code="better", score=9), banked_reason="final")
    merged = merge_into_pool([first], better)
    assert merged[0]["code"] == "better"
    assert merged[0]["version"] == 2


def test_metadata_strips_code_prompts_and_reasoning() -> None:
    public = snapshot_for_metadata(build_snapshot(_state(), banked_reason="final"))
    for key in ("code", "gen_system", "gen_user", "first_turn_reasoning", "final_turn_reasoning"):
        assert key not in public
    assert public["code_sha256"]
    assert public["origin"] == "live_api"
