"""Store transaction, strict-gate, and canonical-export regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cuda_sft.config import Settings
from cuda_sft.core.types import build_snapshot
from cuda_sft.store import Store


def _kernel_state(**overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "kind": "kernel",
        "question_id": 1,
        "question": "vector add",
        "user_prompt": "vector add",
        "code": "__global__ void k() {}\n",
        "dialect": "cuda",
        "metadata": {
            "refval": {"status": "pass", "cases_run": 2, "manifest_hash": "manifest"},
            "critic": {"status": "verified", "passed": True, "issues": []},
        },
    }
    state.update(overrides)
    metadata = state.get("metadata")
    refval = metadata.get("refval") if isinstance(metadata, dict) else None
    state["compile_ok"] = True
    state["refval_status"] = refval.get("status", "skip") if isinstance(refval, dict) else "skip"
    state["candidate_ctx"] = {
        "gen_system": "generation system",
        "gen_user": state["user_prompt"],
        "prompt_variant": {"temperature": 0.2},
    }
    state["selected"] = build_snapshot(state, banked_reason="final")
    return state


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"refval_strict": True, "cot_enabled": False}
    values.update(overrides)
    return Settings(**values)


def test_strict_missing_evidence_is_quarantined(tmp_path: Path) -> None:
    store = Store(tmp_path, allow_test_sources=True)
    with patch("cuda_sft.store.get_settings", return_value=_settings()):
        saved = store.write_success(_kernel_state(metadata={}), model_name="m")
    assert saved is False
    assert not store.sft_path.exists()
    abandoned = [json.loads(line) for line in store.abandoned_path.read_text().splitlines()]
    assert abandoned[0]["reason"] == "strict_quality_gate"


def test_strict_rejects_unverified_and_failed_critic(tmp_path: Path) -> None:
    store = Store(tmp_path, allow_test_sources=True)
    for status in ("unverified", "failed"):
        state = _kernel_state(
            question_id=10 if status == "unverified" else 11,
            metadata={
                "refval": {"status": "pass", "cases_run": 1, "manifest_hash": "m"},
                "critic": {"status": status, "passed": status == "verified"},
            },
        )
        with patch("cuda_sft.store.get_settings", return_value=_settings()):
            assert store.write_success(state, model_name="m") is False
    assert not store.sft_path.exists()


def test_critic_block_setting_matches_selection_at_save(tmp_path: Path) -> None:
    store = Store(tmp_path, allow_test_sources=True)
    state = _kernel_state(
        metadata={
            "refval": {"status": "pass", "cases_run": 3, "manifest_hash": "m"},
            "critic": {"status": "failed", "passed": False, "issues": ["wrong axis"]},
        }
    )
    with patch("cuda_sft.store.get_settings", return_value=_settings(kernel_critic_blocks_save=True)):
        assert store.write_success(state, model_name="m") is False
    assert not store.sft_path.exists()
    state["question_id"] = 2
    with patch("cuda_sft.store.get_settings", return_value=_settings(kernel_critic_blocks_save=False)):
        assert store.write_success(state, model_name="m") is True
    state["question_id"] = 3
    with patch(
        "cuda_sft.store.get_settings",
        return_value=_settings(refval_strict=False, kernel_critic_blocks_save=True),
    ):
        assert store.write_success(state, model_name="m") is False


def test_store_preserves_oracle_verification_tier(tmp_path: Path) -> None:
    store = Store(tmp_path, allow_test_sources=True)
    state = _kernel_state(
        metadata={
            "refval": {
                "status": "pass", "cases_run": 3, "manifest_hash": "m",
                "verification_tier": "model_consistency", "oracle_origin": "model_extract",
            },
            "critic": {"status": "verified", "passed": True},
        }
    )
    with patch("cuda_sft.store.get_settings", return_value=_settings()):
        assert store.write_success(state, model_name="m") is True
    metadata = json.loads(store.sft_path.read_text().splitlines()[0])["metadata"]
    assert metadata["verification_tier"] == "model_consistency"
    assert metadata["oracle_origin"] == "model_extract"
    assert metadata["refval"]["verification_tier"] == "model_consistency"


def test_knowledge_without_judge_evidence_is_quarantined(tmp_path: Path) -> None:
    store = Store(tmp_path, allow_test_sources=True)
    state = {
        "kind": "knowledge",
        "question_id": 2,
        "question": "Explain warps",
        "user_prompt": "Explain warps",
        "answer": "A warp is a group of threads.",
        "topic": "architecture",
        "track": "knowledge:architecture",
        "metadata": {},
    }
    with patch("cuda_sft.store.get_settings", return_value=_settings()):
        assert store.write_success(state, model_name="m") is False
    assert not store.sft_path.exists()
    abandoned = json.loads(store.abandoned_path.read_text().splitlines()[0])
    assert abandoned["reason"] == "strict_quality_gate"


def test_duplicate_identity_requires_question_code_and_dialect(tmp_path: Path) -> None:
    store = Store(tmp_path, allow_test_sources=True)
    with patch("cuda_sft.store.get_settings", return_value=_settings(refval_strict=False)):
        assert store.write_success(_kernel_state(), model_name="m") is True
        assert store.write_success(_kernel_state(), model_name="m") is True
        assert store.write_success(_kernel_state(question="different"), model_name="m") is True
        assert store.write_success(_kernel_state(dialect="triton"), model_name="m") is True
    assert len(store.sft_path.read_text().splitlines()) == 3


def test_duplicate_call_recovers_missing_progress_and_exports(tmp_path: Path) -> None:
    store = Store(tmp_path, allow_test_sources=True)
    state = _kernel_state()
    with patch("cuda_sft.store.get_settings", return_value=_settings(refval_strict=False)):
        assert store.write_success(state, model_name="m") is True
    store.progress_path.unlink()
    store.swift_path.unlink()
    store.openrlhf_path.unlink()
    with patch("cuda_sft.store.get_settings", return_value=_settings(refval_strict=False)):
        assert store.write_success(state, model_name="m") is True
    assert len(store.sft_path.read_text().splitlines()) == 1
    assert len(store.progress_path.read_text().splitlines()) == 1
    assert len(store.swift_path.read_text().splitlines()) == 1
    assert len(store.openrlhf_path.read_text().splitlines()) == 1


def test_production_store_requires_live_winning_candidate(tmp_path: Path) -> None:
    store = Store(tmp_path)
    state = _kernel_state(origin="live_api", provenance={"origin": "replay"})
    state["selected"]["origin"] = "replay"
    with patch("cuda_sft.store.get_settings", return_value=_settings(refval_strict=False)):
        with pytest.raises(RuntimeError, match="without live API origin: replay"):
            store.write_success(state, model_name="m")
    assert not store.sft_path.exists()
    assert not store.progress_path.exists()


def test_live_origin_is_persisted_from_selected_snapshot(tmp_path: Path) -> None:
    store = Store(tmp_path)
    state = _kernel_state(origin="live_api", provenance={"origin": "replay"})
    with patch("cuda_sft.store.get_settings", return_value=_settings(refval_strict=False)):
        assert store.write_success(state, model_name="m") is True
    row = json.loads(store.sft_path.read_text().splitlines()[0])
    assert row["metadata"]["provenance"]["origin"] == "live_api"


def test_production_store_requires_live_knowledge_answer(tmp_path: Path) -> None:
    store = Store(tmp_path)
    state = {
        "kind": "knowledge",
        "question_id": 2,
        "question": "Explain warps",
        "answer": "A warp is a group of threads.",
        "origin": "replay",
    }
    with pytest.raises(RuntimeError, match="without live API origin: replay"):
        store.write_success(state, model_name="m")
    assert not store.sft_path.exists()
