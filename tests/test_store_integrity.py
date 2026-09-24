"""Store transaction, strict-gate, and canonical-export regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from cuda_sft.config import Settings
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
    return state


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {"refval_strict": True, "cot_enabled": False}
    values.update(overrides)
    return Settings(**values)


def test_strict_missing_evidence_is_quarantined(tmp_path: Path) -> None:
    store = Store(tmp_path)
    with patch("cuda_sft.store.get_settings", return_value=_settings()):
        saved = store.write_success(_kernel_state(metadata={}), model_name="m")
    assert saved is False
    assert not store.sft_path.exists()
    abandoned = [json.loads(line) for line in store.abandoned_path.read_text().splitlines()]
    assert abandoned[0]["reason"] == "strict_quality_gate"


def test_strict_rejects_unverified_and_failed_critic(tmp_path: Path) -> None:
    store = Store(tmp_path)
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


def test_knowledge_bypasses_kernel_strict_gate(tmp_path: Path) -> None:
    store = Store(tmp_path)
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
        assert store.write_success(state, model_name="m") is True
    assert store.sft_path.exists()


def test_duplicate_identity_requires_question_code_and_dialect(tmp_path: Path) -> None:
    store = Store(tmp_path)
    with patch("cuda_sft.store.get_settings", return_value=_settings(refval_strict=False)):
        assert store.write_success(_kernel_state(), model_name="m") is True
        assert store.write_success(_kernel_state(), model_name="m") is True
        assert store.write_success(_kernel_state(question="different"), model_name="m") is True
        assert store.write_success(_kernel_state(dialect="triton"), model_name="m") is True
    assert len(store.sft_path.read_text().splitlines()) == 3


def test_duplicate_call_recovers_missing_progress_and_exports(tmp_path: Path) -> None:
    store = Store(tmp_path)
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
