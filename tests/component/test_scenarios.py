"""Run every YAML scenario under tests/fixtures/scenarios (T0.4)."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from cuda_sft.parse import extract_cuda_source
from cuda_sft.testing.scenario import (
    assert_expectations,
    load_scenario,
    run_scenario,
)

SCEN_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "scenarios"


def _cases():
    for path in sorted(SCEN_DIR.glob("*.yaml")):
        sc = load_scenario(path)
        for schedule in sc.get("schedules") or ["serial", "parallel"]:
            marks = [pytest.mark.component]
            if sc.get("xfail"):
                marks.append(pytest.mark.xfail(strict=True, reason=str(sc["xfail"])))
            yield pytest.param(path, schedule, id=f"{sc['id']}-{schedule}", marks=marks)


@pytest.mark.parametrize(("path", "schedule"), list(_cases()))
def test_scenario(
    path: Path, schedule: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sc = load_scenario(path)
    result = run_scenario(sc, tmp_path, monkeypatch, schedule=schedule)
    assert_expectations(result, sc)
    if sc["id"] in {"R05", "R06"}:
        final = result.finals[0]
        assert final["repair_idx"] == 0
        assert final["oracle_retry_idx"] == 2
        assert final["oracle_blocked"] is True
        snapshot = final["candidate_reports"][0]
        assert snapshot["oracle_blocked"] is True
        assert snapshot["banked_reason"] == "oracle_blocked"
        assert snapshot["repairs"] == 0
        assert snapshot["refval_owner"] == ("INFRA" if sc["id"] == "R05" else "ORACLE")
        assert len(result.refval.calls) == 3
        assert len({call["code_sha8"] for call in result.refval.calls}) == 1
        assert result.sleeper.calls == [5.0, 20.0]
        assert len(result.abandoned) == 1 and not result.samples
        abandoned = result.abandoned[0]
        assert abandoned["id"] == sc["job"]["question_id"]
        assert abandoned["track"] == "cuda"
        assert abandoned["abandon_reason"] == abandoned["reason"] == "oracle_unavailable"
        assert abandoned["question_hash"]
    if sc["id"] == "R03":
        sample = result.samples[-1]
        metadata = sample["metadata"]
        assistant = next(message["content"] for message in sample["messages"] if message["role"] == "assistant")
        actual_code = extract_cuda_source(assistant)
        assert metadata["selected_code_sha256"] == hashlib.sha256(actual_code.encode()).hexdigest()
        winner_call = next(call for call in result.llm.calls_for("generator") if call.key == "c1r0")
        assert metadata["generation"]["system"] == winner_call.system
        assert metadata["generation"]["user"] == winner_call.user_text
    if sc["id"] == "R08":
        sample = result.samples[-1]
        assistant = next(message["content"] for message in sample["messages"] if message["role"] == "assistant")
        code_hash = hashlib.sha256(extract_cuda_source(assistant).encode()).hexdigest()
        assert sample["metadata"]["selected_code_sha256"] == code_hash
        assert len(result.compiler.calls) == len(result.refval.calls) == 1
        assert result.compiler.calls[0]["code_sha8"] == code_hash[:8]
        assert result.refval.calls[0]["code_sha8"] == code_hash[:8]
    if sc["id"] == "K01":
        first_call = result.llm.calls_for("knowledge_generator")[0]
        sample = result.samples[-1]
        system = next(message["content"] for message in sample["messages"] if message["role"] == "system")
        assert result.finals[0]["gen_system"] == first_call.system
        assert sample["metadata"]["generation"]["system"] == first_call.system
        assert system == first_call.system
        assert sample["metadata"]["source_line"] == sc["job"]["source_line"]


@pytest.mark.component
def test_critic_raise_patch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sc = deepcopy(load_scenario(SCEN_DIR / "R01_happy_path.yaml"))
    sc["settings"]["KERNEL_LLM_CRITIC"] = "always"
    sc["settings"]["CRITIC_RETRY_ON_ERROR"] = "1"
    sc["patches"] = {"critic_raise": True}
    result = run_scenario(sc, tmp_path, monkeypatch)
    assert len(result.llm.calls_for("critic")) == 2
    assert not result.llm.calls_for("repair.*")
    assert result.finals[0]["repair_idx"] == 0
    assert result.finals[0]["metadata"]["critic"]["status"] == "unverified"
