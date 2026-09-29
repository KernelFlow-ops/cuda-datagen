"""Question-supplied oracle validation and report provenance."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cuda_sft import graph
from cuda_sft.config import Settings
from cuda_sft.core.gates import FailureOwner, owner_of_refval
from cuda_sft.refval import runner
from cuda_sft.refval.cross_dialect import canonical_tasks, make_manifest, task_source
from cuda_sft.refval.oracle import resolve_oracle_manifest
from cuda_sft.refval.spec import RefvalReport
from cuda_sft.runtime import deps


@pytest.mark.parametrize("dialect", ["cuda", "cutlass", "triton", "tilelang"])
def test_supplied_oracle_is_bound_to_question_and_source(dialect: str) -> None:
    task = canonical_tasks()["elementwise_add"]
    manifest = make_manifest(task, question_id=7, dialect=dialect)
    supplied = {"oracle_manifests": {dialect: manifest.to_dict()}}
    resolved = resolve_oracle_manifest(
        supplied, dialect=dialect, question_id=7, code=task_source(task, dialect)
    )
    assert resolved is not None
    assert resolved.extracted_from == "independent"
    assert resolved.provenance["oracle_origin"] == "question_input"


def test_invalid_oracle_never_downgrades_to_model_extraction() -> None:
    task = canonical_tasks()["elementwise_add"]
    manifest = make_manifest(task, question_id=7, dialect="cuda").to_dict()
    with pytest.raises(ValueError, match="expected q8/cuda"):
        resolve_oracle_manifest(
            {"oracle_manifests": {"cuda": manifest}},
            dialect="cuda", question_id=8, code=task_source(task, "cuda"),
        )
    with pytest.raises(ValueError, match="ABI does not match source"):
        resolve_oracle_manifest(
            {"oracle_manifests": {"cuda": manifest}},
            dialect="cuda", question_id=7, code="extern \"C\" void other() {}",
        )
    with pytest.raises(ValueError, match="no cutlass entry"):
        resolve_oracle_manifest(
            {"oracle_manifests": {"cuda": manifest}},
            dialect="cutlass", question_id=7, code="",
        )
    assert resolve_oracle_manifest({}, dialect="cuda", question_id=7, code="") is None


def test_report_tier_survives_serialization() -> None:
    report = RefvalReport(
        status="pass", dialect="cuda", cases_run=3,
        oracle_origin="independent", verification_tier="independent",
    )
    assert RefvalReport.from_dict(report.to_dict()).verification_tier == "independent"
    assert report.to_metadata()["oracle_origin"] == "independent"


def test_graph_passes_question_oracle_to_refval(tmp_path) -> None:
    task = canonical_tasks()["elementwise_add"]
    manifest = make_manifest(task, question_id=7, dialect="cuda").to_dict()
    seen = {}
    settings = Settings(refval_enabled=True, refval_strict=True, work_dir=str(tmp_path))

    def refval_fn(**kwargs):
        seen.update(kwargs)
        return RefvalReport(
            status="pass", dialect="cuda", cases_run=3,
            manifest_hash="independent-hash", oracle_origin="independent",
            verification_tier="independent",
        )

    spec = SimpleNamespace(name="cuda", refval_spec=lambda _settings: object())
    dialect_agent = SimpleNamespace(nest_workdir=lambda *_args: False)
    state = {
        "question_id": 7, "question": "vector add", "dialect": "cuda",
        "code": task_source(task, "cuda"), "metadata": {},
        "quality_status": {"contract": "skip"},
        "input_metadata": {"oracle_manifests": {"cuda": manifest}},
    }
    with patch.object(graph, "get_settings", return_value=settings), patch.object(
        graph, "_spec", return_value=spec
    ), patch.object(graph, "get_dialect_agent", return_value=dialect_agent), deps.use(
        deps.Deps(refval_fn=refval_fn)
    ):
        result = graph.validate(state)
    assert seen["oracle_manifests"] == {"cuda": manifest}
    assert result["quality_status"]["contract"] == "pass"
    assert result["metadata"]["refval"]["verification_tier"] == "independent"


def test_input_oracle_skips_speculative_model_extract() -> None:
    settings = Settings(refval_enabled=True, async_llm_enabled=True, kernel_fast_mode=False)
    spec = SimpleNamespace(name="cuda", extract=lambda raw: raw)
    enqueue = Mock(side_effect=AssertionError("unexpected LLM oracle extract"))
    with patch.object(graph, "get_settings", return_value=settings), patch.object(
        graph, "_spec", return_value=spec
    ), patch.object(runner, "enqueue_speculative_extract", enqueue):
        result = graph.extract({
            "question_id": 7, "raw_response": "source", "input_metadata": {
                "oracle_manifests": {"cuda": {}}
            },
        })
    assert result["code"] == "source"
    enqueue.assert_not_called()


@pytest.mark.parametrize("strict", [True, False])
def test_invalid_input_oracle_fails_before_model_extract(tmp_path, monkeypatch, strict: bool) -> None:
    task = canonical_tasks()["elementwise_add"]
    wrong = make_manifest(task, question_id=8, dialect="cuda").to_dict()
    monkeypatch.setattr(runner, "toolchain_status", lambda *_args: ("ok", ""))
    monkeypatch.setattr(
        runner, "obtain_manifest", lambda **_kwargs: pytest.fail("model fallback was called")
    )
    report = runner.run_refval(
        question="vector add", code=task_source(task, "cuda"), question_id=7,
        dialect="cuda", dialect_spec=SimpleNamespace(dialect="cuda", runner="nvcc_link"),
        settings=Settings(refval_enabled=True, refval_strict=strict, work_dir=str(tmp_path)),
        workdir=tmp_path, oracle_manifests={"cuda": wrong},
    )
    assert report.status == "fail"
    assert report.verification_tier == "none"
    assert "expected q7/cuda" in report.reason
    assert runner.refval_blocks_save(report, Settings(refval_strict=strict))
    assert owner_of_refval(report)[0] is FailureOwner.ORACLE
    assert graph.route_after_validate({"oracle_blocked": True, "last_gate": {"owner": "ORACLE"}}) == "collect_candidate"
