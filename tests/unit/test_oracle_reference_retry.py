"""Model references can be repaired once; supplied references cannot."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from cuda_sft.config import Settings
from cuda_sft.refval import runner
from cuda_sft.refval.cross_dialect import canonical_tasks, make_manifest, task_source


@pytest.mark.parametrize("source, expected_retries", [("llm", 1), ("independent", 0)])
def test_reference_self_check_retry_uses_refval_role(
    tmp_path, monkeypatch, source: str, expected_retries: int
) -> None:
    task = canonical_tasks()["elementwise_add"]
    valid = make_manifest(task, question_id=7, dialect="cuda")
    broken = replace(valid, extracted_from=source, reference_source="broken")
    repaired = replace(valid, extracted_from="llm")
    calls = []
    cached = []
    monkeypatch.setattr(runner, "toolchain_status", lambda *_args: ("ok", ""))
    monkeypatch.setattr(runner, "obtain_manifest", lambda **_kwargs: broken)
    monkeypatch.setattr(runner, "load_reference_fn", lambda text, _name: text)
    monkeypatch.setattr(
        runner, "validate_reference_fn",
        lambda fn, _abi, *, seed: "reference failed" if fn == "broken" else None,
    )
    monkeypatch.setattr(runner, "_store_cache", lambda _path, manifest: cached.append(manifest))

    def complete(_user, _system, _settings, *, meta):
        calls.append(meta)
        return json.dumps(repaired.to_dict())

    monkeypatch.setattr(runner, "_llm_complete", complete)
    monkeypatch.setattr(
        runner, "prepare_testdir", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("stop after reference check"))
    )
    report = runner.run_refval(
        question="vector add", code=task_source(task, "cuda"), question_id=7,
        dialect="cuda", dialect_spec=SimpleNamespace(
            dialect="cuda", runner="nvcc_link", source_filename="solution.cu"
        ),
        settings=Settings(
            refval_enabled=True, refval_strict=True, refval_cache=True,
            refval_cases="smoke", work_dir=str(tmp_path),
        ),
        workdir=tmp_path,
    )
    assert len(calls) == expected_retries
    assert all(call.role == "refval_extract" and call.purpose == "reference_retry" for call in calls)
    assert len(cached) == expected_retries
    if source == "independent":
        assert report.status == "fail" and report.error_class == "reference_error"
