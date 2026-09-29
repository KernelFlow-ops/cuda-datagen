"""A partial harness run cannot publish a numeric pass."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuda_sft.config import Settings
from cuda_sft.refval import runner
from cuda_sft.refval.cross_dialect import (
    canonical_plans,
    canonical_tasks,
    make_manifest,
    task_source,
)
from cuda_sft.refval.spec import CaseResult


@pytest.mark.parametrize("exit_code,reported", [(1, 3), (0, 2)])
def test_harness_requires_successful_exit_and_every_case(
    tmp_path, monkeypatch, exit_code: int, reported: int
) -> None:
    task = canonical_tasks()["elementwise_add"]
    plans = canonical_plans(task, question_id=7, suite="smoke")
    payload = {
        "ok": True,
        "cases": [{"name": plan.name, "ok": True} for plan in plans[:reported]],
    }

    class Lock:
        def __init__(self, *_args):
            pass

        def acquire(self):
            return True

        def release(self):
            pass

    monkeypatch.setattr(runner, "toolchain_status", lambda *_args: ("ok", ""))
    monkeypatch.setattr(runner, "load_reference_fn", lambda *_args: object())
    monkeypatch.setattr(runner, "validate_reference_fn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "GpuFileLock", Lock)
    monkeypatch.setattr(runner, "run_binary", lambda *_args, **_kwargs: (exit_code, "", payload))
    monkeypatch.setattr(runner, "bind_and_call", lambda *_args, **_kwargs: {"c": [1.0]})
    monkeypatch.setattr(runner, "load_gpu_outputs", lambda *_args, **_kwargs: {"c": [1.0]})
    monkeypatch.setattr(
        runner, "compare_outputs",
        lambda _got, _expected, _abi, plan, _tols: CaseResult(
            name=plan.name, ok=True, status="pass", n_compared=1
        ),
    )
    report = runner.run_refval(
        question="vector add", code=task_source(task, "triton"), question_id=7,
        dialect="triton", dialect_spec=SimpleNamespace(
            dialect="triton", runner="python_import", source_filename="solution.py"
        ),
        settings=Settings(refval_enabled=True, refval_strict=True, work_dir=str(tmp_path)),
        workdir=tmp_path, manifest=make_manifest(task, question_id=7, dialect="triton"),
        case_plans=plans,
    )
    assert report.status == "fail"
    assert report.error_class == ("crash" if exit_code else "output_missing")
