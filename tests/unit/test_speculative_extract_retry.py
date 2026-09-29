"""Invalid prefetched oracle text gets one diagnostic retry."""

from __future__ import annotations

import json
from unittest.mock import patch

from cuda_sft import graph
from cuda_sft.config import Settings
from cuda_sft.refval import runner
from cuda_sft.refval.cross_dialect import canonical_tasks, make_manifest, task_source


def test_invalid_prefetch_does_not_repeat_the_same_full_prompt(monkeypatch, tmp_path) -> None:
    task = canonical_tasks()["elementwise_add"]
    valid = make_manifest(task, question_id=7, dialect="cuda")
    calls = []
    monkeypatch.setattr(runner, "_try_speculative", lambda *_args: '{"invalid": true}')

    def complete(user, _system, _settings, meta=None):
        calls.append(user)
        return json.dumps(valid.to_dict())

    monkeypatch.setattr(runner, "_llm_complete", complete)
    result = runner.obtain_manifest(
        question="vector add", code=task_source(task, "cuda"),
        question_id=7, dialect="cuda", speculative_id="q7_cuda_c1_r0_refval",
        settings=Settings(refval_cache=False, work_dir=str(tmp_path)),
    )
    assert result is not None and result.reference_source
    assert len(calls) == 1
    assert "previous ABI/reference JSON failed" in calls[0]


def test_terminal_candidate_clears_prefetch_ownership() -> None:
    with patch.object(graph, "cancel_speculative") as cancel, patch.object(
        graph, "_bank_candidate", return_value={"candidate_reports": []}
    ):
        result = graph.collect_candidate({
            "speculative_requests": ["q7_cuda_c1_r0_refval"],
            "compile_ok": False, "refval_ok": False,
        })
    cancel.assert_called_once_with(["q7_cuda_c1_r0_refval"])
    assert result["speculative_requests"] == []
