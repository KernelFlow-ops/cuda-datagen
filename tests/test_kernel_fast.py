"""Fast kernel deadline and terminal timeout accounting."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

from cuda_sft.config import Settings
from cuda_sft.main import run_job_list
from cuda_sft.tasks.kinds import Job


def test_deadline_records_abandoned(tmp_path) -> None:
    class SlowGraph:
        def invoke(self, _state, _config):
            time.sleep(2)
            raise AssertionError("deadline did not interrupt graph")

    settings = Settings(
        kernel_fast_mode=True,
        kernel_deadline_sec=1,
        trace_enabled=False,
        cuda_arch="sm_86",
        gpu_name="test GPU",
        cuda_home="/usr/local/cuda",
        data_dir=str(tmp_path),
        work_dir=str(tmp_path / "work"),
    )
    job = Job(question_id=1, question="vector add", kind="kernel", track="cuda")
    with patch("cuda_sft.main.get_settings", return_value=settings), patch(
        "cuda_sft.main.build_graph", return_value=SlowGraph()
    ):
        counts = run_job_list([job], data_dir=tmp_path, show_progress=False)

    assert counts == (0, 1, 0)
    progress = [json.loads(line) for line in (tmp_path / "progress.jsonl").read_text().splitlines()]
    assert progress[-1]["status"] == "abandoned"
    assert progress[-1]["reason"] == "kernel_deadline_exceeded"
