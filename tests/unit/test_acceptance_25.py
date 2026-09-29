"""Batch acceptance uses the full wall clock and committed samples."""

import json

from scripts.run_acceptance_25 import audit


def test_batch_audit_rejects_missing_output_even_with_good_compute_time(tmp_path) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        "".join(json.dumps({"question": f"Explain CUDA topic {n}", "source_line": n}) + "\n"
                for n in range(25)),
        encoding="utf-8",
    )
    data = tmp_path / "data"
    data.mkdir()
    report = audit("knowledge", source, data, elapsed_s=500.0, returncode=0)
    assert report["seconds_per_question"] is None
    assert report["seconds_per_published"] is None
    assert report["performance_pass"] is True
    assert report["quality_pass"] is False
    assert report["accepted"] is False
    assert len([key for key in report["issues"] if key.isdecimal()]) == 25
    probe = audit("knowledge", source, data, elapsed_s=20.0, returncode=0, count=1)
    assert probe["mode"] == "probe" and probe["expected"] == 1
    assert probe["accepted"] is False
