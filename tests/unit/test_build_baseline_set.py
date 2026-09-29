"""Baseline selection is deterministic and stratified."""

import json

import pytest

from scripts import build_baseline_set as baseline
from scripts.build_baseline_set import build


def test_baseline_is_deterministic(tmp_path) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text(
        "".join(
            json.dumps({"question": question}) + "\n"
            for question in [
                "Implement vector add",
                "Implement matmul",
                "Implement transpose",
                "Implement ReLU",
            ]
        ),
        encoding="utf-8",
    )
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    assert build(source, first, limit=3)["selected_rows"] == 3
    build(source, second, limit=3)
    assert first.read_bytes() == second.read_bytes()
    rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert len({(row["family"], row["difficulty"]) for row in rows}) >= 2
    stats = json.loads(first.with_suffix(".stats.json").read_text(encoding="utf-8"))
    assert sum(stats["source_strata"].values()) == 4
    assert sum(stats["strata"].values()) == 3
    assert all(row["question"] for row in rows)
    assert all(row["source_line"] > 0 and len(row["question_hash"]) == 64 for row in rows)


def test_negative_limit_is_rejected(tmp_path) -> None:
    source = tmp_path / "questions.jsonl"
    source.write_text('{"question": "add"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-negative"):
        build(source, tmp_path / "out.jsonl", limit=-1)


def test_first_twenty_preserve_round_robin_strata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(baseline, "_operation_family", lambda question: question.split("/")[0])
    monkeypatch.setattr(baseline, "kernel_difficulty", lambda question: question.split("/")[1])
    source = tmp_path / "questions.jsonl"
    source.write_text(
        "".join(
            json.dumps({"question": f"f{family}/{difficulty}/{depth}"}) + "\n"
            for depth in range(3)
            for family in range(6)
            for difficulty in ("hard", "medium", "simple")
        ),
        encoding="utf-8",
    )
    output = tmp_path / "baseline.jsonl"
    build(source, output, limit=30)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    strata = [(row["family"], row["difficulty"]) for row in rows[:20]]
    assert len(set(strata)) == 18
    assert strata[18:20] == strata[:2]
    for stratum in set(strata):
        hashes = [
            row["question_hash"] for row in rows if (row["family"], row["difficulty"]) == stratum
        ]
        assert hashes == sorted(hashes)


def test_knowledge_selection_infers_topics_and_skips_duplicate_questions(tmp_path) -> None:
    source = tmp_path / "knowledge.jsonl"
    source.write_text(
        "".join(json.dumps({"question": text}) + "\n" for text in (
            "Explain occupancy and latency hiding.",
            "Explain occupancy and latency hiding.",
            "Explain CUDA memory hierarchy.",
            "Derive the roofline formula.",
        )), encoding="utf-8",
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    stats = build(source, first, limit=4, task="knowledge")
    build(source, second, limit=4, task="knowledge")
    rows = [json.loads(line) for line in first.read_text(encoding="utf-8").splitlines()]
    assert first.read_bytes() == second.read_bytes()
    assert stats["selected_rows"] == 3
    assert {row["source_line"] for row in rows} <= {1, 2, 3, 4}
    assert all(row["task"] == "knowledge" and row["topic"] for row in rows)
    assert len({row["question_hash"] for row in rows}) == 3
