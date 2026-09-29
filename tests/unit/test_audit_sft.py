"""Historical SFT audit labels defects without changing the source file."""

import json
from pathlib import Path

import pytest

from cuda_sft.testing.scenario import load_scenario, run_scenario
from scripts.audit_sft import audit, flags_for, main


def _row(
    number: int,
    *,
    system: str = "training system",
    assistant: str = "__global__ void kernel() {}\n",
    metadata: dict | None = None,
) -> dict:
    return {
        "id": number,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "vector add"},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {"dialect": "cuda", "refval": {"status": "pass"}, **(metadata or {})},
    }


def test_eight_rows_cover_all_labels_composite_and_invalid_json(tmp_path) -> None:
    rows = [
        _row(1, system="You are a REPAIRER"),
        _row(2, system="The downstream checker only runs nvcc -c"),
        _row(3, assistant="<think>修复上一版</think>\n__global__ void kernel() {}"),
        _row(4, assistant="<think>Call `missing_kernel`.</think>\n__global__ void kernel() {}"),
        _row(5, metadata={"candidate_pool": {"selected_candidate": 3, "count": 2}}),
        _row(
            6,
            system="repairer; Do not explain",
            assistant=(
                "<think>修复 `ghost`; BLOCK=64.</think>\n"
                "```cuda\n#define BLOCK 32\n__global__ void kernel() {}\n```"
            ),
            metadata={
                "candidate": 2,
                "candidate_pool": {"selected_candidate": 1, "count": 2},
                "refval": {"status": "skip"},
                "question_hash": "original-question-hash",
            },
        ),
        _row(7),
    ]
    clean_line = json.dumps(rows[-1], ensure_ascii=False, separators=(",", ":")) + "\n"
    source_text = (
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[:-1])
        + clean_line
        + "{broken\n"
    )
    source = tmp_path / "historical.jsonl"
    source.write_text(source_text, encoding="utf-8")
    out = tmp_path / "audit"

    report = audit(source, out)

    assert source.read_text(encoding="utf-8") == source_text
    assert report["total"] == 8 and report["flagged"] == 6 and report["clean"] == 2
    assert report["flag_counts"] == {
        "repair_system_leak": 2,
        "protocol_system_leak": 2,
        "cot_repair_leak": 2,
        "cot_entity_mismatch": 1,
        "pool_mismatch_risk": 2,
        "no_refval_evidence": 1,
        "invalid_json": 1,
    }
    assert report["flag_ratio"]["no_refval_evidence"] == 0.125
    regen = [
        json.loads(line)
        for line in (out / "needs_regen.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(regen) == 6
    assert set(regen[4]["flags"]) == {
        "repair_system_leak",
        "protocol_system_leak",
        "cot_repair_leak",
        "cot_entity_mismatch",
        "pool_mismatch_risk",
        "no_refval_evidence",
    }
    assert regen[4]["question_hash"] == "original-question-hash"
    assert regen[5] == {
        "id": None,
        "dialect": None,
        "question_hash": None,
        "flags": ["invalid_json"],
        "source_line": 8,
    }
    expected_clean = json.dumps(rows[3], ensure_ascii=False) + "\n" + clean_line
    assert (out / "clean_sft.jsonl").read_text(encoding="utf-8") == expected_clean
    assert json.loads((out / "audit_report.json").read_text(encoding="utf-8")) == report


def test_legacy_top_level_fields_and_builtins() -> None:
    row = {
        "id": 9,
        "question": "vector add",
        "system": "compile-fix",
        "assistant": "<think>`threadIdx` indexes data.</think>\n__global__ void kernel() {}",
        "metadata": {"refval": {"status": "pass"}},
    }
    assert flags_for(row) == ["repair_system_leak"]


def test_constant_mismatch_without_missing_identifier() -> None:
    row = _row(
        10,
        assistant=(
            "<think>Use BLOCK=64 threads.</think>\n"
            "```cuda\n#define BLOCK 32\n__global__ void kernel() {}\n```"
        ),
    )
    assert flags_for(row) == ["cot_entity_mismatch"]


def test_r03_valid_earlier_winner_is_not_a_pool_mismatch(tmp_path, monkeypatch) -> None:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "scenarios" / "R03_winner_not_last.yaml"
    result = run_scenario(load_scenario(path), tmp_path, monkeypatch)
    assert result.samples
    row = result.samples[0]
    pool = row["metadata"]["candidate_pool"]
    assert (row["metadata"]["candidate"], pool["selected_candidate"], pool["count"]) == (1, 1, 2)
    assert "pool_mismatch_risk" not in flags_for(row)


@pytest.mark.parametrize(("selected", "count", "candidate"), [
    (0, 2, 0),
    (3, 2, 3),
    (1, 0, 1),
    (1, 2, 2),
    (1, 2, False),
    ("invalid", 2, 1),
])
def test_invalid_pool_selection_is_flagged(selected: object, count: object, candidate: object) -> None:
    row = _row(12, metadata={
        "candidate": candidate,
        "candidate_pool": {"selected_candidate": selected, "count": count},
    })
    assert flags_for(row) == ["pool_mismatch_risk"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"task": "knowledge"},
        {"language": "prose"},
        {"track": "knowledge:architecture"},
    ],
)
def test_knowledge_does_not_require_kernel_refval(metadata: dict) -> None:
    row = _row(11, metadata=metadata)
    del row["metadata"]["refval"]
    assert flags_for(row) == []
    row["messages"][0]["content"] = "You are a compile-fix repairer."
    assert flags_for(row) == ["repair_system_leak"]


def test_cli_replaces_outputs_deterministically(tmp_path, capsys) -> None:
    source = tmp_path / "old.jsonl"
    source.write_text(json.dumps(_row(1)) + "\n", encoding="utf-8")
    out = tmp_path / "audit"
    assert main([str(source), "--out", str(out)]) == 0
    first = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
    assert main([str(source), "--out", str(out)]) == 0
    assert first == {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}
    assert '"clean": 1' in capsys.readouterr().out


def test_audit_cannot_replace_its_input(tmp_path) -> None:
    source = tmp_path / "clean_sft.jsonl"
    original = json.dumps(_row(1)) + "\n"
    source.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="replace the input"):
        audit(source, tmp_path)
    assert source.read_text(encoding="utf-8") == original
