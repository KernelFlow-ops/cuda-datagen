"""L6 S1 dataset gates and command exit status."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from cuda_sft.observability.dataset_checks import check_dataset
from cuda_sft.testing.scenario import load_scenario, run_scenario

pytestmark = pytest.mark.dataset

ROOT = Path(__file__).resolve().parents[2]
CODE = "#include <cuda_runtime.h>\n__global__ void add() {}\n"
THINK = "Use `add` to process each element."


def sample() -> dict:
    return {
        "id": 1,
        "messages": [
            {"role": "system", "content": "Write a CUDA kernel."},
            {"role": "user", "content": "Add arrays."},
            {"role": "assistant", "content": f"<think>\n{THINK}\n</think>\n{CODE}"},
        ],
        "metadata": {
            "dataset_version": "cuda-sft-2",
            "sample_key": "sample-1",
            "question_hash": "question-1",
            "source_line": 1,
            "track": "cuda",
            "dialect": "cuda",
            "language": "cuda-cpp",
            "candidate": 1,
            "repairs": 0,
            "selected_code_sha256": hashlib.sha256(CODE.encode()).hexdigest(),
            "system_mode": "fixed",
            "user_mode": "raw_question",
            "generation": {"system": "generator", "user": "Add arrays.", "prompt_variant": {}},
            "refval": {"status": "pass", "cases_run": 12, "manifest_hash": "fake-abc"},
            "critic": {"status": "skipped"},
            "judge": {},
            "cot": {"source": "agent", "policy": "synthetic", "consistency_issues": [], "chars": len(THINK)},
            "candidate_pool": {"count": 1, "selected_candidate": 1, "eligible_count": 1, "reports": []},
            "release_tier": "strict",
            "provenance": {"model": "fake", "provider": "fake", "prompt_packs": {}, "run_id": "test", "created_at": "2026-09-24"},
        },
    }


def knowledge_sample() -> dict:
    row = sample()
    answer = (
        "## Answer\nA CUDA warp groups 32 threads for instruction scheduling on an SM. "
        "Several warps can reside on one SM, subject to registers, shared memory, and block limits."
    )
    row["messages"][-1]["content"] = answer
    md = row["metadata"]
    md.update({
        "task": "knowledge", "topic": "architecture", "track": "knowledge:architecture",
        "dialect": "knowledge:architecture", "language": "prose", "release_tier": "strict",
        "code_hash": hashlib.sha256(answer.encode()).hexdigest(),
        "knowledge_judge": {
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "pass": True, "unavailable": False, "skipped_llm": False,
            "hard_gate_failed": False, "judge_error": "", "must_fix": [],
            "topic": "architecture", "overall": 8.0,
            "dimensions": {key: 8.0 for key in (
                "factual", "completeness", "derivation", "terminology", "structure", "grounding"
            )},
        },
    })
    for key in ("selected_code_sha256", "refval", "critic", "judge", "candidate_pool"):
        del md[key]
    return row


def run_check(tmp_path: Path, *rows: dict | str) -> dict:
    source = tmp_path / "sft.jsonl"
    source.write_text("".join((row if isinstance(row, str) else json.dumps(row, ensure_ascii=False)) + "\n" for row in rows), encoding="utf-8")
    return check_dataset(source)


def assert_failure(report: dict, rule: str) -> None:
    assert not report["ok"]
    assert report["checks"][rule]["fail"] == 1, report["failures"]


def test_valid_s1_sample_passes_all_applicable_gates(tmp_path: Path) -> None:
    report = run_check(tmp_path, sample())
    assert report["ok"], report["failures"]
    for rule in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H13"):
        assert report["checks"][rule]["pass"] == 1
    for rule in ("H9", "H10", "H11", "H12"):
        assert report["checks"][rule]["n/a"] == 1


@pytest.mark.parametrize("bad", ['{broken', json.dumps({"messages": [], "metadata": {}})])
def test_h1_rejects_bad_json_or_roles(tmp_path: Path, bad: str) -> None:
    assert_failure(run_check(tmp_path, bad), "H1")


def test_h2_rejects_repair_system(tmp_path: Path) -> None:
    row = sample()
    row["messages"][0]["content"] = "You are a compile-fix repairer."
    assert_failure(run_check(tmp_path, row), "H2")


def test_h3_only_checks_fixed_system(tmp_path: Path) -> None:
    row = sample()
    row["messages"][0]["content"] = "Run nvcc -c first."
    assert_failure(run_check(tmp_path, row), "H3")
    row["metadata"]["system_mode"] = "generation"
    report = run_check(tmp_path, row)
    assert report["ok"] and report["checks"]["H3"]["n/a"] == 1


@pytest.mark.parametrize("assistant", [
    "Here is the code:\n```cuda\n" + CODE + "```",
    "```cuda\n" + CODE + "```\n```cuda\n" + CODE + "```",
    "<think>first</think><think>second</think>" + CODE,
    CODE + "Here is why this code works.",
    "just an explanation",
])
def test_h4_rejects_extra_text_or_multiple_sources(tmp_path: Path, assistant: str) -> None:
    row = sample()
    row["messages"][-1]["content"] = assistant
    assert_failure(run_check(tmp_path, row), "H4")


def test_h4_accepts_one_fence_without_think(tmp_path: Path) -> None:
    row = sample()
    row["messages"][-1]["content"] = f"```cuda\n{CODE}```"
    assert run_check(tmp_path, row)["ok"]


def test_h5_rejects_code_hash_mismatch(tmp_path: Path) -> None:
    row = sample()
    row["metadata"]["selected_code_sha256"] = "0" * 64
    assert_failure(run_check(tmp_path, row), "H5")


def test_h6_rejects_repair_narrative(tmp_path: Path) -> None:
    row = sample()
    row["messages"][-1]["content"] = f"<think>修复上一版代码</think>\n{CODE}"
    assert_failure(run_check(tmp_path, row), "H6")


def test_h6_accepts_fixed_as_a_property(tmp_path: Path) -> None:
    row = sample()
    row["messages"][-1]["content"] = f"<think>Bandwidth is not fixed across systems.</think>\n{CODE}"
    assert run_check(tmp_path, row)["checks"]["H6"]["pass"] == 1


def test_h6_accepts_numeric_fix_and_repair_terms(tmp_path: Path) -> None:
    row = sample()
    row["messages"][-1]["content"] = (
        "<think>Higher precision cannot fix ill-conditioned problems or repair "
        f"rounding errors.</think>\n{CODE}"
    )
    assert run_check(tmp_path, row)["checks"]["H6"]["pass"] == 1


def test_h7_checks_only_when_consistency_key_exists(tmp_path: Path) -> None:
    row = sample()
    code = "const int BLOCK = 128;\n" + CODE
    row["messages"][-1]["content"] = f"<think>BLOCK = 256 threads.</think>\n{code}"
    row["metadata"]["selected_code_sha256"] = hashlib.sha256(code.encode()).hexdigest()
    assert_failure(run_check(tmp_path, row), "H7")
    del row["metadata"]["cot"]["consistency_issues"]
    report = run_check(tmp_path, row)
    assert report["ok"] and report["checks"]["H7"]["n/a"] == 1


@pytest.mark.parametrize(("key", "value"), [("status", "fail"), ("cases_run", 2), ("manifest_hash", "")])
def test_h8_rejects_missing_strict_refval_evidence(tmp_path: Path, key: str, value: object) -> None:
    row = sample()
    row["metadata"]["refval"][key] = value
    assert_failure(run_check(tmp_path, row), "H8")


def test_h8_not_applicable_to_compile_only(tmp_path: Path) -> None:
    row = sample()
    row["metadata"]["release_tier"] = "compile_only"
    row["metadata"]["refval"] = {"status": "skip"}
    report = run_check(tmp_path, row)
    assert report["ok"] and report["checks"]["H8"]["n/a"] == 1


@pytest.mark.parametrize("field", ["critic", "candidate_pool", "selected_code_sha256", "provenance", "generation"])
def test_h13_rejects_missing_required_field(tmp_path: Path, field: str) -> None:
    row = sample()
    del row["metadata"][field]
    assert_failure(run_check(tmp_path, row), "H13")


def test_h13_rejects_missing_nested_field(tmp_path: Path) -> None:
    row = sample()
    del row["metadata"]["candidate_pool"]["selected_candidate"]
    assert_failure(run_check(tmp_path, row), "H13")


def test_old_version_skips_versioned_checks(tmp_path: Path) -> None:
    row = sample()
    row["metadata"] = {"dataset_version": "cuda-sft-1", "system_mode": "fixed", "release_tier": "compile_only"}
    report = run_check(tmp_path, row)
    assert report["ok"]
    assert report["checks"]["H5"]["n/a"] == 1
    assert report["checks"]["H13"]["n/a"] == 1


def test_s1_knowledge_skips_kernel_fields(tmp_path: Path) -> None:
    report = run_check(tmp_path, knowledge_sample())
    assert report["ok"], report["failures"]
    assert report["checks"]["H4"]["n/a"] == 1
    assert report["checks"]["H8"]["n/a"] == 1
    assert report["checks"]["H13"]["pass"] == 1


def test_knowledge_h13_accepts_legacy_optional_review_fields(tmp_path: Path) -> None:
    row = knowledge_sample()
    del row["metadata"]["knowledge_judge"]["topic"]
    del row["metadata"]["knowledge_judge"]["judge_error"]
    report = run_check(tmp_path, row)
    assert report["ok"], report["failures"]


@pytest.mark.parametrize(("field", "value"), [
    ("pass", False),
    ("unavailable", True),
    ("skipped_llm", True),
    ("hard_gate_failed", True),
    ("judge_error", "provider error"),
    ("must_fix", ["wrong fact"]),
    ("answer_sha256", "0" * 64),
    ("overall", 0.0),
    ("overall", 10**1000),
    ("dimensions", {"factual": 8.0}),
])
def test_knowledge_h13_rejects_invalid_judge_evidence(
    tmp_path: Path, field: str, value: object
) -> None:
    row = knowledge_sample()
    row["metadata"]["knowledge_judge"][field] = value
    assert_failure(run_check(tmp_path, row), "H13")


def test_knowledge_h13_rejects_missing_or_stale_review(tmp_path: Path) -> None:
    row = knowledge_sample()
    del row["metadata"]["knowledge_judge"]
    assert_failure(run_check(tmp_path, row), "H13")
    row = knowledge_sample()
    row["messages"][-1]["content"] += "\nAltered after review."
    assert_failure(run_check(tmp_path, row), "H13")


def test_cli_exit_codes_and_report(tmp_path: Path) -> None:
    good = sample()
    source = tmp_path / "sft.jsonl"
    out = tmp_path / "report.json"
    for row, expected in ((good, 0), (deepcopy(good), 1)):
        if expected:
            row["metadata"]["critic"] = {}
        source.write_text(json.dumps(row) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "check_dataset.py"), str(source), "--out", str(out)],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        assert proc.returncode == expected, proc.stderr
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["ok"] is (expected == 0)
        assert report["rows"] == 1


def test_empty_file_fails_h1(tmp_path: Path) -> None:
    report = run_check(tmp_path)
    assert_failure(report, "H1")


def test_blank_line_inside_dataset_fails_h1(tmp_path: Path) -> None:
    report = run_check(tmp_path, sample(), "")
    assert_failure(report, "H1")
    assert report["rows"] == 2


def test_r01_s1_scenario_output_passes_l6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = load_scenario(ROOT / "tests" / "fixtures" / "scenarios" / "R01_happy_path.yaml")
    result = run_scenario(scenario, tmp_path, monkeypatch)
    assert result.samples and result.finals[0]["status"] == "success"
    report = check_dataset(tmp_path / "data" / "sft.jsonl")
    assert report["ok"], report["failures"]


def test_r02_repaired_sample_passes_l6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = load_scenario(ROOT / "tests" / "fixtures" / "scenarios" / "R02_compile_fail_then_repair.yaml")
    result = run_scenario(scenario, tmp_path, monkeypatch)
    assert result.samples and result.finals[0]["status"] == "success"
    report = check_dataset(tmp_path / "data" / "sft.jsonl")
    assert report["ok"], report["failures"]


def test_r03_winner_sample_passes_l6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = load_scenario(ROOT / "tests" / "fixtures" / "scenarios" / "R03_winner_not_last.yaml")
    result = run_scenario(scenario, tmp_path, monkeypatch)
    assert result.samples and result.finals[0]["status"] == "success"
    report = check_dataset(tmp_path / "data" / "sft.jsonl")
    assert report["ok"], report["failures"]


def test_r08_unoptimized_sample_passes_l6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = load_scenario(ROOT / "tests" / "fixtures" / "scenarios" / "R08_judge_optimization_ignored.yaml")
    result = run_scenario(scenario, tmp_path, monkeypatch)
    assert result.samples and result.finals[0]["status"] == "success"
    report = check_dataset(tmp_path / "data" / "sft.jsonl")
    assert report["ok"], report["failures"]


def test_k01_knowledge_sample_passes_l6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = load_scenario(ROOT / "tests" / "fixtures" / "scenarios" / "K01_knowledge_repair_then_pass.yaml")
    result = run_scenario(scenario, tmp_path, monkeypatch)
    assert result.samples and result.finals[0]["status"] == "success"
    report = check_dataset(tmp_path / "data" / "sft.jsonl")
    assert report["ok"], report["failures"]
    assert report["checks"]["H8"]["n/a"] == 1
    assert report["checks"]["H13"]["pass"] == 1
