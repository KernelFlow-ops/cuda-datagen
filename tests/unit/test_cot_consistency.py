"""CoT prose may describe only the selected source without repair history."""

import pytest

from cuda_sft.core.cot import cot_consistency_issues, cot_consistency_report


@pytest.mark.parametrize(
    "text",
    ["编译错误后修复", "修复上一版", "The previous attempt failed", "debugging the kernel", "nvcc -c failed", "I fixed the kernel"],
)
def test_repair_narrative_is_rejected(text: str) -> None:
    assert any(issue.startswith("repair leak:") for issue in cot_consistency_issues(text, "int kernel;"))


def test_unknown_backticked_identifier_is_a_soft_issue() -> None:
    hard, soft = cot_consistency_report("Use `missing_kernel` for each element.", "void actual_kernel() {}")
    assert hard == []
    assert soft == ["unknown identifier: missing_kernel"]


def test_builtin_and_source_identifiers_are_allowed() -> None:
    assert cot_consistency_report(
        "Use `threadIdx` and `vector_add_kernel`.",
        "__global__ void vector_add_kernel() { int i = threadIdx.x; }",
    ) == ([], [])


def test_constant_mismatch_is_rejected() -> None:
    issues = cot_consistency_issues("BLOCK = 256 threads.", "const int BLOCK = 128;")
    assert "constant mismatch: BLOCK cot=256 code=128" in issues


def test_words_containing_fix_are_not_repair_leaks() -> None:
    assert cot_consistency_issues("prefix scan over each row", "int prefix = 0;") == []
    assert cot_consistency_issues("Bandwidth is not fixed across systems.", "int kernel;") == []
    assert cot_consistency_issues(
        "Higher precision cannot fix ill-conditioned problems or repair rounding errors.",
        "int kernel;",
    ) == []
