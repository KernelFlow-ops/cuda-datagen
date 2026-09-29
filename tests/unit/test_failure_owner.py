"""Failure ownership follows the ordered taxonomy, not the surface status."""

from types import SimpleNamespace

import pytest

from cuda_sft.compile import CompileResult
from cuda_sft.core import gates
from cuda_sft.core.gates import (
    ErrorClass,
    FailureOwner,
    GateResult,
    owner_of_compile,
    owner_of_refval,
)
from cuda_sft.refval.runner import (
    REASON_EXTRACT_FAILED,
    REASON_GPU_LOCK_TIMEOUT,
    REASON_PREPARE_HARNESS,
    REASON_TIMEOUT_BEFORE_GPU,
)
from cuda_sft.refval.spec import RefvalReport


@pytest.mark.parametrize(
    ("status", "reason", "error_class", "expected"),
    [
        ("pass", "", "", (None, "")),
        ("skip", "no GPU (nvidia-smi missing)", "", (FailureOwner.INFRA, "toolchain_missing")),
        ("skip", "refval disabled", "", (None, "")),
        (
            "reference_error",
            REASON_EXTRACT_FAILED,
            "reference_error",
            (FailureOwner.ORACLE, "extract_failed"),
        ),
        (
            "reference_error",
            "reference raised",
            "reference_error",
            (FailureOwner.ORACLE, "reference_error"),
        ),
        ("fail", "reference raised", "reference_error", (FailureOwner.ORACLE, "reference_error")),
        ("fail", REASON_TIMEOUT_BEFORE_GPU, "timeout", (FailureOwner.INFRA, "extract_timeout")),
        ("fail", REASON_GPU_LOCK_TIMEOUT, "timeout", (FailureOwner.INFRA, "gpu_slot_timeout")),
        (
            "fail",
            f"{REASON_PREPARE_HARNESS}: bad ABI",
            "crash",
            (FailureOwner.INFRA, "harness_prep_error"),
        ),
        (
            "fail",
            "harness.cu(42): error: generated harness failed",
            "signature_mismatch",
            (FailureOwner.INFRA, "harness_build_error"),
        ),
        (
            "fail",
            "solution.cu(42): error: bad signature",
            "signature_mismatch",
            (FailureOwner.KERNEL, "signature_mismatch"),
        ),
        ("fail", "wrong values", "", (FailureOwner.KERNEL, "numeric_mismatch")),
    ],
)
def test_refval_taxonomy(status, reason, error_class, expected) -> None:
    report = RefvalReport(status=status, dialect="cuda", reason=reason, error_class=error_class)
    assert owner_of_refval(report) == expected


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (
            "nvcc harness.cu -o refval_bin\nundefined reference to `cufftExecC2C'",
            (FailureOwner.INFRA, "harness_build_error"),
        ),
        (
            "nvcc harness.cu -o refval_bin -lcufft\nundefined reference to `cufftExecC2C'",
            (FailureOwner.KERNEL, "signature_mismatch"),
        ),
        ("undefined reference to `my_kernel'", (FailureOwner.KERNEL, "signature_mismatch")),
        ("nvcc timed out after 60s", (FailureOwner.INFRA, "harness_build_error")),
        ("solution compile timeout", (FailureOwner.KERNEL, "signature_mismatch")),
        ("failed to launch nvcc: executable missing", (FailureOwner.INFRA, "toolchain_missing")),
        ("FileNotFoundError: nvcc", (FailureOwner.INFRA, "toolchain_missing")),
        ("solution.cu: FileNotFoundError", (FailureOwner.KERNEL, "signature_mismatch")),
    ],
)
def test_cufft_linkage_exception_is_narrow(reason, expected) -> None:
    report = RefvalReport(
        status="fail", dialect="cuda", reason=reason, error_class="signature_mismatch"
    )
    assert owner_of_refval(report) == expected


@pytest.mark.parametrize(
    ("ok", "output", "error_text", "expected"),
    [
        (True, "", "", (None, "")),
        (False, "solution.cu(4): error: expected a ';'", "", (FailureOwner.KERNEL, "syntax")),
        (False, "failed to launch nvcc: missing", "", (FailureOwner.INFRA, "toolchain_missing")),
        (False, "", "nvcc timed out", (FailureOwner.INFRA, "toolchain_missing")),
        (False, "nvcc timed out after 10s", "", (FailureOwner.KERNEL, "other")),
    ],
)
def test_compile_taxonomy(ok, output, error_text, expected) -> None:
    result = CompileResult(ok=ok, command=["nvcc"], output=output, used_rdc=False)
    assert owner_of_compile(result, error_text) == expected


def test_gate_result_serializes_owner_and_bounds_evidence(monkeypatch) -> None:
    monkeypatch.setattr(gates, "get_settings", lambda: SimpleNamespace(repair_error_max_chars=5))
    result = GateResult(
        gate="refval",
        passed=False,
        owner=FailureOwner.ORACLE,
        error_class=ErrorClass.REFERENCE_ERROR.value,
        evidence="long evidence",
        metrics={"cases_run": 2},
    )
    assert result.to_dict() == {
        "gate": "refval",
        "passed": False,
        "owner": "ORACLE",
        "error_class": "reference_error",
        "evidence": "long ",
        "metrics": {"cases_run": 2},
    }
