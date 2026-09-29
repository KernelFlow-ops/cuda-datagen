"""Classify gate failures by the component that can fix them."""

from __future__ import annotations

import re
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cuda_sft.agents.repairer import classify_compile_error
from cuda_sft.compile import CompileResult
from cuda_sft.config import get_settings
from cuda_sft.refval.runner import (
    REASON_EXTRACT_FAILED,
    REASON_GPU_LOCK_TIMEOUT,
    REASON_PREPARE_HARNESS,
    REASON_TIMEOUT_BEFORE_GPU,
)
from cuda_sft.refval.spec import RefvalReport


class FailureOwner(str, Enum):
    KERNEL = "KERNEL"
    ORACLE = "ORACLE"
    INFRA = "INFRA"


class ErrorClass(str, Enum):
    EMPTY_SOURCE = "empty_source"
    SYNTAX = "syntax"
    UNDECLARED = "undeclared"
    MISSING_HEADER = "missing_header"
    TEMPLATE = "template"
    DIALECT_VIOLATION = "dialect_violation"
    IMPORT_TIME = "import_time"
    JIT_COMPILE = "jit_compile"
    CONTRACT_MISMATCH = "contract_mismatch"
    SIGNATURE_MISMATCH = "signature_mismatch"
    NUMERIC_MISMATCH = "numeric_mismatch"
    NAN_INF = "nan_inf"
    CRASH = "crash"
    ILLEGAL_MEMORY = "cuda_illegal_memory"
    KERNEL_TIMEOUT = "timeout"
    OUTPUT_MISSING = "output_missing"
    SEMANTIC_MUST_FIX = "semantic_must_fix"
    OTHER_COMPILE = "other"
    REFERENCE_ERROR = "reference_error"
    EXTRACT_FAILED = "extract_failed"
    SPEC_UNCERTAIN = "oracle_uncertain"
    GPU_SLOT_TIMEOUT = "gpu_slot_timeout"
    EXTRACT_TIMEOUT = "extract_timeout"
    HARNESS_BUILD_ERROR = "harness_build_error"
    HARNESS_PREP_ERROR = "harness_prep_error"
    TOOLCHAIN_MISSING = "toolchain_missing"
    LLM_UNAVAILABLE = "llm_unavailable"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    owner: FailureOwner | None
    error_class: str
    evidence: str
    metrics: dict[str, Any]

    def __post_init__(self) -> None:
        limit = int(get_settings().repair_error_max_chars)
        if len(self.evidence) > limit:
            object.__setattr__(self, "evidence", self.evidence[:limit])

    def to_dict(self) -> dict[str, Any]:
        """Return trace-safe values, with the enum serialized as its value."""
        return {
            "gate": self.gate,
            "passed": self.passed,
            "owner": self.owner.value if self.owner is not None else None,
            "error_class": self.error_class,
            "evidence": self.evidence,
            "metrics": dict(self.metrics),
        }


def _harness_only_errors(reason: str) -> bool:
    error_lines = [line for line in reason.splitlines() if re.search(r"\berror\b", line, re.I)]
    return bool(error_lines) and all(
        "harness.cu" in line.lower() and "solution.cu" not in line.lower() for line in error_lines
    )


def _missing_cufft_link(reason: str) -> bool:
    return "-lcufft" not in reason and bool(
        re.search(r"undefined reference to\s+[`'\"]?cufft[A-Za-z0-9_]*", reason, re.I)
    )


def owner_of_refval(report: RefvalReport) -> tuple[FailureOwner | None, str]:
    """Apply failure-taxonomy rules in their specified precedence order."""
    status = report.status
    reason = report.reason or ""
    error_class = report.error_class or ""
    if status == "pass":
        return None, ""
    if status == "skip":
        if any(term in reason.lower() for term in ("not installed", "no gpu", "not available")):
            return FailureOwner.INFRA, ErrorClass.TOOLCHAIN_MISSING.value
        return None, ""
    if reason == REASON_EXTRACT_FAILED:
        return FailureOwner.ORACLE, ErrorClass.EXTRACT_FAILED.value
    if error_class == "invalid_oracle":
        return FailureOwner.ORACLE, error_class
    if status == "reference_error" or error_class == "reference_error":
        return FailureOwner.ORACLE, ErrorClass.REFERENCE_ERROR.value
    if reason == REASON_TIMEOUT_BEFORE_GPU:
        return FailureOwner.INFRA, ErrorClass.EXTRACT_TIMEOUT.value
    if reason == REASON_GPU_LOCK_TIMEOUT or error_class == ErrorClass.GPU_SLOT_TIMEOUT.value:
        return FailureOwner.INFRA, ErrorClass.GPU_SLOT_TIMEOUT.value
    if error_class == "crash" and reason.startswith(REASON_PREPARE_HARNESS):
        return FailureOwner.INFRA, ErrorClass.HARNESS_PREP_ERROR.value
    if error_class == "signature_mismatch" and _harness_only_errors(reason):
        return FailureOwner.INFRA, ErrorClass.HARNESS_BUILD_ERROR.value
    if error_class == "signature_mismatch" and _missing_cufft_link(reason):
        return FailureOwner.INFRA, ErrorClass.HARNESS_BUILD_ERROR.value
    if error_class == "signature_mismatch" and reason.startswith("nvcc timed out"):
        return FailureOwner.INFRA, ErrorClass.HARNESS_BUILD_ERROR.value
    if error_class == "signature_mismatch" and reason.startswith(
        ("failed to launch nvcc", "FileNotFoundError")
    ):
        return FailureOwner.INFRA, ErrorClass.TOOLCHAIN_MISSING.value
    if status == "fail":
        return FailureOwner.KERNEL, error_class or ErrorClass.NUMERIC_MISMATCH.value
    return None, ""


def owner_of_compile(result: CompileResult, error_text: str) -> tuple[FailureOwner | None, str]:
    """Treat compiler launch failures as infrastructure, other errors as kernel."""
    if result.ok:
        return None, ""
    details = error_text or result.output or ""
    lowered = details.lower()
    if (
        "failed to launch nvcc" in lowered
        or "filenotfounderror" in lowered
        or (not (result.output or "").strip() and ("timeout" in lowered or "timed out" in lowered))
    ):
        return FailureOwner.INFRA, ErrorClass.TOOLCHAIN_MISSING.value
    code = ""
    if result.source_path is not None:
        with suppress(OSError):
            code = result.source_path.read_text(encoding="utf-8")
    return FailureOwner.KERNEL, classify_compile_error(details, dialect=result.dialect, code=code)
