"""CUDA harness: correct kernel passes, deliberately wrong kernel is caught.

GPU tests skip when nvcc or nvidia-smi is missing. Compare-level catching of
a wrong kernel is always asserted so CI without a GPU still gates the logic.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from cuda_sft.config import Settings
from cuda_sft.dialects.cuda import CudaDialect
from cuda_sft.refval.cases import numpy_available
from cuda_sft.refval.runner import obtain_manifest, run_refval
from cuda_sft.refval.spec import CasePlan, KernelABI, KernelParam, RefManifest, seed_for

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "refval"

ADD_REF = """
def reference(a, b, n):
    return {"c": np.asarray(a) + np.asarray(b)}
"""


def _add_abi() -> KernelABI:
    return KernelABI(
        entry="launch_add",
        params=(
            KernelParam("a", "input", "f32", rank=1, shape_from=("n",)),
            KernelParam("b", "input", "f32", rank=1, shape_from=("n",)),
            KernelParam("c", "output", "f32", rank=1, shape_from=("n",)),
            KernelParam("n", "size", "i32", rank=0),
        ),
        dtype="f32",
    )


def _gpu_ready() -> str | None:
    if shutil.which("nvcc") is None:
        return "nvcc missing"
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi missing"
    if not numpy_available():
        return "numpy missing"
    return None


def _manifest(qid: int = 1) -> RefManifest:
    return RefManifest(
        question_id=qid,
        dialect="cuda",
        abi=_add_abi(),
        reference_source=ADD_REF,
        reference_fn_name="reference",
        seed=seed_for(qid, "cuda"),
        extracted_from="injected",
    )


class MockWrongKernelTests(unittest.TestCase):
    def test_compare_catches_plus_one(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        import numpy as np

        from cuda_sft.refval.compare import compare_arrays

        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        got = a + b + 1.0
        exp = a + b
        result = compare_arrays(
            got, exp, dtype="f32", tolerances=None, allows_nan=False,
            sort_mode="elementwise", name="wrong", seed=1,
        )
        self.assertFalse(result.ok)


class GpuHarnessTests(unittest.TestCase):
    def test_correct_add_passes(self) -> None:
        reason = _gpu_ready()
        if reason:
            self.skipTest(reason)
        source = (FIXTURES / "add_ok.cu").read_text(encoding="utf-8")
        settings = Settings(
            refval_enabled=True,
            refval_cases="smoke",
            refval_timeout_sec=45,
            async_llm_enabled=False,
            workers=1,
        )
        dialect = CudaDialect().refval_spec(settings)
        with tempfile.TemporaryDirectory() as tmp:
            report = run_refval(
                question="vector add",
                code=source,
                question_id=1,
                dialect="cuda",
                dialect_spec=dialect,
                settings=settings,
                workdir=Path(tmp),
                manifest=_manifest(1),
            )
        self.assertEqual(report.status, "pass", report.reason or report.evidence)
        self.assertGreaterEqual(report.cases_run, 1)

    def test_wrong_add_fails(self) -> None:
        reason = _gpu_ready()
        if reason:
            self.skipTest(reason)
        source = (FIXTURES / "add_wrong.cu").read_text(encoding="utf-8")
        settings = Settings(
            refval_enabled=True,
            refval_cases="smoke",
            refval_timeout_sec=45,
            async_llm_enabled=False,
            workers=1,
        )
        dialect = CudaDialect().refval_spec(settings)
        with tempfile.TemporaryDirectory() as tmp:
            report = run_refval(
                question="vector add",
                code=source,
                question_id=2,
                dialect="cuda",
                dialect_spec=dialect,
                settings=settings,
                workdir=Path(tmp),
                manifest=_manifest(2),
            )
        self.assertEqual(report.status, "fail", report.reason)
        self.assertEqual(report.error_class, "numeric_mismatch")
        self.assertTrue(report.failed_case)


class SkipTests(unittest.TestCase):
    def test_disabled_skips(self) -> None:
        settings = Settings(refval_enabled=False, async_llm_enabled=False)
        dialect = CudaDialect().refval_spec(settings)
        report = run_refval(
            question="q",
            code="int x;",
            question_id=1,
            dialect="cuda",
            dialect_spec=dialect,
            settings=settings,
            manifest=_manifest(),
        )
        self.assertEqual(report.status, "skip")

    def test_valid_cache_manifest_is_reused(self) -> None:
        settings = Settings(
            refval_enabled=True,
            refval_cache=True,
            async_llm_enabled=False,
            work_dir=tempfile.mkdtemp(),
        )
        cached = _manifest(9)
        with patch("cuda_sft.refval.runner._load_cache", return_value=cached), patch(
            "cuda_sft.refval.runner._llm_complete",
            side_effect=AssertionError("valid cache must not call LLM"),
        ):
            result = obtain_manifest(
                question="vector add",
                code=(FIXTURES / "add_ok.cu").read_text(encoding="utf-8"),
                question_id=9,
                dialect="cuda",
                settings=settings,
            )
        self.assertIs(result, cached)


class ExtractRetryTests(unittest.TestCase):
    def test_abi_only_json_is_retried_before_heuristic(self) -> None:
        source = (FIXTURES / "add_ok.cu").read_text(encoding="utf-8")
        abi_only = json.dumps({"abi": _add_abi().to_dict(), "reference_source": ""})
        good = json.dumps(
            {
                "abi": _add_abi().to_dict(),
                "reference_fn_name": "reference",
                "reference_source": ADD_REF,
            }
        )
        settings = Settings(
            refval_enabled=True,
            refval_cache=False,
            async_llm_enabled=False,
            work_dir=tempfile.mkdtemp(),
        )
        with patch("cuda_sft.refval.runner._try_speculative", return_value=""), patch(
            "cuda_sft.refval.runner._llm_complete",
            side_effect=[abi_only, good],
        ) as complete:
            result = obtain_manifest(
                question="vector add",
                code=source,
                question_id=11,
                dialect="cuda",
                settings=settings,
            )
        self.assertEqual(complete.call_count, 2)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.extracted_from, "llm")
        self.assertIn("def reference", result.reference_source)
        self.assertNotEqual(result.extracted_from, "heuristic")

    def test_unusable_extract_still_falls_back_to_heuristic(self) -> None:
        source = (FIXTURES / "add_ok.cu").read_text(encoding="utf-8")
        settings = Settings(
            refval_enabled=True,
            refval_cache=False,
            async_llm_enabled=False,
            work_dir=tempfile.mkdtemp(),
        )
        with patch("cuda_sft.refval.runner._try_speculative", return_value=""), patch(
            "cuda_sft.refval.runner._llm_complete",
            side_effect=["not json", "still not json"],
        ):
            result = obtain_manifest(
                question="vector add",
                code=source,
                question_id=12,
                dialect="cuda",
                settings=settings,
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.extracted_from, "heuristic")
        self.assertEqual(result.reference_source, "")


class ReferenceCaseErrorTests(unittest.TestCase):
    def test_reference_reshape_does_not_crash_the_question(self) -> None:
        source = (FIXTURES / "add_ok.cu").read_text(encoding="utf-8")
        settings = Settings(
            refval_enabled=True,
            refval_cases="smoke",
            refval_timeout_sec=30,
            async_llm_enabled=False,
            workers=1,
            refval_cache=False,
        )
        dialect = CudaDialect().refval_spec(settings)
        plan = CasePlan(
            name="smoke",
            kind="smoke",
            shapes={"a": (4,), "b": (4,), "c": (4,)},
            scalars={"n": 4},
            seed=1,
        )
        payload = {"cases": [{"name": "smoke", "ok": True}]}
        with tempfile.TemporaryDirectory() as tmp, patch(
            "cuda_sft.refval.runner.validate_reference_fn", return_value=None
        ), patch("cuda_sft.refval.runner.prepare_testdir"), patch(
            "cuda_sft.refval.runner.run_prepared", return_value=(0, "", payload)
        ), patch(
            "cuda_sft.refval.runner.bind_and_call",
            side_effect=ValueError("cannot reshape array of size 15 into shape (15,7)"),
        ):
            report = run_refval(
                question="vector add",
                code=source,
                question_id=15,
                dialect="cuda",
                dialect_spec=dialect,
                settings=settings,
                workdir=Path(tmp),
                manifest=_manifest(15),
                case_plans=[plan],
            )
        self.assertEqual(report.status, "fail")
        self.assertEqual(report.error_class, "reference_error")
        self.assertIn("cannot reshape", report.reason)


if __name__ == "__main__":
    unittest.main()
