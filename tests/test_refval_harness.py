"""CUDA harness: correct kernel passes, deliberately wrong kernel is caught.

GPU tests skip when nvcc or nvidia-smi is missing. Compare-level catching of
a wrong kernel is always asserted so CI without a GPU still gates the logic.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from concurrent.futures import TimeoutError as FuturesTimeoutError
from pathlib import Path
from unittest.mock import patch

import pytest

from cuda_sft.config import Settings
from cuda_sft.dialects.cuda import CudaDialect
from cuda_sft.parse import abi_matches_source
from cuda_sft.refval.cases import numpy_available
from cuda_sft.refval.harness import compile_cuda_harness, render_cuda_harness
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


def test_aliased_output_zero_guard_uses_allocated_input() -> None:
    plan = CasePlan(
        name="inplace",
        kind="inplace",
        shapes={"a": (33,), "b": (33,), "c": (33,)},
        scalars={"n": 33},
        seed=1,
        alias={"c": "a"},
    )
    harness = render_cuda_harness(_add_abi(), [plan])
    assert "ne_c" not in harness
    assert "solution(d_a, d_b, d_a, 33)" not in harness
    assert "launch_add(d_a, d_b, d_a, 33)" in harness


def test_cuda_harness_links_only_libraries_used_by_solution() -> None:
    import subprocess

    settings = Settings()
    library_source = (FIXTURES / "cuda_library_link.cu").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        testdir = Path(tmp)
        (testdir / "harness.cu").write_text('#include "solution.cu"\nint main() { return 0; }\n')
        for source, expected in (
            ("void solution() {}", set()),
            (library_source, {"-lcufft", "-lcurand", "-lcublas"}),
        ):
            (testdir / "solution.cu").write_text(source)
            with patch("cuda_sft.refval.harness.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess([], 0, "", "")
                ok, _, _ = compile_cuda_harness(testdir, settings=settings)
            assert ok
            cmd = run.call_args.args[0]
            assert {arg for arg in cmd if arg in {"-lcufft", "-lcurand", "-lcublas"}} == expected


@pytest.mark.gpu
def test_cuda_harness_links_cuda_library_fixture() -> None:
    reason = _gpu_ready()
    if reason:
        pytest.skip(reason)
    with tempfile.TemporaryDirectory() as tmp:
        testdir = Path(tmp)
        (testdir / "solution.cu").write_text(
            (FIXTURES / "cuda_library_link.cu").read_text(encoding="utf-8")
        )
        (testdir / "harness.cu").write_text('#include "solution.cu"\nint main() { return 0; }\n')
        ok, output, binary = compile_cuda_harness(testdir, settings=Settings())
        assert ok, output
        assert binary is not None and binary.is_file()


@pytest.mark.gpu
def test_cublas_handle_abi_runs_on_gpu() -> None:
    reason = _gpu_ready()
    if reason:
        pytest.skip(reason)
    source = (FIXTURES / "cublas_handle_gemm.cu").read_text(encoding="utf-8")
    abi = KernelABI(
        entry="gemm_with_init",
        params=(
            KernelParam("handle", "scalar", "cublas_handle", rank=0),
            KernelParam("A", "input", "f64", rank=2, shape_from=("m", "k")),
            KernelParam("B", "input", "f64", rank=2, shape_from=("k", "n")),
            KernelParam("C", "output", "f64", rank=2, shape_from=("m", "n")),
            KernelParam("m", "size", "i32", rank=0),
            KernelParam("n", "size", "i32", rank=0),
            KernelParam("k", "size", "i32", rank=0),
            KernelParam("init_value", "scalar", "i32", rank=0),
        ),
        dtype="f64",
        returns="i32",
    )
    assert abi_matches_source(abi, source) == []
    wrong = KernelABI(abi.entry, (KernelParam("handle", "scalar", "f32", rank=0), *abi.params[1:]), dtype="f64", returns="i32")
    assert any("type mismatch" in issue for issue in abi_matches_source(wrong, source))
    reference = """def reference(handle, A, B, C, m, n, k, init_value):
    result = np.asarray(A).reshape(m, k) @ np.asarray(B).reshape(k, n)
    return {"C": (result + init_value).astype(np.float64), "__return__": np.int32(0)}
"""
    settings = Settings(refval_enabled=True, refval_cases="smoke", async_llm_enabled=False)
    with tempfile.TemporaryDirectory() as tmp:
        report = run_refval(
            question="row-major FP64 GEMM with cublasHandle_t",
            code=source,
            question_id=23,
            dialect="cuda",
            dialect_spec=CudaDialect().refval_spec(settings),
            settings=settings,
            workdir=Path(tmp),
            manifest=RefManifest(
                question_id=23,
                dialect="cuda",
                abi=abi,
                reference_source=reference,
                extracted_from="injected",
            ),
        )
    assert report.status == "pass", report.reason or report.evidence
    assert report.cases_run == 3


@pytest.mark.gpu
def test_cufft_2d_handle_and_complex_input_run_on_gpu() -> None:
    reason = _gpu_ready()
    if reason:
        pytest.skip(reason)
    source = (FIXTURES / "cufft_handle_magnitude.cu").read_text(encoding="utf-8")
    abi = KernelABI(
        entry="fft2_magnitude",
        params=(
            KernelParam("plan", "scalar", "cufft_handle", rank=0),
            KernelParam("input", "input", "c64", rank=2, shape_from=("rows", "cols")),
            KernelParam("output", "output", "f32", rank=2, shape_from=("rows", "cols")),
            KernelParam("rows", "size", "i32", rank=0),
            KernelParam("cols", "size", "i32", rank=0),
        ),
    )
    assert abi_matches_source(abi, source) == []
    reference = """def reference(plan, input, output, rows, cols):
    if rows == 0 or cols == 0:
        return {"output": np.zeros((rows, cols), dtype=np.float32)}
    spectrum = np.fft.fft2(np.asarray(input, dtype=np.complex64))
    return {"output": np.abs(spectrum).astype(np.float32)}
"""
    settings = Settings(refval_enabled=True, refval_cases="smoke", async_llm_enabled=False)
    with tempfile.TemporaryDirectory() as tmp:
        report = run_refval(
            question="2D complex FFT magnitude using provided cufftHandle",
            code=source,
            question_id=20,
            dialect="cuda",
            dialect_spec=CudaDialect().refval_spec(settings),
            settings=settings,
            workdir=Path(tmp),
            manifest=RefManifest(
                question_id=20,
                dialect="cuda",
                abi=abi,
                reference_source=reference,
                extracted_from="injected",
            ),
        )
    assert report.status == "pass", report.reason or report.evidence
    assert report.cases_run == 3


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
            got,
            exp,
            dtype="f32",
            tolerances=None,
            allows_nan=False,
            sort_mode="elementwise",
            name="wrong",
            seed=1,
        )
        self.assertFalse(result.ok)


@pytest.mark.gpu
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
        with (
            patch("cuda_sft.refval.runner._load_cache", return_value=cached),
            patch(
                "cuda_sft.refval.runner._llm_complete",
                side_effect=AssertionError("valid cache must not call LLM"),
            ),
        ):
            result = obtain_manifest(
                question="vector add",
                code=(FIXTURES / "add_ok.cu").read_text(encoding="utf-8"),
                question_id=9,
                dialect="cuda",
                settings=settings,
            )
        self.assertIs(result, cached)


def test_speculative_future_timeout_is_not_reissued_as_a_new_extract() -> None:
    class TimedOutPool:
        calls = 0

        def get(self, _request_id: str, *, timeout_sec: float) -> None:
            self.calls += 1
            raise FuturesTimeoutError("still running")

    pool = TimedOutPool()
    settings = Settings(
        refval_enabled=True,
        refval_cache=False,
        async_llm_enabled=True,
        refval_extract_timeout_sec=5,
        work_dir=tempfile.mkdtemp(),
    )
    source = (FIXTURES / "add_ok.cu").read_text(encoding="utf-8")
    with (
        patch("cuda_sft.llm_async.get_async_pool", return_value=pool),
        patch(
            "cuda_sft.refval.runner._llm_complete",
            side_effect=AssertionError("timed out request must not trigger a duplicate call"),
        ),
    ):
        result = obtain_manifest(
            question="vector add",
            code=source,
            question_id=17,
            dialect="cuda",
            settings=settings,
            speculative_id="q17_cuda_c1_refval",
        )
    assert result is None
    assert pool.calls == 1


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
        with (
            patch("cuda_sft.refval.runner._try_speculative", return_value=""),
            patch(
                "cuda_sft.refval.runner._llm_complete",
                side_effect=[abi_only, good],
            ) as complete,
        ):
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
        with (
            patch("cuda_sft.refval.runner._try_speculative", return_value=""),
            patch(
                "cuda_sft.refval.runner._llm_complete",
                side_effect=["not json", "still not json"],
            ),
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
        payload = {"ok": True, "cases": [{"name": "smoke", "ok": True}]}
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("cuda_sft.refval.runner.validate_reference_fn", return_value=None),
            patch("cuda_sft.refval.runner.prepare_testdir"),
            patch(
                "cuda_sft.refval.runner.compile_cuda_harness",
                return_value=(True, "", Path("refval_bin")),
            ),
            patch("cuda_sft.refval.runner.GpuFileLock.acquire", return_value=True),
            patch("cuda_sft.refval.runner.run_binary", return_value=(0, "", payload)),
            patch(
                "cuda_sft.refval.runner.bind_and_call",
                side_effect=ValueError("cannot reshape array of size 15 into shape (15,7)"),
            ),
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
