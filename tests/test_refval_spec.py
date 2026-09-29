"""Frozen dataclass round-trips and seed stability."""

from __future__ import annotations

import json
import unittest

from cuda_sft.refval.spec import (
    CasePlan,
    CaseResult,
    KernelABI,
    KernelParam,
    RefManifest,
    RefvalReport,
    dumps,
    loads_manifest,
    normalize_dtype,
    cpp_type,
    dtype_nbytes,
    numpy_dtype_name,
    seed_for,
)


class SeedTests(unittest.TestCase):
    def test_stable_across_calls(self) -> None:
        self.assertEqual(seed_for(42, "cuda"), seed_for(42, "cuda"))
        self.assertNotEqual(seed_for(42, "cuda"), seed_for(42, "triton"))
        self.assertNotEqual(seed_for(1, "cuda"), seed_for(2, "cuda"))
        self.assertGreaterEqual(seed_for(1, "cuda"), 0)


class DtypeTests(unittest.TestCase):
    def test_aliases(self) -> None:
        self.assertEqual(normalize_dtype("float"), "f32")
        self.assertEqual(normalize_dtype("float32"), "f32")
        self.assertEqual(normalize_dtype("int32_t"), "i32")
        self.assertEqual(normalize_dtype("__half"), "f16")
        self.assertEqual(normalize_dtype("unsigned short"), "u16")
        self.assertEqual(normalize_dtype("uint16_t"), "u16")
        self.assertEqual(normalize_dtype("cublasHandle_t"), "cublas_handle")
        self.assertEqual(normalize_dtype("cufftHandle"), "cufft_handle")
        self.assertEqual(normalize_dtype("cufftComplex"), "c64")
        self.assertEqual(numpy_dtype_name("c64"), "complex64")
        self.assertEqual(dtype_nbytes("c64"), 8)
        self.assertEqual(cpp_type("cublas_handle"), "cublasHandle_t")
        self.assertEqual(
            KernelParam.from_dict({"name": "handle", "kind": "scalar", "dtype": "cublasHandle_t"}).dtype,
            "cublas_handle",
        )
        self.assertEqual(cpp_type("u16", pointer=True), "unsigned short*")
        self.assertEqual(numpy_dtype_name("u16"), "uint16")
        self.assertEqual(dtype_nbytes("u16"), 2)


class RoundTripTests(unittest.TestCase):
    def test_abi_and_manifest(self) -> None:
        abi = KernelABI(
            entry="launch_add",
            params=(
                KernelParam("a", "input", "f32", rank=1, shape_from=("n",)),
                KernelParam("b", "input", "f32", rank=1, shape_from=("n",)),
                KernelParam("c", "output", "f32", rank=1, shape_from=("n",)),
                KernelParam("n", "size", "i32", rank=0),
            ),
            dtype="f32",
        )
        self.assertEqual(abi.issues(), [])
        self.assertEqual(abi.result_names(), ("c",))
        manifest = RefManifest(
            question_id=1,
            dialect="cuda",
            abi=abi,
            reference_source="def reference(a, b, n):\n    return {'c': a + b}\n",
            seed=seed_for(1, "cuda"),
        )
        loaded = loads_manifest(dumps(manifest))
        self.assertEqual(loaded.abi.entry, "launch_add")
        self.assertEqual(loaded.abi.params[0].shape_from, ("n",))
        extra = json.loads(dumps(manifest))
        extra["unknown_future_field"] = 1
        again = RefManifest.from_dict(extra)
        self.assertEqual(again.abi.entry, "launch_add")

    def test_report_metadata_subset(self) -> None:
        report = RefvalReport(
            status="fail",
            dialect="cuda",
            cases_run=3,
            failed_case="odd_7",
            error_class="numeric_mismatch",
            seed=9,
            results=[CaseResult(name="odd_7", ok=False, status="fail")],
        )
        meta = report.to_metadata()
        self.assertEqual(meta["status"], "fail")
        self.assertEqual(meta["failed_case"], "odd_7")
        self.assertNotIn("results", meta)
        self.assertNotIn("evidence", meta)

    def test_case_plan_round_trip(self) -> None:
        plan = CasePlan(
            name="n1",
            kind="n1",
            shapes={"a": (1,)},
            scalars={"n": 1},
            seed=3,
        )
        loaded = CasePlan.from_dict(plan.to_dict())
        self.assertEqual(loaded.shapes["a"], (1,))


if __name__ == "__main__":
    unittest.main()
