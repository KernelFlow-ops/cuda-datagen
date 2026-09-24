"""Numeric compare: tolerances, NaN/Inf, sort-flatten."""

from __future__ import annotations

import unittest

from cuda_sft.refval.cases import numpy_available
from cuda_sft.refval.compare import compare_arrays, compare_outputs
from cuda_sft.refval.spec import CasePlan, KernelABI, KernelParam


def _add_abi(**kwargs: object) -> KernelABI:
    return KernelABI(
        entry="launch_add",
        params=(
            KernelParam("a", "input", "f32", rank=1, shape_from=("n",)),
            KernelParam("c", "output", "f32", rank=1, shape_from=("n",)),
            KernelParam("n", "size", "i32", rank=0),
        ),
        **kwargs,  # type: ignore[arg-type]
    )


class CompareTests(unittest.TestCase):
    def setUp(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        import numpy as np

        self.np = np

    def test_close_floats_pass(self) -> None:
        got = self.np.array([1.0, 2.0], dtype="float32")
        exp = self.np.array([1.0 + 1e-6, 2.0], dtype="float32")
        result = compare_arrays(
            got, exp, dtype="f32", tolerances=None, allows_nan=False, sort_mode="elementwise",
            name="t", seed=1,
        )
        self.assertTrue(result.ok)

    def test_wrong_kernel_is_caught(self) -> None:
        got = self.np.array([2.0, 3.0], dtype="float32")
        exp = self.np.array([1.0, 2.0], dtype="float32")
        result = compare_arrays(
            got, exp, dtype="f32", tolerances=None, allows_nan=False, sort_mode="elementwise",
            name="t", seed=1,
        )
        self.assertFalse(result.ok)
        self.assertGreater(result.n_mismatch, 0)

    def test_nan_forbidden(self) -> None:
        got = self.np.array([self.np.nan], dtype="float32")
        exp = self.np.array([1.0], dtype="float32")
        result = compare_arrays(
            got, exp, dtype="f32", tolerances=None, allows_nan=False, sort_mode="elementwise",
            name="t", seed=1,
        )
        self.assertFalse(result.ok)
        self.assertIn("NaN", result.error)

    def test_sort_flatten_ignores_order(self) -> None:
        got = self.np.array([3.0, 1.0, 2.0], dtype="float32")
        exp = self.np.array([1.0, 2.0, 3.0], dtype="float32")
        elem = compare_arrays(
            got, exp, dtype="f32", tolerances=None, allows_nan=False, sort_mode="elementwise",
            name="t", seed=1,
        )
        sorted_ = compare_arrays(
            got, exp, dtype="f32", tolerances=None, allows_nan=False, sort_mode="sorted",
            name="t", seed=1,
        )
        self.assertFalse(elem.ok)
        self.assertTrue(sorted_.ok)

    def test_compare_outputs_merges_tensor_name(self) -> None:
        abi = _add_abi()
        plan = CasePlan(name="n1", kind="n1", shapes={"c": (2,)}, scalars={"n": 2}, seed=1)
        result = compare_outputs(
            {"c": self.np.array([0.0, 0.0])},
            {"c": self.np.array([1.0, 1.0])},
            abi,
            plan,
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.mismatches)
        self.assertEqual(result.mismatches[0]["tensor"], "c")


if __name__ == "__main__":
    unittest.main()
