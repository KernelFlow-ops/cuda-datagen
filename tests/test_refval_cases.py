"""Adversarial case planner."""

from __future__ import annotations

import unittest

from cuda_sft.refval.cases import build_case_plans, materialize_arrays, numpy_available
from cuda_sft.refval.cross_dialect import canonical_tasks
from cuda_sft.refval.spec import KernelABI, KernelParam


def _add_abi() -> KernelABI:
    return KernelABI(
        entry="launch_add",
        params=(
            KernelParam("a", "input", "f32", rank=1, shape_from=("n",)),
            KernelParam("b", "input", "f32", rank=1, shape_from=("n",)),
            KernelParam("c", "output", "f32", rank=1, shape_from=("n",)),
            KernelParam("n", "size", "i32", rank=0),
        ),
    )


class CasePlanTests(unittest.TestCase):
    def test_smoke_has_empty_and_unit(self) -> None:
        plans = build_case_plans(_add_abi(), question_id=1, dialect="cuda", suite="smoke")
        names = [p.name for p in plans]
        self.assertIn("n0", names)
        self.assertIn("n1", names)
        n0 = next(p for p in plans if p.name == "n0")
        self.assertEqual(n0.scalars["n"], 0)
        self.assertEqual(n0.shapes["a"], (0,))

    def test_standard_covers_adversarial_kinds(self) -> None:
        plans = build_case_plans(_add_abi(), question_id=8, dialect="cuda", suite="standard")
        kinds = {p.kind for p in plans}
        self.assertIn("non_tile", kinds)
        self.assertIn("non_pow2", kinds)
        self.assertIn("large", kinds)
        self.assertIn("extreme", kinds)
        self.assertIn("dup", kinds)
        large = next(p for p in plans if p.kind == "large")
        numel = 1
        for dim in large.shapes["a"]:
            numel *= dim
        self.assertGreaterEqual(numel, 1_000)
        self.assertLessEqual(numel, 4_000_000)

    def test_same_seed_materializes_identically(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        plans = build_case_plans(_add_abi(), question_id=3, dialect="cuda", suite="smoke")
        plan = next(p for p in plans if p.name == "odd_7")
        a = materialize_arrays(plan, _add_abi())
        b = materialize_arrays(plan, _add_abi())
        self.assertTrue((a["a"] == b["a"]).all())

    def test_inplace_case_shares_reference_and_gpu_inputs(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        abi = KernelABI(
            entry="solution",
            params=(
                KernelParam("x", "input", "f32", rank=1, shape_from=("n",)),
                KernelParam("y", "inout", "f32", rank=1, shape_from=("n",)),
                KernelParam("n", "size", "i32", rank=0),
            ),
            in_place=True,
        )
        plan = next(
            p for p in build_case_plans(abi, question_id=125, dialect="cuda")
            if p.name == "inplace"
        )
        self.assertEqual(plan.alias, {"y": "x"})
        arrays = materialize_arrays(plan, abi)
        self.assertIs(arrays["y"], arrays["x"])

    def test_matrix_row_sum_uses_rectangular_case_without_broadcast(self) -> None:
        abi = KernelABI(
            entry="matrix_row_sum_fp16",
            params=(
                KernelParam("A", "input", "u16", rank=2, shape_from=("rows", "cols")),
                KernelParam("sums", "output", "u16", rank=1, shape_from=("rows",)),
                KernelParam("rows", "size", "i32", rank=0),
                KernelParam("cols", "size", "i32", rank=0),
            ),
            dtype="f16",
        )
        plans = build_case_plans(abi, question_id=63, dialect="cuda")
        self.assertNotIn("broadcast", {plan.name for plan in plans})
        rectangular = next(plan for plan in plans if plan.name == "rectangular")
        rows, cols = rectangular.scalars["rows"], rectangular.scalars["cols"]
        self.assertEqual((rows, cols), (15, 7))
        self.assertEqual(rectangular.shapes["A"], (rows, cols))
        self.assertEqual(rectangular.shapes["sums"], (rows,))

    def test_convolution_filter_size_stays_small_and_odd(self) -> None:
        abi = KernelABI(
            entry="solution",
            params=(
                KernelParam("input", "input", "f32", rank=2, shape_from=("height", "width")),
                KernelParam("filter", "input", "f32", rank=2, shape_from=("kernel_size", "kernel_size")),
                KernelParam("output", "output", "f32", rank=2, shape_from=("height", "width")),
                KernelParam("width", "size", "i32"),
                KernelParam("height", "size", "i32"),
                KernelParam("kernel_size", "size", "i32"),
            ),
        )
        plans = build_case_plans(abi, question_id=89, dialect="cuda")
        self.assertEqual({p.scalars["kernel_size"] for p in plans}, {3, 5, 7})
        for plan in plans:
            kernel_size = plan.scalars["kernel_size"]
            self.assertEqual(plan.shapes["filter"], (kernel_size, kernel_size))
            self.assertEqual(
                plan.shapes["input"],
                (plan.scalars["height"], plan.scalars["width"]),
            )
        large = next(plan for plan in plans if plan.name == "large")
        self.assertGreaterEqual(large.scalars["width"], 1000)
        self.assertLessEqual(large.scalars["kernel_size"], 7)

    def test_explicit_broadcast_support_still_shrinks_input_axis(self) -> None:
        abi = KernelABI(
            entry="broadcast_add",
            params=(
                KernelParam("a", "input", "f32", rank=2, shape_from=("rows", "cols")),
                KernelParam("out", "output", "f32", rank=2, shape_from=("rows", "cols")),
                KernelParam("rows", "size", "i32", rank=0),
                KernelParam("cols", "size", "i32", rank=0),
            ),
            supports_broadcast=True,
        )
        plans = build_case_plans(abi, question_id=64, dialect="cuda")
        broadcast = next(plan for plan in plans if plan.name == "broadcast")
        self.assertEqual(broadcast.shapes["a"], (15, 1))
        self.assertEqual(broadcast.shapes["out"], (15, 7))

    def test_strided_case_passes_matching_scalar_strides(self) -> None:
        tasks = canonical_tasks()
        add = next(
            plan for plan in build_case_plans(
                tasks["elementwise_add"].abi, question_id=7, dialect="cuda"
            ) if plan.name == "strided"
        )
        for tensor in ("a", "b", "c"):
            self.assertEqual(add.scalars[f"{tensor}_stride"], add.strides[tensor][0])

        row_sum = next(
            plan for plan in build_case_plans(
                tasks["row_sum"].abi, question_id=7, dialect="cuda"
            ) if plan.name == "strided"
        )
        self.assertEqual(row_sum.scalars["x_row_stride"], row_sum.strides["x"][0])
        self.assertEqual(row_sum.scalars["x_col_stride"], row_sum.strides["x"][1])
        self.assertGreater(row_sum.scalars["x_col_stride"], 1)
        self.assertEqual(row_sum.scalars["out_stride"], row_sum.strides["out"][0])


if __name__ == "__main__":
    unittest.main()
