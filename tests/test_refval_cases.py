"""Adversarial case planner."""

from __future__ import annotations

import unittest

from cuda_sft.refval.cases import build_case_plans, materialize_arrays, numpy_available
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


if __name__ == "__main__":
    unittest.main()
