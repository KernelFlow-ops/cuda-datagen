"""Reference sandbox: forbidden I/O, determinism, 2σ."""

from __future__ import annotations

import unittest
from typing import Any

from cuda_sft.refval.cases import numpy_available
from cuda_sft.refval.reference import (
    bind_and_call,
    check_forbidden,
    load_reference_fn,
    validate_reference_fn,
)
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


ADD_REF = """
def reference(a, b, n):
    import numpy as np
    return {"c": np.asarray(a) + np.asarray(b)}
"""

ZERO_REF = """
def reference(a, b, n):
    import numpy as np
    return {"c": np.zeros_like(np.asarray(a))}
"""

HANG_REF = """
def reference(a, b, n):
    while True:
        pass
"""


def _daemon_reference_worker(result: Any) -> None:
    """Module-level so a daemon Process can pickle it."""
    import numpy as np

    fn = load_reference_fn(ADD_REF)
    try:
        out = bind_and_call(
            fn,
            _add_abi(),
            {"a": np.ones(2, dtype=np.float32), "b": np.full(2, 2.0, dtype=np.float32)},
            {"n": 2},
        )
        result.put(("ok", [float(v) for v in out["c"]]))
    except Exception as exc:
        result.put(("err", f"{type(exc).__name__}: {exc}"))


class ForbiddenTests(unittest.TestCase):
    def test_rejects_open(self) -> None:
        self.assertIsNotNone(check_forbidden("def reference():\n    open('/tmp/x','w')\n"))

    def test_rejects_os(self) -> None:
        self.assertIsNotNone(check_forbidden("import os\ndef reference():\n    os.system('id')\n"))

    def test_allows_numpy(self) -> None:
        self.assertIsNone(check_forbidden(ADD_REF))

    def test_allows_parameter_named_input(self) -> None:
        source = "def reference(input, n):\n    return {'out': input}\n"
        self.assertIsNone(check_forbidden(source))

    def test_rejects_input_call(self) -> None:
        self.assertIsNotNone(check_forbidden("def reference():\n    return input('x')\n"))

    def test_restricted_import_builtin_rejects_system_modules(self) -> None:
        source = "def reference():\n    return __builtins__['__import__']('os')\n"
        fn = load_reference_fn(source)
        with self.assertRaisesRegex(RuntimeError, "reference import is not allowed: os"):
            fn()


class LoadTests(unittest.TestCase):
    def test_add_reference_is_sensitive(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        fn = load_reference_fn(ADD_REF)
        issue = validate_reference_fn(fn, _add_abi(), seed=1)
        self.assertIsNone(issue, issue)

    def test_constant_zero_fails_2sigma(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        fn = load_reference_fn(ZERO_REF)
        issue = validate_reference_fn(fn, _add_abi(), seed=1)
        self.assertIsNotNone(issue)
        self.assertIn("2σ", issue or "")

    def test_infinite_reference_is_hard_timed_out(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        fn = load_reference_fn(HANG_REF)
        issue = validate_reference_fn(fn, _add_abi(), seed=1)
        self.assertIsNotNone(issue)
        self.assertIn("timed out", issue or "")

    def test_bind_and_call_preserves_callable_contract(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        fn = load_reference_fn(ADD_REF)
        import numpy as np

        out = bind_and_call(
            fn,
            _add_abi(),
            {"a": np.ones(2, dtype=np.float32), "b": np.ones(2, dtype=np.float32)},
            {"n": 2},
        )
        self.assertEqual(list(out), ["c"])
        self.assertEqual(out["c"].shape, (2,))

    def test_extreme_numpy_reference_does_not_require_import_builtin(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        import numpy as np

        source = (
            "def reference(input, rows, cols):\n"
            "    x = np.asarray(input, dtype=np.float32).reshape((rows, cols))\n"
            "    return {'output': 1.0 / (1.0 + np.exp(-x))}\n"
        )
        abi = KernelABI(
            entry="solution",
            params=(
                KernelParam("input", "input", "f32", rank=2, shape_from=("rows", "cols")),
                KernelParam("output", "output", "f32", rank=2, shape_from=("rows", "cols")),
                KernelParam("rows", "size", "i32", rank=0),
                KernelParam("cols", "size", "i32", rank=0),
            ),
        )
        out = bind_and_call(
            load_reference_fn(source), abi,
            {"input": np.array([[1000.0, -1000.0]], dtype=np.float32)},
            {"rows": 1, "cols": 2},
        )
        np.testing.assert_array_equal(out["output"], [[1.0, 0.0]])

    def test_output_argument_is_not_dropped(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        source = (
            "def reference(A, B, C, alpha, beta, M, N):\n"
            "    A_np = np.asarray(A).reshape(M, N)\n"
            "    B_np = np.asarray(B).reshape(M, N)\n"
            "    return {'C': (alpha * A_np + beta * B_np).reshape(-1)}\n"
        )
        abi = KernelABI(
            entry="matrix_scale",
            params=(
                KernelParam("A", "input", "f32", rank=2, shape_from=("M", "N")),
                KernelParam("B", "input", "f32", rank=2, shape_from=("M", "N")),
                KernelParam("C", "output", "f32", rank=2, shape_from=("M", "N")),
                KernelParam("alpha", "scalar", "f32", rank=0),
                KernelParam("beta", "scalar", "f32", rank=0),
                KernelParam("M", "size", "i32", rank=0),
                KernelParam("N", "size", "i32", rank=0),
            ),
            dtype="f32",
        )
        issue = validate_reference_fn(load_reference_fn(source), abi, seed=3)
        self.assertIsNone(issue, issue)

    def test_half_bits_reference_self_check_uses_numeric_inputs(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        source = (
            "def reference(A, rows, cols):\n"
            "    values = np.asarray(A).view(np.float16).reshape(rows, cols)\n"
            "    result = values.astype(np.float32).sum(axis=1).astype(np.float16)\n"
            "    return {'sums': result.view(np.uint16)}\n"
        )
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
        issue = validate_reference_fn(load_reference_fn(source), abi, seed=63)
        self.assertIsNone(issue, issue)

    def test_bind_and_call_works_from_daemon_worker(self) -> None:
        if not numpy_available():
            self.skipTest("numpy missing")
        import multiprocessing as mp

        ctx = mp.get_context("fork")
        queue = ctx.Queue()
        proc = ctx.Process(target=_daemon_reference_worker, args=(queue,), daemon=True)
        proc.start()
        proc.join(60)
        self.assertFalse(proc.is_alive(), "daemon reference worker hung")
        status, payload = queue.get(timeout=5)
        self.assertEqual(status, "ok", payload)
        self.assertEqual(payload, [3.0, 3.0])


if __name__ == "__main__":
    unittest.main()
