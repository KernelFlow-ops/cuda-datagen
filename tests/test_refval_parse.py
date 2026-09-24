"""ABI JSON parse + heuristic host-signature scan."""

from __future__ import annotations

import unittest

from cuda_sft.parse import abi_matches_source, heuristic_cuda_abi, parse_refval_manifest


ADD_SOURCE = """
#include <cuda_runtime.h>
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
void launch_add(const float* a, const float* b, float* c, int n) {
  int t = 256;
  int g = (n + t - 1) / t;
  if (n > 0) add_kernel<<<g, t>>>(a, b, c, n);
}
"""

JSON_REPLY = """
```json
{
  "abi": {
    "entry": "launch_add",
    "dtype": "f32",
    "layout": "contiguous",
    "returns": "void",
    "params": [
      {"name": "a", "kind": "input", "dtype": "f32", "rank": 1, "shape_from": ["n"]},
      {"name": "b", "kind": "input", "dtype": "f32", "rank": 1, "shape_from": ["n"]},
      {"name": "c", "kind": "output", "dtype": "f32", "rank": 1, "shape_from": ["n"]},
      {"name": "n", "kind": "size", "dtype": "i32", "rank": 0}
    ]
  },
  "reference_fn_name": "reference",
  "reference_source": "def reference(a, b, n):\\n    return {'c': a + b}\\n"
}
```
"""


class ParseManifestTests(unittest.TestCase):
    def test_json_fence(self) -> None:
        manifest = parse_refval_manifest(JSON_REPLY, question_id=1, dialect="cuda", seed=1)
        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest.abi.entry, "launch_add")
        self.assertEqual(manifest.abi.result_names(), ("c",))
        self.assertIn("reference", manifest.reference_source)

    def test_self_check(self) -> None:
        manifest = parse_refval_manifest(JSON_REPLY, question_id=1, dialect="cuda")
        assert manifest is not None
        self.assertEqual(abi_matches_source(manifest.abi, ADD_SOURCE), [])
        issues = abi_matches_source(manifest.abi, "__global__ void other() {}")
        self.assertTrue(issues)

    def test_manifest_with_invalid_abi_is_rejected(self) -> None:
        malformed = '{"abi": {"entry": "launch", "params": []}, "reference_source": "def reference(): return {}"}'
        self.assertIsNone(parse_refval_manifest(malformed, question_id=1, dialect="cuda"))


class HeuristicAbiTests(unittest.TestCase):
    def test_picks_host_launcher_not_kernel(self) -> None:
        abi = heuristic_cuda_abi(ADD_SOURCE)
        self.assertIsNotNone(abi)
        assert abi is not None
        self.assertEqual(abi.entry, "launch_add")
        names = [p.name for p in abi.params]
        self.assertEqual(names, ["a", "b", "c", "n"])
        kinds = {p.name: p.kind for p in abi.params}
        self.assertEqual(kinds["a"], "input")
        self.assertEqual(kinds["c"], "output")
        self.assertEqual(kinds["n"], "size")

    def test_cuda_signature_count_and_type_are_checked(self) -> None:
        abi = heuristic_cuda_abi(ADD_SOURCE)
        assert abi is not None
        bad = ADD_SOURCE.replace("const float* b", "double* b").replace(", int n", "")
        issues = abi_matches_source(abi, bad)
        self.assertTrue(any("count mismatch" in issue for issue in issues))
        self.assertTrue(any("type mismatch" in issue for issue in issues))

    def test_multiletter_scalar_names_match(self) -> None:
        source = """
__global__ void sigmoid_kernel(float* matrix, int rows, int cols) {
  int idx = blockIdx.x;
  if (idx < rows * cols) matrix[idx] = matrix[idx];
}
void sigmoid(float* matrix, int rows, int cols) {
  sigmoid_kernel<<<1, 1>>>(matrix, rows, cols);
}
"""
        abi = heuristic_cuda_abi(source)
        assert abi is not None
        self.assertEqual([p.name for p in abi.params], ["matrix", "rows", "cols"])
        self.assertEqual(abi_matches_source(abi, source), [])

    def test_concrete_overload_matches_when_template_precedes_it(self) -> None:
        from cuda_sft.refval.spec import KernelABI, KernelParam

        source = """
template <typename T>
void scale(const T* in, T* out, int rows, int cols) {
  (void)in; (void)out; (void)rows; (void)cols;
}
void scale(const float* in, float* out, int rows, int cols) {
  (void)in; (void)out; (void)rows; (void)cols;
}
"""
        abi = KernelABI(
            entry="scale",
            params=(
                KernelParam("in", "input", "f32", rank=1, shape_from=("rows", "cols")),
                KernelParam("out", "output", "f32", rank=1, shape_from=("rows", "cols")),
                KernelParam("rows", "size", "i32", rank=0),
                KernelParam("cols", "size", "i32", rank=0),
            ),
        )
        self.assertEqual(abi_matches_source(abi, source), [])

    def test_explicit_instantiation_matches_template_host(self) -> None:
        from cuda_sft.refval.spec import KernelABI, KernelParam

        source = """
template <typename T>
__global__ void matrix_scale_kernel(const T* A, const T* B, T* C, T alpha, T beta, int M, int N) {}

template <typename T>
void matrix_scale(const T* A, const T* B, T* C, T alpha, T beta, int M, int N) {}

template void matrix_scale<float>(const float*, const float*, float*, float, float, int, int);
template void matrix_scale<double>(const double*, const double*, double*, double, double, int, int);
"""
        abi = KernelABI(
            entry="matrix_scale",
            params=(
                KernelParam("A", "input", "f32", rank=1, shape_from=("M", "N")),
                KernelParam("B", "input", "f32", rank=1, shape_from=("M", "N")),
                KernelParam("C", "output", "f32", rank=1, shape_from=("M", "N")),
                KernelParam("alpha", "scalar", "f32", rank=0),
                KernelParam("beta", "scalar", "f32", rank=0),
                KernelParam("M", "size", "i32", rank=0),
                KernelParam("N", "size", "i32", rank=0),
            ),
            dtype="f32",
        )
        self.assertEqual(abi_matches_source(abi, source), [])

    def test_python_ast_signature_is_checked(self) -> None:
        from cuda_sft.refval.spec import KernelABI, KernelParam

        abi = KernelABI(
            entry="launch",
            params=(
                KernelParam("x", "input", "f32", rank=1),
                KernelParam("n", "size", "i32", rank=0),
            ),
        )
        source = "def launch(x: list[float], n: int):\n    return {'out': x}\n"
        issues = abi_matches_source(abi, source)
        self.assertTrue(any("unsupported Python annotation" in issue for issue in issues))


class RepairerClassTests(unittest.TestCase):
    def test_classify_refval_error(self) -> None:
        from cuda_sft.agents.repairer import classify_refval_error

        self.assertEqual(classify_refval_error("timeout after 45s"), "timeout")
        self.assertEqual(
            classify_refval_error("x", error_class="numeric_mismatch"),
            "numeric_mismatch",
        )
        self.assertEqual(classify_refval_error("NaN in GPU output"), "nan_inf")


if __name__ == "__main__":
    unittest.main()
