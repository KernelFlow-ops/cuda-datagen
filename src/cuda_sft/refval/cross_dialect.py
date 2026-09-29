"""Canonical cross-dialect GPU operators and their frozen validation contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cuda_sft.refval.spec import CasePlan, KernelABI, KernelParam, RefManifest, stable_hash


@dataclass(frozen=True)
class CanonicalTask:
    """One semantic operation with one source variant per backend."""

    task_id: str
    operation: str
    abi: KernelABI
    reference_source: str
    sources: dict[str, str]
    semantic_contract: dict[str, Any]


def _reference_add() -> str:
    return '''
def reference(a, b, c=None, n=0, a_stride=1, b_stride=1, c_stride=1):
    import numpy as np
    av = np.asarray(a).reshape(-1)[:int(n)]
    bv = np.asarray(b).reshape(-1)[:int(n)]
    return {"c": av + bv}
'''


def _reference_scale() -> str:
    return '''
def reference(x, out=None, scale=1.0, n=0, x_stride=1, out_stride=1):
    import numpy as np
    xv = np.asarray(x).reshape(-1)[:int(n)]
    return {"out": xv * scale}
'''


def _reference_row_sum() -> str:
    return '''
def reference(x, out=None, rows=0, cols=0, x_row_stride=1, x_col_stride=1, out_stride=1):
    import numpy as np
    matrix = np.asarray(x).reshape((int(rows), int(cols)))
    return {"out": np.sum(matrix, axis=1, dtype=np.float32)}
'''


def _cuda_add(entry: str = "launch_add") -> str:
    return f'''#include <cuda_runtime.h>
__global__ void add_kernel(const float* a, const float* b, float* c, int n, int a_stride, int b_stride, int c_stride) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i * c_stride] = a[i * a_stride] + b[i * b_stride];
}}
extern "C" void {entry}(const float* a, const float* b, float* c, int n, int a_stride, int b_stride, int c_stride) {{
  if (n > 0) add_kernel<<<(n + 255) / 256, 256>>>(a, b, c, n, a_stride, b_stride, c_stride);
}}
'''


def _cuda_scale(entry: str = "launch_scale") -> str:
    return f'''#include <cuda_runtime.h>
__global__ void scale_kernel(const float* x, float* out, float scale, int n, int x_stride, int out_stride) {{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i * out_stride] = x[i * x_stride] * scale;
}}
extern "C" void {entry}(const float* x, float* out, float scale, int n, int x_stride, int out_stride) {{
  if (n > 0) scale_kernel<<<(n + 255) / 256, 256>>>(x, out, scale, n, x_stride, out_stride);
}}
'''


def _cuda_row_sum(entry: str = "launch_row_sum") -> str:
    return f'''#include <cuda_runtime.h>
__global__ void row_sum_kernel(const float* x, float* out, int rows, int cols, int x_row_stride, int x_col_stride, int out_stride) {{
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float acc = 0.0f;
  for (int col = 0; col < cols; ++col) acc += x[r * x_row_stride + col * x_col_stride];
  out[r * out_stride] = acc;
}}
extern "C" void {entry}(const float* x, float* out, int rows, int cols, int x_row_stride, int x_col_stride, int out_stride) {{
  if (rows > 0 && cols > 0) row_sum_kernel<<<(rows + 127) / 128, 128>>>(x, out, rows, cols, x_row_stride, x_col_stride, out_stride);
}}
'''


def _cutlass_add(entry: str = "launch_add") -> str:
    return f'''#include <cuda_runtime.h>
#include <cute/tensor.hpp>
__global__ void cute_add_kernel(const float* a, const float* b, float* c, int n, int a_stride, int b_stride, int c_stride) {{
  auto la = cute::make_layout(cute::make_shape(n), cute::make_stride(a_stride));
  auto lb = cute::make_layout(cute::make_shape(n), cute::make_stride(b_stride));
  auto lc = cute::make_layout(cute::make_shape(n), cute::make_stride(c_stride));
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[lc(i)] = a[la(i)] + b[lb(i)];
}}
extern "C" void {entry}(const float* a, const float* b, float* c, int n, int a_stride, int b_stride, int c_stride) {{
  if (n > 0) cute_add_kernel<<<(n + 255) / 256, 256>>>(a, b, c, n, a_stride, b_stride, c_stride);
}}
'''


def _cutlass_scale(entry: str = "launch_scale") -> str:
    return f'''#include <cuda_runtime.h>
#include <cute/tensor.hpp>
__global__ void cute_scale_kernel(const float* x, float* out, float scale, int n, int x_stride, int out_stride) {{
  auto lx = cute::make_layout(cute::make_shape(n), cute::make_stride(x_stride));
  auto lo = cute::make_layout(cute::make_shape(n), cute::make_stride(out_stride));
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[lo(i)] = x[lx(i)] * scale;
}}
extern "C" void {entry}(const float* x, float* out, float scale, int n, int x_stride, int out_stride) {{
  if (n > 0) cute_scale_kernel<<<(n + 255) / 256, 256>>>(x, out, scale, n, x_stride, out_stride);
}}
'''


def _cutlass_row_sum(entry: str = "launch_row_sum") -> str:
    return f'''#include <cuda_runtime.h>
#include <cute/tensor.hpp>
__global__ void cute_row_sum_kernel(const float* x, float* out, int rows, int cols, int x_row_stride, int x_col_stride, int out_stride) {{
  auto lx = cute::make_layout(cute::make_shape(rows, cols), cute::make_stride(x_row_stride, x_col_stride));
  auto lo = cute::make_layout(cute::make_shape(rows), cute::make_stride(out_stride));
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows) return;
  float acc = 0.0f;
  for (int col = 0; col < cols; ++col) acc += x[lx(r, col)];
  out[lo(r)] = acc;
}}
extern "C" void {entry}(const float* x, float* out, int rows, int cols, int x_row_stride, int x_col_stride, int out_stride) {{
  if (rows > 0 && cols > 0) cute_row_sum_kernel<<<(rows + 127) / 128, 128>>>(x, out, rows, cols, x_row_stride, x_col_stride, out_stride);
}}
'''


def _triton_add(entry: str = "launch_add") -> str:
    return f'''import triton
import triton.language as tl
@triton.jit
def add_kernel(a, b, c, n, a_stride, b_stride, c_stride, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(c + off * c_stride, tl.load(a + off * a_stride, mask=mask) + tl.load(b + off * b_stride, mask=mask), mask=mask)
def {entry}(a, b, c, n, a_stride, b_stride, c_stride):
    if n > 0: add_kernel[(triton.cdiv(n, 256),)](a, b, c, n, a_stride, b_stride, c_stride, BLOCK=256)
'''


def _triton_scale(entry: str = "launch_scale") -> str:
    return f'''import triton
import triton.language as tl
@triton.jit
def scale_kernel(x, out, scale, n, x_stride, out_stride, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(out + off * out_stride, tl.load(x + off * x_stride, mask=mask) * scale, mask=mask)
def {entry}(x, out, scale, n, x_stride, out_stride):
    if n > 0: scale_kernel[(triton.cdiv(n, 256),)](x, out, scale, n, x_stride, out_stride, BLOCK=256)
'''


def _triton_row_sum(entry: str = "launch_row_sum") -> str:
    return f'''import triton
import triton.language as tl
@triton.jit
def row_sum_kernel(x, out, rows, cols, x_row_stride, x_col_stride, out_stride, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    vals = tl.load(x + r * x_row_stride + col * x_col_stride, mask=(r < rows) & (col < cols), other=0.0)
    tl.store(out + r * out_stride, tl.sum(vals, axis=0), mask=r < rows)
def {entry}(x, out, rows, cols, x_row_stride, x_col_stride, out_stride):
    if rows > 0 and cols > 0: row_sum_kernel[(rows,)](x, out, rows, cols, x_row_stride, x_col_stride, out_stride, BLOCK=512)
'''


def _tilelang_source() -> str:
    """TileLang source with explicit positional launch wrappers."""
    return '''import torch
import tilelang
import tilelang.language as T

def _add_program(n, block=256):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32"),
             astride: T.int32, bstride: T.int32, cstride: T.int32):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                j = bx * block + i
                if j < n: C[j * cstride] = A[j * astride] + B[j * bstride]
    return main
def launch_add(a, b, c, n, a_stride, b_stride, c_stride):
    if n <= 0: return
    aa, bb = a.contiguous(), b.contiguous()
    cc = c if c.is_contiguous() else torch.empty((n,), device=c.device, dtype=c.dtype)
    tilelang.compile(_add_program(n), target="cuda")(aa, bb, cc, 1, 1, 1)
    if cc is not c: c.copy_(cc)

def _scale_program(n, block=256):
    @T.prim_func
    def main(X: T.Tensor((n,), "float32"), O: T.Tensor((n,), "float32"), scale: T.float32,
             xstride: T.int32, ostride: T.int32):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for i in T.Parallel(block):
                j = bx * block + i
                if j < n: O[j * ostride] = X[j * xstride] * scale
    return main
def launch_scale(x, out, scale, n, x_stride, out_stride):
    if n <= 0: return
    xx = x.contiguous()
    oo = out if out.is_contiguous() else torch.empty((n,), device=out.device, dtype=out.dtype)
    tilelang.compile(_scale_program(n), target="cuda")(xx, oo, scale, 1, 1)
    if oo is not out: out.copy_(oo)

def _row_sum_program(rows, cols, block=512):
    @T.prim_func
    def main(X: T.Tensor((rows, cols), "float32"), O: T.Tensor((rows,), "float32"),
             xr: T.int32, xc: T.int32, os: T.int32):
        with T.Kernel(rows, threads=block) as br:
            vals = T.alloc_fragment((block,), "float32")
            reduced = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(block):
                if j < cols:
                    vals[j] = X[br, j]
                else:
                    vals[j] = 0.0
            T.reduce_sum(vals, reduced, dim=0)
            if T.get_thread_binding() == 0:
                O[br] = reduced[0]
    return main
def launch_row_sum(x, out, rows, cols, x_row_stride, x_col_stride, out_stride):
    if rows <= 0 or cols <= 0: return
    xx = x.contiguous()
    oo = out if out.is_contiguous() else torch.empty((rows,), device=out.device, dtype=out.dtype)
    tilelang.compile(_row_sum_program(rows, cols), target="cuda")(xx, oo, cols, 1, 1)
    if oo is not out: out.copy_(oo)

# Public compile-gate factories are deliberately separate from the ABI
# wrappers.  The gate can lower one PrimFunc at import time, while refval
# invokes only launch_* with the frozen positional signature.
add_program = _add_program
scale_program = _scale_program
row_sum_program = _row_sum_program
'''


def _vector_abi(entry: str, *, scale: bool = False) -> KernelABI:
    if scale:
        params = (
            KernelParam("x", "input", "f32", rank=1, shape_from=("n",)), KernelParam("out", "output", "f32", rank=1, shape_from=("n",)),
            KernelParam("scale", "scalar", "f32"), KernelParam("n", "size", "i32"), KernelParam("x_stride", "size", "i32"), KernelParam("out_stride", "size", "i32"),
        )
    else:
        params = (
            KernelParam("a", "input", "f32", rank=1, shape_from=("n",)), KernelParam("b", "input", "f32", rank=1, shape_from=("n",)), KernelParam("c", "output", "f32", rank=1, shape_from=("n",)),
            KernelParam("n", "size", "i32"), KernelParam("a_stride", "size", "i32"), KernelParam("b_stride", "size", "i32"), KernelParam("c_stride", "size", "i32"),
        )
    return KernelABI(entry=entry, params=params, supports_strided=True, zero_size_strategy="call")


def _row_sum_abi() -> KernelABI:
    return KernelABI(entry="launch_row_sum", params=(
        KernelParam("x", "input", "f32", rank=2, shape_from=("rows", "cols")), KernelParam("out", "output", "f32", rank=1, shape_from=("rows",)),
        KernelParam("rows", "size", "i32"), KernelParam("cols", "size", "i32"), KernelParam("x_row_stride", "size", "i32"), KernelParam("x_col_stride", "size", "i32"), KernelParam("out_stride", "size", "i32"),
    ), supports_strided=True, zero_size_strategy="call")


def canonical_tasks() -> dict[str, CanonicalTask]:
    common = {"target_arch": "sm_86", "dtype": "f32", "case_suite": "canonical-v2"}
    return {
        "elementwise_add": CanonicalTask("elementwise_add", "elementwise_add", _vector_abi("launch_add"), _reference_add(), {"cuda": _cuda_add(), "cutlass": _cutlass_add(), "triton": _triton_add(), "tilelang": _tilelang_source()}, {**common, "layout": "strided-1d", "semantics": "c[i]=a[i]+b[i]"}),
        "scale": CanonicalTask("scale", "scale", _vector_abi("launch_scale", scale=True), _reference_scale(), {"cuda": _cuda_scale(), "cutlass": _cutlass_scale(), "triton": _triton_scale(), "tilelang": _tilelang_source()}, {**common, "layout": "strided-1d", "semantics": "out[i]=x[i]*scale"}),
        "row_sum": CanonicalTask("row_sum", "row_sum", _row_sum_abi(), _reference_row_sum(), {"cuda": _cuda_row_sum(), "cutlass": _cutlass_row_sum(), "triton": _triton_row_sum(), "tilelang": _tilelang_source()}, {**common, "layout": "strided-2d", "semantics": "out[r]=sum_c(x[r,c])"}),
    }


def _plan(task: CanonicalTask, name: str, question_id: int) -> CasePlan:
    if task.operation == "row_sum":
        rows, cols = {"empty": (0, 257), "one": (1, 1), "odd7": (3, 7), "tail257": (3, 257), "strided": (5, 7), "extreme": (4, 9)}[name]
        shapes = {"x": (rows, cols), "out": (rows,)}
        strides = {"x": (cols, 1), "out": (1,)}
        scalars = {"rows": rows, "cols": cols, "x_row_stride": cols, "x_col_stride": 1, "out_stride": 1}
        if name == "strided":
            strides, scalars = {"x": (16, 2), "out": (2,)}, {"rows": rows, "cols": cols, "x_row_stride": 16, "x_col_stride": 2, "out_stride": 2}
    else:
        n = {"empty": 0, "one": 1, "odd7": 7, "tail257": 257, "strided": 33, "extreme": 33}[name]
        tensors = ("a", "b", "c") if task.operation == "elementwise_add" else ("x", "out")
        shapes, strides = {key: (n,) for key in tensors}, {key: (1,) for key in tensors}
        if name == "strided": strides = {key: (2,) for key in tensors}
        scalars = {"n": n}
        if task.operation == "elementwise_add": scalars.update(a_stride=strides["a"][0], b_stride=strides["b"][0], c_stride=strides["c"][0])
        else: scalars.update(scale=1.25, x_stride=strides["x"][0], out_stride=strides["out"][0])
    return CasePlan(name=name, kind={"empty": "n0", "one": "n1", "odd7": "non_tile", "tail257": "tail", "strided": "strided", "extreme": "extreme"}[name], shapes=shapes, scalars=scalars, seed=(question_id * 1009 + len(name)) & 0x7FFFFFFF, generate_denorm=name == "extreme", strides=strides)


def canonical_plans(task: CanonicalTask, question_id: int, suite: str = "canonical") -> list[CasePlan]:
    """Return the same six semantic cases for every backend."""
    names = ("empty", "one", "odd7") if suite == "smoke" else ("empty", "one", "odd7", "tail257", "strided", "extreme")
    return [_plan(task, name, int(question_id)) for name in names]


def make_manifest(task: CanonicalTask, *, question_id: int, dialect: str) -> RefManifest:
    semantic = dict(task.semantic_contract)
    semantic["operation"] = task.operation
    semantic_hash = stable_hash(semantic)
    backend = {"dialect": dialect, "entry": task.abi.entry, "call_style": "positional"}
    return RefManifest(question_id=question_id, dialect=dialect, abi=task.abi, reference_source=task.reference_source, extracted_from="independent", task_spec={"operation": task.operation, "semantic_contract_hash": semantic_hash, **semantic}, oracle_spec={"oracle_id": task.operation, "oracle_hash": stable_hash(task.reference_source)}, semantic_contract=semantic, backend_contract=backend, provenance={"canonical": True, "task_id": task.task_id, "contract_hash": semantic_hash})


def task_source(task: CanonicalTask, dialect: str, *, mutant: bool = False) -> str:
    source = task.sources.get(dialect, "")
    if not mutant:
        return source
    if task.operation == "elementwise_add":
        source = source.replace("+ b[", "- b[").replace("+ tl.load(b", "- tl.load(b").replace("+ B[j * bstride]", "- B[j * bstride]").replace("i * a_stride", "i").replace("off * a_stride", "off").replace("j * astride", "j")
    elif task.operation == "scale":
        source = source.replace("* scale", "+ scale").replace("* scale, mask", "+ scale, mask").replace("i * x_stride", "i").replace("off * x_stride", "off").replace("j * xstride", "j")
    else:
        source = source.replace("col < cols", "col < cols - 1").replace("col * x_col_stride", "col").replace("col * xc", "col")
        source = source.replace("X[br, j]", "X[br, j] + 1.0")
    return source


def group_hash(task: CanonicalTask) -> str:
    return stable_hash({"task_id": task.task_id, "semantic": task.semantic_contract, "reference": task.reference_source})
