"""Build and run a numeric harness for one kernel.

CUDA path: generate ``harness.cu`` (``#include "solution.cu"`` + ``main``),
``nvcc``-compile, run, parse stdout JSON. Python path: generate a driver that
imports ``solution.py`` and calls the host entry with torch tensors.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from cuda_sft.compile import ensure_stubs
from cuda_sft.config import Settings, get_settings
from cuda_sft.refval.spec import (
    CasePlan,
    DialectRefvalSpec,
    FLOAT_DTYPES,
    KernelABI,
    KernelParam,
    cpp_type,
    dtype_nbytes,
    normalize_dtype,
)

_MAIN_RE = re.compile(r"\bint\s+main\s*\(", re.MULTILINE)


def has_main(source: str) -> bool:
    """True when the translation unit already defines ``main``."""
    return bool(_MAIN_RE.search(source or ""))


def _c_ident(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name or "arg")
    if cleaned and cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned or "arg"


def _c_scalar(value: Any, dtype: str) -> str:
    canon = normalize_dtype(dtype)
    if canon == "bool":
        return "true" if value else "false"
    if canon in FLOAT_DTYPES:
        number = float(value)
        if canon == "f64":
            return repr(number)
        return f"{number}f"
    return str(int(value))


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= max(0, int(dim))
    return n


def _shape_of(param: KernelParam, plan: CasePlan) -> tuple[int, ...]:
    if param.name in plan.shapes:
        return tuple(int(x) for x in plan.shapes[param.name])
    if param.shape_from:
        dims = []
        for name in param.shape_from:
            dims.append(int(plan.scalars.get(name, 0)))
        return tuple(dims)
    if param.rank <= 1:
        n = 0
        for value in plan.scalars.values():
            if isinstance(value, int):
                n = int(value)
                break
        return (n,)
    return tuple(int(next(iter(plan.scalars.values()), 0)) for _ in range(param.rank))


def _strides_of(param: KernelParam, plan: CasePlan, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return element strides, defaulting to contiguous row-major strides."""
    explicit = tuple(int(x) for x in (plan.strides.get(param.name) or ()))
    if explicit:
        if len(explicit) != len(shape) or any(x <= 0 for x in explicit):
            raise ValueError(f"invalid strides for {param.name}: {explicit}")
        return explicit
    out: list[int] = []
    stride = 1
    for dim in reversed(shape):
        out.append(stride)
        stride *= max(0, int(dim))
    return tuple(reversed(out))


def _storage_numel(shape: tuple[int, ...], strides: tuple[int, ...]) -> int:
    if not shape or any(int(dim) == 0 for dim in shape):
        return 0 if shape else 1
    return 1 + sum((int(dim) - 1) * int(stride) for dim, stride in zip(shape, strides))


def _emit_index_loop(var: str, shape: tuple[int, ...], strides: tuple[int, ...], body: str) -> str:
    """Emit a flat logical-index loop mapping into a padded physical buffer."""
    if any(int(dim) == 0 for dim in shape):
        return "    (void)0;"
    if not shape:
        return body.replace("__OFF__", "0").replace("__FLAT__", "0")
    lines = [f"    for (size_t flat = 0; flat < ne_{var}; ++flat) {{", "      size_t rem = flat;", "      size_t off = 0;"]
    for dim, stride in zip(reversed(shape), reversed(strides)):
        lines.append(f"      off += (rem % {int(dim)}ull) * {int(stride)}ull;")
        lines.append(f"      rem /= {int(dim)}ull;")
    lines.append("      " + body.replace("__OFF__", "off").replace("__FLAT__", "flat"))
    lines.append("    }")
    return "\n".join(lines)


def _needs_half(abi: KernelABI) -> bool:
    dtypes = {normalize_dtype(p.dtype) for p in abi.params}
    dtypes.add(normalize_dtype(abi.dtype))
    if abi.returns and abi.returns != "void":
        dtypes.add(normalize_dtype(abi.returns))
    return bool(dtypes & {"f16", "bf16"})


def _entry_typedef(abi: KernelABI) -> str:
    parts: list[str] = []
    for param in abi.params:
        if param.is_tensor:
            const = param.kind == "input"
            parts.append(cpp_type(param.dtype, pointer=True, const=const))
        else:
            parts.append(cpp_type(param.dtype, pointer=False))
    ret = "void" if not abi.returns or abi.returns == "void" else cpp_type(abi.returns)
    return f"using RefvalEntry = {ret}(*)({', '.join(parts) or 'void'});"


def _call_args(abi: KernelABI, plan: CasePlan, alias: Mapping[str, str]) -> str:
    args: list[str] = []
    for param in abi.params:
        ident = _c_ident(param.name)
        if param.is_tensor:
            src = alias.get(param.name) or param.alias_of
            if src:
                ident = _c_ident(src)
            ptr = f"d_{ident}" if param.memory != "host" else f"h_{ident}.data()"
            args.append(ptr)
        else:
            dtype = param.dtype if param.kind != "size" else (param.dtype or "i32")
            value = plan.scalars.get(param.name, 0)
            args.append(_c_scalar(value, dtype))
    return ", ".join(args)


def _emit_case(abi: KernelABI, plan: CasePlan) -> str:
    alias = dict(plan.alias)
    lines: list[str] = []
    lines.append(f"  // case {plan.name}")
    lines.append("  {")
    lines.append(f'    const char* case_name = "{plan.name}";')
    lines.append("    bool case_ok = true;")
    allocated: list[str] = []
    skipped_outputs_aliased: set[str] = set(alias.keys())
    for param in abi.tensor_params():
        if param.name in skipped_outputs_aliased:
            continue
        ident = _c_ident(param.name)
        shape = _shape_of(param, plan)
        ne = _prod(shape)
        strides = _strides_of(param, plan, shape)
        storage_ne = _storage_numel(shape, strides)
        nbytes = ne * dtype_nbytes(param.dtype)
        storage_nbytes = storage_ne * dtype_nbytes(param.dtype)
        ctype = cpp_type(param.dtype)
        lines.append(f"    size_t ne_{ident} = {ne}ull;")
        lines.append(f"    size_t nb_{ident} = {nbytes}ull;")
        lines.append(f"    size_t storage_ne_{ident} = {storage_ne}ull;")
        lines.append(f"    size_t storage_nb_{ident} = {storage_nbytes}ull;")
        lines.append(f"    std::vector<{ctype}> h_{ident}(storage_ne_{ident});")
        if param.memory != "host":
            lines.append(f"    {ctype}* d_{ident} = nullptr;")
            allocated.append(ident)
        if param.is_input and nbytes:
            rel = f"cases/{plan.name}/{param.name}.bin"
            lines.append(f"    std::vector<{ctype}> logical_{ident}(ne_{ident});")
            lines.append(
                f'    if (nb_{ident} && !read_bin("{rel}", logical_{ident}.data(), nb_{ident})) '
                f'{{ case_ok = false; emit_case(case_name, false, "read {param.name}"); }}'
            )
            lines.append(_emit_index_loop(ident, shape, strides, f"h_{ident}[__OFF__] = logical_{ident}[__FLAT__];"))
        if param.memory != "host":
            lines.append(
                f"    if (case_ok && storage_nb_{ident}) CUDA_DIE(cudaMalloc(&d_{ident}, storage_nb_{ident}), case_name);"
            )
            if param.is_input:
                lines.append(
                    f"    if (case_ok && storage_nb_{ident}) CUDA_DIE(cudaMemcpy(d_{ident}, h_{ident}.data(), "
                    f"storage_nb_{ident}, cudaMemcpyHostToDevice), case_name);"
                )
    call_args = _call_args(abi, plan, alias)
    ret_void = not abi.returns or abi.returns == "void"
    # Zero-sized tensors are a legal contract case.  The default strategy is
    # to skip the launch and validate the initialized empty output; manifests
    # may opt into ``zero_size_strategy=call`` for kernels that require it.
    zero_expr = " || ".join(f"(ne_{_c_ident(p.name)} == 0)" for p in abi.tensor_params()) or "false"
    should_call = "case_ok" if getattr(abi, "zero_size_strategy", "skip") == "call" else f"(case_ok && !({zero_expr}))"
    if ret_void:
        lines.append(f"    if ({should_call}) {abi.entry}({call_args});")
    else:
        rtype = cpp_type(abi.returns)
        lines.append(f"    {rtype} refval_ret = {rtype}{{}};")
        lines.append(f"    if ({should_call}) refval_ret = {abi.entry}({call_args});")
    lines.append("    if (case_ok) CUDA_DIE(cudaGetLastError(), case_name);")
    lines.append("    if (case_ok) CUDA_DIE(cudaDeviceSynchronize(), case_name);")
    for param in abi.output_params():
        src_name = alias.get(param.name) or param.alias_of or param.name
        ident = _c_ident(src_name)
        out_shape = _shape_of(param, plan)
        out_strides = _strides_of(param, plan, out_shape)
        nbytes = _prod(out_shape) * dtype_nbytes(param.dtype)
        rel = f"out/{plan.name}/{param.name}.bin"
        if param.memory != "host":
            lines.append(
                f"    if (case_ok && storage_nb_{ident}) CUDA_DIE(cudaMemcpy(h_{ident}.data(), d_{ident}, "
                f"storage_nb_{ident}, cudaMemcpyDeviceToHost), case_name);"
            )
            lines.append(f"    std::vector<{cpp_type(param.dtype)}> logical_out_{_c_ident(param.name)}(ne_{ident});")
            lines.append(_emit_index_loop(ident, out_shape, out_strides, f"logical_out_{_c_ident(param.name)}[__FLAT__] = h_{ident}[__OFF__];"))
            lines.append(
                f'    if (case_ok && !write_bin("{rel}", logical_out_{_c_ident(param.name)}.data(), {nbytes}ull)) '
                f'{{ case_ok = false; emit_case(case_name, false, "write {param.name}"); }}'
            )
        else:
            host_ident = _c_ident(src_name)
            lines.append(f"    std::vector<{cpp_type(param.dtype)}> logical_out_{_c_ident(param.name)}(ne_{ident});")
            lines.append(_emit_index_loop(ident, out_shape, out_strides, f"logical_out_{_c_ident(param.name)}[__FLAT__] = h_{host_ident}[__OFF__];"))
            lines.append(
                f'    if (case_ok && !write_bin("{rel}", logical_out_{_c_ident(param.name)}.data(), {nbytes}ull)) '
                f'{{ case_ok = false; emit_case(case_name, false, "write {param.name}"); }}'
            )
    if not ret_void:
        lines.append(
            f'    if (case_ok && !write_bin("out/{plan.name}/__return__.bin", &refval_ret, sizeof(refval_ret))) '
            f'{{ case_ok = false; emit_case(case_name, false, "write __return__"); }}'
        )
    for ident in allocated:
        lines.append(f"    if (d_{ident}) cudaFree(d_{ident});")
    lines.append("    if (case_ok) emit_case(case_name, true, \"\");")
    lines.append("  }")
    return "\n".join(lines)


def _signature_check(abi: KernelABI) -> tuple[str, str]:
    """Compile-time ABI check that also accepts a function template.

    ``&entry`` is ill-formed when ``entry`` is a template. A file-scope
    wrapper with the ABI's concrete signature calls ``entry``, so deduction
    or overload resolution still fails closed on a mismatched ABI. The
    wrapper cannot live inside ``main`` (C++ has no nested functions).

    Returns ``(file_scope, main_stmt)``.
    """
    typedef = _entry_typedef(abi)
    formals: list[str] = []
    actuals: list[str] = []
    for index, param in enumerate(abi.params):
        if param.is_tensor:
            ctype = cpp_type(param.dtype, pointer=True, const=param.kind == "input")
        else:
            ctype = cpp_type(param.dtype or "i32", pointer=False)
        formals.append(f"{ctype} refval_a{index}")
        actuals.append(f"refval_a{index}")
    ret = "void" if not abi.returns or abi.returns == "void" else cpp_type(abi.returns)
    call = f"{abi.entry}({', '.join(actuals)})"
    body = f"{call};" if ret == "void" else f"return {call};"
    joined = ", ".join(formals)
    file_scope = (
        f"{typedef}\n"
        f"static {ret} refval_invoke_entry({joined}) {{ {body} }}\n"
    )
    main_stmt = (
        "  RefvalEntry refval_entry_check = &refval_invoke_entry;\n"
        "  (void)refval_entry_check;\n"
    )
    return file_scope, main_stmt


def render_cuda_harness(abi: KernelABI, plans: list[CasePlan]) -> str:
    """Return ``harness.cu`` text that includes ``solution.cu`` and runs ``plans``."""
    headers = [
        "#include <cuda_runtime.h>",
        "#include <cstdio>",
        "#include <cstdlib>",
        "#include <cstdint>",
        "#include <vector>",
        "#include <string>",
    ]
    if _needs_half(abi):
        headers.append("#include <cuda_fp16.h>")
        headers.append("#include <cuda_bf16.h>")
    headers.append('#include "solution.cu"')
    prelude = r'''
static bool g_ok = true;
static bool g_first_case = true;

static bool read_bin(const char* path, void* dst, size_t n) {
  if (n == 0) return true;
  FILE* f = fopen(path, "rb");
  if (!f) return false;
  size_t got = fread(dst, 1, n, f);
  fclose(f);
  return got == n;
}

static bool write_bin(const char* path, const void* src, size_t n) {
  FILE* f = fopen(path, "wb");
  if (!f) return false;
  if (n == 0) { fclose(f); return true; }
  size_t got = fwrite(src, 1, n, f);
  fclose(f);
  return got == n;
}

static void emit_case(const char* name, bool ok, const char* err) {
  if (!g_first_case) fputc(',', stdout);
  g_first_case = false;
  if (ok) {
    printf("{\"name\":\"%s\",\"ok\":true}", name);
  } else {
    g_ok = false;
    printf("{\"name\":\"%s\",\"ok\":false,\"error\":\"%s\"}", name, err ? err : "");
  }
}

static void fail_case(const char* name, const char* err) {
  emit_case(name, false, err);
}

#define CUDA_DIE(call, case_name)                                              \
  do {                                                                         \
    if (!case_ok) break;                                                       \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s: %s\n", (case_name),                      \
              cudaGetErrorString(err__));                                      \
      case_ok = false;                                                         \
      emit_case((case_name), false, cudaGetErrorString(err__));                 \
      cudaGetLastError();                                                      \
    }                                                                          \
  } while (0)
'''
    # Force a compile-time signature check so a wrong ABI fails as signature_mismatch.
    # A direct ``&entry`` is ill-formed for function templates (``&scale`` does
    # not name a single function). Call through a concrete wrapper instead:
    # template argument deduction still rejects a mismatched ABI.
    sig_scope, sig_check = _signature_check(abi)
    case_blocks = "\n".join(_emit_case(abi, plan) for plan in plans)
    body = f"""
{sig_scope}
int main() {{
{sig_check}
  printf("{{\\\"ok\\\":true,\\\"cases\\\":[");
{case_blocks}
  printf("]}}\\n");
  return g_ok ? 0 : 1;
}}
"""
    return "\n".join(headers) + "\n" + prelude + body


def render_python_driver(
    abi: KernelABI,
    plans: list[CasePlan],
    source_filename: str,
    *,
    dialect: str = "triton",
) -> str:
    """Return a strict CUDA-only Python driver for a frozen Triton ABI.

    The generated driver intentionally has one call path.  A ``TypeError`` is
    an ABI/shape contract failure; it is never interpreted as a factory API.
    ``call_style`` is carried by newer manifests and defaults to positional for
    the historical ``launch_*`` ABI.
    """
    param_list = [
        {
            "name": p.name,
            "kind": p.kind,
            "dtype": p.dtype,
            "memory": p.memory,
            "rank": p.rank,
            "layout": p.layout,
            "shape_from": list(p.shape_from),
            "optional": bool(p.optional),
            "alias_of": p.alias_of,
        }
        for p in abi.params
    ]
    payload = {
        "entry": abi.entry,
        "returns": abi.returns,
        "params": param_list,
        "result_names": list(abi.result_names()),
        "source": source_filename,
        "call_style": str(
            getattr(abi, "call_style", "auto" if dialect == "tilelang" else "positional")
            or ("auto" if dialect == "tilelang" else "positional")
        ),
        "dialect": dialect,
        "cases": [plan.to_dict() for plan in plans],
    }
    blob = json.dumps(payload, ensure_ascii=False)
    return f'''# Auto-generated strict Triton refval driver
from __future__ import annotations
import importlib.util
import inspect
import json
from pathlib import Path
import numpy as np
import torch

SPEC = json.loads({blob!r})
DTYPES = {{
    "f32": torch.float32, "f64": torch.float64, "f16": torch.float16,
    "bf16": torch.bfloat16, "i32": torch.int32, "i64": torch.int64,
    "i8": torch.int8, "u8": torch.uint8, "u32": torch.uint32,
    "bool": torch.bool,
}}
NP_DTYPES = {{
    "f32": "float32", "f64": "float64", "f16": "float16",
    "bf16": "float32", "i32": "int32", "i64": "int64",
    "i8": "int8", "u8": "uint8", "u32": "uint32", "bool": "bool",
}}

class ContractError(RuntimeError):
    pass

def _load(path: Path):
    spec = importlib.util.spec_from_file_location("_refval_kernel", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot load {{path}}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _classify(exc):
    text = str(exc).lower()
    if isinstance(exc, ContractError) or isinstance(exc, TypeError):
        return "shape_contract"
    if any(tok in text for tok in ("illegal memory", "out of bounds", "misaligned address", "device-side assert")):
        return "cuda_illegal_memory"
    if isinstance(exc, (RuntimeError, ValueError, AssertionError)):
        return "launch_runtime"
    return "launch_runtime"

def _shape(case, p):
    shape = tuple(int(x) for x in (case.get("shapes") or {{}}).get(p["name"], ()))
    if p["kind"] in ("input", "output", "inout") and not shape:
        raise ContractError(f"missing shape for {{p['name']}}")
    if any(x < 0 for x in shape):
        raise ContractError(f"negative shape for {{p['name']}}: {{shape}}")
    if len(shape) != int(p.get("rank") or 0):
        raise ContractError(f"rank mismatch for {{p['name']}}: got {{len(shape)}} expected {{p.get('rank')}}")
    return shape

def _tensor_from_bin(testdir, case, p):
    name = p["name"]
    shape = _shape(case, p)
    src = (case.get("alias") or {{}}).get(name) or name
    path = testdir / "cases" / case["name"] / f"{{src}}.bin"
    if not path.is_file():
        path = testdir / "cases" / case["name"] / f"{{name}}.bin"
    if not path.is_file():
        raise ContractError(f"missing input binding {{name}}")
    if p["dtype"] == "bf16":
        raw = np.fromfile(str(path), dtype="<u2")
        arr = (raw.astype(np.uint32) << 16).view(np.float32)
    else:
        arr = np.fromfile(str(path), dtype=np.dtype(NP_DTYPES[p["dtype"]]))
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if arr.size != expected:
        raise ContractError(f"input numel mismatch {{name}}: got {{arr.size}} expected {{expected}}")
    arr = arr.reshape(shape)
    base = torch.from_numpy(np.ascontiguousarray(arr)).to(device="cuda", dtype=DTYPES[p["dtype"]])
    strides = tuple(int(x) for x in (case.get("strides") or {{}}).get(name, ()))
    if strides:
        if len(strides) != len(shape) or any(x <= 0 for x in strides):
            raise ContractError(f"invalid strides for {{name}}: {{strides}}")
        storage = 0 if any(d == 0 for d in shape) else (1 + sum((d - 1) * s for d, s in zip(shape, strides)) if shape else 1)
        raw = torch.empty(storage, device="cuda", dtype=base.dtype)
        if storage:
            raw.fill_(float("nan") if base.dtype.is_floating_point else 0x5A)
        view = torch.as_strided(raw, shape, strides)
        view.copy_(base)
        return view
    return base

def _new_output(case, p):
    shape = _shape(case, p)
    strides = tuple(int(x) for x in (case.get("strides") or {{}}).get(p["name"], ()))
    if strides:
        if len(strides) != len(shape) or any(x <= 0 for x in strides):
            raise ContractError(f"invalid strides for {{p['name']}}: {{strides}}")
        storage = 0 if any(d == 0 for d in shape) else (1 + sum((d - 1) * s for d, s in zip(shape, strides)) if shape else 1)
        raw = torch.zeros(storage, device="cuda", dtype=DTYPES[p["dtype"]])
        if storage:
            raw.fill_(float("nan") if raw.dtype.is_floating_point else 0x5A)
        return torch.as_strided(raw, shape, strides)
    return torch.zeros(shape, device="cuda", dtype=DTYPES[p["dtype"]])

def _validate_callable(fn):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"cannot inspect host entry: {{exc}}") from exc
    params = SPEC["params"]
    names = [p.name for p in sig.parameters.values()
             if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    required = [p["name"] for p in params if not p.get("optional")]
    if SPEC.get("dialect") == "tilelang" and SPEC.get("call_style") in {
        "auto", "lazy", "factory", "primfunc_factory", "prim_func_factory",
    }:
        return sig
    if len(names) != len(params) or names != [p["name"] for p in params]:
        raise ContractError(f"ABI parameter names/order mismatch: got {{names}} expected {{[p['name'] for p in params]}}")
    if any(name not in names for name in required):
        raise ContractError("missing required ABI parameter")
    return sig

def _tilelang_factory(fn, tensors, scalars):
    import tilelang
    try:
        sig = inspect.signature(fn)
        names = [p.name for p in sig.parameters.values()
                 if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    except (TypeError, ValueError):
        names = []
    size_kwargs = {{name: scalars[name] for name in names if name in scalars}}
    # TileLang factories commonly spell the logical extent ``N`` while the
    # frozen contract uses canonical lower-case ``n``. Resolve this adapter
    # mapping explicitly instead of guessing a different ABI.
    for name in names:
        if name not in size_kwargs and name.lower() in scalars:
            size_kwargs[name] = scalars[name.lower()]
    if size_kwargs:
        try:
            built = fn(**size_kwargs)
        except TypeError:
            built = fn(*(scalars[name] for name in names if name in scalars))
    else:
        built = fn()
    if "PrimFunc" in type(built).__name__:
        built = tilelang.compile(built, target="cuda")
    if not callable(built):
        raise ContractError(f"TileLang factory returned {{type(built).__name__}}")
    values = [tensors[p["name"]] for p in SPEC["params"]
              if p["kind"] in ("input", "output", "inout")]
    return built(*values)

def _call(fn, tensors, scalars):
    params = SPEC["params"]
    values = []
    for p in params:
        if p["kind"] in ("input", "output", "inout"):
            if p["name"] not in tensors:
                raise ContractError(f"missing tensor binding {{p['name']}}")
            values.append(tensors[p["name"]])
        else:
            if p["name"] not in scalars:
                raise ContractError(f"missing scalar binding {{p['name']}}")
            values.append(scalars[p["name"]])
    style = SPEC.get("call_style", "positional")
    if style in {"positional", "eager"}:
        return fn(*values)
    if style == "kwargs":
        return fn(**{{p["name"]: value for p, value in zip(params, values)}})
    if SPEC.get("dialect") == "tilelang" and style in {
        "auto", "lazy", "factory", "primfunc_factory", "prim_func_factory",
    }:
        try:
            return fn(*values)
        except TypeError:
            return _tilelang_factory(fn, tensors, scalars)
    raise ContractError(f"unsupported call_style {{style!r}}")

def _check_output(value, p):
    if not torch.is_tensor(value):
        raise ContractError(f"output {{p['name']}} is not a torch.Tensor")
    shape = _shape(CURRENT_CASE, p)
    expected_strides = tuple(int(x) for x in (CURRENT_CASE.get("strides") or {{}}).get(p["name"], ()))
    if value.device.type != "cuda" or value.dtype != DTYPES[p["dtype"]] or tuple(value.shape) != shape:
        raise ContractError(f"output binding mismatch {{p['name']}}: device={{value.device}} dtype={{value.dtype}} shape={{tuple(value.shape)}}")
    if expected_strides and tuple(value.stride()) != expected_strides:
        raise ContractError(f"output stride mismatch {{p['name']}}: got={{tuple(value.stride())}} expected={{expected_strides}}")

def main() -> int:
    if not torch.cuda.is_available():
        print(json.dumps({{"ok": False, "error_class": "launch_runtime", "error": "CUDA device is required"}}))
        return 2
    testdir = Path(__file__).resolve().parent
    try:
        mod = _load(testdir / SPEC["source"])
        fn = getattr(mod, SPEC["entry"], None)
        if fn is None or not callable(fn):
            raise ContractError(f"entry {{SPEC['entry']}} not found")
        _validate_callable(fn)
    except Exception as exc:
        print(json.dumps({{"ok": False, "error_class": _classify(exc), "error": str(exc)}}))
        return 2
    results = []
    all_ok = True
    global CURRENT_CASE
    for case in SPEC["cases"]:
        name = case["name"]
        CURRENT_CASE = case
        tensors = {{}}
        try:
            alias = case.get("alias") or {{}}
            scalars = case.get("scalars") or {{}}
            for p in SPEC["params"]:
                if p["kind"] in ("input", "inout"):
                    tensors[p["name"]] = _tensor_from_bin(testdir, case, p)
                elif p["kind"] == "output":
                    src = alias.get(p["name"]) or p.get("alias_of")
                    tensors[p["name"]] = tensors[src] if src else _new_output(case, p)
            ret = _call(fn, tensors, scalars)
            torch.cuda.synchronize()
            for p in SPEC["params"]:
                if p["kind"] in ("output", "inout"):
                    _check_output(tensors[p["name"]], p)
            out_dir = testdir / "out" / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for rname in SPEC["result_names"]:
                if rname == "__return__":
                    if SPEC["returns"] == "void" or ret is None:
                        raise ContractError("missing scalar return")
                    if torch.is_tensor(ret):
                        if ret.device.type != "cuda" or ret.numel() != 1 or ret.dtype != DTYPES[SPEC["returns"]]:
                            raise ContractError("scalar return binding mismatch")
                        if SPEC["returns"] == "bf16":
                            value = ret.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
                        else:
                            value = ret.detach().cpu().numpy()
                    else:
                        if SPEC["returns"] == "bf16":
                            bits = np.asarray(ret, dtype=np.float32).view(np.uint32)
                            value = ((bits + ((bits >> 16) & 1) + 0x7FFF) >> 16).astype("<u2")
                        else:
                            value = np.asarray(ret, dtype=np.dtype(NP_DTYPES[SPEC["returns"]]))
                    np.ascontiguousarray(value).tofile(out_dir / "__return__.bin")
                    continue
                value = tensors.get(rname)
                if value is None:
                    raise ContractError(f"missing output binding {{rname}}")
                pmeta = next((p for p in SPEC["params"] if p["name"] == rname), None)
                if pmeta is not None and pmeta["dtype"] == "bf16":
                    value.detach().contiguous().view(torch.int16).cpu().numpy().view(np.uint16).tofile(out_dir / f"{{rname}}.bin")
                else:
                    np.ascontiguousarray(value.detach().cpu().numpy()).tofile(out_dir / f"{{rname}}.bin")
            results.append({{"name": name, "ok": True}})
        except Exception as exc:
            all_ok = False
            results.append({{"name": name, "ok": False, "error_class": _classify(exc), "error": str(exc)}})
    classes = [r.get("error_class") for r in results if not r.get("ok")]
    payload = {{"ok": all_ok, "cases": results}}
    if classes:
        payload["error_class"] = classes[0]
    print(json.dumps(payload))
    return 0 if all_ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
'''


def _nvcc_env(extra_includes: list[str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    extras = [p for p in (extra_includes or []) if p]
    if extras:
        joined = os.pathsep.join(extras)
        env["CPATH"] = joined
        env["CPLUS_INCLUDE_PATH"] = joined
        env["C_INCLUDE_PATH"] = joined
    return env


def compile_cuda_harness(
    testdir: Path,
    *,
    settings: Settings | None = None,
    extra_includes: list[str] | None = None,
    std: str = "c++17",
    used_rdc: bool = False,
    timeout_sec: int = 60,
) -> tuple[bool, str, Path | None]:
    """``nvcc`` ``harness.cu`` (which includes ``solution.cu``) to ``refval_bin``."""
    settings = settings or get_settings()
    testdir = Path(testdir).resolve()
    testdir.mkdir(parents=True, exist_ok=True)
    ensure_stubs(testdir)
    source = testdir / "harness.cu"
    binary = testdir / "refval_bin"
    cxx = (std or "c++17").strip()
    if not cxx.startswith("c++"):
        cxx = "c++17"
    cmd = [
        settings.nvcc_bin,
        source.name,
        "-o",
        binary.name,
        f"-std={cxx}",
        f"-arch={settings.resolved_cuda_arch}",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        f"-I{settings.resolved_cuda_home}/include",
        f"-I{testdir}",
        f"-I{testdir / 'include'}",
        "-lcudart",
    ]
    for include in extra_includes or []:
        path = str(include).strip()
        if path:
            cmd.append(f"-I{path}")
    if used_rdc:
        cmd.extend(["-rdc=true", "-lcudadevrt"])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout_sec)),
            check=False,
            env=_nvcc_env(extra_includes),
            cwd=str(testdir),
        )
    except subprocess.TimeoutExpired:
        return False, f"nvcc timed out after {timeout_sec}s", None
    except OSError as exc:
        return False, f"failed to launch nvcc: {exc}", None
    merged = "\n".join(
        part for part in ((proc.stdout or "").strip(), (proc.stderr or "").strip()) if part
    )
    if proc.returncode != 0:
        return False, f"{shlex.join(cmd)}\n{merged}".strip(), None
    return True, merged, binary


def _parse_stdout_json(text: str) -> dict[str, Any]:
    blob = (text or "").strip()
    if not blob:
        return {}
    # Last JSON object on stdout (kernels may printf debug lines).
    for line in reversed(blob.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
    try:
        start = blob.find("{")
        end = blob.rfind("}")
        if start >= 0 and end > start:
            payload = json.loads(blob[start : end + 1])
            if isinstance(payload, dict):
                return payload
    except json.JSONDecodeError:
        return {}
    return {}


def run_binary(
    binary: Path,
    testdir: Path,
    *,
    timeout_sec: int,
    extra_env: Mapping[str, str] | None = None,
) -> tuple[int, str, dict[str, Any]]:
    """Run a harness binary/driver. Returns ``(exit_code, combined_output, json)``."""
    testdir = Path(testdir).resolve()
    binary = Path(binary)
    if not binary.is_absolute():
        binary = testdir / binary
    env = os.environ.copy()
    if extra_env:
        env.update({str(k): str(v) for k, v in extra_env.items()})
    try:
        proc = subprocess.run(
            [str(binary)] if binary.suffix != ".py" else [sys.executable, str(binary)],
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
            cwd=str(testdir),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        tail = ""
        if exc.stderr:
            tail = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", "replace")
        return 124, f"timed out after {timeout_sec}s\n{tail}".strip(), {"ok": False, "error": "timeout"}
    except OSError as exc:
        return 127, f"failed to launch harness: {exc}", {"ok": False, "error": str(exc)}
    merged = "\n".join(
        part for part in ((proc.stdout or "").strip(), (proc.stderr or "").strip()) if part
    )
    payload = _parse_stdout_json(proc.stdout or "") or _parse_stdout_json(merged)
    return proc.returncode, merged, payload


def prepare_testdir(
    testdir: Path,
    *,
    source: str,
    filename: str,
    abi: KernelABI,
    plans: list[CasePlan],
    dialect_spec: DialectRefvalSpec,
) -> None:
    """Write source, stubs, harness/driver, and case input bins."""
    testdir = Path(testdir).resolve()
    testdir.mkdir(parents=True, exist_ok=True)
    (testdir / "out").mkdir(parents=True, exist_ok=True)
    for plan in plans:
        (testdir / "out" / plan.name).mkdir(parents=True, exist_ok=True)
    ensure_stubs(testdir)
    (testdir / filename).write_text(source, encoding="utf-8")
    from cuda_sft.refval.cases import write_cases

    write_cases(testdir, abi, plans)
    if dialect_spec.runner == "python_import":
        driver = render_python_driver(abi, plans, filename, dialect=dialect_spec.dialect)
        (testdir / "driver.py").write_text(driver, encoding="utf-8")
    else:
        harness = render_cuda_harness(abi, plans)
        (testdir / "harness.cu").write_text(harness, encoding="utf-8")


def run_prepared(
    testdir: Path,
    dialect_spec: DialectRefvalSpec,
    *,
    settings: Settings | None = None,
    used_rdc: bool = False,
    timeout_sec: int = 30,
) -> tuple[int, str, dict[str, Any]]:
    """Compile (CUDA) if needed and run the prepared harness."""
    settings = settings or get_settings()
    testdir = Path(testdir).resolve()
    if dialect_spec.runner == "python_import":
        driver = testdir / "driver.py"
        return run_binary(driver, testdir, timeout_sec=timeout_sec)
    ok, log, binary = compile_cuda_harness(
        testdir,
        settings=settings,
        extra_includes=list(dialect_spec.extra_includes),
        std=dialect_spec.cxx_std,
        used_rdc=used_rdc,
        timeout_sec=min(timeout_sec, max(5, int(getattr(settings, "nvcc_timeout_sec", 60)))),
    )
    if not ok or binary is None:
        return 2, log, {"ok": False, "error": log, "error_class": "signature_mismatch"}
    code, output, payload = run_binary(binary, testdir, timeout_sec=timeout_sec)
    if not payload:
        payload = {"ok": False, "error": output}
    if code != 0 and "error_class" not in payload:
        lowered = (output or "").lower()
        if "timeout" in lowered:
            payload["error_class"] = "timeout"
        elif "error:" in lowered:
            payload.setdefault("error_class", "crash")
    payload.setdefault("compile_log", log)
    return code, (log + "\n" + output).strip(), payload
