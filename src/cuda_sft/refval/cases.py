"""Adversarial / random case plans and numpy (or list) materialization.

CPU reference and GPU harness consume the same seed so both sides see
identical inputs. Large cases stay in the 1e6–4e6 element band so an
RTX 3060 launch stays in the millisecond range.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any, Callable, Mapping

from cuda_sft.refval.spec import (
    CasePlan,
    FLOAT_DTYPES,
    INT_DTYPES,
    KernelABI,
    KernelParam,
    dtype_nbytes,
    case_seed_for,
    cases_hash,
    normalize_dtype,
    numpy_dtype_name,
    seed_for,
)

NON_TILE_SIZES = (1, 7, 33, 63, 255, 257)
NON_POW2_SIZES = (3, 6, 10, 12, 24, 48, 100, 200, 1000)
SMOKE_SIZES = (0, 1, 7)


def _try_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError:
        return None


def numpy_available() -> bool:
    """True when ``numpy`` can be imported."""
    return _try_numpy() is not None


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= max(0, int(dim))
    return n


def _primary_n(kind: str, *, rank: int, max_elements: int) -> int:
    if kind == "n0":
        return 0
    if kind == "n1":
        return 1
    if kind == "large":
        if rank <= 1:
            return min(max_elements, 1_000_000)
        side = int(max(1, math.sqrt(min(max_elements, 4_000_000))))
        return max(2, side)
    return 8


def _size_names(abi: KernelABI) -> list[str]:
    names = [p.name for p in abi.size_params()]
    if names:
        return names
    inferred: list[str] = []
    for param in abi.tensor_params():
        for item in param.shape_from:
            if item not in inferred:
                inferred.append(item)
    return inferred or ["n"]


def _rank_for(abi: KernelABI) -> int:
    ranks = [max(1, int(p.rank or 1)) for p in abi.tensor_params()]
    return max(ranks) if ranks else 1


def _shape_for_param(
    param: KernelParam,
    sizes: Mapping[str, int],
    *,
    default_n: int,
    broadcast: bool = False,
) -> tuple[int, ...]:
    if not param.is_tensor:
        return ()
    if param.shape_from:
        dims: list[int] = []
        for name in param.shape_from:
            dims.append(int(sizes.get(name, default_n)))
        if broadcast and dims:
            # Shrink the last axis of non-output inputs so broadcast bugs surface.
            if param.kind == "input" and len(dims) >= 2:
                dims[-1] = 1 if dims[-1] > 1 else dims[-1]
        return tuple(max(0, d) for d in dims)
    rank = max(1, int(param.rank or 1))
    if rank == 1:
        return (max(0, default_n),)
    if broadcast and param.kind == "input":
        return (max(0, default_n),) + (1,) * (rank - 1)
    return tuple(max(0, default_n) for _ in range(rank))


def _cap_shapes(
    shapes: dict[str, tuple[int, ...]],
    sizes: dict[str, int],
    max_elements: int,
) -> None:
    biggest = 0
    for shape in shapes.values():
        biggest = max(biggest, _prod(shape))
    if biggest <= max_elements or biggest <= 0:
        return
    scale = (max_elements / float(biggest)) ** (1.0 / max(1, max(len(s) for s in shapes.values()) or 1))
    for key, value in list(sizes.items()):
        sizes[key] = max(0, int(value * scale))
    for name, shape in list(shapes.items()):
        shapes[name] = tuple(max(0, int(dim * scale)) for dim in shape)


def _scalar_defaults(abi: KernelABI, sizes: Mapping[str, int], rng_seed: int) -> dict[str, Any]:
    out: dict[str, Any] = dict(sizes)
    # Tiny deterministic mix so scale/threshold kernels are not identity-on-1.
    mix = (int(rng_seed) % 97) / 97.0
    for param in abi.scalar_params():
        dtype = normalize_dtype(param.dtype)
        name = param.name.lower()
        if dtype in INT_DTYPES:
            if "stride" in name:
                out[param.name] = 1
            elif "scale" in name or "factor" in name:
                out[param.name] = 2 if mix < 0.5 else 3
            elif "threshold" in name:
                out[param.name] = 1
            else:
                out[param.name] = 1 + int(mix * 5)
        elif dtype == "bool":
            out[param.name] = True
        else:
            if "scale" in name or "alpha" in name or "factor" in name:
                out[param.name] = 1.5 if mix < 0.5 else -2.0
            elif "mu" in name:
                out[param.name] = 0.0
            elif "sigma" in name:
                out[param.name] = 1.0
            elif "threshold" in name:
                out[param.name] = 0.5
            else:
                out[param.name] = 1.25 + mix
    for param in abi.size_params():
        if "stride" in param.name.lower():
            out[param.name] = 1
        else:
            out.setdefault(param.name, int(sizes.get(param.name, 0)))
    return out


def _inplace_alias(abi: KernelABI) -> dict[str, str]:
    # Aliasing is opt-in.  Auto-pairing an arbitrary input/output made every
    # ordinary out-of-place ABI accidentally run as an in-place case.
    alias = {p.name: p.alias_of for p in abi.params if p.alias_of}
    if alias or abi.in_place or getattr(abi, "supports_inplace", False):
        if alias:
            return alias
        outputs = list(abi.output_params())
        inputs = [p for p in abi.input_params() if p.kind == "input"]
        for out_p in outputs:
            for in_p in inputs:
                if in_p.name != out_p.name and normalize_dtype(in_p.dtype) == normalize_dtype(out_p.dtype):
                    return {out_p.name: in_p.name}
    return {}


def build_case_plans(
    abi: KernelABI,
    *,
    question_id: int,
    dialect: str,
    suite: str = "standard",
    max_elements: int = 4_000_000,
) -> list[CasePlan]:
    """Build the adversarial suite for one ABI.

    Args:
        abi: Host ABI.
        question_id: Used for ``seed_for``.
        dialect: Used for ``seed_for``.
        suite: ``smoke`` / ``standard`` / ``full``.
        max_elements: Cap on tensor numel (large case).
    """
    suite_name = (suite or "standard").strip().lower()
    rank = _rank_for(abi)
    size_names = _size_names(abi)
    plans: list[CasePlan] = []

    def _sizes_for(n: int, *, vary_tail: bool = False) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for index, name in enumerate(size_names):
            if index == 0:
                sizes[name] = int(n)
            elif vary_tail and n > 1:
                sizes[name] = max(1, int(n) // 2)
            else:
                sizes[name] = int(n)
        return sizes

    def _plan(
        name: str,
        kind: str,
        n: int,
        *,
        broadcast: bool = False,
        inplace: bool = False,
        large: bool = False,
        **flags: bool,
    ) -> CasePlan:
        sizes = _sizes_for(n, vary_tail=broadcast or kind == "broadcast")
        shapes: dict[str, tuple[int, ...]] = {}
        for param in abi.tensor_params():
            shapes[param.name] = _shape_for_param(
                param, sizes, default_n=n, broadcast=broadcast
            )
        _cap_shapes(shapes, sizes, max_elements)
        # The same semantic case must produce identical bytes for CUDA,
        # CUTLASS and Python backends.
        seed = case_seed_for(question_id, name)
        scalars = _scalar_defaults(abi, sizes, seed)
        alias = _inplace_alias(abi) if inplace else {}
        strides: dict[str, tuple[int, ...]] = {}
        if kind == "strided":
            for param in abi.tensor_params():
                shape = shapes.get(param.name) or ()
                if len(shape) == 1 and shape[0] > 0:
                    strides[param.name] = (2,)
                elif len(shape) >= 2 and all(int(x) > 0 for x in shape):
                    strides[param.name] = tuple(int(x) * 2 for x in shape[1:]) + (1,)
        dtypes = {param.name: normalize_dtype(param.dtype) for param in abi.tensor_params()}
        return CasePlan(
            name=name,
            kind=kind,
            shapes=shapes,
            scalars=scalars,
            seed=seed,
            generate_nan=bool(flags.get("generate_nan", False)) and abi.allows_nan,
            generate_inf=bool(flags.get("generate_inf", False)),
            generate_denorm=bool(flags.get("generate_denorm", False)),
            generate_dup=bool(flags.get("generate_dup", False)),
            strides=strides,
            alias=alias,
            large=large,
            dtypes=dtypes,
        )

    plans.append(_plan("n0", "n0", 0))
    plans.append(_plan("n1", "n1", 1))

    if suite_name == "smoke":
        plans.append(_plan("odd_7", "non_tile", 7))
        return plans

    for n in NON_TILE_SIZES:
        if n in {0, 1}:
            continue
        plans.append(_plan(f"odd_{n}", "non_tile", n))

    plans.append(_plan("non_pow2_100", "non_pow2", 100))
    plans.append(_plan("non_pow2_24", "non_pow2", 24))

    large_n = _primary_n("large", rank=rank, max_elements=max_elements)
    plans.append(_plan("large", "large", large_n, large=True))

    plans.append(
        _plan(
            "extreme",
            "extreme",
            33 if rank <= 1 else 17,
            generate_denorm=True,
            generate_inf=False,
            generate_nan=bool(abi.allows_nan),
        )
    )
    plans.append(_plan("dup", "dup", 32, generate_dup=True))

    if abi.supports_strided or any(p.layout == "strided" for p in abi.params) or abi.layout == "strided":
        plans.append(_plan("strided", "strided", 33))

    if abi.supports_broadcast or len(size_names) > 1 or any(len(p.shape_from) > 1 for p in abi.tensor_params()):
        plans.append(_plan("broadcast", "broadcast", 15, broadcast=True))

    if abi.supports_inplace or abi.in_place or any(p.alias_of for p in abi.params):
        plans.append(_plan("inplace", "inplace", 33, inplace=True))

    if rank >= 2:
        plans.append(_plan("row_major_tail", "row_major_tail", 33))
        plans.append(_plan("tail_17", "tail", 17))
    else:
        plans.append(_plan("tail_33", "tail", 33))

    if suite_name == "full":
        plans.append(_plan("non_pow2_1000", "non_pow2", 1000))
        plans.append(_plan("odd_511", "non_tile", 511))

    # Drop duplicate names while keeping order.
    seen: set[str] = set()
    unique: list[CasePlan] = []
    for plan in plans:
        if plan.name in seen:
            continue
        seen.add(plan.name)
        unique.append(plan)
    return unique


def case_plan_hash(plans: list[CasePlan]) -> str:
    """Stable hash of case semantics, excluding generated files."""
    return cases_hash(plans)


def _pack_floats(values: list[float], dtype: str) -> bytes:
    code = {"f32": "<f", "f64": "<d", "f16": None}.get(dtype, "<f")
    if dtype == "f16":
        np = _try_numpy()
        if np is not None:
            return np.asarray(values, dtype=np.float16).tobytes()
        # Truncate to f32 storage if numpy is missing; harness still uses f32 path.
        return b"".join(struct.pack("<f", float(v)) for v in values)
    fmt = code or "<f"
    return b"".join(struct.pack(fmt, float(v)) for v in values)


def _pack_ints(values: list[int], dtype: str) -> bytes:
    fmt = {
        "i32": "<i",
        "i64": "<q",
        "i8": "<b",
        "u8": "<B",
        "u32": "<I",
        "bool": "<?",
    }.get(dtype, "<i")
    return b"".join(struct.pack(fmt, int(v)) for v in values)


def _fill_list(
    n: int,
    dtype: str,
    rng: Any,
    *,
    plan: CasePlan,
) -> list[Any]:
    if n <= 0:
        return []
    if dtype in FLOAT_DTYPES:
        if plan.generate_dup:
            base = [0.0, 1.0, 1.0, 2.0, 2.0, 3.0]
            return [base[i % len(base)] for i in range(n)]
        if plan.kind == "extreme":
            pattern = [
                0.0,
                -0.0,
                1.0,
                -1.0,
                1.0e-20,
                -1.0e-20,
                1.0e10,
                -1.0e10,
            ]
            if plan.generate_denorm:
                pattern.extend([1.0e-40, -1.0e-40])
            if plan.generate_nan:
                pattern.append(float("nan"))
            if plan.generate_inf:
                pattern.extend([float("inf"), float("-inf")])
            return [pattern[i % len(pattern)] for i in range(n)]
        # Uniform-ish deterministic floats in (-2, 2).
        out = []
        for i in range(n):
            u = rng.random() if hasattr(rng, "random") else ((i * 1103515245 + rng) % 2**31) / 2**31
            out.append(float(u) * 4.0 - 2.0)
        return out
    if dtype == "bool":
        return [bool(i % 2) for i in range(n)]
    if plan.generate_dup:
        base = [1, 1, 2, 2, 3, 4, 4]
        return [base[i % len(base)] for i in range(n)]
    lo, hi = (-8, 8) if dtype in {"i8", "u8"} else (-50, 50)
    if dtype.startswith("u"):
        lo = 0
    out = []
    for i in range(n):
        if hasattr(rng, "randrange"):
            out.append(rng.randrange(lo, hi + 1))
        else:
            out.append((i * 17 + int(plan.seed)) % (hi - lo + 1) + lo)
    return out


def materialize_arrays(
    plan: CasePlan,
    abi: KernelABI,
) -> dict[str, Any]:
    """Build host tensors as numpy arrays (or nested lists if numpy is missing).

    Only input/inout tensors are filled; outputs are zeros of the right shape.
    """
    import random

    np = _try_numpy()
    rng = random.Random(int(plan.seed))
    arrays: dict[str, Any] = {}
    for param in abi.tensor_params():
        shape = tuple(int(x) for x in (plan.shapes.get(param.name) or ()))
        n = _prod(shape)
        dtype = normalize_dtype(param.dtype)
        if param.is_input:
            values = _fill_list(n, dtype, rng, plan=plan)
        else:
            values = [0] * n if dtype not in FLOAT_DTYPES else [0.0] * n
        if np is not None:
            arr = np.asarray(values, dtype=numpy_dtype_name(dtype))
            if shape:
                try:
                    arr = arr.reshape(shape)
                except ValueError:
                    arr = arr.reshape(-1)
            arrays[param.name] = arr
        else:
            arrays[param.name] = values
    return arrays


def write_tensor_bin(path: Path, array: Any, dtype: str) -> int:
    """Write a contiguous little-endian tensor. Returns byte count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np = _try_numpy()
    canon = normalize_dtype(dtype)
    if np is not None and hasattr(array, "tobytes"):
        arr = np.ascontiguousarray(array)
        if canon == "f16":
            payload = np.asarray(arr, dtype=np.float16).tobytes()
        elif canon == "bf16":
            # numpy has no stable bfloat16 dtype.  Store the contract's native
            # 16-bit representation (round-to-nearest-even) rather than a
            # misleading 32-bit payload.
            bits = np.asarray(arr, dtype=np.float32).view(np.uint32)
            rounded = bits + ((bits >> 16) & 1) + 0x7FFF
            payload = (rounded >> 16).astype("<u2").tobytes()
        else:
            payload = np.asarray(arr, dtype=numpy_dtype_name(canon)).tobytes()
        path.write_bytes(payload)
        return len(payload)
    flat: list[Any]
    if isinstance(array, (list, tuple)):
        flat = list(array)
    else:
        flat = list(array)
    if canon in FLOAT_DTYPES:
        if canon == "bf16":
            payload = b"".join(
                struct.pack(
                    "<H",
                    (lambda bits: (bits + ((bits >> 16) & 1) + 0x7FFF) >> 16)(
                        struct.unpack("<I", struct.pack("<f", float(x)))[0]
                    ),
                )
                for x in flat
            )
        else:
            payload = _pack_floats([float(x) for x in flat], canon)
    else:
        payload = _pack_ints([int(x) for x in flat], canon)
    path.write_bytes(payload)
    return len(payload)


def read_tensor_bin(path: Path, dtype: str, shape: tuple[int, ...]) -> Any:
    """Read a tensor written by :func:`write_tensor_bin`."""
    np = _try_numpy()
    payload = path.read_bytes()
    canon = normalize_dtype(dtype)
    n = _prod(shape) if shape else (len(payload) // max(1, dtype_nbytes(canon)))
    if np is not None:
        dt = numpy_dtype_name(canon)
        if canon == "bf16":
            raw = np.frombuffer(payload, dtype="<u2").astype(np.uint32)
            arr = (raw << 16).view(np.float32)
        else:
            arr = np.frombuffer(payload, dtype=dt)
        if shape and arr.size == _prod(shape):
            return arr.reshape(shape)
        return arr
    return payload


def write_cases(
    testdir: Path,
    abi: KernelABI,
    plans: list[CasePlan],
    arrays_by_case: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write ``cases.jsonl`` plus per-case ``*.bin`` inputs under ``testdir``.

    Returns:
        Path to ``cases.jsonl``.
    """
    testdir.mkdir(parents=True, exist_ok=True)
    # A reused qN/test directory must never let a prior case satisfy a later
    # comparison.  Inputs and outputs are cheap to regenerate and live below
    # this attempt directory, so clear both trees before materializing.
    for stale in (testdir / "cases", testdir / "out"):
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
    (testdir / "out").mkdir(parents=True, exist_ok=True)
    jsonl = testdir / "cases.jsonl"
    materialize: Callable[[CasePlan, KernelABI], dict[str, Any]] = materialize_arrays
    with jsonl.open("w", encoding="utf-8") as handle:
        for plan in plans:
            arrays = (arrays_by_case or {}).get(plan.name)
            if arrays is None:
                arrays = materialize(plan, abi)
            case_dir = testdir / "cases" / plan.name
            files: dict[str, str] = {}
            (testdir / "out" / plan.name).mkdir(parents=True, exist_ok=True)
            for param in abi.input_params():
                array = arrays.get(param.name)
                if array is None:
                    continue
                rel = Path("cases") / plan.name / f"{param.name}.bin"
                write_tensor_bin(testdir / rel, array, param.dtype)
                files[param.name] = rel.as_posix()
            row = plan.to_dict()
            row["files"] = files
            row["outputs"] = list(abi.result_names())
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (testdir / "cases.hash").write_text(case_plan_hash(plans) + "\n", encoding="ascii")
    return jsonl


def load_case_plans(jsonl: Path) -> list[CasePlan]:
    """Read case plans from ``cases.jsonl``."""
    plans: list[CasePlan] = []
    if not jsonl.is_file():
        return plans
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            plans.append(CasePlan.from_dict(row))
    return plans
