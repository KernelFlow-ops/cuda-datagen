"""Numeric compare: dtype-banded tolerances, NaN/Inf rules, optional sort-flatten."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from cuda_sft.refval.cases import read_tensor_bin
from cuda_sft.refval.spec import (
    CasePlan,
    CaseResult,
    FLOAT_DTYPES,
    KernelABI,
    RefvalReport,
    normalize_dtype,
    tolerances_for,
)


def _try_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError:
        return None


def _flatten_for_compare(array: Any, *, sort_mode: str):
    np = _try_numpy()
    if np is None:
        raise RuntimeError("numpy is required for numeric compare")
    flat = np.ascontiguousarray(array).reshape(-1)
    if sort_mode == "sorted":
        return np.sort(flat, axis=None)
    if sort_mode == "set":
        return np.sort(np.unique(flat), axis=None)
    return flat


def compare_arrays(
    got: Any,
    expected: Any,
    *,
    dtype: str,
    tolerances: Mapping[str, Any] | None,
    allows_nan: bool,
    sort_mode: str,
    name: str,
    seed: int,
    max_mismatches: int = 8,
    expected_shape: tuple[int, ...] | None = None,
) -> CaseResult:
    """Compare one output tensor. ``sort_mode`` is ``elementwise``/``sorted``/``set``."""
    np = _try_numpy()
    if np is None:
        return CaseResult(name=name, ok=False, status="error", error="numpy is not installed", seed=seed)
    exp = np.asarray(expected)
    got_arr = np.asarray(got)
    got_shape = tuple(int(x) for x in got_arr.shape)
    exp_shape = tuple(int(x) for x in exp.shape)
    shape = got_shape
    contract_shape = tuple(int(x) for x in expected_shape) if expected_shape is not None else exp_shape
    if got_shape != contract_shape or exp_shape != contract_shape:
        return CaseResult(
            name=name, ok=False, status="fail",
            error=f"shape mismatch got={got_shape} expected={contract_shape} reference={exp_shape}",
            shape=got_shape, seed=seed,
        )
    canon = normalize_dtype(dtype)
    # bfloat16 is represented as float32 by numpy after decoding its native
    # 16-bit storage.  Other contract dtypes must match exactly; flattening
    # arrays with an accidental cast would otherwise hide ABI bugs.
    if canon == "bf16":
        dtype_ok = got_arr.dtype.kind == "f" and got_arr.dtype.itemsize == 4
    else:
        expected_np = {
            "f32": np.dtype("float32"), "f64": np.dtype("float64"),
            "f16": np.dtype("float16"), "i32": np.dtype("int32"),
            "i64": np.dtype("int64"), "i8": np.dtype("int8"),
            "u8": np.dtype("uint8"), "u32": np.dtype("uint32"),
            "bool": np.dtype("bool"),
        }.get(canon)
        dtype_ok = expected_np is None or got_arr.dtype == expected_np
    if not dtype_ok:
        return CaseResult(
            name=name, ok=False, status="fail",
            error=f"dtype mismatch got={got_arr.dtype} expected={canon}",
            shape=got_shape, seed=seed, n_mismatch=1,
            mismatches=[{"tensor": name, "dtype": str(got_arr.dtype), "expected_dtype": canon}],
        )
    mode = sort_mode if sort_mode in {"sorted", "set"} else "elementwise"
    try:
        g = _flatten_for_compare(got_arr, sort_mode=mode)
        e = _flatten_for_compare(exp, sort_mode=mode)
    except Exception as exc:
        return CaseResult(
            name=name, ok=False, status="error", error=f"reshape failed: {exc}", seed=seed, shape=shape
        )
    n = int(min(g.size, e.size))
    if g.size != e.size:
        return CaseResult(
            name=name,
            ok=False,
            status="fail",
            error=f"numel mismatch got={g.size} expected={e.size}",
            n_compared=n,
            shape=shape,
            seed=seed,
        )
    if n == 0:
        return CaseResult(name=name, ok=True, status="pass", n_compared=0, shape=shape, seed=seed)

    # Integer outputs (notably int64 indices/counts) must never pass through
    # float64: adjacent values above 2**53 otherwise become indistinguishable.
    if canon not in FLOAT_DTYPES:
        bad = g != e
        count = int(np.count_nonzero(bad))
        return CaseResult(
            name=name, ok=count == 0, status="pass" if count == 0 else "fail",
            n_compared=n, n_mismatch=count, shape=shape, seed=seed,
            error="" if count == 0 else f"{count} exact integer mismatches",
        )
    g = g[:n].astype(np.float64, copy=False)
    e = e[:n].astype(np.float64, copy=False)
    tol = tolerances_for(canon, tolerances)

    got_nan = np.isnan(g)
    exp_nan = np.isnan(e)
    got_inf = np.isinf(g)
    exp_inf = np.isinf(e)

    if not allows_nan and bool(got_nan.any()):
        return CaseResult(
            name=name,
            ok=False,
            status="fail",
            error="NaN in GPU output (question does not allow NaN)",
            n_compared=n,
            n_mismatch=int(got_nan.sum()),
            shape=shape,
            seed=seed,
        )

    # Inf is allowed only where the CPU reference also produced Inf of the same sign.
    inf_mismatch = (got_inf != exp_inf) | (got_inf & exp_inf & (np.signbit(g) != np.signbit(e)))
    if bool(inf_mismatch.any()):
        idx = int(np.argmax(inf_mismatch))
        return CaseResult(
            name=name,
            ok=False,
            status="fail",
            error=f"Inf mismatch at {idx}: got={g[idx]} expected={e[idx]}",
            n_compared=n,
            n_mismatch=int(inf_mismatch.sum()),
            shape=shape,
            seed=seed,
            mismatches=[{"index": idx, "got": float(g[idx]), "exp": float(e[idx])}],
        )

    if allows_nan:
        nan_mismatch = got_nan != exp_nan
        if bool(nan_mismatch.any()):
            idx = int(np.argmax(nan_mismatch))
            return CaseResult(
                name=name,
                ok=False,
                status="fail",
                error=f"NaN mismatch at {idx}",
                n_compared=n,
                n_mismatch=int(nan_mismatch.sum()),
                shape=shape,
                seed=seed,
            )
        finite = ~(got_nan | exp_nan | got_inf | exp_inf)
    else:
        finite = ~(got_inf | exp_inf)

    atol = float(tol["atol"])
    rtol = float(tol["rtol"])
    abs_err = np.abs(g - e)
    denom = np.maximum(np.abs(e), 1e-30)
    rel_err = abs_err / denom
    if finite.any():
        abs_err = np.where(finite, abs_err, 0.0)
        rel_err = np.where(finite, rel_err, 0.0)
        bad = finite & (abs_err > atol + rtol * np.abs(e))
    else:
        bad = np.zeros(n, dtype=bool)

    n_bad = int(bad.sum())
    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    max_rel = float(rel_err.max()) if rel_err.size else 0.0
    mismatches: list[dict[str, Any]] = []
    if n_bad:
        indices = np.nonzero(bad)[0][:max_mismatches]
        for idx in indices:
            i = int(idx)
            mismatches.append(
                {
                    "index": i,
                    "got": float(g[i]),
                    "exp": float(e[i]),
                    "abs": float(abs_err[i]),
                    "rel": float(rel_err[i]),
                }
            )
    return CaseResult(
        name=name,
        ok=n_bad == 0,
        status="pass" if n_bad == 0 else "fail",
        max_abs=max_abs,
        max_rel=max_rel,
        n_mismatch=n_bad,
        n_compared=n,
        shape=shape,
        seed=seed,
        mismatches=mismatches,
        error="" if n_bad == 0 else f"{n_bad} mismatches max_abs={max_abs:.4g} max_rel={max_rel:.4g}",
    )


def compare_outputs(
    got: Mapping[str, Any],
    expected: Mapping[str, Any],
    abi: KernelABI,
    plan: CasePlan,
    tolerances: Mapping[str, Any] | None = None,
) -> CaseResult:
    """Compare every ABI output for one case; merge into a single CaseResult."""
    # Elementwise comparison is the contract default.  Legacy ``sort_outputs``
    # is intentionally ignored unless the manifest explicitly opts into a
    # sorted/set mode, preventing accidental order-insensitive validation.
    sort_mode = abi.compare_mode if abi.compare_mode in {"sorted", "set"} else "elementwise"
    merged = CaseResult(name=plan.name, ok=True, status="pass", seed=plan.seed)
    names = list(abi.result_names()) or list(expected.keys())
    for name in names:
        if name not in expected:
            merged.ok = False
            merged.status = "fail"
            merged.error = f"missing reference output {name}"
            return merged
        if name not in got:
            merged.ok = False
            merged.status = "fail"
            merged.error = f"missing GPU output {name}"
            return merged
        param = next((p for p in abi.params if p.name == name), None)
        dtype = param.dtype if param is not None else (abi.returns if name == "__return__" else abi.dtype)
        one = compare_arrays(
            got[name],
            expected[name],
            dtype=dtype,
            tolerances=tolerances,
            allows_nan=abi.allows_nan,
            sort_mode=sort_mode,
            name=plan.name,
            seed=plan.seed,
            expected_shape=(tuple(plan.shapes.get(name)) if name in plan.shapes else None),
        )
        merged.n_compared += one.n_compared
        merged.n_mismatch += one.n_mismatch
        merged.max_abs = max(merged.max_abs, one.max_abs)
        merged.max_rel = max(merged.max_rel, one.max_rel)
        merged.mismatches.extend(
            [{**row, "tensor": name} for row in one.mismatches]
        )
        if one.shape:
            merged.shape = one.shape
        if not one.ok:
            merged.ok = False
            merged.status = "fail"
            merged.error = f"{name}: {one.error}"
            return merged
    return merged


def load_gpu_outputs(
    testdir: Path,
    abi: KernelABI,
    plan: CasePlan,
) -> dict[str, Any]:
    """Load ``out/{case}/{name}.bin`` tensors written by the harness."""
    out: dict[str, Any] = {}
    for name in abi.result_names():
        path = testdir / "out" / plan.name / f"{name}.bin"
        if not path.is_file():
            continue
        param = next((p for p in abi.params if p.name == name), None)
        dtype = param.dtype if param is not None else (abi.returns if name == "__return__" else abi.dtype)
        shape = tuple(plan.shapes.get(name) or ())
        if name == "__return__":
            shape = ()
        out[name] = read_tensor_bin(path, dtype, shape)
    return out


def dump_artifacts(
    testdir: Path,
    *,
    report: RefvalReport,
    manifest_text: str | None = None,
    log_text: str | None = None,
) -> None:
    """Write ``refval.log`` (and optional extra JSON) under ``testdir``."""
    testdir.mkdir(parents=True, exist_ok=True)
    if not report.cases_hash:
        hash_path = testdir / "cases.hash"
        if hash_path.is_file():
            try:
                report.cases_hash = hash_path.read_text(encoding="ascii").strip()
            except OSError:
                pass
    (testdir / "report.json").write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"status={report.status}",
        f"dialect={report.dialect}",
        f"cases_run={report.cases_run}",
        f"cases_hash={report.cases_hash or '-'}",
        f"failed_case={report.failed_case or '-'}",
        f"error_class={report.error_class or '-'}",
        f"elapsed_sec={report.elapsed_sec:.3f}",
        f"reason={report.reason or '-'}",
        "",
    ]
    for item in report.results:
        flag = "PASS" if item.ok else "FAIL"
        lines.append(
            f"  [{flag}] {item.name} n={item.n_compared} "
            f"mismatch={item.n_mismatch} max_abs={item.max_abs:.4g} "
            f"{item.error}".rstrip()
        )
    if log_text:
        lines.extend(["", "--- harness ---", log_text.rstrip(), ""])
    (testdir / "refval.log").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    if manifest_text:
        (testdir / "manifest.json").write_text(manifest_text, encoding="utf-8")


def classify_from_results(results: list[CaseResult], *, harness_error: str = "") -> str:
    """Pick an error_class from case results plus optional harness stderr."""
    text = (harness_error or "").lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(
        tok in text
        for tok in ("undeclared", "not declared", "no matching function", "too many arguments", "too few arguments")
    ):
        return "signature_mismatch"
    if any(tok in text for tok in ("error:", "cuda error", "segfault", "aborted", "illegal")):
        if "error:" in text and "nvcc" in text:
            return "signature_mismatch"
        return "crash"
    for item in results:
        err = (item.error or "").lower()
        if err.startswith("reference raised:"):
            return "reference_error"
        if "nan" in err or "inf" in err:
            return "nan_inf"
        if not item.ok:
            return "numeric_mismatch"
    return "numeric_mismatch"
