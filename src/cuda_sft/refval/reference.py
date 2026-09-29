"""Load and sanity-check a CPU reference callable.

The reference is LLM-generated Python. This module execs it in a restricted
namespace (numpy + math only), checks dtype/shape, determinism, a 2σ
input-sensitivity test, and rejects I/O or network.
"""

from __future__ import annotations

import ast
import inspect
import math
import multiprocessing as mp
import os
import pickle
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from cuda_sft.refval.spec import (
    COMPLEX_DTYPES,
    FLOAT_DTYPES,
    KernelABI,
    KernelParam,
    numpy_dtype_name,
    tolerances_for,
)

FORBIDDEN_CALLS = (
    "open",
    "exec",
    "eval",
    "__import__",
    "compile",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "help",
    "memoryview",
)
FORBIDDEN_MODULES = (
    "os",
    "sys",
    "socket",
    "subprocess",
    "shutil",
    "pathlib",
    "requests",
    "urllib",
    "http",
    "ctypes",
    "multiprocessing",
    "threading",
    "pickle",
    "importlib",
    "builtins",
    "io",
    "tempfile",
    "glob",
    "signal",
    "fcntl",
    "mmap",
)
FORBIDDEN_ATTR = re.compile(
    r"\b(os|sys|socket|subprocess|shutil|pathlib|requests|urllib|http|ctypes)\s*\.",
    re.IGNORECASE,
)


def _safe_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Allow NumPy's internal imports without exposing system modules to references."""
    if level or name.split(".", 1)[0] not in {"numpy", "math", "warnings"}:
        raise ImportError(f"reference import is not allowed: {name}")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}

# Child startup and NumPy import can exceed five seconds on a loaded GPU host.
REFERENCE_CALL_TIMEOUT_SEC = max(
    0.1, float(os.environ.get("REFVAL_REFERENCE_TIMEOUT_SEC", "20") or "20")
)
REFERENCE_VALIDATION_TIMEOUT_SEC = max(
    REFERENCE_CALL_TIMEOUT_SEC,
    float(os.environ.get("REFVAL_REFERENCE_VALIDATION_TIMEOUT_SEC", "30") or "30"),
)


@dataclass(frozen=True)
class _ReferenceProgram:
    """Deferred reference source; compilation and execution happen in a child."""

    source: str
    fn_name: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Direct callers still receive the same callable API and hard timeout.
        return _run_isolated(
            _call_program_worker,
            (self.source, self.fn_name, args, kwargs),
            timeout_sec=REFERENCE_CALL_TIMEOUT_SEC,
        )


def _try_numpy():
    try:
        import numpy as np  # type: ignore

        return np
    except ImportError:
        return None


def strip_reference_fence(source: str) -> str:
    """Drop markdown fences around a reference body."""
    text = (source or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def check_forbidden(source: str) -> str | None:
    """Return a reason if the source uses I/O, imports, or dynamic exec."""
    text = strip_reference_fence(source)
    if not text.strip():
        return "empty reference source"
    if FORBIDDEN_ATTR.search(text):
        return "reference uses a forbidden module attribute"
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return f"reference syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            else:
                names = [(node.module or "").split(".")[0]]
            for name in names:
                if name in FORBIDDEN_MODULES:
                    return f"reference imports forbidden module {name!r}"
                if name not in {"math", "numpy", "np", "operator", "functools", "itertools", "typing", "__future__"}:
                    # numpy is allowed; other third-party (torch) is allowed only if already mentioned.
                    if name in {"torch", "tl", "triton"}:
                        continue
                    if name and name not in {"math", "numpy", "np"}:
                        return f"reference imports {name!r} (only math/numpy/torch allowed)"
        # Flag calls only. A tensor parameter named ``input`` is a normal ABI
        # name; treating every Name as a builtin call rejected valid references.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            return f"reference uses forbidden name {node.func.id!r}"
        if isinstance(node, ast.Attribute) and node.attr in {"system", "popen", "walk", "remove", "unlink"}:
            return f"reference uses forbidden attribute {node.attr!r}"
    return None


def load_reference_fn(
    source: str,
    fn_name: str = "reference",
) -> Callable[..., Any]:
    """Exec ``source`` and return the callable named ``fn_name``.

    Raises:
        ValueError: Forbidden constructs, missing numpy, or missing callable.
    """
    reason = check_forbidden(source)
    if reason:
        raise ValueError(reason)
    text = strip_reference_fence(source)
    # Compile catches syntax errors without executing model-authored top-level
    # statements.  Actual exec is deferred into the isolated worker.
    try:
        compile(text, "<reference>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"reference syntax error: {exc}") from exc
    return _ReferenceProgram(text, fn_name)


def _load_reference_fn_inline(source: str, fn_name: str) -> Callable[..., Any]:
    """Load a reference inside an already-isolated worker process."""
    text = strip_reference_fence(source)
    # Prevalidated import lines are dropped; safe modules are bound below.
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    np = _try_numpy()
    if np is None:
        raise ValueError("numpy is not installed")
    ns: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "math": math,
        "np": np,
        "numpy": np,
    }
    if any(isinstance(node, ast.Name) and node.id == "torch" for node in ast.walk(ast.parse(text))):
        try:
            import torch  # type: ignore

            ns["torch"] = torch
        except ImportError:
            pass
    try:
        exec(compile(text, "<reference>", "exec"), ns, ns)  # noqa: S102 — sandboxed
    except Exception as exc:
        raise ValueError(f"reference exec failed: {exc}") from exc
    fn = ns.get(fn_name) or ns.get("reference") or ns.get("cpu_reference")
    if not callable(fn):
        callables = [k for k, v in ns.items() if callable(v) and not k.startswith("_")]
        if len(callables) == 1:
            fn = ns[callables[0]]
        else:
            raise ValueError(
                f"reference callable {fn_name!r} not found (have {callables[:8]})"
            )
    return fn


def _call_program_worker(
    source: str,
    fn_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    fn = _load_reference_fn_inline(source, fn_name)
    return fn(*args, **kwargs)


def _isolated_entry(conn: Any, target: Callable[..., Any], args: tuple[Any, ...]) -> None:
    """Child entrypoint that serializes either a value or a bounded error."""
    try:
        np = _try_numpy()
        if np is None:
            result = target(*args)
        else:
            with np.errstate(all="ignore"):
                result = target(*args)
        conn.send(("ok", result))
    except BaseException as exc:  # child must report model-code failures
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def _isolation_env() -> dict[str, str]:
    """Child env that can import this package even when PYTHONPATH was unset."""
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[2])
    previous = env.get("PYTHONPATH", "")
    parts = [part for part in previous.split(os.pathsep) if part]
    if src not in parts:
        env["PYTHONPATH"] = os.pathsep.join([src, *parts]) if parts else src
    return env


def _kill_subprocess_group(proc: subprocess.Popen[bytes]) -> None:
    """Kill a ``start_new_session`` child and anything it spawned."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        proc.kill()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        proc.kill()


def _run_isolated_subprocess(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    timeout_sec: float,
) -> Any:
    """Run isolation in a fresh interpreter.

    Pool workers are daemonic, and ``multiprocessing.Process.start`` refuses
    to create grandchildren. A subprocess is allowed from a daemon and does
    not fork the worker's LLM thread pool.
    """
    deadline = max(0.1, float(timeout_sec))
    payload = pickle.dumps((target, args), protocol=pickle.HIGHEST_PROTOCOL)
    out_fd, out_path = tempfile.mkstemp(prefix="refval-ref-", suffix=".pkl")
    os.close(out_fd)
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "cuda_sft.refval.reference", out_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=_isolation_env(),
            start_new_session=True,
        )
        try:
            _stdout, stderr = proc.communicate(payload, timeout=deadline)
        except subprocess.TimeoutExpired:
            _kill_subprocess_group(proc)
            raise TimeoutError(f"reference execution timed out after {timeout_sec:.2f}s") from None
        if not Path(out_path).is_file() or Path(out_path).stat().st_size == 0:
            err = (stderr or b"").decode("utf-8", "replace")[:500]
            raise RuntimeError(
                f"reference worker exited without a result (exitcode={proc.returncode}) {err}".strip()
            )
        try:
            status, result = pickle.loads(Path(out_path).read_bytes())
        except Exception as exc:
            err = (stderr or b"").decode("utf-8", "replace")[:500]
            raise RuntimeError(f"reference worker returned unreadable result: {exc}; {err}") from exc
        if status != "ok":
            raise RuntimeError(str(result))
        return result
    finally:
        if proc is not None and proc.poll() is None:
            _kill_subprocess_group(proc)
        try:
            os.unlink(out_path)
        except OSError:
            pass


def _run_isolated(
    target: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    timeout_sec: float,
) -> Any:
    """Run one reference operation in a killable process with a hard deadline."""
    # Spawn-pool workers are daemons. Fork isolation below cannot start there.
    if mp.current_process().daemon:
        return _run_isolated_subprocess(target, args, timeout_sec=timeout_sec)
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("reference isolation requires multiprocessing 'fork'")
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_isolated_entry, args=(child_conn, target, args), daemon=True)
    proc.start()
    child_conn.close()
    try:
        if not parent_conn.poll(max(0.1, float(timeout_sec))):
            proc.terminate()
            proc.join(timeout=0.25)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.25)
            raise TimeoutError(f"reference execution timed out after {timeout_sec:.2f}s")
        try:
            status, payload = parent_conn.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"reference worker exited without a result (exitcode={proc.exitcode})"
            ) from exc
        if status != "ok":
            raise RuntimeError(str(payload))
        return payload
    finally:
        parent_conn.close()
        if proc.is_alive():
            proc.join(timeout=0.25)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.25)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=0.25)


def _as_numpy(value: Any):
    np = _try_numpy()
    if np is None:
        return value
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def bind_and_call(
    fn: Callable[..., Any],
    abi: KernelABI,
    tensors: Mapping[str, Any],
    scalars: Mapping[str, Any],
    *,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    """Call ``fn`` in an isolated process and normalize its result to a dict."""
    return _run_isolated(
        _bind_and_call_inline,
        (fn, abi, dict(tensors), dict(scalars)),
        timeout_sec=REFERENCE_CALL_TIMEOUT_SEC if timeout_sec is None else timeout_sec,
    )


def _bind_and_call_inline(
    fn: Callable[..., Any],
    abi: KernelABI,
    tensors: Mapping[str, Any],
    scalars: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind/call implementation used only inside an isolated child process."""
    if isinstance(fn, _ReferenceProgram):
        fn = _load_reference_fn_inline(fn.source, fn.fn_name)
    sig = None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        sig = None
    kwargs: dict[str, Any] = {}
    args: list[Any] = []
    if sig is None or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        for param in abi.params:
            if param.is_tensor and param.kind != "output":
                if param.name in tensors:
                    kwargs[param.name] = tensors[param.name]
            elif param.kind in {"scalar", "size"}:
                if param.name in scalars:
                    kwargs[param.name] = scalars[param.name]
        kwargs.setdefault("inputs", dict(tensors))
        kwargs.setdefault("scalars", dict(scalars))
        try:
            result = fn(**{k: v for k, v in kwargs.items() if k in sig.parameters}) if sig else fn(**kwargs)
        except TypeError:
            result = fn(dict(tensors), dict(scalars))
    else:
        abi_by_name = {param.name: param for param in abi.params}
        extent = _fallback_extent(scalars)
        for name, parameter in sig.parameters.items():
            if name in tensors:
                kwargs[name] = tensors[name]
            elif name in scalars:
                kwargs[name] = scalars[name]
            elif name in {"inputs", "tensors"}:
                kwargs[name] = dict(tensors)
            elif name == "scalars":
                kwargs[name] = dict(scalars)
            elif parameter.default is not inspect.Parameter.empty:
                continue
            elif parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue
            else:
                # Host ABI includes the output buffer. Dropping it and then
                # calling positionally shifts every later argument.
                abi_param = abi_by_name.get(name)
                if abi_param is not None and abi_param.is_tensor:
                    kwargs[name] = _zero_like_param(abi_param, extent)
                continue
        try:
            result = fn(**kwargs)
        except TypeError:
            for param in abi.params:
                if param.is_tensor:
                    if param.name in tensors:
                        args.append(tensors[param.name])
                    else:
                        args.append(_zero_like_param(param, extent))
                else:
                    args.append(scalars.get(param.name))
            result = fn(*args)
    return _normalize_result(result, abi)


def _normalize_result(result: Any, abi: KernelABI) -> dict[str, Any]:
    names = list(abi.result_names())
    if isinstance(result, Mapping):
        out = {str(k): _as_numpy(v) for k, v in result.items()}
        if names and not any(n in out for n in names) and len(out) == 1 and len(names) == 1:
            out = {names[0]: next(iter(out.values()))}
        return out
    if isinstance(result, (tuple, list)):
        arrays = [_as_numpy(item) for item in result]
        if not names:
            return {f"out{i}": arr for i, arr in enumerate(arrays)}
        if len(arrays) == 1 and len(names) == 1:
            return {names[0]: arrays[0]}
        return {names[i]: arrays[i] for i in range(min(len(names), len(arrays)))}
    if names:
        return {names[0]: _as_numpy(result)}
    return {"out": _as_numpy(result)}


def _fallback_extent(scalars: Mapping[str, Any]) -> int:
    """Pick a positive size already chosen for this call."""
    for value in scalars.values():
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return 8


def _zero_like_param(param: KernelParam, n: int):
    np = _try_numpy()
    shape = (n,) if param.rank <= 1 else tuple([n] * max(1, param.rank))
    dt = numpy_dtype_name(param.dtype)
    if np is None:
        return [0] * (n if param.rank <= 1 else n ** max(1, param.rank))
    return np.zeros(shape, dtype=dt)


def _random_like_param(
    param: KernelParam, n: int, rng: random.Random, *, compute_dtype: str = ""
):
    np = _try_numpy()
    shape = (n,) if param.rank <= 1 else tuple([max(1, n)] * max(1, param.rank))
    dt = numpy_dtype_name(param.dtype)
    if np is None:
        return [rng.random() for _ in range(_prod(shape))]
    if param.dtype == "u16" and compute_dtype == "f16":
        values = [rng.random() * 4.0 - 2.0 for _ in range(_prod(shape))]
        return np.asarray(values, dtype=np.float16).reshape(shape).view(np.uint16)
    if param.dtype in COMPLEX_DTYPES:
        values = [complex(rng.random() * 2.0 - 1.0, rng.random() * 2.0 - 1.0) for _ in range(_prod(shape))]
    elif param.dtype in FLOAT_DTYPES:
        values = [rng.random() * 2.0 - 1.0 for _ in range(_prod(shape))]
    elif param.dtype == "bool":
        values = [bool(rng.getrandbits(1)) for _ in range(_prod(shape))]
    elif param.dtype.startswith("u"):
        values = [rng.randrange(0, 11) for _ in range(_prod(shape))]
    else:
        values = [rng.randrange(-10, 11) for _ in range(_prod(shape))]
    return np.asarray(values, dtype=dt).reshape(shape)


def _prod(shape: tuple[int, ...]) -> int:
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def validate_reference_fn(
    fn: Callable[..., Any],
    abi: KernelABI,
    *,
    seed: int,
) -> str | None:
    """Return a reason if the reference is unusable; None if it looks sound.

    Checks: callable on a tiny case, dtype/shape, determinism, 2σ sensitivity
    (two different inputs must not collapse to the same output within 2σ of
    the dtype tolerance — catches constant/zeros references).
    """
    np = _try_numpy()
    if np is None:
        return "numpy is not installed"
    n = 8
    sizes = {p.name: n for p in abi.size_params()}
    if not sizes:
        sizes = {"n": n}
    scalars: dict[str, Any] = dict(sizes)
    for param in abi.scalar_params():
        if param.dtype in {"cublas_handle", "cufft_handle"}:
            scalars[param.name] = None
        elif param.dtype in FLOAT_DTYPES:
            scalars[param.name] = 1.25
        elif param.dtype == "bool":
            scalars[param.name] = True
        else:
            scalars[param.name] = 2

    def _inputs(tag: int) -> dict[str, Any]:
        rng = random.Random(int(seed) + tag)
        tensors: dict[str, Any] = {}
        for param in abi.input_params():
            tensors[param.name] = _random_like_param(param, n, rng, compute_dtype=abi.dtype)
        return tensors

    validation_deadline = time.monotonic() + REFERENCE_VALIDATION_TIMEOUT_SEC

    def _checked_call(inputs: Mapping[str, Any]) -> dict[str, Any]:
        remaining = validation_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"reference validation timed out after {REFERENCE_VALIDATION_TIMEOUT_SEC:.2f}s"
            )
        return bind_and_call(
            fn,
            abi,
            inputs,
            scalars,
            timeout_sec=min(REFERENCE_CALL_TIMEOUT_SEC, remaining),
        )

    try:
        a = _inputs(1)
        out1 = _checked_call(a)
        out1b = _checked_call(a)
    except Exception as exc:
        return f"reference raised: {exc}"

    names = list(abi.result_names())
    if names and not any(name in out1 for name in names):
        return f"reference did not return expected keys {names}; got {list(out1)}"

    for name, arr in out1.items():
        other = out1b.get(name)
        if other is None:
            return f"reference is not deterministic (missing {name} on second call)"
        if np.asarray(arr).shape != np.asarray(other).shape:
            return f"reference shape changed across calls for {name}"
        if not np.allclose(
            np.asarray(arr),
            np.asarray(other),
            equal_nan=True,
            atol=0.0,
            rtol=0.0,
        ):
            # Integer exact; floats must still be bit-stable for a CPU ref.
            if not np.allclose(
                np.asarray(arr),
                np.asarray(other),
                equal_nan=True,
                atol=1e-12,
                rtol=1e-12,
            ):
                return f"reference is not deterministic on {name}"

    try:
        b = _inputs(2)
        out2 = _checked_call(b)
    except Exception as exc:
        return f"reference raised on second input: {exc}"

    # 2σ distinguishability: skip when all inputs are empty or outputs are
    # legitimately independent of the varied tensors (rare). Require at least
    # one output to move more than 2σ of the dtype noise if inputs differ.
    moved = False
    compared = False
    for name in out1:
        if name not in out2:
            continue
        param = next((p for p in abi.params if p.name == name), None)
        first = np.asarray(out1[name])
        second = np.asarray(out2[name])
        half_bits = param is not None and param.dtype == "u16" and abi.dtype == "f16"
        if half_bits and first.dtype == np.uint16 and second.dtype == np.uint16:
            first = first.view(np.float16)
            second = second.view(np.float16)
        x = np.asarray(first).reshape(-1)
        y = np.asarray(second).reshape(-1)
        ncmp = min(x.size, y.size)
        if ncmp == 0:
            continue
        # Scalar reductions can legitimately stay put (e.g. count-if with a
        # high threshold). Only require 2σ movement on vector outputs.
        if ncmp < 4:
            continue
        compared = True
        dtype = "f16" if half_bits else (param.dtype if param is not None else abi.dtype)
        tol = tolerances_for(dtype)
        sigma = float(tol["atol"]) + float(tol["rtol"]) * max(
            float(np.nanmax(np.abs(x[:ncmp]))) if ncmp else 0.0,
            1.0,
        )
        delta = np.nanmax(np.abs(x[:ncmp] - y[:ncmp])) if ncmp else 0.0
        if delta > 2.0 * max(sigma, 1e-12):
            moved = True
            break
    if compared and not moved:
        return "reference output does not change with inputs (2σ distinguishability)"
    return None


def _subprocess_isolation_main() -> None:
    """Entry point for ``python -m cuda_sft.refval.reference <result-path>``."""
    out_path = Path(sys.argv[1])
    target, args = pickle.load(sys.stdin.buffer)
    try:
        payload = ("ok", target(*args))
    except BaseException as exc:  # child must report model-code failures
        payload = ("error", f"{type(exc).__name__}: {exc}")
    out_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    _subprocess_isolation_main()
