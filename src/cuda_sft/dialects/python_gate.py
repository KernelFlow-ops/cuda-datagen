"""Subprocess compile gates for Triton / TileLang Python kernels."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import sys
from pathlib import Path


def _load_module(path: Path):
    """Import ``path`` as an isolated module (``__name__`` is not ``__main__``)."""
    spec = importlib.util.spec_from_file_location("_cuda_sft_kernel", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def check_triton(path: Path) -> str:
    """Parse + import a Triton file; require at least one ``@triton.jit`` function.

    Args:
        path: ``solution.py`` written by the compile node.

    Returns:
        One-line success summary.

    Raises:
        RuntimeError: Missing package, syntax error, or no jit kernel.
    """
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    try:
        import triton
        from triton.runtime.jit import JITFunction
    except ImportError as exc:
        raise RuntimeError("triton package is not installed") from exc

    module = _load_module(path)
    jit_fns = [
        name
        for name, value in vars(module).items()
        if isinstance(value, JITFunction) or type(value).__name__ == "JITFunction"
    ]
    if not jit_fns:
        raise RuntimeError("no @triton.jit kernel found after import")
    return f"triton ok jit={','.join(jit_fns)} version={getattr(triton, '__version__', '?')}"


def check_tilelang(path: Path) -> str:
    """Parse + import a TileLang file; require a prim_func or jit kernel.

    Args:
        path: ``solution.py`` written by the compile node.

    Raises:
        RuntimeError: Missing package or no TileLang kernel marker.
    """
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError(f"unavailable: tilelang dependency import failed: {exc}") from exc

    # Importing TileLang is not a compile gate.  Lower at least one real
    # PrimFunc/JIT entry for CUDA so syntax-only candidates cannot pass.
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(f"unavailable: torch is required for TileLang CUDA: {exc}") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("unavailable: torch.cuda.is_available() is false")

    module = _load_module(path)
    hints = []
    primfuncs = []
    callables = []
    for name, value in vars(module).items():
        type_name = type(value).__name__
        if "PrimFunc" in type_name or "JITKernel" in type_name or "TileLang" in type_name:
            hints.append(name)
        if "PrimFunc" in type_name:
            primfuncs.append((name, value))
        if callable(value) and getattr(value, "__name__", "") == "main":
            hints.append(name)
        if callable(value) and not name.startswith("_") and getattr(value, "__module__", "") == module.__name__:
            callables.append((name, value))
    text = source
    if "T.prim_func" not in text and "@T.prim_func" not in text and "tilelang.jit" not in text:
        if not hints:
            raise RuntimeError("no TileLang prim_func / jit kernel found")

    lowered = []
    last = "no candidate PrimFunc/JIT callable found"
    # Prefer explicit host names, then a PrimFunc, then the first local
    # callable.  Factories are invoked with a small canonical N and lowered.
    callables.sort(key=lambda item: (0 if item[0] in {"main", "kernel", "launch"} else 1, item[0]))
    for name, value in primfuncs + callables:
        try:
            kernel, style = _tilelang_lower(value, tilelang, T)
            lowered.append(f"{name}:{style}:{type(kernel).__name__}")
            # A launch is intentionally opt-in for the compile gate.  The
            # refval Python driver always launches and synchronizes CUDA.
            if os.environ.get("CUDA_SFT_TILELANG_LAUNCH", "0") == "1":
                _tilelang_launch(kernel, torch)
            break
        except Exception as exc:
            last = f"{name}: {type(exc).__name__}: {exc}"
    else:
        raise RuntimeError(f"CUDA lowering failed; unavailable or invalid TileLang entry ({last})")
    return (
        f"tilelang ok version={getattr(tilelang, '__version__', '?')} "
        f"symbols={','.join(hints) or 'source-ok'} lowering=cuda "
        f"entry={lowered[0]}"
    )


def _tilelang_lower(value, tilelang, T):
    """Lower a PrimFunc, lazy factory, or eager JIT callable for CUDA."""
    type_name = type(value).__name__
    if "JITKernel" in type_name:
        return value, "compiled"
    if "PrimFunc" in type_name:
        return tilelang.compile(value, target="cuda"), "primfunc"
    if not callable(value):
        raise TypeError(f"unsupported TileLang symbol {type_name}")
    try:
        sig = inspect.signature(value)
        params = list(sig.parameters.values())
    except (TypeError, ValueError):
        params = []
    # A generated factory conventionally accepts N (and optional tile sizes)
    # and returns a T.prim_func.  Try keyword arguments first, then positional.
    kwargs = {}
    args = []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.default is not p.empty:
            continue
        lname = p.name.lower()
        if lname in {"n", "m", "length", "size", "dim"} or p.annotation in (int, "int"):
            kwargs[p.name] = 128
        else:
            # Eager JIT entries need tensors and are handled below.
            kwargs = {}
            break
    if kwargs or (params and all(p.default is not p.empty for p in params)):
        try:
            built = value(**kwargs)
        except TypeError:
            args = [128 for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty]
            built = value(*args)
        if "PrimFunc" in type(built).__name__:
            return tilelang.compile(built, target="cuda"), "primfunc_factory"
        if "JITKernel" in type(built).__name__:
            return built, "lazy"
    # Eager JIT callables are lowered by invoking with CUDA tensors inferred
    # from the signature; this also catches import-only decorators.
    if getattr(value, "__class__", type(value)).__name__ in {"JITImpl", "JITFunction"}:
        import torch
        tensors = [torch.zeros((128,), device="cuda", dtype=torch.float32) for _ in params if _ .name.lower() not in {"n", "m", "size", "length"}]
        vals = []
        for p in params:
            vals.append(128 if p.name.lower() in {"n", "m", "size", "length"} else (tensors.pop(0) if tensors else torch.zeros((128,), device="cuda")))
        result = value(*vals)
        if "JITKernel" in type(result).__name__:
            return result, "lazy"
        return value, "eager"
    raise TypeError("callable did not produce PrimFunc/JITKernel")


def _tilelang_launch(kernel, torch):
    """Launch a lowered kernel with CUDA tensors and wait for completion."""
    import inspect
    params = getattr(kernel, "params", None)
    n = len(params) if params is not None else 3
    if not n:
        n = 3
    tensors = [torch.zeros((128,), device="cuda", dtype=torch.float32) for _ in range(max(1, n))]
    kernel(*tensors)
    torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m cuda_sft.dialects.python_gate triton|tilelang PATH``.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: python -m cuda_sft.dialects.python_gate triton|tilelang PATH", file=sys.stderr)
        return 2
    kind, raw_path = args[0].strip().lower(), Path(args[1])
    try:
        if kind == "triton":
            print(check_triton(raw_path))
        elif kind == "tilelang":
            print(check_tilelang(raw_path))
        else:
            print(f"unknown kind {kind}", file=sys.stderr)
            return 2
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
