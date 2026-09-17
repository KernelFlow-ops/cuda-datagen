"""Subprocess compile gates for Triton / TileLang Python kernels."""

from __future__ import annotations

import ast
import importlib.util
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
    """Parse + import a Triton file; require at least one ``@triton.jit`` function."""
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
    """Parse + import a TileLang file; require a prim_func or jit kernel."""
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    try:
        import tilelang  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("tilelang package is not installed") from exc

    module = _load_module(path)
    hints = []
    for name, value in vars(module).items():
        type_name = type(value).__name__
        if "PrimFunc" in type_name or "JITKernel" in type_name or "TileLang" in type_name:
            hints.append(name)
        if callable(value) and getattr(value, "__name__", "") == "main":
            hints.append(name)
    text = source
    if "T.prim_func" not in text and "@T.prim_func" not in text and "tilelang.jit" not in text:
        if not hints:
            raise RuntimeError("no TileLang prim_func / jit kernel found")
    return f"tilelang ok symbols={','.join(hints) or 'source-ok'}"


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m cuda_sft.dialects.python_gate triton|tilelang PATH``."""
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
