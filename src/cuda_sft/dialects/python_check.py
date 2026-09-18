"""Run Triton/TileLang compile gates in a subprocess."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cuda_sft.compile import CompileResult
from cuda_sft.config import PROJECT_ROOT


def _gate_env() -> dict[str, str]:
    """Ensure the gate subprocess can import ``cuda_sft`` via ``PYTHONPATH=src``."""
    env = os.environ.copy()
    src = str(PROJECT_ROOT / "src")
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if src not in parts:
        env["PYTHONPATH"] = os.pathsep.join([src, *parts])
    return env


def run_python_gate(
    kind: str,
    source: str,
    workdir: Path,
    *,
    filename: str,
    timeout_sec: int,
    dialect: str,
) -> CompileResult:
    """Write ``filename`` and run ``python -m cuda_sft.dialects.python_gate``.

    Args:
        kind: ``triton`` or ``tilelang``.
        source: Full ``solution.py`` text.
        workdir: Per-attempt directory.
        filename: Usually ``solution.py``.
        timeout_sec: Subprocess timeout (import/JIT can be slow).
        dialect: Recorded on :class:`CompileResult`.

    Returns:
        Compile-only result; ``ok`` means parse+import succeeded.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    source_path = workdir / filename
    source_path.write_text(source or "", encoding="utf-8")
    cmd = [sys.executable, "-m", "cuda_sft.dialects.python_gate", kind, str(source_path)]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=max(5, int(timeout_sec)),
            check=False,
            env=_gate_env(),
        )
    except subprocess.TimeoutExpired as exc:
        tail = ""
        if exc.stderr:
            tail = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode(
                "utf-8", "replace"
            )
        output = f"{kind} timed out after {timeout_sec}s\n{tail}".strip()
        return CompileResult(
            False, cmd, output, used_rdc=False, source_path=source_path, dialect=dialect
        )
    except OSError as exc:
        return CompileResult(
            False,
            cmd,
            f"failed to launch python gate: {exc}",
            used_rdc=False,
            source_path=source_path,
            dialect=dialect,
        )
    merged = "\n".join(
        part for part in ((proc.stdout or "").strip(), (proc.stderr or "").strip()) if part
    )
    ok = proc.returncode == 0
    return CompileResult(
        ok, cmd, merged, used_rdc=False, source_path=source_path, dialect=dialect
    )
