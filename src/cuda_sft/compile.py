"""Compile generated CUDA with ``nvcc -c`` (stubs + optional ``-rdc=true``)."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cuda_sft.config import Settings, get_settings

HELPERS_STUB = """#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \\
  do {                                                                         \\
    cudaError_t err__ = (call);                                                \\
    if (err__ != cudaSuccess) {                                                \\
      fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__,           \\
              cudaGetErrorString(err__));                                      \\
    }                                                                          \\
  } while (0)
#endif

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) CUDA_CHECK(call)
#endif

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(call) CUDA_CHECK(call)
#endif

#ifndef CUDA_ERROR_CHECK
#define CUDA_ERROR_CHECK(call) CUDA_CHECK(call)
#endif
"""

SOLUTION_HEADER_STUB = """#pragma once
// Intentionally minimal: generated kernels should define host functions
// in solution.cu. This stub only exists so leftover includes still compile.
"""

RDC_HINTS = (
    "relocatable device code",
    "-rdc=true",
    "rdc=true",
    "dynamic parallelism",
    "cudalaunchdevice",
    "__global__ function from a __device__",
    "calling a __global__ function from device",
    "cannot call a __global__ function from device",
    "kernel launch from device",
    "kernel launch from __device__",
    "device-side kernel launch",
    "separate compilation mode",
    "cdp",
    "cudaDeviceSynchronize from device",
)

SMOKE_SOURCE = r"""
#include <cuda_runtime.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void launch_add(const float* a, const float* b, float* c, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  add_kernel<<<blocks, threads>>>(a, b, c, n);
}
"""


@dataclass
class CompileResult:
    """Outcome of one ``nvcc -c`` invocation.

    Attributes:
        ok: True if the compiler returned 0.
        command: Exact nvcc argv used for the last attempt.
        output: Combined stdout/stderr.
        used_rdc: True if the successful (or last) compile used ``-rdc=true``.
        source_path: Path of the written ``solution.cu``, if any.
    """

    ok: bool
    command: list[str]
    output: str
    used_rdc: bool
    source_path: Path | None = None


def ensure_stubs(workdir: Path) -> None:
    """Write permissive ``helpers.h`` / ``solution_header.h`` stubs into ``workdir``.

    Args:
        workdir: Per-attempt directory that will hold ``solution.cu``.
    """
    include_dir = workdir / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    (include_dir / "helpers.h").write_text(HELPERS_STUB, encoding="utf-8")
    (include_dir / "solution_header.h").write_text(SOLUTION_HEADER_STUB, encoding="utf-8")
    (workdir / "helpers.h").write_text(HELPERS_STUB, encoding="utf-8")
    (workdir / "solution_header.h").write_text(SOLUTION_HEADER_STUB, encoding="utf-8")


def looks_like_rdc_error(output: str) -> bool:
    """Return True if nvcc output suggests relocatable device code is required.

    Args:
        output: Compiler stdout/stderr.
    """
    lowered = output.lower()
    return any(hint in lowered for hint in RDC_HINTS)


def _nvcc_command(
    settings: Settings,
    source: Path,
    output: Path,
    *,
    rdc: bool,
) -> list[str]:
    """Build an ``nvcc -c`` command line.

    Args:
        settings: Runtime CUDA paths and arch.
        source: Path to ``solution.cu``.
        output: Path to the ``.o`` file.
        rdc: If True, add ``-rdc=true`` for dynamic parallelism.

    Returns:
        Argument list suitable for ``subprocess.run``.
    """
    workdir = source.parent
    cmd = [
        settings.nvcc_bin,
        "-c",
        str(source),
        "-o",
        str(output),
        "-std=c++17",
        f"-arch={settings.resolved_cuda_arch}",
        "--expt-relaxed-constexpr",
        "--extended-lambda",
        f"-I{settings.resolved_cuda_home}/include",
        f"-I{workdir}",
        f"-I{workdir / 'include'}",
    ]
    if rdc:
        cmd.append("-rdc=true")
    return cmd


def _run_nvcc(cmd: list[str], timeout: int) -> tuple[int, str]:
    """Run nvcc and return ``(exit_code, combined_output)``.

    Args:
        cmd: nvcc argv.
        timeout: Kill the process after this many seconds.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        tail = ""
        if exc.stderr:
            tail = exc.stderr if isinstance(exc.stderr, str) else exc.stderr.decode("utf-8", "replace")
        return 124, f"nvcc timed out after {timeout}s\n{tail}".strip()
    except OSError as exc:
        return 127, f"failed to launch nvcc: {exc}"

    merged = "\n".join(
        part for part in ((proc.stdout or "").strip(), (proc.stderr or "").strip()) if part
    )
    return proc.returncode, merged


def compile_cuda_source(
    source: str,
    workdir: Path,
    settings: Settings | None = None,
) -> CompileResult:
    """Write ``source`` to ``workdir/solution.cu`` and compile with ``nvcc -c``.

    On dynamic-parallelism errors, retries once with ``-rdc=true``.

    Args:
        source: Full CUDA translation unit.
        workdir: Scratch directory for this attempt.
        settings: CUDA toolchain config; defaults to :func:`get_settings`.

    Returns:
        Compile result (success or last failure).
    """
    settings = settings or get_settings()
    workdir.mkdir(parents=True, exist_ok=True)
    ensure_stubs(workdir)

    source_path = workdir / "solution.cu"
    object_path = workdir / "solution.o"
    source_path.write_text(source, encoding="utf-8")
    if object_path.exists():
        object_path.unlink()

    cmd = _nvcc_command(settings, source_path, object_path, rdc=False)
    code, output = _run_nvcc(cmd, settings.nvcc_timeout_sec)
    if code == 0:
        return CompileResult(True, cmd, output, used_rdc=False, source_path=source_path)

    if looks_like_rdc_error(output):
        rdc_cmd = _nvcc_command(settings, source_path, object_path, rdc=True)
        rdc_code, rdc_output = _run_nvcc(rdc_cmd, settings.nvcc_timeout_sec)
        if rdc_code == 0:
            return CompileResult(
                True, rdc_cmd, rdc_output, used_rdc=True, source_path=source_path
            )
        combined = f"{output}\n\n[retry with -rdc=true]\n{rdc_output}".strip()
        return CompileResult(
            False, rdc_cmd, combined, used_rdc=True, source_path=source_path
        )

    return CompileResult(False, cmd, output, used_rdc=False, source_path=source_path)


def attempt_workdir(
    settings: Settings,
    question_id: int,
    candidate_idx: int,
    repair_idx: int,
) -> Path:
    """Return the directory used to compile one attempt.

    ``WORK_KEEP=simple`` overwrites ``work/q{id}/`` in place.
    ``WORK_KEEP=detailed`` uses ``work/q{id}/c{c}/r{r}/``.
    """
    base = settings.work_path / f"q{question_id}"
    if settings.work_keep == "simple":
        return base
    return base / f"c{candidate_idx}" / f"r{repair_idx}"


def finalize_question_work(
    settings: Settings,
    question_id: int,
    *,
    code: str,
    success: bool,
) -> None:
    """In simple mode, keep only the last ``solution.cu`` (and ``nvcc.log`` if failed).

    Args:
        settings: Pipeline settings (``work_keep``).
        question_id: Question id.
        code: Last CUDA source for this question.
        success: True if compile passed (drop nvcc.log); False keeps the last log.
    """
    if settings.work_keep != "simple":
        return
    qdir = settings.work_path / f"q{question_id}"
    qdir.mkdir(parents=True, exist_ok=True)
    if (code or "").strip():
        (qdir / "solution.cu").write_text(code, encoding="utf-8")
    for child in list(qdir.iterdir()):
        name = child.name
        if child.is_dir() and (name.startswith("c") or name == "include"):
            shutil.rmtree(child, ignore_errors=True)
        elif name in {"helpers.h", "solution_header.h", "solution.o"}:
            child.unlink(missing_ok=True)
        elif name == "nvcc.log" and success:
            child.unlink(missing_ok=True)


def smoke_compile(settings: Settings | None = None, workdir: Path | None = None) -> CompileResult:
    """Compile a tiny built-in kernel to verify nvcc is usable.

    Args:
        settings: Optional settings override.
        workdir: Optional scratch dir; defaults to ``work/_smoke``.
    """
    settings = settings or get_settings()
    target = workdir or (settings.work_path / "_smoke")
    return compile_cuda_source(SMOKE_SOURCE, target, settings=settings)
