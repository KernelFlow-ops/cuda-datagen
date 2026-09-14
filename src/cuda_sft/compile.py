from __future__ import annotations

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
    "device-side kernel launch",
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
    ok: bool
    command: list[str]
    output: str
    used_rdc: bool
    source_path: Path | None = None


def ensure_stubs(workdir: Path) -> None:
    include_dir = workdir / "include"
    include_dir.mkdir(parents=True, exist_ok=True)
    (include_dir / "helpers.h").write_text(HELPERS_STUB, encoding="utf-8")
    (include_dir / "solution_header.h").write_text(SOLUTION_HEADER_STUB, encoding="utf-8")
    (workdir / "helpers.h").write_text(HELPERS_STUB, encoding="utf-8")
    (workdir / "solution_header.h").write_text(SOLUTION_HEADER_STUB, encoding="utf-8")


def looks_like_rdc_error(output: str) -> bool:
    lowered = output.lower()
    return any(hint in lowered for hint in RDC_HINTS)


def _nvcc_command(
    settings: Settings,
    source: Path,
    output: Path,
    *,
    rdc: bool,
) -> list[str]:
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


def smoke_compile(settings: Settings | None = None, workdir: Path | None = None) -> CompileResult:
    settings = settings or get_settings()
    target = workdir or (settings.work_path / "_smoke")
    return compile_cuda_source(SMOKE_SOURCE, target, settings=settings)
