"""CUTLASS 4.x + CuTe dialect (nvcc -c with CUTLASS headers)."""

from __future__ import annotations

import re
from pathlib import Path

from cuda_sft.compile import CompileResult, compile_cuda_source
from cuda_sft.config import Settings
from cuda_sft.judge import JudgeResult
from cuda_sft.parse import extract_fenced_source, looks_like_cuda
from cuda_sft.prompt import SelectedPrompts
from cuda_sft.prompts.selection import language_matched_index, stable_index
from cuda_sft.refval.spec import DialectRefvalSpec

CUTLASS_MAJOR_RE = re.compile(r"#define\s+CUTLASS_MAJOR\s+(\d+)")

SYSTEM_PROMPTS = (
    (
        "You are a CUTLASS 4.x / CuTe kernel engineer. "
        "Write one self-contained CUDA translation unit that compiles with nvcc -c "
        "against CUTLASS 4 headers and CuTe. Target Ampere (sm_86). "
        "Do not use SM90, TMA, cluster launch, or Hopper WGMMA. "
        "Do not wrap kernels in an anonymous namespace (conflicts with CuTe under nvcc). "
        "Do not use CUTLASS 2.x device::Gemm<> templates."
    ),
    (
        "你是 CUTLASS 4.x + CuTe 工程师。输出可 nvcc -c 通过的单文件 .cu。"
        "只使用 CUTLASS 4 / CuTe 头文件；架构 sm_86；禁止 Hopper-only API。"
    ),
    (
        "Act as an expert in NVIDIA CUTLASS 4.x and CuTe layouts. "
        "Prefer cute::Tensor, make_layout, make_tensor for indexing. "
        "One compilable .cu file; CUDA Toolkit + CUTLASS 4 includes only."
    ),
)

USER_SUFFIXES = (
    """## 生成要求（CUTLASS 4.x + CuTe）

请为以上题目生成**等价算子**的 CUTLASS 4 / CuTe 实现（不要死守 solution.cu 路径字面量）。

1. 目标：{gpu_name}，`{cuda_arch}`，CUDA {cuda_version}。CUTLASS **4.x** 头文件已在 include path。
2. 单文件 `solution.cu`：必要 `#include`、`__global__`、host 入口。可用 `<cute/tensor.hpp>`、`<cute/layout.hpp>`、`<cutlass/cutlass.h>`。
3. 禁止 SM90 / TMA / `cute::SM90` / cluster / WGMMA；禁止 CUTLASS 2.x `device::Gemm<>`。禁止匿名 namespace（与 CuTe 冲突）。
4. 不要编造缺失工程头；`include/solution_header.h` 若被引用，环境会提供空 stubs，请把声明写在本文件。
5. 不要 `main()`。门闩是 `nvcc -c`，不跑数值。
6. 只输出一个 ```cuda 代码块。""",
    """## Requirements (CUTLASS 4.x + CuTe)

Implement the **same operator** with CUTLASS 4 / CuTe, compile-gated.

- Target {gpu_name} / `{cuda_arch}` / CUDA {cuda_version}.
- One `solution.cu`. Use CuTe tensors/layouts. No Hopper-only APIs.
- No `main()`. Gate: `nvcc -c` with CUTLASS 4 includes.
- Reply with exactly one ```cuda fence.""",
)

REPAIR_PROMPTS = (
    """上一版 CUTLASS 4 / CuTe 代码未能通过 nvcc 编译。请输出完整修正后的 `solution.cu`。
目标架构 {cuda_arch}。保持算法，只修编译问题。禁止改用 SM90 API。若报 anonymous namespace / cudafe stub 冲突，去掉匿名 namespace。

nvcc 输出：
```
{error}
```

上一版：
```cuda
{code}
```

只输出一个 ```cuda 代码块。""",
    """nvcc failed on the CUTLASS 4.x kernel (arch {cuda_arch}). Return a full corrected `solution.cu`.
Keep Ampere-only CuTe/CUTLASS 4 APIs.

```
{error}
```

```cuda
{code}
```

Exactly one ```cuda fence.""",
)

SMOKE_SOURCE = r"""
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>

__global__ void cute_scale_kernel(float* x, float s, int n) {
  using namespace cute;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    auto tensor = make_tensor(x, make_layout(make_shape(n)));
    tensor(i) = tensor(i) * s;
  }
}

void launch_cute_scale(float* x, float s, int n) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  cute_scale_kernel<<<blocks, threads>>>(x, s, n);
}
"""

COT_SKELETON = """1. Problem restatement — tensors/shapes, host entry, success criteria.
2. Algorithm — formula and numeric type notes.
3. CuTe layout / thread mapping — make_layout, make_tensor, tile vs thread index.
4. Memory and copy — gmem/smem, cute copy atoms if used; no TMA on sm_86.
5. Bounds and edge cases — predicated tails, empty n.
6. Implementation checklist — CUTLASS 4 headers, kernel name, host launcher."""

COT_SKELETON_ZH_CUTLASS = (
    "1. 题意与张量/入口\n"
    "2. 算法与数值类型\n"
    "3. CuTe 布局与线程映射\n"
    "4. 存储与拷贝\n"
    "5. 边界与异常\n"
    "6. 实现清单"
)


def looks_like_cutlass(source: str) -> bool:
    """True if source looks like CUDA or CUTLASS/CuTe C++.

    Args:
        source: Extracted translation unit.
    """
    if looks_like_cuda(source):
        return True
    return any(tok in source for tok in ("cute::", "cutlass::", "make_tensor", "make_layout"))


def read_cutlass_major(home: Path) -> int | None:
    """Parse ``CUTLASS_MAJOR`` from ``include/cutlass/version.h``.

    Args:
        home: CUTLASS root. ``available()`` requires major == 4.
    """
    header = home / "include" / "cutlass" / "version.h"
    if not header.is_file():
        return None
    try:
        text = header.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = CUTLASS_MAJOR_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def candidate_cutlass_homes(settings: Settings | None = None) -> list[Path]:
    """Search order: ``CUTLASS_HOME``, system 4.3.5, project ``third_party/cutlass``."""
    from cuda_sft.config import PROJECT_ROOT

    homes: list[Path] = []
    seen: set[str] = set()
    raw = (settings.cutlass_home if settings is not None else "") or ""
    for item in (
        Path(raw) if raw.strip() else None,
        Path("/usr/local/cutlass-4.3.5"),
        PROJECT_ROOT / "third_party" / "cutlass",
    ):
        if item is None:
            continue
        key = str(item)
        if key not in seen:
            seen.add(key)
            homes.append(item)
    return homes


def is_cutlass4_home(home: Path) -> bool:
    """True if ``home`` looks like CUTLASS 4.x with CuTe headers."""
    if not home.is_dir():
        return False
    if not (home / "include" / "cute").is_dir():
        return False
    major = read_cutlass_major(home)
    return major == 4


def resolved_cutlass_home(settings: Settings) -> Path | None:
    """First valid CUTLASS 4.x tree, or None."""
    for home in candidate_cutlass_homes(settings):
        if is_cutlass4_home(home):
            return home
    return None


def cutlass_include_dirs(settings: Settings) -> list[str]:
    """Include paths for CUTLASS 4.x headers."""
    home = resolved_cutlass_home(settings) or Path(settings.cutlass_home or "")
    includes = [str(home / "include")]
    util = home / "tools" / "util" / "include"
    if util.is_dir():
        includes.append(str(util))
    return includes


class CutlassDialect:
    """CUTLASS 4.x + CuTe C++ compiled with ``nvcc -c``.

    Implements :class:`~cuda_sft.dialects.base.DialectSpec`. Hopper-only
    APIs are rejected by the heuristic judge; ``available()`` requires
    ``CUTLASS_MAJOR==4`` and a CuTe include tree.
    """

    name = "cutlass"
    language = "cuda-cpp"
    source_filename = "solution.cu"
    fence_langs = ("cuda", "cu", "cpp", "c++", "cc", "cxx", "cutlass", "cute")

    def available(self, settings: Settings) -> tuple[bool, str]:
        home = resolved_cutlass_home(settings)
        if home is not None:
            return True, str(home)
        tried = ", ".join(str(p) for p in candidate_cutlass_homes(settings))
        return False, f"no CUTLASS 4.x + CuTe tree (tried {tried})"

    def extract(self, text: str) -> str:
        return extract_fenced_source(
            text, fence_langs=self.fence_langs, looks_like=looks_like_cutlass
        )

    def compile(self, code: str, workdir: Path, settings: Settings) -> CompileResult:
        return compile_cuda_source(
            code,
            workdir,
            settings=settings,
            extra_includes=cutlass_include_dirs(settings),
            std=settings.cutlass_cxx_std,
            filename=self.source_filename,
            dialect="cutlass",
        )

    def select_prompts(
        self,
        question: str,
        *,
        question_id: int,
        candidate_idx: int,
        gpu_name: str,
        cuda_arch: str,
        cuda_version: str,
    ) -> SelectedPrompts:
        sys_i = language_matched_index(
            SYSTEM_PROMPTS, question, question_id, candidate_idx, salt=0
        )
        suf_i = language_matched_index(
            USER_SUFFIXES, question, question_id, candidate_idx, salt=7
        )
        suffix = USER_SUFFIXES[suf_i].format(
            gpu_name=gpu_name, cuda_arch=cuda_arch, cuda_version=cuda_version
        ).strip()
        user = f"{question.rstrip()}\n\n{suffix}\n"
        return SelectedPrompts(
            system=SYSTEM_PROMPTS[sys_i],
            user=user,
            system_index=sys_i,
            suffix_index=suf_i,
        )

    def build_repair(
        self,
        *,
        cuda_arch: str,
        compile_error: str,
        previous_code: str,
        question_id: int,
        candidate_idx: int,
        repair_idx: int,
    ) -> str:
        idx = stable_index(
            len(REPAIR_PROMPTS), question_id, candidate_idx, salt=13 + int(repair_idx)
        )
        return REPAIR_PROMPTS[idx].format(
            cuda_arch=cuda_arch,
            error=compile_error.strip() or "(empty compiler output)",
            code=previous_code.strip() or "(no source extracted)",
        )

    def judge(self, code: str) -> JudgeResult:
        source = code or ""
        issues: list[str] = []
        suggestions: list[str] = []
        score = 10
        lowered = source.lower()
        if "cute::" not in source and "cutlass::" not in source and "make_tensor" not in source:
            issues.append("no CuTe/CUTLASS 4 APIs detected")
            score -= 2
        if any(tok in source for tok in ("SM90", "cute::SM90", "tma_load", "cp.async.bulk")):
            issues.append("Hopper/SM90 API used; target is sm_86")
            score -= 3
        if "cutlass::gemm::device::Gemm" in source:
            suggestions.append("prefer CUTLASS 4 CuTe style over 2.x device::Gemm")
            score -= 1
        if "__global__" not in source:
            issues.append("no __global__ kernel")
            score -= 2
        if "sm90" in lowered or "hopper" in lowered:
            suggestions.append("comments mention Hopper; keep Ampere-only")
        return JudgeResult(
            quality_score=max(1, min(10, score)),
            issues=issues,
            suggestions=suggestions,
        )

    def smoke(self, settings: Settings, workdir: Path) -> CompileResult:
        return self.compile(SMOKE_SOURCE, workdir, settings)

    def cot_skeleton(self, lang: str = "en") -> str:
        """Six-heading CoT outline; ``lang="zh"`` returns the Chinese headings."""
        return COT_SKELETON_ZH_CUTLASS if lang == "zh" else COT_SKELETON

    def refval_spec(self, settings: Settings) -> DialectRefvalSpec:
        """CUTLASS 4.x uses the CUDA nvcc-link path plus extra include dirs."""
        includes = tuple(cutlass_include_dirs(settings))
        return DialectRefvalSpec(
            dialect="cutlass",
            language="cuda-cpp",
            runner="nvcc_link",
            source_filename=self.source_filename,
            extra_includes=includes,
            cxx_std=settings.cutlass_cxx_std or "c++17",
            needs_nvcc=True,
            needs_torch=False,
            timeout_sec=int(getattr(settings, "refval_timeout_sec", 45) or 45),
            host_entry_hint=(
                "CUTLASS/CuTe host launcher; Ampere only. Device pointers unless "
                "the function allocates. Do not emit main()."
            ),
        )
