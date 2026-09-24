"""TileLang Python dialect (optional; skipped if the package is missing)."""

from __future__ import annotations

from pathlib import Path

from cuda_sft.compile import CompileResult
from cuda_sft.config import Settings
from cuda_sft.dialects.python_check import run_python_gate
from cuda_sft.judge import JudgeResult
from cuda_sft.parse import extract_fenced_source
from cuda_sft.prompt import SelectedPrompts, format_nvcc_for_prompt
from cuda_sft.prompts.selection import language_matched_index, stable_index
from cuda_sft.refval.spec import DialectRefvalSpec

SYSTEM_PROMPTS = (
    (
        "You are a TileLang GPU kernel engineer. "
        "Implement the operator with tilelang / T.prim_func / T.Kernel. "
        "Do not write raw CUDA C++ unless TileLang requires a tiny prelude. "
        "Target generic CUDA (Ampere), not TMA-only Hopper paths."
    ),
    (
        "你是 TileLang 算子工程师。用 T.prim_func 与 T.Kernel 实现题面等价算子。"
        "单文件 Python。禁止 Hopper TMA-only。只输出一个 python 代码块。"
    ),
)

USER_SUFFIXES = (
    """## 生成要求（TileLang）

用 **TileLang** 实现以上算子的等价功能。

1. 目标：{gpu_name} / `{cuda_arch}` / CUDA {cuda_version}。
2. 单文件 `solution.py`：`import tilelang` 与 `import tilelang.language as T`，包含 `T.prim_func` 和 `T.Kernel`（或 `@tilelang.jit`）。
3. 可用 `T.copy`、`T.Parallel`；不要依赖 TMA-only。
4. 不要在 import 时跑大测试。门闩是 compile/import，不比对数值。
5. 只输出一个 ```python 代码块。""",
)

REPAIR_PROMPTS = (
    """上一版 TileLang 代码未能通过编译门闩。请输出完整修正后的 `solution.py`。
架构 {cuda_arch}。只修编译/import 问题。

```
{error}
```

```python
{code}
```

只输出一个 ```python 代码块。""",
)

SMOKE_SOURCE = '''
import tilelang
import tilelang.language as T


def elementwise_add(N, block=256, dtype="float32"):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                if gi < N:
                    C[gi] = A[gi] + B[gi]

    return main
'''

COT_SKELETON = """1. Problem restatement — tensors/shapes, host entry.
2. Algorithm — tile-level formula.
3. T.Kernel mapping — grid, threads, T.Parallel.
4. Memory — T.copy / shared vs global.
5. Bounds and edge cases — ceildiv tails.
6. Implementation checklist — prim_func name, jit/compile entry."""


def looks_like_tilelang(source: str) -> bool:
    """True if source looks like TileLang."""
    return any(
        tok in source
        for tok in ("T.prim_func", "T.Kernel", "tilelang.language", "@tilelang.jit")
    )


class TileLangDialect:
    """TileLang DSL; ``available()`` is false when the package is missing."""

    name = "tilelang"
    language = "python"
    source_filename = "solution.py"
    fence_langs = ("python", "py", "tilelang")

    def available(self, settings: Settings) -> tuple[bool, str]:
        try:
            import tilelang  # noqa: F401
        except ImportError:
            return False, "tilelang package is not installed"
        missing = []
        for dep in ("tvm_ffi", "torch_c_dlpack_ext", "z3"):
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        if missing:
            return False, "tilelang transitive dependencies missing: " + ", ".join(missing)
        try:
            import torch
        except ImportError:
            return False, "unavailable: torch is not installed"
        if not torch.cuda.is_available():
            return False, "unavailable: torch.cuda.is_available() is false"
        return True, f"tilelang {getattr(tilelang, '__version__', '?')} + CUDA device"

    def extract(self, text: str) -> str:
        return extract_fenced_source(
            text, fence_langs=self.fence_langs, looks_like=looks_like_tilelang
        )

    def compile(self, code: str, workdir: Path, settings: Settings) -> CompileResult:
        return run_python_gate(
            "tilelang",
            code,
            workdir,
            filename=self.source_filename,
            timeout_sec=settings.tilelang_timeout_sec,
            dialect="tilelang",
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
        return SelectedPrompts(
            system=SYSTEM_PROMPTS[sys_i],
            user=f"{question.rstrip()}\n\n{suffix}\n",
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
            error=format_nvcc_for_prompt(compile_error),
            code=previous_code.strip() or "(no source extracted)",
        )

    def judge(self, code: str) -> JudgeResult:
        source = code or ""
        issues: list[str] = []
        suggestions: list[str] = []
        score = 10
        if "T.prim_func" not in source and "@T.prim_func" not in source:
            issues.append("missing T.prim_func")
            score -= 3
        if "T.Kernel" not in source:
            issues.append("missing T.Kernel")
            score -= 2
        if "tma" in source.lower() or "SM90" in source:
            suggestions.append("avoid TMA/SM90-only TileLang paths on sm_86")
            score -= 1
        return JudgeResult(quality_score=max(1, min(10, score)), issues=issues, suggestions=suggestions)

    def smoke(self, settings: Settings, workdir: Path) -> CompileResult:
        return self.compile(SMOKE_SOURCE, workdir, settings)

    def cot_skeleton(self) -> str:
        return COT_SKELETON

    def refval_spec(self, settings: Settings) -> DialectRefvalSpec:
        """Import ``solution.py``; factory(N) then kernel(tensors) is allowed."""
        return DialectRefvalSpec(
            dialect="tilelang",
            language="python",
            runner="python_import",
            source_filename=self.source_filename,
            needs_nvcc=False,
            needs_torch=True,
            timeout_sec=int(getattr(settings, "tilelang_timeout_sec", 180) or 180),
            host_entry_hint=(
                "TileLang host entry call_style is one of eager, lazy, or "
                "primfunc_factory: eager JIT accepts torch CUDA tensors; lazy and "
                "PrimFunc factories accept sizes, then return/invoke a compiled "
                "kernel. Driver must launch with Torch CUDA tensors and call "
                "torch.cuda.synchronize(). No import-time run."
            ),
        )
