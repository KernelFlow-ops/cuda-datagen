"""Triton Python dialect."""

from __future__ import annotations

from pathlib import Path

from cuda_sft.compile import CompileResult
from cuda_sft.config import Settings
from cuda_sft.dialects.python_check import run_python_gate
from cuda_sft.judge import JudgeResult
from cuda_sft.parse import extract_fenced_source
from cuda_sft.prompt import SelectedPrompts
from cuda_sft.prompts.selection import language_matched_index, stable_index
from cuda_sft.refval.spec import DialectRefvalSpec

SYSTEM_PROMPTS = (
    (
        "You are a Triton GPU kernel engineer. "
        "Implement the operator as Python with @triton.jit kernels plus a host launcher. "
        "Do not launch the kernel at import time. Do not write CUDA C++."
    ),
    (
        "你是 Triton 算子工程师。用 Python + @triton.jit 实现题面等价算子。"
        "同一文件提供 host launch 函数。禁止 import 时执行 kernel。只输出一个 python 代码块。"
    ),
)

USER_SUFFIXES = (
    """## 生成要求（Triton）

用 **Triton** 实现以上算子的等价功能（不要写 CUDA C++，不要死守 solution.cu 文件名）。

1. 目标 GPU：{gpu_name}，架构 {cuda_arch}，CUDA {cuda_version}。
2. 单文件 `solution.py`：`import triton` / `import triton.language as tl`，至少一个 `@triton.jit` kernel，以及 host 入口（例如 `launch_*`）。
3. 用 `tl.program_id`、mask 处理边界；BLOCK 为 2 的幂 constexpr。
4. 禁止在模块顶层或 `if __name__ == "__main__"` 里真正 launch（编译门闩会 import 该文件）。
5. 不要 `main()` 测试循环。门闩是语法 + import + 存在 jit kernel，不比对数值。
6. 只输出一个 ```python 代码块。""",
    """## Requirements (Triton)

Implement the **same operator** in Triton.

- Target {gpu_name} / `{cuda_arch}` / CUDA {cuda_version}.
- One `solution.py`: `@triton.jit` + host launcher. No import-time launch.
- Mask OOB loads/stores. BLOCK power-of-two.
- Output exactly one ```python fence.""",
)

REPAIR_PROMPTS = (
    """上一版 Triton 代码未能通过编译门闩。请输出完整修正后的 `solution.py`。
目标架构 {cuda_arch}。保持算法，只修 import/JIT 问题。禁止改写为 CUDA C++。

编译器输出：
```
{error}
```

上一版：
```python
{code}
```

只输出一个 ```python 代码块。""",
)

SMOKE_SOURCE = '''
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def launch_add(x_ptr, y_ptr, out_ptr, n, block=1024):
    grid = (triton.cdiv(n, block),)
    add_kernel[grid](x_ptr, y_ptr, out_ptr, n, BLOCK=block)
'''

COT_SKELETON = """1. Problem restatement — tensors/shapes, host entry.
2. Algorithm — elementwise / reduction / gemm pattern.
3. Program-id mapping — tl.program_id, BLOCK constexpr, grid.
4. Memory and mask — tl.load/store masks, coalescing.
5. Bounds and edge cases — n not multiple of BLOCK, empty n.
6. Implementation checklist — jit kernel name, host launcher, constexprs."""

COT_SKELETON_ZH_TRITON = (
    "1. 题意与张量/入口\n"
    "2. 算法\n"
    "3. program_id 与 BLOCK 映射\n"
    "4. 访存与 mask\n"
    "5. 边界与异常\n"
    "6. 实现清单"
)


def looks_like_triton(source: str) -> bool:
    """True if source looks like a Triton kernel file."""
    return "@triton.jit" in source or "triton.language" in source or "tl.program_id" in source


class TritonDialect:
    """Python Triton kernels gated by parse + import and strict CUDA refval.

    Implements :class:`~cuda_sft.dialects.base.DialectSpec`. ``available()``
    is False when the ``triton`` package is missing. Numeric validation is
    performed by the shared Python driver with CUDA tensors and an explicit
    frozen ABI call style; import-time launches remain forbidden.
    """

    name = "triton"
    language = "python"
    source_filename = "solution.py"
    fence_langs = ("python", "py", "triton")

    def available(self, settings: Settings) -> tuple[bool, str]:
        """Return whether the Triton package can be imported.

        Args:
            settings: Unused; protocol compatibility.
        """
        try:
            import triton  # noqa: F401
        except ImportError:
            return False, "triton package is not installed"
        return True, ""

    def extract(self, text: str) -> str:
        """Pull a Python/Triton fence from the model reply.

        Args:
            text: Raw assistant text.
        """
        return extract_fenced_source(
            text, fence_langs=self.fence_langs, looks_like=looks_like_triton
        )

    def compile(self, code: str, workdir: Path, settings: Settings) -> CompileResult:
        """Parse + import gate in a subprocess (no numeric tests).

        Args:
            code: Extracted ``solution.py``.
            workdir: Per-attempt directory.
            settings: ``triton_timeout_sec``.
        """
        return run_python_gate(
            "triton",
            code,
            workdir,
            filename=self.source_filename,
            timeout_sec=settings.triton_timeout_sec,
            dialect="triton",
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
            error=compile_error.strip() or "(empty compiler output)",
            code=previous_code.strip() or "(no source extracted)",
        )

    def judge(self, code: str) -> JudgeResult:
        """Heuristic Triton checks (jit kernel, program_id, no CUDA leak).

        Args:
            code: Compile-passing Python source.
        """
        source = code or ""
        issues: list[str] = []
        suggestions: list[str] = []
        score = 10
        if "@triton.jit" not in source:
            issues.append("missing @triton.jit kernel")
            score -= 3
        if "tl.program_id" not in source and "program_id" not in source:
            issues.append("no tl.program_id mapping")
            score -= 2
        if "mask" not in source:
            suggestions.append("add load/store masks for OOB threads")
            score -= 1
        if "__global__" in source:
            issues.append("CUDA C++ leaked into a Triton sample")
            score -= 2
        return JudgeResult(quality_score=max(1, min(10, score)), issues=issues, suggestions=suggestions)

    def smoke(self, settings: Settings, workdir: Path) -> CompileResult:
        return self.compile(SMOKE_SOURCE, workdir, settings)

    def cot_skeleton(self, lang: str = "en") -> str:
        """Six-heading CoT outline; ``lang="zh"`` returns the Chinese headings."""
        return COT_SKELETON_ZH_TRITON if lang == "zh" else COT_SKELETON

    def refval_spec(self, settings: Settings) -> DialectRefvalSpec:
        """Import ``solution.py`` and call the host launcher with torch CUDA tensors."""
        return DialectRefvalSpec(
            dialect="triton",
            language="python",
            runner="python_import",
            source_filename=self.source_filename,
            needs_nvcc=False,
            needs_torch=True,
            timeout_sec=int(getattr(settings, "triton_timeout_sec", 90) or 90),
            host_entry_hint=(
                "Python host launcher with the frozen ABI parameter order and "
                "call_style (positional or kwargs). Every tensor argument must "
                "be a torch CUDA tensor with the manifest dtype/shape/stride; "
                "outputs are preallocated and mutated in place. Do not launch "
                "at import time or guess a factory call after TypeError."
            ),
        )
