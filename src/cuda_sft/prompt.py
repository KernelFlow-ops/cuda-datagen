"""System / user / repair prompt pools and stable per-question selection."""

from __future__ import annotations

from typing import NamedTuple

from cuda_sft.prompts.nvcc_log import format_nvcc_for_prompt, truncate_compile_error
from cuda_sft.prompts.selection import (
    CANDIDATE_TEMPERATURES,
    candidate_temperature,
    language_matched_index,
    looks_chinese,
)
from cuda_sft.prompts.selection import stable_index as _stable_index

# Generator system pool: compile-first role, mixed EN/ZH. Selection matches
# question CJK-ness via language_matched_index (not random).
SYSTEM_PROMPTS: tuple[str, ...] = (
    (
        "You are a senior CUDA kernel engineer. "
        "Write complete, high-quality CUDA source that compiles with nvcc -c "
        "on the specified GPU architecture. "
        "Do not omit includes, kernel definitions, or the host entry function. "
        "Do not invent missing project headers."
    ),
    (
        "You are a compiler-first CUDA implementer. "
        "Ship a single self-contained .cu file that nvcc can compile with -c. "
        "Prefer correct, boring, compilable code over clever micro-optimizations. "
        "Never include headers that are not in the CUDA Toolkit."
    ),
    (
        "你是资深 CUDA 算子工程师，负责写出可独立编译的高质量 kernel。 "
        "必须包含完整 #include、__global__ 与 host 入口。 "
        "禁止编造本地工程头文件；只使用 CUDA Toolkit 提供的头文件。"
    ),
    (
        "You write production CUDA kernels for NVIDIA GPUs. "
        "Respect the requested SM architecture, handle bounds, and keep indexing correct. "
        "Output only compilable source: one translation unit, no test harness, no missing symbols."
    ),
    (
        "You are a CUDA coding agent. Your job is to emit a complete solution.cu. "
        "The downstream checker only runs `nvcc -c`; if it does not compile, you failed. "
        "Do not explain. Do not invent include/solution_header.h or include/helpers.h."
    ),
    (
        "你是高性能计算方向的 CUDA 开发者。 "
        "在保证能编译的前提下，注意合并访存、合理 block 大小和边界判断。 "
        "不要把简单题写得过重；不要依赖评测框架或本地缺失头文件。"
    ),
    (
        "Act as an expert GPU programmer who answers with source code only. "
        "Produce high-quality CUDA that would pass a compile-only gate on the target arch. "
        "Include every symbol you use. Keep the host launcher consistent with the problem."
    ),
    (
        "你是只输出 CUDA 源码的助手。 "
        "目标是高质量、可 nvcc -c 通过的单文件实现。 "
        "题面未给精确签名时，自行给出清晰的 host 入口，仍须覆盖题面要求。"
    ),
)

SYSTEM_PROMPT = SYSTEM_PROMPTS[0]

TRAINING_SYSTEM_PROMPT_ZH = (
    "你是一名资深 GPU 内核工程师。请先分析题目要求、数据规模与硬件特性，再给出完整、正确、高效且可直接编译的实现。"
)
TRAINING_SYSTEM_PROMPT_EN = (
    "You are a senior GPU kernel engineer. First analyze the problem requirements, data sizes and hardware characteristics, then provide a complete, correct, efficient implementation that compiles as-is."
)

USER_SUFFIXES: tuple[str, ...] = (
    """## 生成要求（高质量）

请为以上题目生成**高质量** CUDA 算子实现，必须满足：

1. 目标硬件与编译：{gpu_name}，架构 {cuda_arch}（`nvcc -arch={cuda_arch}`），CUDA {cuda_version}。
2. 输出单个完整源文件 `solution.cu`，在同一文件中包含：必要 `#include`、`__global__` kernel、以及与题面一致的 host 入口函数。若题面未给出精确签名，自行定义清晰的 host 入口。
3. 不要 `#include` 本环境不存在的本地头文件（例如 `include/solution_header.h`、`include/helpers.h`）；不要依赖 `test/` 或任何评测框架。允许 CUDA Toolkit 头文件（`cuda_runtime.h`、`cuda_fp16.h`、CUB、Thrust、`cooperative_groups` 等）。
4. 不需要 `main()`。我们只做 `nvcc -c` 编译检查，不运行测试、不核对数值。
5. 正确处理边界条件、线程索引和必要的 `__syncthreads()`。优先保证能编译：避免未声明标识符、错误 launch 语法、缺分号、错误的 `__global__`/`__device__` 用法。
6. 只输出一个 markdown 代码块，语言标记为 `cuda`，不要附加解释文字。""",
    """## Requirements (high-quality CUDA)

Implement a **high-quality**, self-contained CUDA operator for the problem above.

- Target: {gpu_name}, arch `{cuda_arch}` (`nvcc -arch={cuda_arch}`), CUDA {cuda_version}.
- One file `solution.cu`: includes + `__global__` kernel + host entry matching the prompt (invent a clear host signature if none is given).
- Do **not** include missing local headers such as `include/solution_header.h` or `include/helpers.h`. CUDA Toolkit headers are OK.
- No `main()`, no tests. Gate is `nvcc -c` only.
- Handle bounds, thread indexing, and `__syncthreads()` when shared memory is used.
- Reply with exactly one markdown fence tagged `cuda`. No prose before or after.""",
    """## 约束（compile-first，高质量）

在 {gpu_name} / `{cuda_arch}` / CUDA {cuda_version} 上给出能 `nvcc -c -arch={cuda_arch}` 通过的**高质量**实现。

清单：
- [ ] 单文件 `solution.cu`（kernel + host 入口都在同一文件）
- [ ] 不引用本仓库不存在的头（禁止 `include/solution_header.h`、`include/helpers.h`）
- [ ] 可用 `cuda_runtime.h`、`cuda_fp16.h`、CUB、Thrust、`cooperative_groups`
- [ ] 不要 `main()`，不要测试
- [ ] 边界、索引、同步正确；先保证编译再谈优化

只输出一个 ```cuda 代码块。""",
    """## Implementation notes

Write **high-quality** CUDA for: {gpu_name} (SM `{cuda_arch}`, CUDA {cuda_version}).

Constraints:
1. Self-contained `solution.cu` with kernel and host launcher.
2. Never depend on `include/solution_header.h`, `include/helpers.h`, or `test/`.
3. Compile-only evaluation: `nvcc -c -arch={cuda_arch}`. No `main()`.
4. Prefer correct indexing and bounds checks over unsafe tricks.
5. Shared memory must be paired with `__syncthreads()` where needed.

Output format: a single ```cuda block containing the full source, nothing else.""",
    """## 高质量实现说明

请用 CUDA 完成上述算子，标准是**高质量且可编译**。

目标设备 {gpu_name}，请按 `{cuda_arch}`、CUDA {cuda_version} 来写（编译命令等价于 `nvcc -c -arch={cuda_arch}`）。
把 kernel 和 host 启动函数都放进 `solution.cu`。题面若提到本地头文件，那些文件在本环境中不存在，请把声明/定义写在同一文件里。
不要写 `main()`。不要解释思路。

最终只给一个语言标记为 cuda 的 markdown 代码块。""",
    """## Task

Produce a **high-quality** CUDA implementation of the problem.

Hardware: {gpu_name}. Compile with `nvcc -c -arch={cuda_arch}` (CUDA {cuda_version}).
Deliverable: one `solution.cu` translation unit — includes, device kernels, host API.
Forbidden: missing project headers (`include/solution_header.h`, `include/helpers.h`), `test/`, `main()`.
Allowed: CUDA Toolkit headers.

If the prompt is underspecified, choose a simple, complete host signature and still implement the algorithm.

Return only:
```cuda
// full source
```""",
    """## 生成要求

为以上题目写一份**高质量** CUDA 代码（不是伪代码）。

1. 架构：{gpu_name}，`{cuda_arch}`，CUDA {cuda_version}。
2. 单文件可编译：`nvcc -c` 即可，无需链接、无需运行。
3. 自包含：不要 `#include "include/solution_header.h"` / `"include/helpers.h"`。
4. 线程映射与越界检查必须正确；使用 shared memory 时同步完整。
5. 可以点到 coalescing / 合理 block 尺寸，但不要为了炫技引入无法编译的模板或外部库。

输出：仅一个 ```cuda 代码块。""",
    """## Constraints

High-quality CUDA operator, compile-gated.

| Item | Value |
| --- | --- |
| GPU | {gpu_name} |
| Arch | {cuda_arch} |
| CUDA | {cuda_version} |
| File | `solution.cu` (kernel + host) |
| Check | `nvcc -c -arch={cuda_arch}` |
| Local headers | do not use (they are absent) |
| `main()` / tests | not required |

Style: clear indexing, safe bounds, Toolkit-only includes.
Format: exactly one markdown code block with language `cuda`.""",
    """## 请实现（高质量算子）

在 {gpu_name}（`nvcc -arch={cuda_arch}`，CUDA {cuda_version}）上实现题面功能。

要求简要重申：完整 `solution.cu`；高质量；能过编译；不要本地缺失头文件；不要 `main()`；不要评测目录。
host 函数名尽量贴近题面；题面没写死签名时，用直观名称并在同一文件定义。

请直接给出唯一的 ```cuda 源码块，不要附加说明。""",
    """## Output contract

This is a **high-quality** CUDA codegen task for {gpu_name} / `{cuda_arch}` / CUDA {cuda_version}.

The answer must be one complete `solution.cu` that `nvcc -c -arch={cuda_arch}` accepts.
Do not include `include/solution_header.h` or `include/helpers.h`.
Do not emit a `main()`.
Do not emit commentary, bullet recaps, or a second code fence.

```cuda
<entire source here>
```""",
)

REPAIR_PROMPTS: tuple[str, ...] = (
    """上一版 CUDA 代码未能通过 nvcc 编译。请输出**完整修正后**的 `solution.cu`（不要只给 diff）。
目标架构：{cuda_arch}。保持题面要求的算法与函数职责，只修编译问题。

## Evidence
nvcc 输出（过长则已去重摘要）：
```
{error}
```

## Previous solution.cu
```cuda
{code}
```

只输出一个 ```cuda``` 代码块。""",
    """nvcc failed on the previous kernel (arch {cuda_arch}). Return a full corrected `solution.cu`, not a patch.
Keep the same algorithm and host responsibilities; fix compile errors only.

## Evidence
nvcc output (deduplicated summary if over the size limit):
```
{error}
```

## Previous solution.cu
```cuda
{code}
```

Reply with exactly one ```cuda fence.""",
    """请根据 nvcc 报错修复上一版实现，目标 `{cuda_arch}`。
给出整份可编译源码，不要解释，不要省略未改动的函数。

## Evidence
nvcc 输出（过长则已去重摘要）：
```
{error}
```

## Previous solution.cu
```cuda
{code}
```

仅输出一个 ```cuda 代码块。""",
)

# Short compile-first example. Kept tiny so it does not dominate the real problem.
FEW_SHOT_CUDA = """## Tiny valid-shape example (not the solution)
```cuda
#include <cuda_runtime.h>
__global__ void scale_kernel(float* x, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] *= s;
}
void launch_scale(float* x, float s, int n) {
  int t = 256;
  scale_kernel<<<(n + t - 1) / t, t>>>(x, s, n);
}
```
Invalid: `#include "include/helpers.h"` (that file is absent). Invalid: a `main()` test harness."""

COT_NO_REPAIR_RULES_ZH = "严格要求：以第一次独立解题的视角叙述推理过程。不得提及任何编译错误、报错信息、调试、修复、上一版本、之前的尝试或评测框架。推理中出现的函数名、常量值、线程块尺寸、共享内存数组名必须与最终代码完全一致。"
COT_NO_REPAIR_RULES_EN = "Strict rules: narrate the reasoning as a first, independent attempt at the problem. Never mention compile errors, error messages, debugging, fixes, previous versions, earlier attempts, or any grading harness. Every function name, constant value, block size and shared-memory array name mentioned must match the final code exactly."

_COT_SYSTEM_BASE_CUDA = (
    """你是 CUDA SFT 思维链编辑器，不是写代码的人。最终 solution.cu 已经通过编译；你只整理可学习的推理。
You are the CUDA SFT Chain-of-Thought editor.

Your job is NOT to write a new kernel. A compilable solution.cu already exists.
You rewrite the teacher's raw reasoning into a clean, pedagogical CoT that a
student CUDA model should imitate before emitting code.

Role constraints:
- Faithful to the FINAL source: every claim in the CoT must match the given
  solution.cu. Do not invent algorithms, headers, APIs, or optimizations that
  the code does not implement.
- Teaching voice: concise, ordered, technical. Prefer "what we chose and why"
  over inner monologue, self-doubt, or abandoned drafts.
- If raw thinking contradicts the final code, silently follow the code; never
  narrate the correction or replay the wrong path.
- If raw thinking is missing or noisy, reconstruct CoT from the problem and the
  final code only.
- Do not emit CUDA source, markdown fences, diffs, or a second solution.
- Do not mention this editor role, the judge, nvcc, or the data pipeline.
- Match the problem language (Chinese problem → Chinese CoT; English → English).
- Target length: 400–1200 Chinese characters or 250–800 English words; never
  exceed the stated character budget.

Structure: use exactly the six numbered headings given in the user message,
in its language, and keep each heading short. What each section covers:
- the problem: tensors/shapes, host entry, success criteria;
- the algorithm: formula, reduction/scan/gemm pattern, numerical notes;
- the mapping: index math, grid/block, why this layout;
- memory and sync: only the memory and synchronization choices present in
  the final source; if no explicit synchronization is used, say so;
- bounds and edge cases: empty n, misaligned tails, overflow;
- the checklist: 4–8 bullets that map onto the actual code (includes,
  kernel name, host launcher, key locals).
"""
)

_COT_SYSTEM_BASE_PYTHON = """You are the SFT Chain-of-Thought editor for Triton/TileLang kernels.
A compilable solution.py already exists. Rewrite teacher thinking into a
pedagogical CoT. Faithful to the FINAL Python source only. Do not emit
source, fences, or mention this editor. Match the problem language.
Headings come from the user message (program_id / T.Kernel, not CUDA C++).
If raw thinking contradicts the final source, silently follow the source.
"""

COT_SYSTEM_PROMPT = _COT_SYSTEM_BASE_CUDA + "\n" + COT_NO_REPAIR_RULES_ZH
COT_SYSTEM_PYTHON = _COT_SYSTEM_BASE_PYTHON + "\n" + COT_NO_REPAIR_RULES_EN

COT_SKELETON_ZH = (
    "1. 题意与张量/入口\n"
    "2. 算法\n"
    "3. 线程与 block 映射\n"
    "4. 存储与同步\n"
    "5. 边界与异常\n"
    "6. 实现清单"
)
COT_SKELETON_PY_ZH = (
    "1. 题意与张量/入口\n"
    "2. 算法\n"
    "3. program_id / T.Kernel 映射\n"
    "4. 存储与 mask\n"
    "5. 边界\n"
    "6. 实现清单"
)


def cot_system_for(*, dialect: str, question: str = "") -> str:
    """CoT-editor system prompt for a kernel dialect.

    Args:
        dialect: ``cuda`` / ``cutlass`` use the CUDA-C++ editor; Python
            dialects use :data:`COT_SYSTEM_PYTHON`.
        question: Raw problem; when given, the no-repair rules follow its
            language instead of the dialect default.
    """
    name = (dialect or "cuda").strip().lower()
    python = name in {"triton", "tilelang"}
    if not (question or "").strip():
        return COT_SYSTEM_PYTHON if python else COT_SYSTEM_PROMPT
    base = _COT_SYSTEM_BASE_PYTHON if python else _COT_SYSTEM_BASE_CUDA
    rules = COT_NO_REPAIR_RULES_ZH if looks_chinese(question) else COT_NO_REPAIR_RULES_EN
    return base + "\n" + rules


COT_USER_TEMPLATE = """## Problem
{question}

## Dialect
{dialect} ({language}). CoT must match this source, not another language.

Required headings:
{skeleton}

## Final source (compile-passed, do not change)
```{fence}
{code}
```

## Raw teacher thinking (may be empty, noisy, or contradictory)
{raw_reasoning}

## Output
Return ONLY the polished CoT using the six headings. No code fences. Character budget: {max_chars}.
"""


def build_cot_user_prompt(
    *,
    question: str,
    code: str,
    raw_reasoning: str,
    judge_issues: list[str] | None = None,
    judge_suggestions: list[str] | None = None,
    max_chars: int = 8000,
    dialect: str = "cuda",
    language: str = "cuda-cpp",
    skeleton: str = "",
    fence: str = "cuda",
) -> str:
    """Build the CoT-editor user message for one winning sample.

    Args:
        question: Original problem text (not the generation suffix).
        code: Compile-passed source.
        raw_reasoning: Teacher thinking, already truncated/cleaned.
        judge_issues: Ignored. Heuristic nits about the final code pull the
            editor away from a faithful first-attempt narrative.
        judge_suggestions: Ignored, see ``judge_issues``.
        max_chars: Character budget told to the editor.
        dialect: Kernel dialect id.
        language: ``cuda-cpp`` or ``python``.
        skeleton: Six-heading outline for this dialect.
        fence: Markdown fence language for the frozen source.
    """
    del judge_issues, judge_suggestions
    default_skel = (
        "1. Problem restatement\n2. Algorithm\n3. Thread/block mapping\n"
        "4. Memory and sync\n5. Bounds and edge cases\n6. Implementation checklist"
    )
    return COT_USER_TEMPLATE.format(
        question=(question or "").strip() or "(empty problem)",
        dialect=(dialect or "cuda"),
        language=(language or "cuda-cpp"),
        skeleton=(skeleton or default_skel).strip(),
        fence=(fence or "cuda"),
        code=(code or "").strip() or "(no source)",
        raw_reasoning=(raw_reasoning or "").strip()
        or "(none — derive the reasoning from the problem and the final code)",
        max_chars=int(max_chars),
    ).strip()



class SelectedPrompts(NamedTuple):
    """Prompts chosen for one candidate.

    Attributes:
        system: System prompt text.
        user: Full user message (question + suffix).
        system_index: Index into :data:`SYSTEM_PROMPTS`.
        suffix_index: Index into :data:`USER_SUFFIXES`.
    """

    system: str
    user: str
    system_index: int
    suffix_index: int


def select_system_prompt(question_id: int, candidate_idx: int = 1) -> str:
    """Pick a system prompt from :data:`SYSTEM_PROMPTS`.

    Args:
        question_id: Question id used for hashing.
        candidate_idx: Candidate number (later candidates get another variant).
    """
    idx = _stable_index(len(SYSTEM_PROMPTS), question_id, candidate_idx, salt=0)
    return SYSTEM_PROMPTS[idx]


def build_user_prompt(
    question: str,
    *,
    gpu_name: str,
    cuda_arch: str,
    cuda_version: str,
    question_id: int = 1,
    candidate_idx: int = 1,
) -> str:
    """Concatenate the raw question with a varied generation suffix.

    Args:
        question: Original problem text.
        gpu_name: Target GPU name inserted into the suffix.
        cuda_arch: e.g. ``sm_86``.
        cuda_version: e.g. ``12.6``.
        question_id: Used to pick the suffix variant.
        candidate_idx: Later candidates use a different suffix.
    """
    idx = language_matched_index(
        USER_SUFFIXES, question, question_id, candidate_idx, salt=7
    )
    suffix = USER_SUFFIXES[idx].format(
        gpu_name=gpu_name,
        cuda_arch=cuda_arch,
        cuda_version=cuda_version,
    ).strip()
    return f"{question.rstrip()}\n\n{suffix}\n\n{FEW_SHOT_CUDA}\n"


def select_prompts(
    question: str,
    *,
    question_id: int,
    candidate_idx: int,
    gpu_name: str,
    cuda_arch: str,
    cuda_version: str,
) -> SelectedPrompts:
    """Select system + user prompts for this question/candidate pair.

    Args:
        question: Original problem text.
        question_id: 1-based id.
        candidate_idx: 1-based candidate.
        gpu_name: Inserted into the user suffix.
        cuda_arch: Inserted into the user suffix.
        cuda_version: Inserted into the user suffix.
    """
    system_idx = language_matched_index(
        SYSTEM_PROMPTS, question, question_id, candidate_idx, salt=0
    )
    suffix_idx = language_matched_index(
        USER_SUFFIXES, question, question_id, candidate_idx, salt=7
    )
    suffix = USER_SUFFIXES[suffix_idx].format(
        gpu_name=gpu_name,
        cuda_arch=cuda_arch,
        cuda_version=cuda_version,
    ).strip()
    user = f"{question.rstrip()}\n\n{suffix}\n\n{FEW_SHOT_CUDA}\n"
    return SelectedPrompts(
        system=SYSTEM_PROMPTS[system_idx],
        user=user,
        system_index=system_idx,
        suffix_index=suffix_idx,
    )


def build_repair_prompt(
    *,
    cuda_arch: str,
    compile_error: str,
    previous_code: str,
    question_id: int = 1,
    candidate_idx: int = 1,
    repair_idx: int = 1,
) -> str:
    """Build a compile-fix user message (full file, not a diff).

    Args:
        cuda_arch: Target nvcc arch.
        compile_error: Truncated nvcc output.
        previous_code: Last extracted CUDA source.
        question_id: For variant selection.
        candidate_idx: For variant selection.
        repair_idx: 1-based repair round (changes wording).
    """
    error = compile_error.strip() or "(empty compiler output)"
    code = previous_code.strip() or "(no source extracted)"
    idx = _stable_index(
        len(REPAIR_PROMPTS), question_id, candidate_idx, salt=13 + int(repair_idx)
    )
    return REPAIR_PROMPTS[idx].format(
        cuda_arch=cuda_arch,
        error=error,
        code=code,
    )


__all__ = [
    "CANDIDATE_TEMPERATURES",
    "COT_SYSTEM_PROMPT",
    "REPAIR_PROMPTS",
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPTS",
    "USER_SUFFIXES",
    "SelectedPrompts",
    "build_cot_user_prompt",
    "build_repair_prompt",
    "build_user_prompt",
    "candidate_temperature",
    "format_nvcc_for_prompt",
    "select_prompts",
    "select_system_prompt",
    "truncate_compile_error",
]
