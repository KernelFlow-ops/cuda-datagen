from __future__ import annotations

from typing import NamedTuple

# Fallback / first variant. Kept as SYSTEM_PROMPT for older imports.
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

编译错误：
```
{error}
```

上一版代码：
```cuda
{code}
```

只输出一个 ```cuda``` 代码块。""",
    """nvcc failed on the previous kernel (arch {cuda_arch}). Return a full corrected `solution.cu`, not a patch.
Keep the same algorithm and host responsibilities; fix compile errors only.

Compiler output:
```
{error}
```

Previous source:
```cuda
{code}
```

Reply with exactly one ```cuda fence.""",
    """请根据 nvcc 报错修复上一版实现，目标 `{cuda_arch}`。
给出整份可编译源码，不要解释，不要省略未改动的函数。

报错：
```
{error}
```

原代码：
```cuda
{code}
```

仅输出一个 ```cuda 代码块。""",
)

CANDIDATE_TEMPERATURES = (0.2, 0.5, 0.8)


class SelectedPrompts(NamedTuple):
    system: str
    user: str
    system_index: int
    suffix_index: int


def candidate_temperature(candidate_idx: int) -> float:
    if candidate_idx <= 1:
        return CANDIDATE_TEMPERATURES[0]
    if candidate_idx >= len(CANDIDATE_TEMPERATURES):
        return CANDIDATE_TEMPERATURES[-1]
    return CANDIDATE_TEMPERATURES[candidate_idx - 1]


def _stable_index(n: int, question_id: int, candidate_idx: int, salt: int) -> int:
    if n <= 0:
        raise ValueError("empty prompt pool")
    return (int(question_id) * 31 + int(candidate_idx) * 17 + salt) % n


def select_system_prompt(question_id: int, candidate_idx: int = 1) -> str:
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
    idx = _stable_index(len(USER_SUFFIXES), question_id, candidate_idx, salt=7)
    suffix = USER_SUFFIXES[idx].format(
        gpu_name=gpu_name,
        cuda_arch=cuda_arch,
        cuda_version=cuda_version,
    ).strip()
    return f"{question.rstrip()}\n\n{suffix}\n"


def select_prompts(
    question: str,
    *,
    question_id: int,
    candidate_idx: int,
    gpu_name: str,
    cuda_arch: str,
    cuda_version: str,
) -> SelectedPrompts:
    system_idx = _stable_index(len(SYSTEM_PROMPTS), question_id, candidate_idx, salt=0)
    suffix_idx = _stable_index(len(USER_SUFFIXES), question_id, candidate_idx, salt=7)
    suffix = USER_SUFFIXES[suffix_idx].format(
        gpu_name=gpu_name,
        cuda_arch=cuda_arch,
        cuda_version=cuda_version,
    ).strip()
    user = f"{question.rstrip()}\n\n{suffix}\n"
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


def truncate_compile_error(error: str, max_chars: int = 6000) -> str:
    text = (error or "").strip()
    if len(text) <= max_chars:
        return text

    lines = text.splitlines()
    important = [
        line
        for line in lines
        if any(
            token in line.lower()
            for token in ("error:", "fatal error", "undefined", "error ")
        )
    ]
    if important:
        joined = "\n".join(important)
        if len(joined) <= max_chars:
            return joined
        return joined[-max_chars:]
    return text[-max_chars:]
