"""Prompts for knowledge generation, repair, LLM judge, and CoT polish."""

from __future__ import annotations

import re
from typing import NamedTuple

from cuda_sft.tasks.kinds import KNOWN_TOPICS

_CJK_RE = re.compile(r"[\u3400-\u9fff]")

CANDIDATE_TEMPERATURES = (0.2, 0.5, 0.8)

SYSTEM_PROMPTS: tuple[str, ...] = (
    (
        "You are a senior GPU architecture instructor. "
        "Answer CUDA / NVIDIA / CuTe / CUTLASS conceptual questions with precise "
        "terminology, explicit assumptions (GPU generation, CUDA version), and "
        "step-by-step reasoning. Do not emit a compilable kernel as the answer. "
        "Short pseudocode (under 15 lines) is allowed only to illustrate a mechanism."
    ),
    (
        "You teach CUDA internals and NVIDIA GPU architecture. "
        "Prefer correct hardware facts over folklore. When a number depends on "
        "the SM architecture, say so. Write structured answers with headings."
    ),
    (
        "你是 CUDA / NVIDIA GPU 体系结构教师。用准确术语讲解原理、执行模型、存储层次、"
        "CuTe layout 或公式推导。必须写明假设的 GPU 代际与 CUDA 版本。"
        "不要把答案写成可编译 kernel；允许不超过 15 行的示意伪代码。"
    ),
)

USER_SUFFIXES: tuple[str, ...] = (
    """## 答题要求（知识题，不是写算子）

目标硬件上下文（答题时作为前提写明，不是编译目标）：{gpu_name}，架构 {cuda_arch}，CUDA {cuda_version}。
主题 topic={topic}。

1. 先给简短结论，再讲机制 / 推导，最后写适用边界（哪些架构/版本不成立）。
2. 公式写出符号定义、假设、步骤、最终式。
3. 不要假装代码已经 `nvcc` 通过。不要输出完整 `solution.cu`。
4. 用 Markdown 小标题组织。匹配题面语言（中文题用中文）。""",
    """## Requirements (knowledge answer, not a kernel)

Ground the answer in this context: {gpu_name}, arch `{cuda_arch}`, CUDA {cuda_version}.
Topic: {topic}.

- Lead with the conclusion, then the mechanism or derivation, then caveats.
- Define symbols before formulas. State when a fact is architecture-specific.
- Do not ship a compilable `solution.cu`. Pseudocode ≤ 15 lines is OK.
- Match the problem language. Use Markdown headings.""",
)

TOPIC_HINTS: dict[str, str] = {
    "architecture": "Cover the relevant hardware hierarchy (thread/warp/SM/chip) and how it maps to the programming model.",
    "memory": "Trace the data path (registers / smem / L1 / L2 / DRAM). State coalescing or bank-conflict conditions.",
    "execution": "Define occupancy vs utilization. Name the limiting resource if relevant.",
    "formula": "Required: symbol table, assumptions, algebraic steps, final equation, units/complexity, domain of validity.",
    "cute": "Use CuTe vocabulary (layout, shape, stride, tiler, MMA/copy atom). Relate algebra to a kernel tiling story; do not claim the code compiled.",
    "cutlass": "Explain pipeline / collective roles and dataflow, not a CUTLASS 4 source dump.",
    "isa": "Distinguish PTX virtual ISA vs SASS. State which state spaces or constraints matter.",
    "api": "Runtime vs driver, object lifetimes, error model, stream semantics.",
    "worked_example": "A short worked numeric or layout example is the point; still explain why.",
    "general": "Be precise; mark uncertainty; do not invent SM-specific constants.",
}

REPAIR_TEMPLATE = """上一份知识答案未通过质量门闩。请重写整篇答案，不要只打补丁。
The previous knowledge answer failed the quality gate. Rewrite the full answer.

topic={topic}
hard_gate:
{gate}
must_fix:
{must_fix}
other issues:
{issues}

## Previous answer
{answer}

## 重写要求
- 修正 must_fix 与硬门闩问题。
- 保留正确部分，补全缺失推导/前提。
- 仍不要输出完整可编译 kernel。
- 匹配题面语言，使用 Markdown 小标题。
"""

JUDGE_SYSTEM = """You are a strict CUDA/NVIDIA architecture grader, not a generator.
Score the assistant answer against the question. Do not rewrite the answer.
Return ONE JSON object and nothing else. No markdown fence, no essay, no analysis.
Keys:
  "dimensions": object with integer 1-10 scores for
    factual, completeness, derivation, terminology, structure, grounding
  "must_fix": array of blocking errors (empty if none)
  "issues": array of non-blocking nits
  "notes": short string
Scoring:
- factual: hardware/API/CuTe names and invariant numbers.
- completeness: every asked sub-question.
- derivation: formula topics need assumptions → steps → result; else 10.
- terminology: occupancy vs utilization, layout vs shape, atom vs instruction.
- structure: headings, readable formulas, SFT-teacher quality.
- grounding: GPU generation / CUDA version assumptions; when the claim fails.
If the answer invents compile success, dump a full kernel, or contradicts warp=32
on NVIDIA CUDA, put that in must_fix.
"""

JUDGE_USER = """## Topic
{topic}

## Question
{question}

## Answer
{answer}

Reply with the JSON object only."""

COT_SYSTEM = """You are the SFT chain-of-thought editor for CUDA *knowledge* answers.
A structured teacher answer already exists. You rewrite raw model thinking into
a clean pedagogical CoT that a student should imitate *before* the final answer.

Rules:
- Faithful to the final answer. Do not add new hardware facts that the answer
  does not contain. If raw thinking contradicts the answer, trust the answer.
- Teaching voice, ordered, technical. No inner monologue.
- Keep formulas in the CoT as symbols/steps (preserve LaTeX ``$...$`` / ``\\frac``
  from the answer; do not drop equations). Not a second full essay.
- Do not emit a kernel, markdown fences, or mention this editor / the judge / the pipeline.
- Match the problem language.
- Use the required headings for this topic. Stay within the character budget.
"""

COT_USER = """## Problem
{question}

## Topic
{topic}

Required headings:
{skeleton}

## Final answer (already quality-gated; do not change it)
{answer}

## Raw teacher thinking (may be empty or noisy)
{raw_reasoning}

## Optional quality notes (context only)
repairs={repair_idx}; gate={gate}; judge_issues={issues}

## Output
Return ONLY the polished CoT using the required headings. No code fences. Character budget: {max_chars}.
"""

COT_SKELETONS: dict[str, str] = {
    "formula": (
        "1. 题意与要求的量\n"
        "2. 符号与假设（arch / 单位）\n"
        "3. 推导步骤\n"
        "4. 最终公式与检查（量纲/边界）\n"
        "5. 常见误区"
    ),
    "architecture": (
        "1. 题意与要解释的硬件层次\n"
        "2. 相关部件及其关系\n"
        "3. 与 CUDA 编程模型的对应\n"
        "4. 代际差异与前提\n"
        "5. 常见误区"
    ),
    "memory": (
        "1. 题意与数据路径\n"
        "2. 硬件约束（合并/bank/cache）\n"
        "3. 软件对策\n"
        "4. 适用边界\n"
        "5. 常见误区"
    ),
    "execution": (
        "1. 题意与定义（occupancy/utilization 等）\n"
        "2. 限制因素\n"
        "3. 与延迟隐藏的关系\n"
        "4. 适用边界\n"
        "5. 常见误区"
    ),
    "cute": (
        "1. 题意与 CuTe 概念\n"
        "2. layout 代数（shape/stride/tiler）\n"
        "3. 与 kernel tiling 的关系\n"
        "4. 适用边界\n"
        "5. 常见误区"
    ),
    "cutlass": (
        "1. 题意与 CUTLASS 角色\n"
        "2. 数据流 / pipeline\n"
        "3. 与 CUDA 编程模型的关系\n"
        "4. 适用边界\n"
        "5. 常见误区"
    ),
    "isa": (
        "1. 题意与 ISA 层次（PTX vs SASS）\n"
        "2. 相关指令或状态空间\n"
        "3. 约束\n"
        "4. 适用边界\n"
        "5. 常见误区"
    ),
    "api": (
        "1. 题意与 API 层次\n"
        "2. 调用约定与生命周期\n"
        "3. 错误模型 / 流语义\n"
        "4. 适用边界\n"
        "5. 常见误区"
    ),
    "worked_example": (
        "1. 题意与已知量\n"
        "2. 计算或 layout 步骤\n"
        "3. 结果\n"
        "4. 为何如此\n"
        "5. 常见误区"
    ),
    "general": (
        "1. 题意\n"
        "2. 核心机制\n"
        "3. 关键约束\n"
        "4. 适用边界\n"
        "5. 常见误区"
    ),
}

DEFAULT_SKELETON = COT_SKELETONS["general"]


class SelectedPrompts(NamedTuple):
    """System + user prompts for one knowledge candidate."""

    system: str
    user: str
    system_index: int
    suffix_index: int


def candidate_temperature(candidate_idx: int) -> float:
    """Temperature for candidate 1/2/3 (0.2 / 0.5 / 0.8)."""
    if candidate_idx <= 1:
        return CANDIDATE_TEMPERATURES[0]
    if candidate_idx >= len(CANDIDATE_TEMPERATURES):
        return CANDIDATE_TEMPERATURES[-1]
    return CANDIDATE_TEMPERATURES[candidate_idx - 1]


def cot_skeleton(topic: str) -> str:
    """Heading outline for the knowledge CoT editor."""
    name = (topic or "general").strip().lower()
    return COT_SKELETONS.get(name, DEFAULT_SKELETON)


def _looks_chinese(text: str) -> bool:
    """True when the question is primarily Chinese (CJK ideographs present)."""
    return bool(_CJK_RE.search(text or ""))


def _pick_index(question_id: int, candidate_idx: int, stride: int, n: int) -> int:
    if n <= 0:
        return 0
    return (int(question_id) * stride + int(candidate_idx) - 1) % n


def select_prompts(
    question: str,
    *,
    question_id: int,
    candidate_idx: int,
    topic: str,
    gpu_name: str,
    cuda_arch: str,
    cuda_version: str,
) -> SelectedPrompts:
    """Stable per-question variant pick, independent of kernel prompt pools.

    System/user language follows the question (CJK → Chinese prompts).
    """
    chinese = _looks_chinese(question)
    systems = [
        (i, text)
        for i, text in enumerate(SYSTEM_PROMPTS)
        if _looks_chinese(text) == chinese
    ] or list(enumerate(SYSTEM_PROMPTS))
    suffixes = [
        (i, text)
        for i, text in enumerate(USER_SUFFIXES)
        if _looks_chinese(text) == chinese
    ] or list(enumerate(USER_SUFFIXES))
    sys_index, system = systems[_pick_index(question_id, candidate_idx, 1, len(systems))]
    suf_index, suffix_tmpl = suffixes[_pick_index(question_id, candidate_idx, 3, len(suffixes))]
    topic_name = (topic or "general").strip().lower()
    if topic_name not in KNOWN_TOPICS:
        topic_name = "general"
    hint = TOPIC_HINTS.get(topic_name, TOPIC_HINTS["general"])
    suffix = suffix_tmpl.format(
        gpu_name=gpu_name or "NVIDIA GPU",
        cuda_arch=cuda_arch or "sm_86",
        cuda_version=cuda_version or "unknown",
        topic=topic_name,
    )
    user = f"{question.strip()}\n\n{hint}\n\n{suffix}"
    return SelectedPrompts(
        system=system,
        user=user,
        system_index=sys_index,
        suffix_index=suf_index,
    )


def build_repair_prompt(
    *,
    topic: str,
    answer: str,
    gate_reasons: list[str],
    must_fix: list[str],
    issues: list[str],
) -> str:
    """User turn asking for a full rewrite after a failed quality gate."""
    def _lines(items: list[str]) -> str:
        if not items:
            return "(none)"
        return "\n".join(f"- {item}" for item in items)

    return REPAIR_TEMPLATE.format(
        topic=topic or "general",
        gate=_lines(gate_reasons),
        must_fix=_lines(must_fix),
        issues=_lines(issues),
        answer=(answer or "").strip() or "(empty)",
    ).strip()


def build_judge_user(*, question: str, answer: str, topic: str) -> str:
    """User message for the JSON grader."""
    return JUDGE_USER.format(
        topic=topic or "general",
        question=(question or "").strip() or "(empty)",
        answer=(answer or "").strip() or "(empty)",
    )


def build_cot_user(
    *,
    question: str,
    answer: str,
    raw_reasoning: str,
    topic: str,
    repair_idx: int,
    gate: str,
    issues: list[str],
    max_chars: int,
) -> str:
    """User message for the knowledge CoT editor."""
    return COT_USER.format(
        question=(question or "").strip() or "(empty problem)",
        topic=topic or "general",
        skeleton=cot_skeleton(topic),
        answer=(answer or "").strip() or "(empty)",
        raw_reasoning=(raw_reasoning or "").strip() or "(none)",
        repair_idx=int(repair_idx or 0),
        gate=(gate or "").strip() or "(none)",
        issues="; ".join(issues) if issues else "(none)",
        max_chars=int(max_chars),
    ).strip()
