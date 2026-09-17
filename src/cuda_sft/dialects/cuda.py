"""CUDA C++ dialect: existing nvcc -c pipeline."""

from __future__ import annotations

from pathlib import Path

from cuda_sft.compile import CompileResult, compile_cuda_source, smoke_compile
from cuda_sft.config import Settings
from cuda_sft.judge import CudaCodeJudge, JudgeResult
from cuda_sft.parse import extract_cuda_source
from cuda_sft.prompt import (
    SelectedPrompts,
    build_repair_prompt,
    select_prompts,
)

COT_SKELETON = """1. Problem restatement — tensors/shapes, host entry, success criteria.
2. Algorithm — formula, reduction/scan/gemm pattern, numerical notes.
3. Thread/block mapping — index math, grid/block, why this layout.
4. Memory and sync — global/shared/registers, coalescing, __syncthreads__.
5. Bounds and edge cases — empty n, misaligned tails, overflow.
6. Implementation checklist — 4–8 bullets that map onto the actual code."""


class CudaDialect:
    """Raw CUDA C++ (``solution.cu``, ``nvcc -c``)."""

    name = "cuda"
    language = "cuda-cpp"
    source_filename = "solution.cu"
    fence_langs = ("cuda", "cu", "cpp", "c++", "cc", "cxx", "c", "hpp")

    def available(self, settings: Settings) -> tuple[bool, str]:
        return True, ""

    def extract(self, text: str) -> str:
        return extract_cuda_source(text)

    def compile(self, code: str, workdir: Path, settings: Settings) -> CompileResult:
        return compile_cuda_source(code, workdir, settings=settings, dialect="cuda")

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
        return select_prompts(
            question,
            question_id=question_id,
            candidate_idx=candidate_idx,
            gpu_name=gpu_name,
            cuda_arch=cuda_arch,
            cuda_version=cuda_version,
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
        return build_repair_prompt(
            cuda_arch=cuda_arch,
            compile_error=compile_error,
            previous_code=previous_code,
            question_id=question_id,
            candidate_idx=candidate_idx,
            repair_idx=repair_idx,
        )

    def judge(self, code: str) -> JudgeResult:
        return CudaCodeJudge().judge(code)

    def smoke(self, settings: Settings, workdir: Path) -> CompileResult:
        return smoke_compile(settings, workdir)

    def cot_skeleton(self) -> str:
        return COT_SKELETON
