"""Static CUDA quality judge used after a successful ``nvcc -c``."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgeResult:
    """Heuristic quality report for one CUDA translation unit.

    Attributes:
        quality_score: 1-10 (higher is better).
        issues: Blocking or high-severity problems.
        suggestions: Optional improvements that still compiled.
        optimized_code: Lightly rewritten source, or empty if unchanged.
    """

    quality_score: int
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    optimized_code: str = ""


class CudaCodeJudge:
    """Rule-based CUDA quality checks. Conservative to limit false positives."""

    def __init__(self, settings: Any | None = None) -> None:
        """Initialize the judge.

        Args:
            settings: Unused pipeline settings; accepted for graph wiring.
        """
        self.settings = settings

    def judge(self, code: str) -> JudgeResult:
        """Score compiled CUDA source.

        Args:
            code: Extracted ``solution.cu`` text.

        Returns:
            JudgeResult with score, issues, suggestions, and optional optimizations.
        """
        source = code or ""
        lowered = source.lower()
        issues: list[str] = []
        suggestions: list[str] = []
        score = 10

        has_kernel = "__global__" in source
        has_include = bool(re.search(r"^\s*#\s*include\b", source, re.M))
        has_index = ("threadIdx" in source) or ("blockIdx" in source)
        has_bounds = bool(
            re.search(
                r"\b(if|while)\s*\([^)]{0,160}\b(idx|tid|gid|index|i|col|row|x|y|n|size|N)\b",
                source,
            )
        ) or ("<" in source and any(tok in source for tok in ("n;", "N;", "size", "numel")))
        syncthreads_count = source.count("__syncthreads")
        has_shared = "__shared__" in source
        block_lits = [
            int(m.group(1))
            for m in re.finditer(
                r"\b(?:block(?:Dim|Size)|BLOCK(?:_SIZE|DIM)|THREADS(?:_PER_BLOCK)?)\s*=\s*(\d+)",
                source,
            )
        ]
        dim3_lits = [
            int(m.group(1))
            for m in re.finditer(r"\bdim3\s+\w+\s*\(\s*(\d+)", source)
        ]
        block_sizes = block_lits + dim3_lits

        if not has_index and has_kernel:
            issues.append("kernel is missing threadIdx/blockIdx mapping")
            score -= 2

        if has_kernel and not has_bounds:
            issues.append("no obvious bounds check on thread indices")
            score -= 2

        if syncthreads_count >= 4 and not has_shared:
            suggestions.append(
                "multiple __syncthreads() without shared memory; confirm they are required"
            )
            score -= 1

        if has_kernel and ("[threadIdx.y]" in source or "[threadIdx.x * " in lowered):
            suggestions.append("verify global loads are coalesced along threadIdx.x")
            score -= 1

        for bsz in block_sizes:
            if bsz < 64 or bsz > 1024 or (bsz % 32) != 0:
                suggestions.append(
                    f"block size {bsz} is outside 64-1024 or not a multiple of 32"
                )
                score -= 1
                break

        if not has_kernel:
            issues.append("no __global__ kernel in the translation unit")
            score -= 2

        if not has_include:
            issues.append("missing #include (expected at least cuda_runtime.h)")
            score -= 1

        if has_kernel and "cudaMalloc" not in source and "<<<" in source:
            suggestions.append("host launcher present; keep error checks around launches")

        if has_shared and syncthreads_count == 0:
            issues.append("shared memory used without __syncthreads()")
            score -= 1

        if "__syncthreads()" in source:
            suggestions.append("keep __syncthreads() paired with shared-memory phases only")

        # Clamp score to [1, 10]
        score = max(1, min(10, score))

        return JudgeResult(
            quality_score=score,
            issues=issues,
            suggestions=suggestions,
            optimized_code="",
        )
