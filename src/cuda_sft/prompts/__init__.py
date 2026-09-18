"""Shared prompt helpers (variant selection, compiler-log trimming).

Prompt *text* still lives in ``cuda_sft.prompt`` (CUDA) and each dialect /
knowledge module. This package only holds resume-stable indexing and log
formatting so dialects do not import private names from ``prompt.py``.
"""

from cuda_sft.prompts.nvcc_log import format_nvcc_for_prompt, truncate_compile_error
from cuda_sft.prompts.selection import (
    CANDIDATE_TEMPERATURES,
    candidate_temperature,
    looks_chinese,
    stable_index,
)

__all__ = [
    "CANDIDATE_TEMPERATURES",
    "candidate_temperature",
    "format_nvcc_for_prompt",
    "looks_chinese",
    "stable_index",
    "truncate_compile_error",
]
