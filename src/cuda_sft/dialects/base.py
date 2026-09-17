"""Dialect protocol and name aliases."""

from __future__ import annotations

from typing import Any, Protocol

from cuda_sft.compile import CompileResult
from cuda_sft.config import Settings
from cuda_sft.judge import JudgeResult
from cuda_sft.prompt import SelectedPrompts

DIALECT_ALIASES = {
    "cu": "cuda",
    "cuda-cpp": "cuda",
    "cuda_cpp": "cuda",
    "cute": "cutlass",
    "cutlass/cute": "cutlass",
    "cutlass_cute": "cutlass",
    "cutlass4": "cutlass",
    "cutlass4.x": "cutlass",
    "cutlass-4": "cutlass",
    "python-triton": "triton",
    "triton-lang": "triton",
    "tl": "tilelang",
    "tile-lang": "tilelang",
    "tile_lang": "tilelang",
}

KNOWN_DIALECTS = ("cuda", "cutlass", "triton", "tilelang")


def normalize_dialect_name(raw: str) -> str:
    """Map aliases to a canonical dialect id.

    Raises:
        ValueError: Unknown dialect name.
    """
    name = (raw or "").strip().lower().replace(" ", "")
    name = DIALECT_ALIASES.get(name, name)
    if name not in KNOWN_DIALECTS:
        raise ValueError(
            f"unknown kernel dialect {raw!r}; expected one of {', '.join(KNOWN_DIALECTS)}"
        )
    return name


class DialectSpec(Protocol):
    """One kernel language backend managed by KernelDialectAgent."""

    name: str
    language: str
    source_filename: str
    fence_langs: tuple[str, ...]

    def available(self, settings: Settings) -> tuple[bool, str]:
        """Return ``(True, '')`` or ``(False, reason)`` if the toolchain is missing."""
        ...

    def extract(self, text: str) -> str:
        """Pull source out of a model reply."""
        ...

    def compile(
        self,
        code: str,
        workdir: Any,
        settings: Settings,
    ) -> CompileResult:
        """Compile-only gate for this dialect."""
        ...

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
        """System + user prompts for one candidate."""
        ...

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
        """Compile-fix user message."""
        ...

    def judge(self, code: str) -> JudgeResult:
        """Heuristic quality report."""
        ...

    def smoke(self, settings: Settings, workdir: Any) -> CompileResult:
        """Built-in compile smoke for ``--dry-compile``."""
        ...

    def cot_skeleton(self) -> str:
        """Six-heading CoT outline for the editor agent."""
        ...
