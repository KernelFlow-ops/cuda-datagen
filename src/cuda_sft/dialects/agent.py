"""Kernel Dialect Agent: resolve .env dialects and dispatch specs."""

from __future__ import annotations

import logging
from typing import Iterable

from cuda_sft.config import Settings, get_settings
from cuda_sft.dialects.base import (
    KNOWN_DIALECTS,
    normalize_dialect_name,
)
from cuda_sft.dialects.cuda import CudaDialect
from cuda_sft.dialects.cutlass import CutlassDialect
from cuda_sft.dialects.tilelang import TileLangDialect
from cuda_sft.dialects.triton import TritonDialect

logger = logging.getLogger(__name__)


def parse_dialect_list(raw: str) -> list[str]:
    """Split a comma-separated dialect list into canonical unique names."""
    names: list[str] = []
    seen: set[str] = set()
    for part in (raw or "").split(","):
        item = part.strip()
        if not item:
            continue
        name = normalize_dialect_name(item)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


class KernelDialectAgent:
    """Registry + .env resolver for CUDA / CUTLASS 4 / Triton / TileLang."""

    def __init__(self) -> None:
        self._registry = {
            "cuda": CudaDialect(),
            "cutlass": CutlassDialect(),
            "triton": TritonDialect(),
            "tilelang": TileLangDialect(),
        }

    def spec(self, name: str):
        """Return the dialect implementation for a canonical or aliased name."""
        return self._registry[normalize_dialect_name(name)]

    def requested_names(self, settings: Settings | None = None) -> list[str]:
        """Dialect names requested by KERNEL_MODE / KERNEL_DIALECTS (not filtered)."""
        cfg = settings or get_settings()
        parsed = parse_dialect_list(cfg.kernel_dialects) or ["cuda"]
        if cfg.kernel_mode == "single":
            override = (cfg.kernel_dialect or "").strip()
            if override:
                return [normalize_dialect_name(override)]
            return parsed[:1]
        return parsed

    def resolve(self, settings: Settings | None = None) -> list:
        """Requested dialects that pass ``available()``."""
        cfg = settings or get_settings()
        out = []
        for name in self.requested_names(cfg):
            spec = self.spec(name)
            ok, reason = spec.available(cfg)
            if not ok:
                logger.warning("skipping dialect %s: %s", name, reason)
                continue
            out.append(spec)
        if not out:
            raise RuntimeError(
                "no kernel dialects available; check KERNEL_DIALECTS / CUTLASS_HOME / pip packages"
            )
        return out

    def nest_workdir(self, dialect: str, settings: Settings | None = None) -> bool:
        """True when sources live under ``work/q{id}/{dialect}/``."""
        cfg = settings or get_settings()
        names = self.requested_names(cfg)
        if cfg.kernel_mode == "all" or len(names) > 1:
            return True
        return (dialect or "cuda") != "cuda"

    def llm_call_options(self, dialect: str, settings: Settings | None = None) -> dict:
        """Per-dialect sampling overrides (CUTLASS uses a smaller thinking budget)."""
        cfg = settings or get_settings()
        name = (dialect or "cuda").strip().lower()
        if name == "cutlass":
            return {
                "thinking_level": cfg.cutlass_thinking_level or "low",
                "max_output_tokens": int(cfg.cutlass_max_output_tokens or 8192),
                "reasoning_max_tokens": int(cfg.cutlass_reasoning_max_tokens or 0),
            }
        return {}

    def expand_jobs(
        self,
        questions: Iterable[tuple[int, str]],
        settings: Settings | None = None,
    ) -> list[tuple[int, str, str]]:
        """Turn questions into ``(id, question, dialect)`` jobs."""
        specs = self.resolve(settings)
        jobs: list[tuple[int, str, str]] = []
        for question_id, question in questions:
            for spec in specs:
                jobs.append((int(question_id), question, spec.name))
        return jobs


_agent: KernelDialectAgent | None = None


def get_dialect_agent() -> KernelDialectAgent:
    """Process-wide Kernel Dialect Agent."""
    global _agent
    if _agent is None:
        _agent = KernelDialectAgent()
    return _agent


def get_spec(name: str):
    """Shortcut: dialect spec by name."""
    return get_dialect_agent().spec(name)


__all__ = [
    "KNOWN_DIALECTS",
    "KernelDialectAgent",
    "get_dialect_agent",
    "get_spec",
    "normalize_dialect_name",
    "parse_dialect_list",
]
