"""Load ``.env`` settings and detect local CUDA / GPU architecture."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
    """Load project-root ``.env`` if present (does not override existing env)."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


_load_env_file()


def normalize_anthropic_base_url(url: str) -> str:
    """Anthropic SDK calls `{base_url}/v1/messages`. Strip a trailing /v1."""
    cleaned = url.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[: -len("/v1")]
    return cleaned


def _run_text(cmd: list[str]) -> str | None:
    """Run a short command and return stripped stdout, or None on failure.

    Args:
        cmd: argv to execute.
    """
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def detect_gpu_name() -> str:
    """Return the first GPU name from ``nvidia-smi``, or a generic fallback."""
    raw = _run_text(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if not raw:
        return "NVIDIA GPU"
    return raw.splitlines()[0].strip()


def detect_cuda_arch() -> str:
    """Return nvcc arch like ``sm_86`` from compute capability, default ``sm_86``."""
    raw = _run_text(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"]
    )
    if raw:
        cap = raw.splitlines()[0].strip()
        match = re.fullmatch(r"(\d+)\.(\d+)", cap)
        if match:
            return f"sm_{match.group(1)}{match.group(2)}"
    return "sm_86"


def detect_cuda_home() -> str:
    """Locate the CUDA toolkit root from env, ``nvcc`` path, or common prefixes."""
    env_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if env_home and Path(env_home).exists():
        return env_home
    nvcc = shutil.which("nvcc")
    if nvcc:
        # /usr/local/cuda-12.6/bin/nvcc -> /usr/local/cuda-12.6
        return str(Path(nvcc).resolve().parent.parent)
    for candidate in (
        "/usr/local/cuda",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12",
    ):
        if Path(candidate).exists():
            return candidate
    return "/usr/local/cuda"


def detect_cuda_version(cuda_home: str) -> str:
    """Parse ``nvcc --version`` release number (e.g. ``12.6``).

    Args:
        cuda_home: Toolkit root used to find ``bin/nvcc``.
    """
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    binary = str(nvcc) if nvcc.exists() else (shutil.which("nvcc") or "nvcc")
    raw = _run_text([binary, "--version"])
    if not raw:
        return "unknown"
    match = re.search(r"release\s+(\d+\.\d+)", raw)
    return match.group(1) if match else "unknown"


def detect_nvcc() -> str:
    """Return a path to ``nvcc``, or the string ``nvcc`` if not found."""
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    home = detect_cuda_home()
    candidate = Path(home) / "bin" / "nvcc"
    if candidate.exists():
        return str(candidate)
    return "nvcc"


def _split_api_keys(raw: str) -> list[str]:
    """Split one env value into unique keys (comma / whitespace / newline)."""
    keys: list[str] = []
    seen: set[str] = set()
    for part in re.split(r"[\s,;]+", (raw or "").strip()):
        key = part.strip()
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


class WorkerSlot(NamedTuple):
    """One worker process: which provider and which NVIDIA key to use.

    ``label`` is safe to print (``nvidia#2``); ``api_key`` is the secret.
    """

    provider: str
    api_key: str
    label: str


class Settings(BaseSettings):
    """Pipeline config loaded from environment / ``.env``.

    Provider-specific keys (OpenRouter vs NVIDIA NIM) are selected through
    :attr:`llm_provider`. Empty ``MODEL`` falls back to ``OPENROUTER_MODEL`` or
    ``NVIDIA_MODEL``.
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = Field(default="openrouter")
    llm_providers: str = Field(default="")
    workers_per_provider: int = Field(default=0, ge=0)
    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api")
    openrouter_model: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b:free")
    nvidia_api_key: str = Field(default="")
    nvidia_api_key_2: str = Field(default="")
    nvidia_api_key_3: str = Field(default="")
    nvidia_base_url: str = Field(default="https://integrate.api.nvidia.com/v1")
    nvidia_model: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b")
    model: str = Field(default="")
    thinking_level: str = Field(default="medium")
    max_input_tokens: int = Field(default=131072, ge=1024)
    max_output_tokens: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=50000, ge=1)
    top_p: float = Field(default=0.95)
    llm_timeout_sec: float = Field(default=600.0)

    max_candidates: int = Field(default=3, ge=1)
    max_repairs: int = Field(default=3, ge=0)
    workers: int = Field(default=1, ge=1)
    repair_error_max_chars: int = Field(default=6000, ge=500)
    work_keep: str = Field(default="simple")
    judge_enabled: bool = Field(default=True)
    use_judge_optimization: bool = Field(default=False)
    async_llm_enabled: bool = Field(default=True)
    async_llm_max_workers: int = Field(default=2, ge=1, le=4)
    cot_enabled: bool = Field(default=True)
    cot_agent_enabled: bool = Field(default=True)
    cot_in_assistant: bool = Field(default=True)
    cot_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    cot_max_chars: int = Field(default=8000, ge=200)
    cot_raw_max_chars: int = Field(default=24000, ge=200)
    cot_raw_store_max_chars: int = Field(default=32768, ge=0)
    cot_on_empty: str = Field(default="synthetic")
    cot_on_agent_fail: str = Field(default="raw")

    kernel_dialects: str = Field(default="cuda")
    kernel_mode: str = Field(default="single")
    kernel_dialect: str = Field(default="")
    cutlass_home: str = Field(default="/usr/local/cutlass-4.3.5")
    cutlass_cxx_std: str = Field(default="c++17")
    cutlass_thinking_level: str = Field(default="low")
    cutlass_max_output_tokens: int = Field(default=8192, ge=256)
    cutlass_reasoning_max_tokens: int = Field(default=2048, ge=0)
    triton_timeout_sec: int = Field(default=90, ge=5)
    tilelang_timeout_sec: int = Field(default=180, ge=10)

    cuda_arch: str = Field(default="")
    gpu_name: str = Field(default="")
    cuda_home: str = Field(default="")
    nvcc_timeout_sec: int = Field(default=60, ge=5)

    http_referer: str = Field(default="https://github.com/cuda-datagen")
    x_title: str = Field(default="cuda-datagen")

    questions_path: str = Field(default=str(PROJECT_ROOT / "question.jsonl"))
    data_dir: str = Field(default=str(PROJECT_ROOT / "data"))
    work_dir: str = Field(default=str(PROJECT_ROOT / "work"))

    @field_validator("openrouter_base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        """Strip a trailing ``/v1`` so the Anthropic SDK does not double it."""
        return normalize_anthropic_base_url(value)

    @field_validator("thinking_level", "cutlass_thinking_level")
    @classmethod
    def _normalize_thinking(cls, value: str) -> str:
        """Lowercase thinking level (``medium``, ``high``, ``none``, ...)."""
        return value.strip().lower()

    @field_validator("llm_provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        """Map aliases to ``openrouter`` or ``nvidia``.

        Raises:
            ValueError: If the name is not a known provider.
        """
        name = value.strip().lower()
        aliases = {
            "nim": "nvidia",
            "nv": "nvidia",
            "integrate": "nvidia",
            "or": "openrouter",
            "open-router": "openrouter",
        }
        name = aliases.get(name, name)
        if name not in {"openrouter", "nvidia"}:
            raise ValueError("LLM_PROVIDER must be 'openrouter' or 'nvidia'")
        return name

    @field_validator("work_keep")
    @classmethod
    def _normalize_work_keep(cls, value: str) -> str:
        """Map aliases to ``simple`` (last answer only) or ``detailed`` (all attempts)."""
        name = value.strip().lower()
        aliases = {
            "last": "simple",
            "final": "simple",
            "all": "detailed",
            "full": "detailed",
            "debug": "detailed",
        }
        name = aliases.get(name, name)
        if name not in {"simple", "detailed"}:
            raise ValueError("WORK_KEEP must be 'simple' or 'detailed'")
        return name

    @field_validator("kernel_mode")
    @classmethod
    def _normalize_kernel_mode(cls, value: str) -> str:
        """Map aliases to ``single`` or ``all``."""
        name = (value or "single").strip().lower()
        aliases = {
            "one": "single",
            "only": "single",
            "multi": "all",
            "each": "all",
            "every": "all",
        }
        name = aliases.get(name, name)
        if name not in {"single", "all"}:
            raise ValueError("KERNEL_MODE must be 'single' or 'all'")
        return name

    @field_validator("cutlass_cxx_std")
    @classmethod
    def _normalize_cxx_std(cls, value: str) -> str:
        """Normalize ``c++17`` / ``17`` to a ``-std`` value."""
        name = (value or "c++17").strip().lower().replace("gnu++", "c++")
        if name.isdigit():
            name = f"c++{name}"
        if name not in {"c++14", "c++17", "c++20", "c++23"}:
            raise ValueError("CUTLASS_CXX_STD must be c++17 or c++20")
        return name

    @field_validator("cot_on_empty", "cot_on_agent_fail")
    @classmethod
    def _normalize_cot_fallback(cls, value: str) -> str:
        """Map CoT fallback aliases to ``raw``, ``synthetic``, or ``empty``."""
        name = (value or "").strip().lower()
        aliases = {
            "none": "empty",
            "off": "empty",
            "skip": "empty",
            "teacher": "raw",
            "thinking": "raw",
            "rewrite": "synthetic",
            "synth": "synthetic",
        }
        name = aliases.get(name, name)
        if name not in {"raw", "synthetic", "empty"}:
            raise ValueError("CoT fallback must be 'raw', 'synthetic', or 'empty'")
        return name

    @property
    def resolved_cuda_home(self) -> str:
        """CUDA toolkit root: env override or auto-detect."""
        return self.cuda_home or detect_cuda_home()

    @property
    def resolved_cuda_arch(self) -> str:
        """nvcc ``-arch`` value, e.g. ``sm_86``."""
        return self.cuda_arch or detect_cuda_arch()

    @property
    def resolved_gpu_name(self) -> str:
        """Human-readable GPU name for prompts."""
        return self.gpu_name or detect_gpu_name()

    @property
    def resolved_cuda_version(self) -> str:
        """Toolkit version string from ``nvcc --version``."""
        return detect_cuda_version(self.resolved_cuda_home)

    @property
    def nvcc_bin(self) -> str:
        """Path to the nvcc binary."""
        home_nvcc = Path(self.resolved_cuda_home) / "bin" / "nvcc"
        if home_nvcc.exists():
            return str(home_nvcc)
        return detect_nvcc()

    def provider_pool(self) -> list[str]:
        """Providers to run in parallel, e.g. ``[nvidia, openrouter]``.

        ``LLM_PROVIDERS`` is a comma-separated list; empty falls back to
        ``LLM_PROVIDER``.
        """
        raw = (self.llm_providers or "").strip()
        if not raw:
            return [self.llm_provider]
        names: list[str] = []
        for part in raw.split(","):
            item = part.strip().lower()
            if not item:
                continue
            aliases = {
                "nim": "nvidia",
                "nv": "nvidia",
                "integrate": "nvidia",
                "or": "openrouter",
                "open-router": "openrouter",
            }
            item = aliases.get(item, item)
            if item not in {"openrouter", "nvidia"}:
                raise ValueError(f"unknown provider in LLM_PROVIDERS: {part!r}")
            if item not in names:
                names.append(item)
        return names or [self.llm_provider]

    def nvidia_api_keys(self) -> list[str]:
        """Unique NVIDIA NIM keys from ``NVIDIA_API_KEY`` / ``_2`` / ``_3``.

        Each field also accepts comma-separated keys. Order is preserved;
        empty values are skipped. At most three keys are used. Multiple
        keys are separate concurrency slots (NIM rate-limits per key).
        """
        keys: list[str] = []
        seen: set[str] = set()
        for raw in (self.nvidia_api_key, self.nvidia_api_key_2, self.nvidia_api_key_3):
            for key in _split_api_keys(raw):
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
            if len(keys) >= 3:
                break
        return keys[:3]

    def missing_provider_secrets(self, providers: list[str] | None = None) -> list[str]:
        """Env var names that are empty for the active provider pool."""
        missing: list[str] = []
        for name in providers or self.provider_pool():
            if name == "nvidia" and not self.nvidia_api_keys():
                missing.append("NVIDIA_API_KEY")
            elif name == "openrouter" and not self.openrouter_api_key.strip():
                missing.append("OPENROUTER_API_KEY")
        return missing

    def build_worker_assignments(
        self,
        *,
        workers: int,
        workers_per_provider: int = 0,
        providers: list[str] | None = None,
    ) -> list[WorkerSlot]:
        """Map each worker process to a provider and NVIDIA key.

        NVIDIA keys (``NVIDIA_API_KEY`` / ``_2`` / ``_3``) each become one
        nvidia slot. When ``workers_per_provider > 0``, every slot — each
        NVIDIA key and each other provider — gets that many processes, so
        three NVIDIA keys with ``WORKERS_PER_PROVIDER=2`` yield six nvidia
        workers (two per key).

        When ``workers_per_provider`` is 0, ``workers`` is the process
        count and NVIDIA keys are round-robin'd across nvidia workers.

        Args:
            workers: Process count used when ``workers_per_provider`` is 0.
            workers_per_provider: If >0, overrides ``workers``.
            providers: Provider names; defaults to :meth:`provider_pool`.

        Returns:
            One :class:`WorkerSlot` per worker. ``label`` is log-safe.
        """
        names = list(providers or self.provider_pool())
        nvidia_keys = self.nvidia_api_keys()
        openrouter_key = self.openrouter_api_key.strip()

        def nvidia_slots() -> list[tuple[str, str]]:
            """``(api_key, label)`` for each NVIDIA key (one empty slot if none)."""
            if not nvidia_keys:
                return [("", "nvidia")]
            if len(nvidia_keys) == 1:
                return [(nvidia_keys[0], "nvidia")]
            return [
                (key, f"nvidia#{index}")
                for index, key in enumerate(nvidia_keys, start=1)
            ]

        def slots_for(name: str) -> list[WorkerSlot]:
            if name == "nvidia":
                return [
                    WorkerSlot("nvidia", key, label) for key, label in nvidia_slots()
                ]
            key = openrouter_key if name == "openrouter" else ""
            return [WorkerSlot(name, key, name)]

        if workers_per_provider and workers_per_provider > 0:
            out: list[WorkerSlot] = []
            for name in names:
                for slot in slots_for(name):
                    out.extend([slot] * int(workers_per_provider))
            return out

        count = max(1, int(workers))
        if len(names) == 1:
            pool = slots_for(names[0])
            return [pool[i % len(pool)] for i in range(count)]

        nvidia_pool = nvidia_slots()
        nvidia_i = 0
        out = []
        for i in range(count):
            name = names[i % len(names)]
            if name == "nvidia":
                key, label = nvidia_pool[nvidia_i % len(nvidia_pool)]
                nvidia_i += 1
                out.append(WorkerSlot("nvidia", key, label))
            else:
                key = openrouter_key if name == "openrouter" else ""
                out.append(WorkerSlot(name, key, name))
        return out

    @property
    def resolved_max_output_tokens(self) -> int:
        """Max completion tokens: ``MAX_OUTPUT_TOKENS`` or fallback ``MAX_TOKENS``."""
        if self.max_output_tokens > 0:
            return self.max_output_tokens
        return self.max_tokens

    @property
    def resolved_api_key(self) -> str:
        """API key for the active :attr:`llm_provider` (first NVIDIA key)."""
        if self.llm_provider == "nvidia":
            keys = self.nvidia_api_keys()
            return keys[0] if keys else ""
        return self.openrouter_api_key.strip()

    @property
    def resolved_base_url(self) -> str:
        """HTTP base URL for the active provider."""
        if self.llm_provider == "nvidia":
            return self.nvidia_base_url.rstrip("/")
        return self.openrouter_base_url

    @property
    def resolved_model(self) -> str:
        """Model id: ``MODEL`` override, else provider-specific default."""
        if self.model.strip():
            return self.model.strip()
        if self.llm_provider == "nvidia":
            return self.nvidia_model.strip()
        return self.openrouter_model.strip()

    @property
    def data_path(self) -> Path:
        """Directory for sft/progress/abandoned jsonl and ``run.log``."""
        return Path(self.data_dir)

    @property
    def work_path(self) -> Path:
        """Scratch directory for per-attempt ``solution.cu`` files."""
        return Path(self.work_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings, filling empty CUDA fields via detection."""
    settings = Settings()
    updates: dict[str, str] = {}
    if not settings.cuda_arch:
        updates["cuda_arch"] = detect_cuda_arch()
    if not settings.gpu_name:
        updates["gpu_name"] = detect_gpu_name()
    if not settings.cuda_home:
        updates["cuda_home"] = detect_cuda_home()
    if updates:
        settings = settings.model_copy(update=updates)
    return settings
