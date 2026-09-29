"""Load ``.env`` settings and detect local CUDA / GPU architecture."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
    """Load project-root ``.env`` if present (does not override existing env)."""
    if os.environ.get("CUDA_SFT_NO_DOTENV") == "1":
        return
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


def normalize_openai_base_url(url: str) -> str:
    """OpenAI SDK appends ``/chat/completions`` under ``/v1``."""
    cleaned = url.strip().rstrip("/")
    return cleaned if cleaned.endswith("/v1") else f"{cleaned}/v1"


ROLE_CONFIG_PREFIXES = {
    "generator": "generator",
    "repair.compile": "repair_compile",
    "repair.numeric": "repair_numeric",
    "repair.semantic": "repair_semantic",
    "critic": "critic",
    "cot_editor": "cot_editor",
    "refval_extract": "refval_extract",
    "knowledge_judge": "knowledge_judge",
    "knowledge_generator": "knowledge_generator",
    "knowledge_repair": "knowledge_repair",
}


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
    prefixes = [
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        "/usr/local/cuda",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12",
        "/usr/local/cuda-13",
    ]
    for candidate in prefixes:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "bin" / "nvcc").is_file():
            return str(root)
        if str(candidate).startswith("/usr/local/cuda") and root.exists():
            return str(root)
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
    for prefix in (os.environ.get("CONDA_PREFIX"), sys.prefix):
        if not prefix:
            continue
        env_nvcc = Path(prefix) / "bin" / "nvcc"
        if env_nvcc.is_file():
            return str(env_nvcc)
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

    Provider keys are shared; each agent role can override the worker's
    provider, model, and API base URL.
    """

    model_config = SettingsConfigDict(
        env_file=None if os.environ.get("CUDA_SFT_NO_DOTENV") == "1" else str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: str = Field(default="openrouter")
    llm_providers: str = Field(default="")
    workers_per_provider: int = Field(default=0, ge=0)
    openrouter_api_key: str = Field(default="")
    openrouter_model: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b:free")
    nvidia_api_key: str = Field(default="")
    nvidia_api_key_2: str = Field(default="")
    nvidia_api_key_3: str = Field(default="")
    nvidia_model: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b")
    openai_api_key: str = Field(default="")
    openai_model: str = Field(default="gpt-6-luna")
    model: str = Field(default="")
    api_base_url: str = Field(default="")
    generator_provider: str = Field(default="")
    generator_model: str = Field(default="")
    generator_api_base_url: str = Field(default="")
    generator_thinking_level: str = Field(default="")
    repair_compile_provider: str = Field(default="")
    repair_compile_model: str = Field(default="")
    repair_compile_api_base_url: str = Field(default="")
    repair_compile_thinking_level: str = Field(default="")
    repair_numeric_provider: str = Field(default="")
    repair_numeric_model: str = Field(default="")
    repair_numeric_api_base_url: str = Field(default="")
    repair_semantic_provider: str = Field(default="")
    repair_semantic_model: str = Field(default="")
    repair_semantic_api_base_url: str = Field(default="")
    critic_provider: str = Field(default="")
    critic_model: str = Field(default="")
    critic_api_base_url: str = Field(default="")
    cot_editor_provider: str = Field(default="")
    cot_editor_model: str = Field(default="")
    cot_editor_api_base_url: str = Field(default="")
    cot_editor_thinking_level: str = Field(default="")
    refval_extract_provider: str = Field(default="")
    refval_extract_model: str = Field(default="")
    refval_extract_api_base_url: str = Field(default="")
    refval_extract_thinking_level: str = Field(default="")
    knowledge_judge_provider: str = Field(default="")
    knowledge_judge_model: str = Field(default="")
    knowledge_judge_api_base_url: str = Field(default="")
    knowledge_generator_provider: str = Field(default="")
    knowledge_generator_model: str = Field(default="")
    knowledge_generator_api_base_url: str = Field(default="")
    knowledge_generator_thinking_level: str = Field(default="")
    knowledge_repair_provider: str = Field(default="")
    knowledge_repair_model: str = Field(default="")
    knowledge_repair_api_base_url: str = Field(default="")
    thinking_level: str = Field(default="medium")
    max_input_tokens: int = Field(default=131072, ge=1024)
    max_output_tokens: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=50000, ge=1)
    top_p: float = Field(default=0.95)
    llm_timeout_sec: float = Field(default=600.0)

    max_candidates: int = Field(default=3, ge=1)
    max_repairs: int = Field(default=3, ge=0)
    kernel_fast_mode: bool = Field(default=False)
    kernel_deadline_sec: int = Field(default=115, ge=1)
    workers: int = Field(default=1, ge=1)
    max_inflight_jobs: int = Field(default=2, ge=1)
    llm_concurrency: int = Field(default=2, ge=1)
    compile_concurrency: int = Field(default=1, ge=1)
    repair_error_max_chars: int = Field(default=6000, ge=500)
    work_keep: str = Field(default="simple")
    judge_enabled: bool = Field(default=True)
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
    cot_repaired_policy: str = Field(default="synthetic")
    cot_consistency_check: bool = Field(default=True)
    # Total CoT-editor LLM calls per sample: drafts, feedback retries, the
    # synthetic fallback and transport retries all share this budget.
    cot_max_calls: int = Field(default=3, ge=1, le=10)
    sft_user_is_raw_question: bool = Field(default=True)
    sft_system_mode: str = Field(default="fixed")
    kernel_llm_critic: str = Field(default="adaptive")
    critic_retry_on_error: int = Field(default=1, ge=0, le=3)
    kernel_critic_blocks_save: bool = Field(default=False)
    repair_history_mode: str = Field(default="single_turn")
    knowledge_judge_mode: str = Field(default="capped")
    difficulty_aware: bool = Field(default=True)

    task_mode: str = Field(default="kernel")
    knowledge_judge_enabled: bool = Field(default=True)
    knowledge_min_score: float = Field(default=7.0, ge=1.0, le=10.0)
    knowledge_factual_min: float = Field(default=6.0, ge=1.0, le=10.0)
    knowledge_max_candidates: int = Field(default=2, ge=1)
    knowledge_max_repairs: int = Field(default=2, ge=0)
    knowledge_min_answer_chars: int = Field(default=400, ge=50)
    knowledge_require_structure: bool = Field(default=True)
    knowledge_max_output_tokens: int = Field(default=8192, ge=256)
    knowledge_thinking_level: str = Field(default="medium")
    knowledge_on_judge_fail: str = Field(default="retry")

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

    refval_enabled: bool = Field(default=True)
    refval_timeout_sec: int = Field(default=45, ge=5)
    refval_extract_timeout_sec: int = Field(default=180, ge=5)
    refval_build_timeout_sec: int = Field(default=120, ge=5)
    refval_run_timeout_sec: int = Field(default=45, ge=5)
    oracle_retry_max: int = Field(default=2, ge=0, le=5)
    oracle_retry_backoff_s: str = Field(default="5,20")
    refval_cases: str = Field(default="standard")
    # A training release must have executable numeric evidence.  Set
    # REFVAL_STRICT=false explicitly for compile-only exploration.
    refval_strict: bool = Field(default=True)
    refval_max_elements: int = Field(default=4_000_000, ge=1024)
    refval_cache: bool = Field(default=True)
    trace_enabled: bool = Field(default=True)
    trace_dir: str = Field(default="")

    cuda_arch: str = Field(default="")
    gpu_name: str = Field(default="")
    cuda_home: str = Field(default="")
    nvcc_timeout_sec: int = Field(default=60, ge=5)

    http_referer: str = Field(default="https://github.com/cuda-datagen")
    x_title: str = Field(default="cuda-datagen")

    questions_path: str = Field(default=str(PROJECT_ROOT / "question.jsonl"))
    data_dir: str = Field(default=str(PROJECT_ROOT / "data"))
    work_dir: str = Field(default=str(PROJECT_ROOT / "work"))

    @field_validator(
        "thinking_level",
        "cutlass_thinking_level",
        "knowledge_thinking_level",
        "generator_thinking_level",
        "repair_compile_thinking_level",
        "cot_editor_thinking_level",
        "refval_extract_thinking_level",
        "knowledge_generator_thinking_level",
    )
    @classmethod
    def _normalize_thinking(cls, value: str) -> str:
        """Lowercase thinking level (``medium``, ``high``, ``none``, ...)."""
        return value.strip().lower()

    @field_validator("llm_provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        """Map aliases to a supported provider.

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
        if name not in {"openrouter", "nvidia", "openai"}:
            raise ValueError("LLM_PROVIDER must be 'openrouter', 'nvidia', or 'openai'")
        return name

    @field_validator(*(f"{prefix}_provider" for prefix in ROLE_CONFIG_PREFIXES.values()))
    @classmethod
    def _normalize_role_provider(cls, value: str) -> str:
        return cls._normalize_provider(value) if value.strip() else ""

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

    @field_validator("sft_system_mode")
    @classmethod
    def _normalize_sft_system_mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in {"fixed", "generation", "none"}:
            raise ValueError("SFT_SYSTEM_MODE must be fixed, generation, or none")
        return mode

    @field_validator("cot_repaired_policy")
    @classmethod
    def _normalize_cot_repaired_policy(cls, value: str) -> str:
        policy = value.strip().lower()
        if policy not in {"synthetic", "drop_cot", "first_turn"}:
            raise ValueError("COT_REPAIRED_POLICY must be synthetic, drop_cot, or first_turn")
        return policy

    @field_validator("repair_history_mode")
    @classmethod
    def _normalize_repair_history_mode(cls, value: str) -> str:
        mode = value.strip().lower()
        if mode not in {"single_turn", "full"}:
            raise ValueError("REPAIR_HISTORY_MODE must be single_turn or full")
        return mode

    @model_validator(mode="after")
    def _legacy_refval_timeout(self) -> Settings:
        if "USE_JUDGE_OPTIMIZATION" in os.environ:
            warnings.warn("USE_JUDGE_OPTIMIZATION is deprecated and ignored", DeprecationWarning, stacklevel=2)
        if "REFVAL_TIMEOUT_SEC" in os.environ and "REFVAL_RUN_TIMEOUT_SEC" not in os.environ:
            warnings.warn("REFVAL_TIMEOUT_SEC is deprecated; use REFVAL_RUN_TIMEOUT_SEC", DeprecationWarning, stacklevel=2)
            self.refval_run_timeout_sec = self.refval_timeout_sec
        return self

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

    @field_validator("task_mode")
    @classmethod
    def _normalize_task_mode(cls, value: str) -> str:
        """Map aliases to ``kernel``, ``knowledge``, or ``auto``."""
        name = (value or "kernel").strip().lower()
        aliases = {
            "code": "kernel",
            "impl": "kernel",
            "operator": "kernel",
            "cuda": "kernel",
            "theory": "knowledge",
            "explain": "knowledge",
            "concept": "knowledge",
            "prose": "knowledge",
            "mixed": "auto",
            "detect": "auto",
        }
        name = aliases.get(name, name)
        if name not in {"kernel", "knowledge", "auto"}:
            raise ValueError("TASK_MODE must be 'kernel', 'knowledge', or 'auto'")
        return name

    @field_validator("knowledge_on_judge_fail")
    @classmethod
    def _normalize_knowledge_judge_fail(cls, value: str) -> str:
        """Map aliases to ``retry`` or ``abandon``."""
        name = (value or "retry").strip().lower()
        aliases = {
            "again": "retry",
            "rerun": "retry",
            "skip": "abandon",
            "drop": "abandon",
            "fail": "abandon",
        }
        name = aliases.get(name, name)
        if name not in {"retry", "abandon"}:
            raise ValueError("KNOWLEDGE_ON_JUDGE_FAIL must be 'retry' or 'abandon'")
        return name

    @field_validator("refval_cases")
    @classmethod
    def _normalize_refval_cases(cls, value: str) -> str:
        """Map aliases to ``smoke`` / ``standard`` / ``full``."""
        name = (value or "standard").strip().lower()
        aliases = {
            "fast": "smoke",
            "quick": "smoke",
            "default": "standard",
            "all": "full",
        }
        name = aliases.get(name, name)
        if name not in {"smoke", "standard", "full"}:
            raise ValueError("REFVAL_CASES must be 'smoke', 'standard', or 'full'")
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
            if item not in {"openrouter", "nvidia", "openai"}:
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

    def active_llm_roles(self) -> list[str]:
        """Roles reachable under the selected task mode and feature flags."""
        roles: list[str] = []
        if self.task_mode in {"kernel", "auto"}:
            roles.extend(("generator", "repair.compile", "repair.numeric", "repair.semantic"))
            if self.refval_enabled:
                roles.append("refval_extract")
            if self.kernel_llm_critic != "off" and not self.kernel_fast_mode:
                roles.append("critic")
        if self.task_mode in {"knowledge", "auto"}:
            roles.extend(("knowledge_generator", "knowledge_repair"))
            if self.knowledge_judge_enabled:
                roles.append("knowledge_judge")
        if self.cot_enabled and self.cot_agent_enabled and (
            not self.kernel_fast_mode or self.task_mode in {"knowledge", "auto"}
        ):
            roles.append("cot_editor")
        return roles

    def missing_provider_secrets(
        self, providers: list[str] | None = None, *, roles: list[str] | None = None
    ) -> list[str]:
        """Env var names missing for providers reached by enabled roles."""
        missing: list[str] = []
        workers = list(providers or self.provider_pool())
        names: list[str] = []
        if roles is None:
            names = workers
        else:
            for worker_provider in workers:
                worker = self.model_copy(update={"llm_provider": worker_provider})
                for role in roles:
                    provider = worker.for_role(role).llm_provider
                    if provider not in names:
                        names.append(provider)
        for name in names:
            if name == "nvidia" and not self.nvidia_api_keys():
                missing.append("NVIDIA_API_KEY")
            elif name == "openrouter" and not self.openrouter_api_key.strip():
                missing.append("OPENROUTER_API_KEY")
            elif name == "openai" and not self.openai_api_key.strip():
                missing.append("OPENAI_API_KEY")
        return missing

    def for_role(self, role: str) -> Settings:
        """Resolve one role's explicit overrides against the worker defaults."""
        prefix = ROLE_CONFIG_PREFIXES.get(role)
        if prefix is None:
            return self
        provider = getattr(self, f"{prefix}_provider").strip() or self.llm_provider
        model = getattr(self, f"{prefix}_model").strip() or self.model
        base_url = getattr(self, f"{prefix}_api_base_url").strip() or self.api_base_url
        thinking_level = getattr(self, f"{prefix}_thinking_level", "").strip()
        if not thinking_level:
            if prefix.startswith("knowledge_"):
                thinking_level = self.knowledge_thinking_level
            else:
                thinking_level = self.thinking_level
        return self.model_copy(update={
            "llm_provider": provider,
            "model": model,
            "api_base_url": base_url,
            "thinking_level": thinking_level,
        })

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
        provider_keys = {
            "openrouter": openrouter_key,
            "openai": self.openai_api_key.strip(),
        }

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
            key = provider_keys.get(name, "")
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
                key = provider_keys.get(name, "")
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
        if self.llm_provider == "openai":
            return self.openai_api_key.strip()
        return self.openrouter_api_key.strip()

    @property
    def resolved_base_url(self) -> str:
        """HTTP base URL for the active provider."""
        defaults = {
            "openrouter": "https://openrouter.ai/api",
            "nvidia": "https://integrate.api.nvidia.com/v1",
            "openai": "https://www.poke2api.com",
        }
        url = self.api_base_url.strip() or defaults[self.llm_provider]
        if self.llm_provider == "openrouter":
            return normalize_anthropic_base_url(url)
        return normalize_openai_base_url(url)

    @property
    def resolved_model(self) -> str:
        """Model id: ``MODEL`` override, else provider-specific default."""
        if self.model.strip():
            return self.model.strip()
        if self.llm_provider == "nvidia":
            return self.nvidia_model.strip()
        if self.llm_provider == "openai":
            return self.openai_model.strip()
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
