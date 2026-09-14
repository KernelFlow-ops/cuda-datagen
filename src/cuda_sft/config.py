from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
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
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").strip()


def detect_gpu_name() -> str:
    raw = _run_text(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )
    if not raw:
        return "NVIDIA GPU"
    return raw.splitlines()[0].strip()


def detect_cuda_arch() -> str:
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
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    binary = str(nvcc) if nvcc.exists() else (shutil.which("nvcc") or "nvcc")
    raw = _run_text([binary, "--version"])
    if not raw:
        return "unknown"
    match = re.search(r"release\s+(\d+\.\d+)", raw)
    return match.group(1) if match else "unknown"


def detect_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    home = detect_cuda_home()
    candidate = Path(home) / "bin" / "nvcc"
    if candidate.exists():
        return str(candidate)
    return "nvcc"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api")
    model: str = Field(default="nvidia/nemotron-3-ultra-550b-a55b:free")
    thinking_level: str = Field(default="medium")
    max_tokens: int = Field(default=16384)
    llm_timeout_sec: float = Field(default=600.0)

    max_candidates: int = Field(default=3, ge=1)
    max_repairs: int = Field(default=3, ge=0)

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
        return normalize_anthropic_base_url(value)

    @field_validator("thinking_level")
    @classmethod
    def _normalize_thinking(cls, value: str) -> str:
        return value.strip().lower()

    @property
    def resolved_cuda_home(self) -> str:
        return self.cuda_home or detect_cuda_home()

    @property
    def resolved_cuda_arch(self) -> str:
        return self.cuda_arch or detect_cuda_arch()

    @property
    def resolved_gpu_name(self) -> str:
        return self.gpu_name or detect_gpu_name()

    @property
    def resolved_cuda_version(self) -> str:
        return detect_cuda_version(self.resolved_cuda_home)

    @property
    def nvcc_bin(self) -> str:
        home_nvcc = Path(self.resolved_cuda_home) / "bin" / "nvcc"
        if home_nvcc.exists():
            return str(home_nvcc)
        return detect_nvcc()

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def work_path(self) -> Path:
        return Path(self.work_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
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
