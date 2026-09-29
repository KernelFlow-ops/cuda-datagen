"""Shared pytest setup (T0.1). Must not import cuda_sft before env isolation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Isolation from the developer's .env and real toolchain probing (T0.2 adds the
# CUDA_SFT_NO_DOTENV switch to config._load_env_file and Settings.env_file).
os.environ["CUDA_SFT_NO_DOTENV"] = "1"
os.environ.setdefault("CUDA_ARCH", "sm_86")
os.environ.setdefault("GPU_NAME", "FakeGPU")
os.environ.setdefault("TRACE_ENABLED", "false")
for _key in list(os.environ):
    if _key.endswith("API_KEY") or "_API_KEY_" in _key or _key.endswith("API_KEYS"):
        os.environ.pop(_key)
os.environ["NVIDIA_API_KEY"] = "test-key-not-real"
os.environ["OPENROUTER_API_KEY"] = "test-key-not-real"

FIXTURES = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Fresh settings, deps and trace sink per test; no real CUDA probing."""
    from cuda_sft import config
    from cuda_sft.runtime import deps, trace

    monkeypatch.setattr(config, "detect_cuda_version", lambda *_a, **_k: "12.4", raising=False)
    monkeypatch.setattr(
        config, "detect_cuda_home", lambda *_a, **_k: str(tmp_path / "cuda"), raising=False
    )
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    config.get_settings.cache_clear()
    deps.reset()
    prev = trace.set_sink(None)
    yield
    trace.set_sink(prev)
    deps.reset()
    config.get_settings.cache_clear()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Untagged tests are unit tests."""
    layer_marks = {"unit", "component", "gpu", "e2e", "live", "load", "chaos", "dataset"}
    for item in items:
        if not any(m.name in layer_marks for m in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
