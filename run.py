#!/usr/bin/env python3
"""Project launcher: puts ``src/`` on ``sys.path`` then runs the CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _is_setup_cli(argv: list[str]) -> bool:
    """True for environment detect/install invocations."""
    return "--setup" in argv or "--check" in argv


def _bootstrap_core_deps() -> None:
    """Install requirements.txt if the generation CLI cannot import its stack."""
    missing: list[str] = []
    for name in ("pydantic", "pydantic_settings", "dotenv", "langgraph", "tqdm"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if not missing:
        return
    req = ROOT / "requirements.txt"
    print(f"missing Python packages ({', '.join(missing)}); pip install -r {req}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


if __name__ == "__main__":
    if _is_setup_cli(sys.argv[1:]):
        from cuda_sft.setup_env import main as setup_main

        raise SystemExit(setup_main(sys.argv[1:]))
    _bootstrap_core_deps()
    from cuda_sft.main import main

    raise SystemExit(main())
