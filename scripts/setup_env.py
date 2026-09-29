#!/usr/bin/env python3
"""Standalone launcher for dialect environment setup."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_sft.setup_env import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
