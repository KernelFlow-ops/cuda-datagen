#!/usr/bin/env python3
"""Project launcher for the fixed M0 benchmark harness."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cuda_sft.benchmark import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
