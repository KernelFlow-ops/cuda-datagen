#!/usr/bin/env bash
# Detect and auto-install CUDA / CUTLASS 4.x / Triton / TileLang for cuda-datagen.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PY="${CONDA_PREFIX}/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "python3 not found; install Python 3.10+ first" >&2
  exit 1
fi

echo "setup_env: python=$PY"
exec "$PY" "$ROOT/scripts/setup_env.py" "$@"
