#!/usr/bin/env bash
set -euo pipefail
# Conda env that contains langgraph on this machine is named "langchain".
CONDA_ENV="${CONDA_ENV:-langchain}"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
cd "$(dirname "$0")"
exec python run.py "$@"
