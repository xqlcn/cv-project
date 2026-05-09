#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CV_PROJECT_ROOT="${CV_PROJECT_ROOT:-$ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

cd "${ROOT}"
python -m src.training.train_probe --config-name=dino_probe "$@"
