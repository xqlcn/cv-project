#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CV_PROJECT_ROOT="${CV_PROJECT_ROOT:-$ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

cd "${ROOT}"

# Step 1-8: preprocess + render 16 controlled views/mesh
python scripts/preprocess_modelnet40.py --root data/modelnet40 --split train --n-views 16 --image-size 224 "$@"

# Step 9: extract frozen CLIP features
python -m src.training.extract_features --config-name=modelnet_clip

# Step 10: train multiclass linear probe
python -m src.training.train_modelnet_probe --config-name=modelnet_clip
