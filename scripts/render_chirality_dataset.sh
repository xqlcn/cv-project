#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CV_PROJECT_ROOT="${CV_PROJECT_ROOT:-$ROOT}"

BLENDER="${BLENDER:-blender}"

"${BLENDER}" --background --python "${ROOT}/blender/render_dataset.py" -- \
  --config "${ROOT}/configs/render_config.yaml" \
  --project-root "${ROOT}"
