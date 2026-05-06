#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python evaluate.py \
  --generated_dir ./outputs/edge2shoes_100/inference/generated \
  --target_dir ./outputs/edge2shoes_100/inference/target \
  --save_path ./outputs/edge2shoes_100/metrics/metrics.json
