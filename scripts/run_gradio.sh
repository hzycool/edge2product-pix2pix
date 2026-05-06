#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python demo_gradio.py \
  --checkpoint ./outputs/edge2shoes_100/checkpoints/latest_G.pth \
  --device cpu
