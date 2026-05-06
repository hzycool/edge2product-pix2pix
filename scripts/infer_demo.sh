#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python infer.py \
  --dataroot ./data/edges2shoes_100 \
  --checkpoint ./outputs/edge2shoes_100/checkpoints/latest_G.pth \
  --save_dir ./outputs/edge2shoes_100/inference \
  --img_size 256 \
  --direction AtoB \
  --device auto \
  --num_images 20
