#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python make_subset.py \
  --dataroot ./data/edges2shoes \
  --sample_size 100 \
  --output_root ./data/edges2shoes_100

python train.py \
  --dataroot ./data/edges2shoes_100 \
  --save_dir ./outputs/edge2shoes_100 \
  --epochs 5 \
  --batch_size 1 \
  --lr 0.0002 \
  --lambda_l1 100 \
  --img_size 256 \
  --direction AtoB \
  --device auto
