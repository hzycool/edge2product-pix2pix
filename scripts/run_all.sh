#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Step 1/6: checking Python dependencies."
echo "If dependencies are missing, run: pip install -r requirements.txt"

echo "Step 2/6: checking dataset."
if [ ! -d "./data/edges2shoes" ]; then
  echo "Dataset not found at ./data/edges2shoes"
  echo "Please run: bash scripts/download_edges2shoes.sh"
  exit 1
fi

echo "Step 3/6: running automated small experiment."
python run_experiment.py

echo "Step 4/6: plotting loss curve."
python utils/plot_loss.py \
  --csv_path ./outputs/edge2shoes_100/logs/loss.csv \
  --save_path ./outputs/edge2shoes_100/curves/loss_curve.png

echo "Step 5/6: compiling LaTeX report."
if ! bash scripts/compile_report.sh; then
  echo "Report compilation failed. Install TeX Live or MiKTeX, then rerun bash scripts/compile_report.sh"
fi

echo "Step 6/6: final artifacts."
echo "- Checkpoints: $ROOT_DIR/outputs/edge2shoes_100/checkpoints"
echo "- Loss CSV: $ROOT_DIR/outputs/edge2shoes_100/logs/loss.csv"
echo "- Loss curve: $ROOT_DIR/outputs/edge2shoes_100/curves/loss_curve.png"
echo "- Inference grid: $ROOT_DIR/outputs/edge2shoes_100/inference/inference_grid.png"
echo "- Metrics: $ROOT_DIR/outputs/edge2shoes_100/metrics/metrics.json"
echo "- Report PDF: $ROOT_DIR/report/report.pdf"
