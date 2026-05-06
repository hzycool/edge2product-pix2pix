#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
ARCHIVE="$DATA_DIR/edges2shoes.tar.gz"
URL="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/edges2shoes.tar.gz"

mkdir -p "$DATA_DIR"

if [ -d "$DATA_DIR/edges2shoes" ]; then
  echo "Dataset already exists at $DATA_DIR/edges2shoes"
  exit 0
fi

echo "Downloading edges2shoes dataset..."
if command -v wget >/dev/null 2>&1; then
  wget -O "$ARCHIVE" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L "$URL" -o "$ARCHIVE"
else
  echo "Neither wget nor curl is available. Please install one of them and rerun this script."
  exit 1
fi

echo "Extracting dataset..."
tar -xzf "$ARCHIVE" -C "$DATA_DIR"
echo "Dataset ready at $DATA_DIR/edges2shoes"
