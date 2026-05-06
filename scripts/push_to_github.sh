#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="edge2product-pix2pix"

if [ ! -f "README.md" ] || [ ! -d "models" ] || [ ! -d "scripts" ]; then
  echo "Please run this script from the project root directory."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is not installed or not available in PATH. Please install Git and rerun this script."
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is not installed or not available in PATH. Please install gh and rerun this script."
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI 尚未登录，请先运行 gh auth login，然后重新运行 scripts/push_to_github.sh。"
  exit 1
fi

git init
git branch -M main
git add .

if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "Initial commit: Edge2Product Pix2Pix GAN project"
fi

EXISTING_URL="$(gh repo view "$REPO_NAME" --json url -q '.url' 2>/dev/null || true)"
if [ -n "$EXISTING_URL" ]; then
  echo "GitHub repo already exists: $EXISTING_URL"
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "$EXISTING_URL"
  else
    git remote add origin "$EXISTING_URL"
  fi
  git push -u origin main
  echo "GitHub URL: $EXISTING_URL"
else
  gh repo create "$REPO_NAME" --public --source=. --remote=origin --push
  CREATED_URL="$(gh repo view "$REPO_NAME" --json url -q '.url')"
  echo "GitHub URL: $CREATED_URL"
fi
