#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="$ROOT_DIR/report"

cd "$REPORT_DIR"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -xelatex -interaction=nonstopmode report.tex
elif command -v xelatex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode report.tex
  if command -v bibtex >/dev/null 2>&1; then
    bibtex report || true
  fi
  xelatex -interaction=nonstopmode report.tex
  xelatex -interaction=nonstopmode report.tex
else
  echo "LaTeX environment not found. Please install TeX Live or MiKTeX with XeLaTeX/latexmk."
  exit 1
fi

if [ -f "$REPORT_DIR/report.pdf" ]; then
  echo "Compiled report: $REPORT_DIR/report.pdf"
else
  echo "Compilation finished but report.pdf was not found."
  exit 1
fi
