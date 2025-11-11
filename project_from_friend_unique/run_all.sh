#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -d data/train ] && [ -d data/test ] && [ "$(ls -A data/train 2>/dev/null || true)" ]; then
  echo "train/test exist -> skipping prepare"
else
  TAR_CAND=""
  for p in ./20_newsgroups.tar.gz ../20_newsgroups.tar.gz ~/Desktop/nb-classifier/20_newsgroups.tar.gz; do
    [ -f "$p" ] && TAR_CAND="$p" && break
  done

  if [ -n "$TAR_CAND" ]; then
    mkdir -p data/raw
    tar -xzf "$TAR_CAND" -C data/raw
  fi

  SUB=$(find data/raw -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | head -n 1 || true)
  if [ -n "$SUB" ] && [ "$SUB" != "20_newsgroups" ]; then
    mv -v "data/raw/$SUB" "data/raw/20_newsgroups" || true
  fi

  if [ -d data/raw/20_newsgroups ]; then
    python3 bin/prepare_data.py --extracted_root "$(pwd)/data/raw/20_newsgroups" --out data --seed 42 || python3 auto_split.py
  else
    python3 auto_split.py
  fi
fi

python3 main.py --data_root data --alpha 1.0 --report reports/report.md

if command -v pandoc >/dev/null 2>&1; then
  pandoc reports/report.md -o reports/report.pdf --pdf-engine=xelatex --toc || true
fi

echo "Done. report: reports/report.md"
[ -f reports/report.pdf ] && echo "PDF: reports/report.pdf"

