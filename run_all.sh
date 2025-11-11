#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -d data/train ] && [ -d data/test ]; then
  echo "train/test exist -> skipping prepare"
else
  if [ ! -f 20_newsgroups.tar.gz ]; then
    echo "20_newsgroups.tar.gz not found"
    exit 1
  fi
  mkdir -p data/raw
  tar -xzf 20_newsgroups.tar.gz -C data/raw || true
  SUB=$(find data/raw -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | head -n 1 || true)
  if [ -n "$SUB" ] && [ "$SUB" != "20_newsgroups" ]; then mv -v "data/raw/$SUB" "data/raw/20_newsgroups"; fi
  python3 bin/prepare_data.py --extracted_root "$(pwd)/data/raw/20_newsgroups" --out data --seed 42 || python3 auto_split.py
fi
python3 main.py --data_root data --alpha 1.0 --report reports/report.md
sed -n '1,25p' reports/report.md || true
