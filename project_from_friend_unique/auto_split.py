#!/usr/bin/env python3
import shutil, random, sys
from pathlib import Path
random.seed(42)
root = Path('data/raw')
if not root.exists():
    print("ERROR: data/raw missing. Please extract tarball into data/raw first."); sys.exit(1)
cands = [p for p in root.iterdir() if p.is_dir()]
dataset_root = cands[0] if len(cands)==1 else max(cands, key=lambda p: sum(1 for q in p.iterdir() if q.is_dir())) if cands else root
print("Using dataset root:", dataset_root)
classes = [d for d in sorted(dataset_root.iterdir()) if d.is_dir()]
if not classes:
    print("ERROR: no class subfolders found in", dataset_root); sys.exit(1)
out = Path('data'); train = out/'train'; test = out/'test'
shutil.rmtree(train, ignore_errors=True); shutil.rmtree(test, ignore_errors=True)
train.mkdir(parents=True, exist_ok=True); test.mkdir(parents=True, exist_ok=True)
for c in classes:
    files = [p for p in c.iterdir() if p.is_file()]
    random.shuffle(files)
    mid = len(files)//2
    (train/c.name).mkdir(parents=True, exist_ok=True)
    (test/c.name).mkdir(parents=True, exist_ok=True)
    for p in files[:mid]: shutil.copy2(p, train/c.name/p.name)
    for p in files[mid:]: shutil.copy2(p, test/c.name/p.name)
print("Split complete.")
print("Sample counts (first 10 classes):")
for c in sorted(list(train.iterdir())[:10]):
    print(c.name, len(list(c.iterdir())), len(list((test/c.name).iterdir())))
