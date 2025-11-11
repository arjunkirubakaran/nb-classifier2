
#!/usr/bin/env python3
import argparse, tarfile, random, shutil
from pathlib import Path
from typing import Dict, List

def list_files(root: Path):
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        for fp in d.iterdir():
            if fp.is_file():
                yield d.name, fp

def stratified_split(root: Path, out_dir: Path, seed: int = 42):
    random.seed(seed)
    train_dir = out_dir / "train"
    test_dir  = out_dir / "test"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[Path]] = {}
    for label, fp in list_files(root):
        buckets.setdefault(label, []).append(fp)

    for label, files in buckets.items():
        random.shuffle(files)
        mid = len(files) // 2
        tr, te = files[:mid], files[mid:]
        (train_dir/label).mkdir(parents=True, exist_ok=True)
        (test_dir/label).mkdir(parents=True, exist_ok=True)
        for src in tr:
            shutil.copy2(src, train_dir/label/src.name)
        for src in te:
            shutil.copy2(src, test_dir/label/src.name)
    return train_dir, test_dir

def main():
    ap = argparse.ArgumentParser(description="Unpack dataset and create stratified 50/50 train/test directories (robust).")
    ap.add_argument("--tar_gz", type=str, help="Path to 20_newsgroups.tar.gz")
    ap.add_argument("--extracted_root", type=str, default="", help="If already extracted, path to folder with 20 class subfolders.")
    ap.add_argument("--out", type=str, default="data", help="Output root to place train/ and test/")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    if args.extracted_root:
        raw_root = Path(args.extracted_root)
    else:
        assert args.tar_gz, "Provide --tar_gz or --extracted_root"
        tgz = Path(args.tar_gz)
        assert tgz.exists(), f"Not found: {tgz}"
        raw_root = Path(args.extracted_root) if args.extracted_root else (out / "raw" / "20_newsgroups")
        raw_root.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz, "r:gz") as tar:
            tar.extractall(path=out / "raw")
            print('Extracted tarball into', (out / 'raw').resolve())

    print(f"Splitting from: {raw_root.resolve()}")
    tr, te = stratified_split(raw_root, out, seed=args.seed)
    print(f"Wrote train={tr}  test={te}")

if __name__ == "__main__":
    main()
