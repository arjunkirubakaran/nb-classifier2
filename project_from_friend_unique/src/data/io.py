
from pathlib import Path
from typing import List, Tuple

def read_text_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return p.read_text(encoding="latin-1", errors="ignore")

def discover_labels(root: Path) -> List[str]:
    labels = [d.name for d in root.iterdir() if d.is_dir()]
    labels.sort()
    return labels

def load_labeled_docs(root: Path) -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    for label in discover_labels(root):
        sub = root / label
        for fp in sub.rglob("*"):
            if fp.is_file():
                items.append((label, read_text_file(fp)))
    return items
