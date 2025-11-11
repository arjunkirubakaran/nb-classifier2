
from typing import List, Dict, Tuple

def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    return sum(1 for a,b in zip(y_true, y_pred) if a == b) / max(1, len(y_true))

def confusion(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[Tuple[str,str], int]:
    mat = {(t,p): 0 for t in labels for p in labels}
    for t, p in zip(y_true, y_pred):
        mat[(t,p)] += 1
    return mat

def per_class_prf(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}
    for c in labels:
        tp = sum(1 for t,p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t,p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t,p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        res[c] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(1 for t in y_true if t == c)}
    macro = sum(v["f1"] for v in res.values()) / len(labels) if labels else 0.0
    micro = sum(1 for t,p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
    res["_macro"] = {"f1": macro}
    res["_micro"] = {"f1": micro}
    return res

def tsv_confusion(mat: Dict[tuple, int], labels: List[str]) -> str:
    lines = []
    header = ["true\pred"] + labels
    lines.append("\t".join(header))
    for t in labels:
        row = [t] + [str(mat[(t,p)]) for p in labels]
        lines.append("\t".join(row))
    return "\n".join(lines)
