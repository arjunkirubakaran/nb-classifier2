
#!/usr/bin/env python3
import argparse, time, json
from pathlib import Path
from src.data.io import load_labeled_docs, discover_labels
from src.text.processing import preprocess
from src.model.nb_patch import StudentMultinomialNB
from src.eval.metrics import accuracy, confusion, per_class_prf, tsv_confusion

def to_xy(pairs):
    y = [lbl for lbl,_ in pairs]
    X = [preprocess(t) for _,t in pairs]
    return X, y

def main():
    ap = argparse.ArgumentParser(description="Train/Eval Multinomial NB on 20 Newsgroups (from scratch).")
    ap.add_argument("--data_root", type=str, default="data", help="Folder with train/ and test/ subfolders.")
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing parameter.")
    ap.add_argument("--report", type=str, default="reports/report.md", help="Where to write the report.")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    train_root = data_root / "train"
    test_root  = data_root / "test"
    assert train_root.exists() and test_root.exists(), "Expect data/train and data/test to exist. Run bin/prepare_data.py first."

    print("Loading data...")
    train_pairs = load_labeled_docs(train_root)
    test_pairs  = load_labeled_docs(test_root)
    classes = discover_labels(train_root)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train={len(train_pairs)}, Test={len(test_pairs)}")

    Xtr, ytr = to_xy(train_pairs)
    Xte, yte = to_xy(test_pairs)

    print("Training NB...")
    t0 = time.time()
    nb = StudentMultinomialNB(alpha=args.alpha)
    nb.train(Xtr, ytr)
    train_secs = time.time() - t0
    print(f"Done in {train_secs:.2f}s. | Vocab={len(nb.vocab_)}")

    print("Evaluating...")
    yhat = nb.infer(Xte)
    acc = accuracy(yte, yhat)
    mat = confusion(yte, yhat, nb.classes_)
    prf = per_class_prf(yte, yhat, nb.classes_)

    rpt = Path(args.report)
    rpt.parent.mkdir(parents=True, exist_ok=True)
    template = Path("report_template.md").read_text(encoding="utf-8")
    filled = template.format(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        alpha=args.alpha,
        vocab=len(nb.vocab_),
        ntrain=len(Xtr),
        ntest=len(Xte),
        classes=", ".join(nb.classes_),
        accuracy=f"{acc:.4f}",
        macro_f1=f"{prf['_macro']['f1']:.4f}",
        micro_f1=f"{prf['_micro']['f1']:.4f}",
        cm_tsv=tsv_confusion(mat, nb.classes_),
        per_class=json.dumps({k:v for k,v in prf.items() if not k.startswith('_')}, indent=2)
    )
    rpt.write_text(filled, encoding="utf-8")
    print(f"Wrote report to: {rpt.resolve()}")

if __name__ == "__main__":
    main()
