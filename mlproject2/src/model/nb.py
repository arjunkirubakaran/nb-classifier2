

import math
from collections import Counter
from typing import Dict, List

class StudentMultinomialNB:
    """Multinomial Naive Bayes for text classification."""
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: List[str] = []
        self.class_log_prior_: Dict[str, float] = {}
        self.feature_log_prob_: Dict[str, Dict[str, float]] = {}
        self.class_token_totals_: Dict[str, int] = {}
        self.vocab_: set = set()

    def train(self, docs: List[List[str]], labels: List[str]) -> None:
        assert len(docs) == len(labels), "length mismatch"
        self.class_list = sorted(set(labels))
        n_docs = len(labels)

        class_counts = Counter(labels)
        for c in self.classes_:
            self.prior_log_[c] = math.log(class_counts[c] / n_docs)

        token_counts = {c: Counter() for c in self.classes_}
        for toks, y in zip(docs, labels):
            token_counts[y].update(toks)

        vocab = set()
        for c in self.classes_:
            vocab.update(token_counts[c].keys())
        self.vocab_ = vocab

        self.class_token_totals_ = {c: sum(token_counts[c].values()) for c in self.classes_}
        V = len(self.vocab_)

        for c in self.classes_:
            self.feature_log_prob_[c] = {}
            denom = self.class_token_totals_[c] + self.alpha * V
            for w in self.vocab_:
                count = token_counts[c][w]
                self.feature_log_prob_[c][w] = math.log((count + self.alpha) / denom)

    def infer(self, docs: List[List[str]]) -> List[str]:
        from collections import Counter
        preds: List[str] = []
        for toks in docs:
            counts = Counter(toks)
            best_c, best_lp = None, None
            for c in self.classes_:
                lp = self.class_log_prior_[c]
                cond = self.feature_log_prob_[c]
                for w, k in counts.items():
                    if w in cond:
                        lp += k * cond[w]
                if best_lp is None or lp > best_lp:
                    best_c, best_lp = c, lp
            preds.append(best_c if best_c is not None else self.classes_[0])
        return preds
