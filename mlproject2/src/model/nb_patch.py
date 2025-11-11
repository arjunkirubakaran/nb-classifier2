"""
StudentMultinomialNB - Multinomial Naive Bayes with scikit-like attribute aliases.
This class trains on token lists and sets attributes main.py expects:
 - vocab_ (set of tokens)
 - classes_ (sorted list of class labels)
 - class_count_ (dict: class -> doc count)
 - class_log_prior_ (dict: class -> log prior)
 - vocab (same as vocab_) and token_logprob per class
"""
from collections import Counter
import math
from typing import List, Dict

class StudentMultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)
        # internal
        self.vocab = set()
        self.class_word_counts = {}   # class -> Counter(word -> count)
        self.class_total_words = {}   # class -> int
        self.token_logprob = {}
        self.unseen_logprob_cache = {}
        # scikit-like attributes (to be set on train)
        self.vocab_ = set()
        self.classes_ = []
        self.class_count_ = {}
        self.class_log_prior_ = {}

    def train(self, docs: List[List[str]], labels: List[str]) -> None:
        if len(docs) != len(labels):
            raise ValueError("docs and labels must have same length")
        # compute class doc counts
        class_doc_counts = Counter(labels)
        n_docs = len(labels)
        # classes
        self.classes_ = sorted(set(labels))
        # initialize counters
        self.class_word_counts = {c: Counter() for c in self.classes_}
        # build vocab and counts
        for toks, lab in zip(docs, labels):
            if lab not in self.class_word_counts:
                self.class_word_counts[lab] = Counter()
            self.class_word_counts[lab].update(toks)
            for t in toks:
                self.vocab.add(t)
        # totals
        self.class_total_words = {c: sum(self.class_word_counts[c].values()) for c in self.classes_}
        self.vocab_ = set(self.vocab)
        self.vocab = self.vocab_  # alias

        # priors
        self.class_count_ = dict(class_doc_counts)
        self.class_log_prior_ = {c: math.log(self.class_count_.get(c,0) / n_docs) for c in self.classes_}

        # token logprobs per class (Laplace smoothing)
        V = max(1, len(self.vocab_))
        self.token_logprob = {}
        for c in self.classes_:
            total = self.class_total_words.get(c, 0)
            denom = total + self.alpha * V
            base_unseen = math.log(self.alpha / denom) if denom > 0 else math.log(1e-12)
            self.unseen_logprob_cache[c] = base_unseen
            cnts = self.class_word_counts[c]
            mp = {}
            for token, cnt in cnts.items():
                mp[token] = math.log((cnt + self.alpha) / denom)
            self.token_logprob[c] = mp

    # compatibility: main.py may call predict()
    def predict(self, docs: List[List[str]]) -> List[str]:
        return self._infer(docs)

    # keep infer() name as well
    def infer(self, docs: List[List[str]]) -> List[str]:
        return self._infer(docs)

    def _infer(self, docs: List[List[str]]) -> List[str]:
        preds = []
        for toks in docs:
            best_label = None
            best_score = None
            for c in self.classes_:
                score = self.class_log_prior_.get(c, float('-inf'))
                token_log = self.token_logprob.get(c, {})
                unseen = self.unseen_logprob_cache.get(c, math.log(1e-12))
                for t in toks:
                    score += token_log.get(t, unseen)
                if best_score is None or score > best_score:
                    best_score = score
                    best_label = c
            preds.append(best_label if best_label is not None else (self.classes_[0] if self.classes_ else None))
        return preds
