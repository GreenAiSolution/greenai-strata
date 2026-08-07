"""Okapi BM25 over an inverted index, implemented directly on numpy arrays.

Why hand-rolled: the lexical leg of a hybrid system has to be *inspectable*.
When a hybrid ranker misfires you need to be able to say "BM25 gave this doc
7.2, and here are the four terms that produced it" — which means owning the
scoring loop.
"""

from __future__ import annotations

import numpy as np

from .stem import stem_cached
from .text import tokenize


class BM25Index:
    """Classic Okapi BM25 (Robertson/Sparck Jones) with `k1` / `b` exposed.

    Postings are stored as three flat arrays plus an offset table — the same
    layout a real inverted index uses, so scoring is a handful of vectorised
    slices rather than a Python dict walk.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75, stem: bool = False):
        self.k1 = k1
        self.b = b
        #: Porter-stem every term at index and query time, the way Lucene's
        #: EnglishAnalyzer does. Off by default: enabling it changes the
        #: postings and every IDF, so it must never turn on silently under
        #: numbers that were published without it. See strata/stem.py.
        self.stem = stem
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0, dtype=np.float32)
        self.post_doc = np.zeros(0, dtype=np.int32)   # doc id per posting
        self.post_tf = np.zeros(0, dtype=np.float32)  # term freq per posting
        self.offsets = np.zeros(1, dtype=np.int64)    # term -> posting slice
        self.doc_len = np.zeros(0, dtype=np.float32)
        self.avgdl = 0.0
        self.n_docs = 0

    def _terms(self, text: str) -> list[str]:
        """Tokenise, and stem if this index was built with stemming on.

        Index and query must go through the same function or the two vocabularies
        drift apart and queries silently stop matching — the classic analyzer
        mismatch bug, which looks like "retrieval is just bad" rather than like
        a bug.
        """
        tokens = tokenize(text)
        return [stem_cached(t) for t in tokens] if self.stem else tokens

    # ------------------------------------------------------------------ build
    def fit(self, documents: list[str]) -> "BM25Index":
        self.n_docs = len(documents)
        term_docs: dict[str, list[tuple[int, int]]] = {}
        doc_len = np.zeros(self.n_docs, dtype=np.float32)

        for doc_id, doc in enumerate(documents):
            counts: dict[str, int] = {}
            tokens = self._terms(doc)
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            doc_len[doc_id] = len(tokens)
            for term, tf in counts.items():
                term_docs.setdefault(term, []).append((doc_id, tf))

        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if self.n_docs else 0.0

        terms = sorted(term_docs)
        self.vocab = {t: i for i, t in enumerate(terms)}
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        post_doc: list[int] = []
        post_tf: list[float] = []
        idf = np.zeros(len(terms), dtype=np.float32)

        for i, term in enumerate(terms):
            postings = term_docs[term]
            offsets[i + 1] = offsets[i] + len(postings)
            for doc_id, tf in postings:
                post_doc.append(doc_id)
                post_tf.append(tf)
            df = len(postings)
            # Robertson-Sparck Jones IDF with the +0.5 smoothing that keeps
            # very common terms from going negative.
            idf[i] = np.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

        self.offsets = offsets
        self.post_doc = np.asarray(post_doc, dtype=np.int32)
        self.post_tf = np.asarray(post_tf, dtype=np.float32)
        self.idf = idf
        return self

    # ------------------------------------------------------------------ score
    def score(self, query: str) -> np.ndarray:
        """Return a dense score vector over all documents."""
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if not self.n_docs:
            return scores
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len / max(self.avgdl, 1e-9))

        seen: set[int] = set()
        for token in self._terms(query):
            term_id = self.vocab.get(token)
            if term_id is None or term_id in seen:
                continue
            seen.add(term_id)
            lo, hi = self.offsets[term_id], self.offsets[term_id + 1]
            docs = self.post_doc[lo:hi]
            tf = self.post_tf[lo:hi]
            scores[docs] += self.idf[term_id] * (tf * (self.k1 + 1.0)) / (tf + norm[docs])
        return scores

    def explain(self, query: str, doc_id: int, top: int = 6) -> list[tuple[str, float]]:
        """Per-term contribution for one document — the audit trail."""
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_id] / max(self.avgdl, 1e-9))
        out: dict[str, float] = {}
        for token in self._terms(query):
            term_id = self.vocab.get(token)
            if term_id is None or token in out:
                continue
            lo, hi = self.offsets[term_id], self.offsets[term_id + 1]
            docs = self.post_doc[lo:hi]
            hit = np.nonzero(docs == doc_id)[0]
            if hit.size:
                tf = float(self.post_tf[lo:hi][hit[0]])
                out[token] = float(self.idf[term_id] * (tf * (self.k1 + 1.0)) / (tf + norm))
        return sorted(out.items(), key=lambda kv: -kv[1])[:top]
