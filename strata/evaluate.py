"""Measurement. Without this the rest of the repo is just plumbing.

The hard part of evaluating a retrieval system on your own corpus is getting
relevance labels without hand-labelling thousands of pairs — and hand labels
written by the same person who built the ranker are not evidence.

STRATA uses the **Inverse Cloze Task** (Lee et al., 2019) instead. Take a
sentence out of a passage, use the sentence as the query, and the passage it was
removed from is the one correct answer. The sentence is *deleted from the index*
before searching, so the system cannot win by string-matching the query back to
itself. Labels are objective, there are hundreds of them, and no system in the
ablation gets an advantage from how they were made.

Two query variants, because they stress different legs:

* `verbatim` — the sentence as written. Shares vocabulary with its passage, so
  lexical retrieval should do well. This is the easy set.
* `hard` — stopwords stripped, 45% of the remaining content words dropped at
  random, order shuffled. Most of the lexical overlap is destroyed, so this
  approximates the "user asks in their own words" case that breaks BM25.

The honest caveat, stated up front: ICT queries are sentences lifted from the
corpus, not questions typed by a human. They measure *passage discrimination*,
which is what a retriever must do, but they are not a substitute for real query
logs. Treat the numbers as a controlled comparison between systems on identical
data, not as an absolute quality score.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .corpus import Chunk, Corpus
from .fusion import reciprocal_rank_fusion, top_k, weighted_fusion
from .pipeline import SearchEngine
from .text import content_words, sentences


@dataclass
class EvalQuery:
    query: str
    target: int
    variant: str


@dataclass
class Metrics:
    n: int = 0
    recall_at: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: float = 0.0
    doc_recall_10: float = 0.0   # right *file* in the top 10, not just right chunk
    latency_ms: float = 0.0

    def row(self) -> str:
        r = self.recall_at
        return (f"{r.get(1, 0):.3f}  {r.get(5, 0):.3f}  {r.get(10, 0):.3f}  "
                f"{self.mrr:.3f}  {self.ndcg:.3f}  {self.doc_recall_10:.3f}  "
                f"{self.latency_ms:7.1f}")


# --------------------------------------------------------------------------- #
# Query-set construction
# --------------------------------------------------------------------------- #

_CODE_HINTS = ("=>", "{", "}", "</", "/>", "();", "const ", "let ", "import ",
               "function ", "className", "==", "::", "});", "]:", "|---")


def looks_like_prose(sentence: str) -> bool:
    """Reject code, tables and markup when sampling evaluation queries.

    A corpus of technical notes contains fenced source files. A JSX fragment
    makes a terrible query — it is near-identical to every other JSX fragment in
    the corpus, so the "correct" target is genuinely ambiguous and the whole
    ablation ends up measuring noise. Filtering here keeps the query set
    meaningful; it does not touch what is *indexed*.
    """
    if not sentence or any(hint in sentence for hint in _CODE_HINTS):
        return False
    letterish = sum(ch.isalpha() or ch.isspace() for ch in sentence)
    if letterish / len(sentence) < 0.86:
        return False
    words = sentence.split()
    plain = sum(1 for w in words if w.isalpha())
    return plain / max(len(words), 1) >= 0.7


def clean_query(sentence: str) -> str:
    """Strip markdown decoration so the query reads like something a user typed."""
    out = sentence.strip().lstrip("-*>#| ").replace("`", "").replace("**", "")
    out = out.replace("[[", "").replace("]]", "").replace("*", "")
    return " ".join(out.split())


def corpus_idf(corpus: Corpus) -> dict[str, float]:
    df: dict[str, int] = {}
    for chunk in corpus.chunks:
        for term in set(content_words(chunk.indexable())):
            df[term] = df.get(term, 0) + 1
    n = max(len(corpus), 1)
    return {t: math.log((1.0 + n) / (1.0 + d)) for t, d in df.items()}


def build_query_set(corpus: Corpus, *, n: int = 200, variant: str = "verbatim",
                    seed: int = 0, drop_rate: float = 0.45,
                    keywords: int = 3,
                    min_words: int = 10) -> tuple[list[EvalQuery], Corpus]:
    """Return (queries, masked_corpus).

    The masked corpus has each selected sentence removed from its chunk. Build
    the index on *that* corpus — otherwise the query is a literal substring of
    its own answer and every system scores ~1.0.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(corpus))
    idf = corpus_idf(corpus) if variant == "keyword" else {}

    queries: list[EvalQuery] = []
    removals: dict[int, list[str]] = {}

    for idx in order:
        if len(queries) >= n:
            break
        chunk = corpus[int(idx)]
        candidates = [s for s in sentences(chunk.text, min_words=min_words)
                      if len(content_words(s)) >= 5 and looks_like_prose(s)]
        if not candidates:
            continue
        sentence = candidates[int(rng.integers(len(candidates)))]

        if variant == "verbatim":
            query = clean_query(sentence)
        elif variant == "hard":
            words = content_words(sentence)
            keep_n = max(3, int(round(len(words) * (1.0 - drop_rate))))
            keep_idx = rng.choice(len(words), size=min(keep_n, len(words)),
                                  replace=False)
            kept = [words[i] for i in sorted(keep_idx)]
            rng.shuffle(kept)
            query = " ".join(kept)
        elif variant == "keyword":
            # What people actually type: two or three high-information words,
            # no syntax. Short queries are where fusion weighting bites.
            words = sorted(set(content_words(sentence)),
                           key=lambda w: -idf.get(w, 8.0))
            query = " ".join(words[:keywords])
        else:
            raise ValueError(f"unknown variant: {variant!r}")

        if len(query.split()) < 3:
            continue
        queries.append(EvalQuery(query=query, target=int(idx), variant=variant))
        removals.setdefault(int(idx), []).append(sentence)

    masked = Corpus(root=corpus.root, chunks=[])
    for chunk in corpus.chunks:
        # Flatten first. `sentences()` returns whitespace-collapsed text, so a
        # sentence spanning a line break will not match the raw chunk and the
        # removal silently no-ops — which leaks the query into its own answer
        # and inflates every number in the report. Learned the hard way; there
        # is a regression test for exactly this in tests/test_strata.py.
        text = " ".join(chunk.text.split())
        for sentence in removals.get(chunk.id, []):
            text = text.replace(sentence, " ")
        masked.chunks.append(
            Chunk(id=chunk.id, doc_id=chunk.doc_id, title=chunk.title,
                  text=" ".join(text.split()), start_line=chunk.start_line,
                  n_tokens=chunk.n_tokens, project=chunk.project)
        )
    return queries, masked


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def score_rankings(rankings: list[list[int]], targets: list[int], *,
                   cutoffs=(1, 5, 10), ndcg_at: int = 10,
                   groups: list[str] | None = None) -> Metrics:
    """Single relevant document per query, so nDCG collapses to 1/log2(rank+1).

    `groups` maps chunk index -> a coarser key (here: the source file). Chunk
    level is the strict score; document level acknowledges that a corpus with
    sibling sections has genuinely ambiguous chunk labels — retrieving "Phase 13"
    when the target was "Phase 10" of the same file is not the same kind of miss
    as returning an unrelated document.
    """
    metrics = Metrics(n=len(targets))
    hits = {c: 0 for c in cutoffs}
    mrr = ndcg = 0.0
    doc_hits = 0
    for ranked, target in zip(rankings, targets):
        if groups is not None and ranked:
            wanted = groups[target]
            if any(groups[d] == wanted for d in ranked[:10]):
                doc_hits += 1
        rank = None
        for position, doc_id in enumerate(ranked, start=1):
            if doc_id == target:
                rank = position
                break
        if rank is None:
            continue
        for c in cutoffs:
            if rank <= c:
                hits[c] += 1
        if rank <= 10:
            mrr += 1.0 / rank
        if rank <= ndcg_at:
            ndcg += 1.0 / math.log2(rank + 1)
    n = max(len(targets), 1)
    metrics.recall_at = {c: hits[c] / n for c in cutoffs}
    metrics.mrr = mrr / n
    metrics.ndcg = ndcg / n
    metrics.doc_recall_10 = doc_hits / n
    return metrics


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #

def _rank_all(engine: SearchEngine, queries: list[EvalQuery], mode: str,
              alpha: float, depth: int) -> tuple[list[list[int]], float]:
    """Rank every query under one configuration, reusing cached score vectors."""
    rankings: list[list[int]] = []
    start = time.perf_counter()
    query_vectors = engine.embedder.embed_queries([q.query for q in queries])
    for q, qv in zip(queries, query_vectors):
        lexical = engine.bm25.score(q.query)
        semantic = engine.exact.scores(qv)
        if mode == "bm25":
            rankings.append(top_k(lexical, depth))
        elif mode == "vector":
            rankings.append(top_k(semantic, depth))
        elif mode == "rrf":
            fused = reciprocal_rank_fusion(
                {"bm25": top_k(lexical, depth * 3), "vector": top_k(semantic, depth * 3)},
                limit=depth,
            )
            rankings.append([s.doc_id for s in fused])
        elif mode == "hybrid":
            pool = np.union1d(top_k(lexical, depth * 3), top_k(semantic, depth * 3))
            fused = weighted_fusion(lexical, semantic, alpha=alpha, limit=depth,
                                    candidates=pool)
            rankings.append([s.doc_id for s in fused])
        else:
            raise ValueError(mode)
    elapsed = (time.perf_counter() - start) * 1000 / max(len(queries), 1)
    return rankings, elapsed


def run_ablation(engine: SearchEngine, queries: list[EvalQuery], *,
                 alphas=(0.3, 0.5, 0.7), depth: int = 10,
                 reranker=None, rerank_depth: int = 25,
                 rerank_from: tuple[str, float] | None = None,
                 ceilings: dict[int, float] | None = None
                 ) -> dict[str, Metrics]:
    """Compare every retrieval configuration on identical queries.

    Pass a dict as `ceilings` to receive the re-ranking headroom back out:
    `{pool_depth: fraction of queries whose target reached the pool}`.
    """
    targets = [q.target for q in queries]
    groups = [c.doc_id for c in engine.corpus.chunks]
    results: dict[str, Metrics] = {}
    ceilings = {} if ceilings is None else ceilings

    for mode in ("bm25", "vector", "rrf"):
        ranked, ms = _rank_all(engine, queries, mode, 0.5, depth)
        metrics = score_rankings(ranked, targets, groups=groups)
        metrics.latency_ms = ms
        results[mode] = metrics

    best_alpha, best = None, -1.0
    for alpha in alphas:
        ranked, ms = _rank_all(engine, queries, "hybrid", alpha, depth)
        metrics = score_rankings(ranked, targets, groups=groups)
        metrics.latency_ms = ms
        results[f"hybrid α={alpha:g}"] = metrics
        if metrics.ndcg > best:
            best_alpha, best = alpha, metrics.ndcg

    if reranker is not None:
        mode, alpha = rerank_from or ("hybrid", best_alpha or 0.5)
        pooled, _ = _rank_all(engine, queries, mode, alpha, rerank_depth)

        # The ceiling: a re-ranker can only reorder what retrieval handed it.
        # If the target is missing from the pool 12% of the time, no judge —
        # however good — gets past 0.88 recall.
        ceilings[rerank_depth] = (
            sum(1 for pool, target in zip(pooled, targets) if target in pool)
            / max(len(targets), 1)
        )

        ranked_after: list[list[int]] = []
        start = time.perf_counter()
        for q, pool in zip(queries, pooled):
            payload = [
                {"doc_id": d, "title": engine.corpus[d].title,
                 "text": engine.corpus[d].text}
                for d in pool
            ]
            out = reranker.rerank(q.query, payload, top_k=depth)
            ranked_after.append([r.doc_id for r in out])
        ms = (time.perf_counter() - start) * 1000 / max(len(queries), 1)
        metrics = score_rankings(ranked_after, targets, groups=groups)
        metrics.latency_ms = ms
        label = f"{mode} α={alpha:g} + {getattr(reranker, 'name', 'rerank')}"
        results[label] = metrics

    return results


# --------------------------------------------------------------------------- #
# ANN fidelity
# --------------------------------------------------------------------------- #

def ann_recall(engine: SearchEngine, queries: list[EvalQuery], *, k: int = 10,
               efs=(32, 64, 128)) -> dict[int, dict[str, float]]:
    """How much of the exact top-k does HNSW actually return, and how fast?

    This is the number people skip. An ANN index that returns 0.72 of the exact
    neighbours has quietly capped your whole system's ceiling.
    """
    if engine.ann is None:
        return {}
    vectors = engine.embedder.embed_queries([q.query for q in queries])

    exact_sets = []
    start = time.perf_counter()
    for qv in vectors:
        exact_sets.append({doc for doc, _ in engine.exact.search(qv, k=k)})
    exact_ms = (time.perf_counter() - start) * 1000 / max(len(vectors), 1)

    out: dict[int, dict[str, float]] = {}
    for ef in efs:
        overlap = 0
        start = time.perf_counter()
        for qv, truth in zip(vectors, exact_sets):
            approx = {doc for doc, _ in engine.ann.search(qv, k=k, ef=ef)}
            overlap += len(approx & truth)
        ms = (time.perf_counter() - start) * 1000 / max(len(vectors), 1)
        out[ef] = {
            "recall": overlap / max(len(vectors) * k, 1),
            "latency_ms": ms,
            "exact_latency_ms": exact_ms,
            "speedup": exact_ms / ms if ms else float("nan"),
        }
    return out
