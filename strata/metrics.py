"""Ranking metrics, implemented to match `trec_eval` / `pytrec_eval` exactly.

This module exists because a retrieval number is only worth anything if it means
the same thing as everybody else's. BEIR results in the literature come from
`pytrec_eval`, which is a Python binding around NIST's `trec_eval`; if our
nDCG@10 is computed even slightly differently — a different gain function, a
different ideal ranking, a different tie-break — then comparing our table to a
published one is comparing two different quantities that happen to share a name.

So every definition below is written against `trec_eval`'s actual behaviour, and
`tests/test_metrics.py` checks them against fixtures worked out by hand.

The four decisions that people most often get wrong, and what `trec_eval` does:

1. **Gain is linear, not exponential.** `ndcg_cut` uses `rel` as the gain, not
   `2**rel - 1`. The exponential form is the Burges et al. variant and is common
   in learning-to-rank papers, but it is *not* what produced the BEIR tables.

2. **The ideal ranking is drawn from the whole qrels file, not the run.** If a
   query has 40 relevant documents and the system retrieved none of them, IDCG
   is still computed from those 40. Ranking only against what you retrieved
   turns a total miss into a perfect score.

3. **AP divides by the total number of relevant documents**, including ones
   below the cutoff and ones never retrieved — not by the number found.

4. **Unjudged documents count as zero**, they are not skipped. A run that fills
   its top 10 with unjudged documents is penalised, which is the intended
   behaviour under the standard (if debatable) closed-world assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# A run is {query_id: {doc_id: score}}; qrels is {query_id: {doc_id: relevance}}.
Run = dict[str, dict[str, float]]
Qrels = dict[str, dict[str, int]]


def rank_documents(scored: dict[str, float]) -> list[str]:
    """Order documents the way `trec_eval` does: score descending, ties broken
    by document id descending (reverse lexicographic).

    The tie-break looks arbitrary because it is — but it is *deterministic*, and
    on datasets with many identical scores (a BM25 leg where most of the pool
    scores exactly 0.0) an unstable tie-break can move nDCG by a few thousandths
    between runs. Matching trec_eval's choice keeps us comparable.
    """
    return [doc_id for doc_id, _ in
            sorted(scored.items(), key=lambda kv: (-kv[1], _desc_key(kv[0])))]


class _desc_key:
    """Sort helper: wraps a string so that it sorts in descending order."""

    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value

    def __lt__(self, other: "_desc_key") -> bool:
        return self.value > other.value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _desc_key) and self.value == other.value


def dcg(gains: list[float]) -> float:
    """Σ gain_i / log2(i + 1), one-indexed — so rank 1 is undiscounted."""
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1) if g)


def ndcg_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalised DCG at cutoff k, `trec_eval ndcg_cut_k`.

    Negative relevance levels are clamped to zero — trec_eval treats them as
    non-relevant rather than as a penalty.
    """
    gains = [max(relevance.get(doc_id, 0), 0) for doc_id in ranking[:k]]
    ideal = sorted((max(v, 0) for v in relevance.values()), reverse=True)[:k]
    denominator = dcg(ideal)
    if denominator == 0.0:
        return 0.0
    return dcg(gains) / denominator


def recall_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    """Fraction of all relevant documents that appear in the top k."""
    total = sum(1 for v in relevance.values() if v > 0)
    if total == 0:
        return 0.0
    found = sum(1 for doc_id in ranking[:k] if relevance.get(doc_id, 0) > 0)
    return found / total


def precision_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    """Precision at cutoff k. The denominator is k, not len(ranking) — a run
    that returns 3 documents for k=10 is not given credit for the 7 it skipped.
    """
    if k == 0:
        return 0.0
    found = sum(1 for doc_id in ranking[:k] if relevance.get(doc_id, 0) > 0)
    return found / k


def average_precision_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    """Average precision at cutoff k, `trec_eval map_cut_k`.

    Divides by the total relevant count for the query, so relevant documents
    that fall outside the cutoff still drag the score down. That is what makes
    MAP sensitive to the depth of the ranking in a way nDCG@10 is not.
    """
    total = sum(1 for v in relevance.values() if v > 0)
    if total == 0:
        return 0.0
    hits = 0
    running = 0.0
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if relevance.get(doc_id, 0) > 0:
            hits += 1
            running += hits / rank
    return running / total


def reciprocal_rank_at_k(ranking: list[str], relevance: dict[str, int], k: int) -> float:
    """1 / rank of the first relevant document within the top k, else 0."""
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if relevance.get(doc_id, 0) > 0:
            return 1.0 / rank
    return 0.0


@dataclass
class EvalResult:
    """Mean metrics over a query set, plus the per-query nDCG@10 for analysis."""

    n_queries: int
    ndcg: dict[int, float]
    recall: dict[int, float]
    precision: dict[int, float]
    map: dict[int, float]
    mrr: dict[int, float]
    per_query_ndcg10: dict[str, float]

    @property
    def headline(self) -> float:
        """nDCG@10 — the number BEIR tables are sorted by."""
        return self.ndcg.get(10, 0.0)

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "ndcg": {f"NDCG@{k}": round(v, 5) for k, v in sorted(self.ndcg.items())},
            "recall": {f"Recall@{k}": round(v, 5) for k, v in sorted(self.recall.items())},
            "precision": {f"P@{k}": round(v, 5) for k, v in sorted(self.precision.items())},
            "map": {f"MAP@{k}": round(v, 5) for k, v in sorted(self.map.items())},
            "mrr": {f"MRR@{k}": round(v, 5) for k, v in sorted(self.mrr.items())},
        }


DEFAULT_K = (1, 3, 5, 10, 100, 1000)


def evaluate(qrels: Qrels, run: Run, k_values: tuple[int, ...] = DEFAULT_K) -> EvalResult:
    """Score a run against qrels, averaging over the queries **in qrels**.

    Averaging over qrels rather than over the run is deliberate and matters: a
    system that silently returns nothing for a hard query should score 0 on it,
    not have it excluded from the mean. `trec_eval` does the same unless you
    pass `-c`... which is the flag nobody remembers to pass. Doing it this way
    by default removes a whole class of accidentally-flattering results.
    """
    query_ids = sorted(qrels)
    totals = {
        "ndcg": {k: 0.0 for k in k_values},
        "recall": {k: 0.0 for k in k_values},
        "precision": {k: 0.0 for k in k_values},
        "map": {k: 0.0 for k in k_values},
        "mrr": {k: 0.0 for k in k_values},
    }
    per_query_ndcg10: dict[str, float] = {}

    for query_id in query_ids:
        relevance = qrels[query_id]
        ranking = rank_documents(run.get(query_id, {}))
        for k in k_values:
            totals["ndcg"][k] += ndcg_at_k(ranking, relevance, k)
            totals["recall"][k] += recall_at_k(ranking, relevance, k)
            totals["precision"][k] += precision_at_k(ranking, relevance, k)
            totals["map"][k] += average_precision_at_k(ranking, relevance, k)
            totals["mrr"][k] += reciprocal_rank_at_k(ranking, relevance, k)
        if 10 in k_values:
            per_query_ndcg10[query_id] = ndcg_at_k(ranking, relevance, 10)

    n = len(query_ids)
    if n == 0:
        raise ValueError("qrels is empty — nothing to evaluate")

    return EvalResult(
        n_queries=n,
        ndcg={k: v / n for k, v in totals["ndcg"].items()},
        recall={k: v / n for k, v in totals["recall"].items()},
        precision={k: v / n for k, v in totals["precision"].items()},
        map={k: v / n for k, v in totals["map"].items()},
        mrr={k: v / n for k, v in totals["mrr"].items()},
        per_query_ndcg10=per_query_ndcg10,
    )


def paired_bootstrap(per_query_a: dict[str, float], per_query_b: dict[str, float],
                     *, iterations: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Paired bootstrap test on two systems' per-query scores.

    Retrieval query sets are small — SciFact has 300 queries — and a 0.01 gap in
    mean nDCG@10 across 300 queries is very often noise. Reporting a winner
    without a significance test is how ablation tables end up asserting
    improvements that would vanish on a different sample of queries.

    Returns the mean difference (b - a) and a two-sided p-value: the fraction of
    resamples in which the observed difference's sign flips or vanishes.
    """
    import numpy as np

    shared = sorted(set(per_query_a) & set(per_query_b))
    if not shared:
        return {"delta": 0.0, "p_value": 1.0, "n": 0}

    a = np.array([per_query_a[q] for q in shared], dtype=np.float64)
    b = np.array([per_query_b[q] for q in shared], dtype=np.float64)
    diff = b - a
    observed = float(diff.mean())

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(iterations, len(shared)))
    resampled = diff[idx].mean(axis=1)

    # Two-sided: how often does a resample land on the other side of zero (or
    # exactly on it) relative to the observed effect?
    if observed >= 0:
        p = float((resampled <= 0).mean()) * 2
    else:
        p = float((resampled >= 0).mean()) * 2

    return {"delta": observed, "p_value": min(p, 1.0), "n": len(shared)}
