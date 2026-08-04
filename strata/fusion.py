"""Score fusion: how to combine a lexical ranking with a vector ranking.

Two families, and they fail differently:

* **Reciprocal Rank Fusion** ignores the scores entirely and fuses *ranks*.
  It is scale-free, needs no tuning, and is very hard to break — which is why
  it is the sane default when you cannot calibrate.
* **Weighted score fusion** normalises both legs onto a comparable scale and
  interpolates with `alpha`. It can beat RRF when you tune alpha on held-out
  queries, because it keeps the margin information RRF throws away.

STRATA implements both and the evaluation harness sweeps alpha, so the choice
is a measurement rather than an opinion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scored:
    doc_id: int
    score: float
    parts: dict[str, float] = field(default_factory=dict)


def minmax(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def zscore(scores: np.ndarray) -> np.ndarray:
    std = float(scores.std())
    if std < 1e-9:
        return np.zeros_like(scores)
    return (scores - float(scores.mean())) / std


def top_k(scores: np.ndarray, k: int) -> list[int]:
    k = min(k, scores.shape[0])
    if k <= 0:
        return []
    idx = np.argpartition(-scores, k - 1)[:k]
    return [int(i) for i in idx[np.argsort(-scores[idx])]]


def reciprocal_rank_fusion(rankings: dict[str, list[int]], *, k: int = 60,
                           weights: dict[str, float] | None = None,
                           limit: int = 50) -> list[Scored]:
    """RRF: score(d) = Σ_leg w_leg / (k + rank_leg(d)).

    `k` damps the influence of the very top ranks; 60 is the constant from
    Cormack et al. and is a genuinely good default.
    """
    weights = weights or {}
    fused: dict[int, float] = {}
    parts: dict[int, dict[str, float]] = {}
    for leg, ranked in rankings.items():
        weight = weights.get(leg, 1.0)
        for rank, doc_id in enumerate(ranked, start=1):
            contribution = weight / (k + rank)
            fused[doc_id] = fused.get(doc_id, 0.0) + contribution
            parts.setdefault(doc_id, {})[leg] = contribution
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:limit]
    return [Scored(doc_id, score, parts[doc_id]) for doc_id, score in ordered]


def weighted_fusion(lexical: np.ndarray, semantic: np.ndarray, *, alpha: float = 0.5,
                    normaliser=minmax, limit: int = 50,
                    candidates: np.ndarray | None = None) -> list[Scored]:
    """score(d) = alpha·norm(vector) + (1-alpha)·norm(bm25).

    Normalisation is computed over the *candidate pool*, not the whole corpus:
    min-maxing across 3,000 documents where 2,990 score zero squashes the
    interesting range into nothing.
    """
    if candidates is None:
        pool = np.union1d(top_k(lexical, limit * 4), top_k(semantic, limit * 4))
    else:
        pool = candidates
    if pool.size == 0:
        return []
    lex = normaliser(lexical[pool])
    sem = normaliser(semantic[pool])
    combined = alpha * sem + (1.0 - alpha) * lex
    order = np.argsort(-combined)[:limit]
    return [
        Scored(
            int(pool[i]),
            float(combined[i]),
            {"vector": float(alpha * sem[i]), "bm25": float((1.0 - alpha) * lex[i])},
        )
        for i in order
    ]
