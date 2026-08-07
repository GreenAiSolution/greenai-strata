"""Differential test: our metrics against `pytrec_eval`, on randomised inputs.

`tests/test_metrics.py` checks the implementation against hand-worked arithmetic,
which proves it matches *our reading* of the definitions. This file checks it
against NIST's `trec_eval` (via the `pytrec_eval` binding) — the same code that
produced the numbers in the BEIR paper and on the leaderboard. If the two agree
across thousands of randomised queries, our nDCG@10 is the same quantity as
everyone else's, which is the entire point of running a public benchmark.

`pytrec_eval` needs a C++ toolchain, so it is a dev-only extra and this module
skips cleanly when it is absent:

    pip install -e '.[dev]'

The fuzz deliberately includes the cases that break naive implementations:
graded relevance, exact score ties, unjudged documents in the run, judged
documents missing from the run, and queries with no relevant documents at all.
"""

from __future__ import annotations

import pytest

from strata.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    rank_documents,
    recall_at_k,
)

pytrec_eval = pytest.importorskip(
    "pytrec_eval", reason="pytrec_eval-terrier not installed (dev extra)"
)

import numpy as np  # noqa: E402  — after the skip guard


K_VALUES = (1, 3, 5, 10, 100)


def _random_case(seed: int, *, n_queries: int = 25, n_docs: int = 60):
    """Build a (qrels, run) pair with the awkward properties deliberately baked in."""
    rng = np.random.default_rng(seed)
    docs = [f"doc{i}" for i in range(n_docs)]

    qrels: dict[str, dict[str, int]] = {}
    run: dict[str, dict[str, float]] = {}

    for q in range(n_queries):
        qid = f"q{q}"

        # Judged pool: a random subset, with graded relevance including 0s so
        # that judged-nonrelevant is exercised separately from unjudged.
        n_judged = int(rng.integers(1, 15))
        judged = rng.choice(docs, size=n_judged, replace=False)
        levels = rng.integers(0, 4, size=n_judged)
        qrels[qid] = {d: int(v) for d, v in zip(judged, levels)}

        # Retrieved pool: overlaps the judged set only partially, so the run
        # contains unjudged documents and misses some judged ones.
        n_ret = int(rng.integers(1, 40))
        retrieved = rng.choice(docs, size=n_ret, replace=False)

        # Coarse score quantisation on purpose — this manufactures exact ties,
        # which is where tie-breaking policy starts to show up in the numbers.
        scores = np.round(rng.random(n_ret) * 3, 1)
        run[qid] = {d: float(s) for d, s in zip(retrieved, scores)}

    return qrels, run


def _reference(qrels, run):
    measures = {f"ndcg_cut.{','.join(map(str, K_VALUES))}",
                f"recall.{','.join(map(str, K_VALUES))}",
                f"map_cut.{','.join(map(str, K_VALUES))}",
                f"P.{','.join(map(str, K_VALUES))}"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    return evaluator.evaluate(run)


@pytest.mark.parametrize("seed", range(12))
def test_per_query_metrics_match_pytrec_eval(seed: int):
    qrels, run = _random_case(seed)
    reference = _reference(qrels, run)

    for qid, expected in reference.items():
        ranking = rank_documents(run[qid])
        relevance = qrels[qid]
        for k in K_VALUES:
            assert ndcg_at_k(ranking, relevance, k) == pytest.approx(
                expected[f"ndcg_cut_{k}"], abs=1e-9
            ), f"nDCG@{k} diverged on {qid} (seed {seed})"
            assert recall_at_k(ranking, relevance, k) == pytest.approx(
                expected[f"recall_{k}"], abs=1e-9
            ), f"Recall@{k} diverged on {qid} (seed {seed})"
            assert average_precision_at_k(ranking, relevance, k) == pytest.approx(
                expected[f"map_cut_{k}"], abs=1e-9
            ), f"MAP@{k} diverged on {qid} (seed {seed})"
            assert precision_at_k(ranking, relevance, k) == pytest.approx(
                expected[f"P_{k}"], abs=1e-9
            ), f"P@{k} diverged on {qid} (seed {seed})"


def test_matches_on_query_with_no_relevant_documents():
    qrels = {"q1": {"a": 0, "b": 0}}
    run = {"q1": {"a": 1.0, "c": 0.5}}
    reference = _reference(qrels, run)["q1"]
    ranking = rank_documents(run["q1"])
    assert ndcg_at_k(ranking, qrels["q1"], 10) == pytest.approx(reference["ndcg_cut_10"])
    assert recall_at_k(ranking, qrels["q1"], 10) == pytest.approx(reference["recall_10"])


def test_matches_when_run_is_all_ties():
    # Every score identical: the ordering is decided entirely by the tie-break,
    # so any mismatch with trec_eval's policy shows up immediately.
    qrels = {"q1": {f"doc{i}": (i % 3) for i in range(20)}}
    run = {"q1": {f"doc{i}": 1.0 for i in range(20)}}
    reference = _reference(qrels, run)["q1"]
    ranking = rank_documents(run["q1"])
    for k in K_VALUES:
        assert ndcg_at_k(ranking, qrels["q1"], k) == pytest.approx(
            reference[f"ndcg_cut_{k}"], abs=1e-9
        )


def test_matches_on_graded_relevance_with_deep_qrels():
    # Many relevant documents, few retrieved — the case where normalising the
    # ideal ranking against the run instead of the qrels inflates the score.
    qrels = {"q1": {f"doc{i}": 3 for i in range(50)}}
    run = {"q1": {"doc0": 9.0, "unjudged": 8.0}}
    reference = _reference(qrels, run)["q1"]
    ranking = rank_documents(run["q1"])
    assert ndcg_at_k(ranking, qrels["q1"], 10) == pytest.approx(
        reference["ndcg_cut_10"], abs=1e-9
    )
    # One hit at rank 1 out of 50 equally-relevant documents: the gains are all
    # equal, so this reduces to 1 / Σ_{i=1..10} 1/log2(i+1) ≈ 0.2201 — the same
    # constant the hand-worked fixture derives.
    assert ndcg_at_k(ranking, qrels["q1"], 10) == pytest.approx(0.2201, abs=1e-4)
