"""Fixtures for the ranking metrics, worked out by hand.

Every expected value below is written as explicit arithmetic rather than as a
call into `strata.metrics`. A test that checks a function against itself proves
nothing; these check it against the definition.

The fixtures are chosen to catch the four ways an nDCG implementation usually
goes wrong — see the module docstring in `strata/metrics.py`.
"""

from math import log2

import pytest

from strata.metrics import (
    average_precision_at_k,
    evaluate,
    ndcg_at_k,
    paired_bootstrap,
    precision_at_k,
    rank_documents,
    recall_at_k,
    reciprocal_rank_at_k,
)


# --------------------------------------------------------------------------- #
# Binary relevance, relevant documents interleaved with unjudged ones.
# --------------------------------------------------------------------------- #

BINARY_QRELS = {"d1": 1, "d2": 1, "d3": 1}
BINARY_RANKING = ["d1", "x1", "d2", "x2", "d3"]


def test_ndcg_binary_relevance():
    # Gains at ranks 1..5 are 1, 0, 1, 0, 1.
    expected_dcg = 1 / log2(2) + 1 / log2(4) + 1 / log2(6)
    # Ideal ranking puts all three relevant documents first.
    expected_idcg = 1 / log2(2) + 1 / log2(3) + 1 / log2(4)
    assert ndcg_at_k(BINARY_RANKING, BINARY_QRELS, 5) == pytest.approx(
        expected_dcg / expected_idcg
    )
    assert expected_dcg / expected_idcg == pytest.approx(0.8854, abs=1e-4)


def test_unjudged_documents_count_as_zero_not_skipped():
    # If unjudged docs were skipped rather than scored zero, d2 would be treated
    # as rank 2 and d3 as rank 3, giving the ideal DCG and an nDCG of exactly 1.
    assert ndcg_at_k(BINARY_RANKING, BINARY_QRELS, 5) < 1.0


def test_recall_precision_mrr_binary():
    assert recall_at_k(BINARY_RANKING, BINARY_QRELS, 5) == pytest.approx(1.0)
    assert recall_at_k(BINARY_RANKING, BINARY_QRELS, 1) == pytest.approx(1 / 3)
    assert precision_at_k(BINARY_RANKING, BINARY_QRELS, 5) == pytest.approx(3 / 5)
    # Precision divides by k, not by the number of results returned.
    assert precision_at_k(["d1"], BINARY_QRELS, 10) == pytest.approx(0.1)
    assert reciprocal_rank_at_k(BINARY_RANKING, BINARY_QRELS, 5) == pytest.approx(1.0)
    assert reciprocal_rank_at_k(["x1", "d1"], BINARY_QRELS, 5) == pytest.approx(0.5)
    assert reciprocal_rank_at_k(["x1", "x2"], BINARY_QRELS, 5) == pytest.approx(0.0)


def test_average_precision_binary():
    # Relevant documents land at ranks 1, 3 and 5.
    expected = (1 / 1 + 2 / 3 + 3 / 5) / 3
    assert average_precision_at_k(BINARY_RANKING, BINARY_QRELS, 5) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Graded relevance — catches an exponential gain function.
# --------------------------------------------------------------------------- #

GRADED_QRELS = {"a": 2, "b": 1, "c": 2, "d": 0}
GRADED_RANKING = ["b", "d", "a"]


def test_ndcg_uses_linear_gain_not_exponential():
    expected_dcg = 1 / log2(2) + 0 / log2(3) + 2 / log2(4)
    expected_idcg = 2 / log2(2) + 2 / log2(3) + 1 / log2(4)
    expected = expected_dcg / expected_idcg
    assert ndcg_at_k(GRADED_RANKING, GRADED_QRELS, 3) == pytest.approx(expected)
    assert expected == pytest.approx(0.5317, abs=1e-4)

    # Under the 2**rel - 1 gain used by some learning-to-rank papers the same
    # ranking scores noticeably differently. Pin that down so a "harmless"
    # switch of gain function can never pass silently.
    exponential_dcg = 1 / log2(2) + 0 / log2(3) + 3 / log2(4)
    exponential_idcg = 3 / log2(2) + 3 / log2(3) + 1 / log2(4)
    assert exponential_dcg / exponential_idcg != pytest.approx(expected, abs=1e-3)


def test_judged_nonrelevant_excluded_from_recall_denominator():
    # 'd' is judged with relevance 0, so there are three relevant documents.
    assert recall_at_k(GRADED_RANKING, GRADED_QRELS, 3) == pytest.approx(2 / 3)


def test_negative_relevance_clamped_to_zero():
    # Some qrels files use -1 for "explicitly not relevant"; trec_eval treats
    # that as zero gain rather than as a penalty.
    assert ndcg_at_k(["z"], {"z": -1, "y": 1}, 10) == pytest.approx(0.0)
    assert recall_at_k(["z"], {"z": -1, "y": 1}, 10) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# The ideal ranking comes from the whole qrels file, not from the run.
# --------------------------------------------------------------------------- #

TEN_RELEVANT = {f"r{i}": 1 for i in range(1, 11)}


def test_idcg_is_built_from_all_qrels_not_just_retrieved():
    ranking = ["r1"] + [f"x{i}" for i in range(9)]
    expected_idcg = sum(1 / log2(i + 1) for i in range(1, 11))
    assert ndcg_at_k(ranking, TEN_RELEVANT, 10) == pytest.approx(1 / expected_idcg)
    assert 1 / expected_idcg == pytest.approx(0.2201, abs=1e-4)

    # The failure mode this guards against: normalising by the ideal of what was
    # actually retrieved would score this single lucky hit as a perfect ranking.
    assert ndcg_at_k(ranking, TEN_RELEVANT, 10) < 0.25


def test_complete_miss_scores_zero():
    assert ndcg_at_k(["x", "y", "z"], TEN_RELEVANT, 10) == pytest.approx(0.0)


def test_average_precision_divides_by_total_relevant():
    four_relevant = {"r1": 1, "r2": 1, "r3": 1, "r4": 1}
    ranking = ["r1", "x", "r2"]
    # Two of four relevant documents found, at ranks 1 and 3.
    assert average_precision_at_k(ranking, four_relevant, 3) == pytest.approx(
        (1 / 1 + 2 / 3) / 4
    )
    # Dividing by the number *found* would give 0.833 — a total miss on half the
    # relevant set would look like a strong result.
    assert average_precision_at_k(ranking, four_relevant, 3) < 0.5


def test_query_with_no_relevant_documents_scores_zero():
    assert ndcg_at_k(["a"], {"a": 0}, 10) == pytest.approx(0.0)
    assert recall_at_k(["a"], {"a": 0}, 10) == pytest.approx(0.0)
    assert average_precision_at_k(["a"], {"a": 0}, 10) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Ordering and aggregation
# --------------------------------------------------------------------------- #

def test_ties_break_by_descending_document_id():
    # trec_eval's tie-break. Deterministic, and it matters on a BM25 leg where
    # most of the candidate pool scores exactly zero.
    assert rank_documents({"a": 1.0, "b": 1.0, "c": 2.0}) == ["c", "b", "a"]


def test_ranking_is_score_descending():
    assert rank_documents({"a": 0.1, "b": 9.0, "c": 3.0}) == ["b", "c", "a"]


def test_missing_query_scores_zero_and_still_counts():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    run = {"q1": {"d1": 5.0}}  # nothing returned for q2 at all
    result = evaluate(qrels, run, k_values=(10,))
    assert result.n_queries == 2
    # A silently-dropped query must drag the mean down, not be excluded from it.
    assert result.ndcg[10] == pytest.approx(0.5)


def test_evaluate_averages_over_queries():
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
    run = {"q1": {"d1": 5.0}, "q2": {"x": 5.0, "d2": 1.0}}
    result = evaluate(qrels, run, k_values=(1, 10))
    assert result.ndcg[1] == pytest.approx(0.5)         # perfect, then miss
    assert result.ndcg[10] == pytest.approx((1.0 + 1 / log2(3)) / 2)
    assert result.mrr[10] == pytest.approx((1.0 + 0.5) / 2)
    assert result.headline == result.ndcg[10]


def test_evaluate_rejects_empty_qrels():
    with pytest.raises(ValueError):
        evaluate({}, {})


# --------------------------------------------------------------------------- #
# Significance testing
# --------------------------------------------------------------------------- #

def test_bootstrap_reports_no_effect_for_identical_systems():
    scores = {f"q{i}": i / 100 for i in range(100)}
    out = paired_bootstrap(scores, dict(scores), iterations=500)
    assert out["delta"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["n"] == 100


def test_bootstrap_detects_a_consistent_improvement():
    a = {f"q{i}": 0.5 for i in range(200)}
    b = {f"q{i}": 0.7 for i in range(200)}
    out = paired_bootstrap(a, b, iterations=500)
    assert out["delta"] == pytest.approx(0.2)
    assert out["p_value"] < 0.01


def test_bootstrap_handles_disjoint_query_sets():
    out = paired_bootstrap({"q1": 1.0}, {"q2": 1.0})
    assert out["n"] == 0
    assert out["p_value"] == 1.0
