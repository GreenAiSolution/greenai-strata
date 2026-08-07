"""Guards on the transcribed published baselines.

These numbers are typed in by hand from two papers, so the failure mode is a
transcription slip that quietly changes what we claim to have reproduced. The
tests below pin the values this project actually cites in its writeup, and check
the structural properties that would reveal a column having been pasted twice or
shifted by a row.
"""

from __future__ import annotations

import pytest

from strata.reference import (
    BASELINES,
    CITATIONS,
    KAMALLOO_FLAT,
    KAMALLOO_MULTIFIELD,
    PRIMARY,
    THAKUR_2021,
    comparison_table,
    published,
    spread,
)


def test_thakur_values_cited_in_the_writeup():
    # Thakur et al. 2021, Table 2, BM25 column.
    assert THAKUR_2021["scifact"] == 0.665
    assert THAKUR_2021["nfcorpus"] == 0.325
    assert THAKUR_2021["fiqa"] == 0.236
    assert THAKUR_2021["scidocs"] == 0.158
    assert THAKUR_2021["arguana"] == 0.315


def test_kamalloo_values_cited_in_the_writeup():
    # Kamalloo et al. 2024, Table 3.
    assert KAMALLOO_MULTIFIELD["arguana"] == 0.414
    assert KAMALLOO_FLAT["arguana"] == 0.397
    assert KAMALLOO_FLAT["scifact"] == 0.679
    assert KAMALLOO_FLAT["scidocs"] == 0.149
    assert KAMALLOO_FLAT["nfcorpus"] == 0.322
    assert KAMALLOO_FLAT["fiqa"] == 0.236


def test_the_arguana_disagreement_is_real_and_large():
    # This single dataset is why the module refuses to collapse the references
    # into one "BM25 baseline" column: the published values differ by ~0.1,
    # which is larger than almost any improvement claimed over BM25.
    assert spread("arguana") == pytest.approx(0.099, abs=1e-6)
    assert spread("arguana") > spread("scifact")


def test_columns_are_not_accidental_duplicates():
    # A copy-paste of one column over another would make these identical.
    assert KAMALLOO_FLAT != KAMALLOO_MULTIFIELD
    assert THAKUR_2021 != KAMALLOO_FLAT
    differing = [d for d in KAMALLOO_FLAT
                 if KAMALLOO_FLAT[d] != KAMALLOO_MULTIFIELD.get(d)]
    assert len(differing) >= 8


def test_thakur_covers_all_eighteen_beir_datasets_plus_msmarco():
    assert len(THAKUR_2021) == 19          # 18 BEIR tasks + in-domain MS MARCO
    assert "msmarco" in THAKUR_2021


def test_every_score_is_a_plausible_ndcg():
    for source, table in BASELINES.items():
        for dataset, value in table.items():
            assert 0.0 < value < 1.0, f"{source}/{dataset} = {value}"


def test_every_source_is_cited():
    from strata.reference import DENSE_BASELINES

    assert set(CITATIONS) == set(BASELINES) | set(DENSE_BASELINES)
    assert PRIMARY in BASELINES
    # STRATA concatenates title and body, so the flat variant is the honest
    # like-for-like reference rather than the more flattering multifield one.
    assert PRIMARY == "kamalloo-flat"


def test_published_returns_none_for_unknown_datasets():
    assert published("not-a-dataset") is None
    assert published("scifact", source="not-a-source") is None
    assert spread("not-a-dataset") is None


def test_comparison_table_reports_the_gap_and_names_its_sources():
    rendered = comparison_table({"scifact": 0.6613, "arguana": 0.4204})
    assert "scifact" in rendered and "arguana" in rendered
    assert "+0.0234" in rendered or "-0.0177" in rendered   # deltas vs flat
    assert "mean |Δ| vs flat" in rendered
    # The BM25 comparison table cites the BM25 sources; the dense baseline is
    # a separate registry and deliberately does not appear here.
    for source in BASELINES:
        assert CITATIONS[source] in rendered


def test_comparison_table_handles_a_dataset_with_no_published_number():
    rendered = comparison_table({"scifact": 0.66, "made-up": 0.5})
    assert "made-up" in rendered
    assert "—" in rendered


# --------------------------------------------------------------------------- #
# Dense baselines
# --------------------------------------------------------------------------- #

def test_bge_values_match_the_official_model_card():
    # Read from the machine-readable model-index at
    # huggingface.co/BAAI/bge-base-en-v1.5. Percentages there, fractions here.
    from strata.reference import BGE_BASE_EN_V15, published_dense

    assert BGE_BASE_EN_V15["nfcorpus"] == 0.37389
    assert BGE_BASE_EN_V15["scifact"] == 0.74039
    assert BGE_BASE_EN_V15["arguana"] == 0.63605
    assert BGE_BASE_EN_V15["scidocs"] == 0.21731
    assert BGE_BASE_EN_V15["fiqa"] == 0.40646
    assert published_dense("scifact") == 0.74039
    assert published_dense("scifact", model="not-a-model") is None
    assert published_dense("not-a-dataset") is None


def test_dense_and_lexical_baselines_stay_in_separate_registries():
    # Comparing a lexical run against a neural reference by accident is the
    # most common way a retrieval table misleads. Keep them apart.
    from strata.reference import BASELINES, DENSE_BASELINES

    assert set(BASELINES) & set(DENSE_BASELINES) == set()
    assert "bge-base-en-v1.5" not in BASELINES


def test_dense_baselines_are_plausible_ndcg():
    from strata.reference import DENSE_BASELINES

    for model, table in DENSE_BASELINES.items():
        for dataset, value in table.items():
            assert 0.0 < value < 1.0, f"{model}/{dataset} = {value}"
