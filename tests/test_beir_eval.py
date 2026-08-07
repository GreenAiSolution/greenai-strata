"""Tests for the BEIR adapter and runner.

The load-bearing claim in `strata/beir_eval.py` is that the benchmark ranks with
the engine's real fusion code rather than a lookalike reimplementation. That
claim is worth nothing unless something checks it, so the first test here pins
the benchmark's weighted fusion to `strata.fusion.weighted_fusion` output.

The rest cover the adapter decisions that silently inflate scores when wrong:
document-level retrieval units, self-match exclusion, and split filtering.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from strata import beir
from strata.beir import Dataset, DatasetSpec, _read_qrels, load
from strata.beir_eval import _fuse, _to_run_entry, _weighted
from strata.fusion import top_k, weighted_fusion


# --------------------------------------------------------------------------- #
# The benchmark must rank with the engine's fusion, not its own
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_benchmark_weighted_fusion_matches_the_engine(seed: int, alpha: float):
    rng = np.random.default_rng(seed)
    n_docs = 500
    depth = 50

    # BM25 leaves most of the corpus at exactly zero; reproduce that, because a
    # dense random vector would hide any min-max normalisation difference.
    lexical = np.zeros(n_docs, dtype=np.float32)
    hot = rng.choice(n_docs, size=80, replace=False)
    lexical[hot] = rng.random(80).astype(np.float32) * 12
    semantic = rng.random(n_docs).astype(np.float32)

    mine = _weighted(lexical, semantic, alpha=alpha, depth=depth)

    pool = np.union1d(top_k(lexical, depth), top_k(semantic, depth))
    theirs = weighted_fusion(lexical, semantic, alpha=alpha, limit=depth,
                             candidates=pool)

    assert [d for d, _ in mine] == [s.doc_id for s in theirs]
    for (_, score), scored in zip(mine, theirs):
        assert score == pytest.approx(scored.score, abs=1e-6)


def test_fuse_dispatches_every_mode():
    rng = np.random.default_rng(0)
    lexical = rng.random(200).astype(np.float32)
    semantic = rng.random(200).astype(np.float32)
    for mode in ("bm25", "vector", "rrf", "hybrid"):
        ranked = _fuse(mode, lexical, semantic, alpha=0.5, depth=20)
        assert len(ranked) == 20
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True), f"{mode} is not sorted"


def test_fuse_rejects_unknown_mode():
    with pytest.raises(ValueError):
        _fuse("magic", np.zeros(4), np.zeros(4), alpha=0.5, depth=2)


def test_bm25_mode_ignores_the_vector_leg():
    lexical = np.array([0.0, 5.0, 0.0], dtype=np.float32)
    semantic = np.array([9.0, 0.0, 9.0], dtype=np.float32)
    assert _fuse("bm25", lexical, semantic, alpha=0.5, depth=1)[0][0] == 1
    assert _fuse("vector", lexical, semantic, alpha=0.5, depth=1)[0][0] in (0, 2)


# --------------------------------------------------------------------------- #
# Self-match exclusion
# --------------------------------------------------------------------------- #

def _dataset(drops: bool) -> Dataset:
    spec = DatasetSpec("arguana" if drops else "scifact", drops_self_matches=drops)
    return Dataset(spec=spec, doc_ids=["q1", "d2", "d3"])


def test_self_match_is_dropped_when_queries_come_from_the_corpus():
    ranked = [(0, 9.9), (1, 5.0), (2, 1.0)]      # index 0 is the query's own doc
    entry = _to_run_entry(ranked, ["q1", "d2", "d3"], _dataset(True), "q1")
    assert "q1" not in entry
    assert entry == {"d2": 5.0, "d3": 1.0}


def test_self_match_is_kept_when_the_dataset_does_not_ask_for_it():
    ranked = [(0, 9.9), (1, 5.0)]
    entry = _to_run_entry(ranked, ["q1", "d2", "d3"], _dataset(False), "q1")
    assert entry == {"q1": 9.9, "d2": 5.0}


def test_registry_flags_the_self_retrieval_datasets():
    # ArguAna and Quora draw queries from their own corpora. If this ever
    # regresses, those two datasets silently score far too well.
    assert beir.REGISTRY["arguana"].drops_self_matches
    assert beir.REGISTRY["quora"].drops_self_matches
    assert not beir.REGISTRY["scifact"].drops_self_matches


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #

def test_qrels_parser_reads_header_and_grades(tmp_path):
    path = tmp_path / "test.tsv"
    path.write_text("query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\nq2\td3\t1\n")
    assert _read_qrels(path) == {"q1": {"d1": 2, "d2": 0}, "q2": {"d3": 1}}


def test_qrels_parser_survives_a_missing_header(tmp_path):
    # Some mirrors ship headerless qrels; the first row must not be eaten.
    path = tmp_path / "test.tsv"
    path.write_text("q1\td1\t1\nq2\td2\t1\n")
    assert _read_qrels(path) == {"q1": {"d1": 1}, "q2": {"d2": 1}}


def _fake_dataset(root, name="scifact"):
    """Write a minimal BEIR layout so `load` can be tested without the network."""
    directory = root / name
    (directory / "qrels").mkdir(parents=True)
    with (directory / "corpus.jsonl").open("w") as fh:
        for i in range(5):
            fh.write(json.dumps({"_id": f"d{i}", "title": f"Title {i}",
                                 "text": f"body text {i}"}) + "\n")
    with (directory / "queries.jsonl").open("w") as fh:
        # q9 belongs to the train split and must not leak into the test run.
        for qid in ("q0", "q1", "q9"):
            fh.write(json.dumps({"_id": qid, "text": f"question {qid}"}) + "\n")
    (directory / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq0\td0\t1\nq1\td3\t2\n"
    )
    return directory


def test_load_uses_only_queries_with_judgements_in_the_split(tmp_path):
    _fake_dataset(tmp_path)
    dataset = load("scifact", cache_dir=tmp_path, verbose=False)
    assert set(dataset.queries) == {"q0", "q1"}      # q9 has no test qrels
    assert len(dataset) == 5


def test_load_maps_one_document_to_one_chunk(tmp_path):
    _fake_dataset(tmp_path)
    dataset = load("scifact", cache_dir=tmp_path, verbose=False)
    corpus = dataset.to_corpus()

    # BEIR scores at document level, so chunking would break the id mapping.
    assert len(corpus) == len(dataset) == 5
    assert [c.doc_id for c in corpus.chunks] == dataset.doc_ids
    # Title is indexed alongside the body, matching BEIR's title + text convention.
    assert "Title 3" in corpus[3].indexable()
    assert "body text 3" in corpus[3].indexable()


def test_load_reports_relevant_documents_per_query(tmp_path):
    _fake_dataset(tmp_path)
    dataset = load("scifact", cache_dir=tmp_path, verbose=False)
    assert dataset.relevant_count() == pytest.approx(1.0)


def test_load_rejects_an_unknown_dataset(tmp_path):
    with pytest.raises(KeyError):
        load("not-a-real-dataset", cache_dir=tmp_path, verbose=False)


def test_load_reports_available_splits_when_the_split_is_missing(tmp_path):
    _fake_dataset(tmp_path)
    with pytest.raises(FileNotFoundError, match="test"):
        load("scifact", split="train", cache_dir=tmp_path, verbose=False)


def test_truncation_drops_queries_left_without_relevant_documents(tmp_path):
    _fake_dataset(tmp_path)
    # Keep only d0-d1; q1's only relevant document (d3) disappears, so q1 can no
    # longer be scored fairly and must be removed rather than counted as a miss.
    dataset = load("scifact", cache_dir=tmp_path, max_docs=2, verbose=False)
    assert len(dataset) == 2
    assert set(dataset.qrels) == {"q0"}
    assert set(dataset.queries) == {"q0"}
