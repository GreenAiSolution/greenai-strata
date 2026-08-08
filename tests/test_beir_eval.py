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
from strata.beir_eval import (_dense_scores, _fuse, _to_run_entry, _weighted,
                              lexical_candidates)
from strata.fusion import top_k, weighted_fusion


# --------------------------------------------------------------------------- #
# The benchmark must rank with the engine's fusion, not its own
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_benchmark_weighted_fusion_matches_the_engine(seed: int, alpha: float):
    """Given the same candidate pool, the benchmark's fusion *is* the engine's.

    The scoring arithmetic must be identical or the benchmark measures something
    the product does not do. The pool the two build differs by design — see
    `test_benchmark_pool_excludes_unmatched_lexical_documents` — so the pool is
    passed in explicitly here to isolate the arithmetic from that choice.
    """
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

    pool = np.union1d(lexical_candidates(lexical, depth), top_k(semantic, depth))
    theirs = weighted_fusion(lexical, semantic, alpha=alpha, limit=depth,
                             candidates=pool)

    assert [d for d, _ in mine] == [s.doc_id for s in theirs]
    for (_, score), scored in zip(mine, theirs):
        assert score == pytest.approx(scored.score, abs=1e-6)


def test_benchmark_pool_excludes_unmatched_lexical_documents():
    """The one deliberate divergence from the engine's pool construction.

    `fusion.weighted_fusion` takes the lexical leg's top-k as given. The
    benchmark first drops documents BM25 scored at exactly zero, because a run
    file must contain only retrieved documents. A document can still reach the
    pool through the dense leg, which genuinely scored it — the filter removes
    it as a *lexical* candidate, not from consideration entirely.
    """
    lexical = np.array([4.0, 0.0, 0.0, 0.0], dtype=np.float32)
    semantic = np.array([0.1, 0.9, 0.2, 0.3], dtype=np.float32)

    assert list(lexical_candidates(lexical, depth=4)) == [0]
    # The engine's top_k keeps the unmatched documents; their order among
    # themselves is whatever numpy's sort happens to produce, which is exactly
    # why letting them into a run was a lottery.
    assert len(top_k(lexical, 4)) == 4
    assert set(top_k(lexical, 4)) == {0, 1, 2, 3}

    # Document 1 is unmatched lexically but is the dense leg's top hit, so it
    # must still appear — via the dense leg. The filter removes a document as a
    # lexical candidate, it does not blacklist it. (A cosine of 0.0 is a real
    # similarity the dense leg computed; a BM25 of 0.0 is the absence of a
    # posting. Only the second means "not retrieved".)
    pooled = {d for d, _ in _weighted(lexical, semantic, alpha=0.5, depth=4)}
    assert pooled == {0, 1, 2, 3}

    # Where it bites is a shallow pool: with depth 1 the engine's lexical leg
    # would nominate an unmatched document, and the benchmark nominates none.
    assert list(lexical_candidates(lexical, depth=1)) == [0]
    assert lexical_candidates(np.zeros(4, dtype=np.float32), depth=4).size == 0


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


def test_bm25_mode_never_returns_unmatched_documents():
    """A BM25 score of exactly zero means no query term occurs in the document.

    Returning those pads the run with documents the system did not retrieve,
    which Lucene cannot do — there is no posting to return. It is not harmless:
    on NFCorpus, where 79 of 323 queries match fewer than ten documents at all,
    padding let relevant documents scoring 0.0 reach the top ten on the
    document-id tie-break and inflated nDCG@10 by 0.0038.
    """
    lexical = np.array([3.0, 0.0, 1.5, 0.0, 0.0], dtype=np.float32)
    semantic = np.zeros(5, dtype=np.float32)

    ranked = _fuse("bm25", lexical, semantic, alpha=0.5, depth=1000)

    assert [d for d, _ in ranked] == [0, 2]
    assert all(score > 0 for _, score in ranked)


def test_bm25_result_count_does_not_change_ndcg_at_10():
    # nDCG@10 must depend only on the top ten. If asking for more results can
    # change it, the run is being padded with things that were never retrieved.
    rng = np.random.default_rng(0)
    lexical = np.zeros(400, dtype=np.float32)
    lexical[rng.choice(400, size=4, replace=False)] = rng.random(4) + 0.5
    semantic = np.zeros(400, dtype=np.float32)

    shallow = _fuse("bm25", lexical, semantic, alpha=0.5, depth=10)
    deep = _fuse("bm25", lexical, semantic, alpha=0.5, depth=400)
    assert [d for d, _ in shallow] == [d for d, _ in deep]


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


# --------------------------------------------------------------------------- #
# Regression: a query that matches nothing must not crash the fusion
# --------------------------------------------------------------------------- #

def test_every_mode_survives_a_query_with_no_lexical_match():
    """25 of NFCorpus's 323 queries match no document at all.

    When the lexical filter leaves nothing, `np.union1d` on an empty *Python
    list* returns a float64 array, and indexing with it raises IndexError. The
    filter therefore has to return a dtype-pinned integer array so the empty
    case is not a special case. This crashed `strata beir nfcorpus` in the
    default mode set until it was caught.
    """
    lexical = np.zeros(500, dtype=np.float32)          # nothing matched
    semantic = np.linspace(0, 1, 500).astype(np.float32)

    assert lexical_candidates(lexical, 1000).size == 0
    assert lexical_candidates(lexical, 1000).dtype == np.int64

    assert _fuse("bm25", lexical, semantic, alpha=0.5, depth=1000) == []
    for mode in ("vector", "rrf", "hybrid"):
        ranked = _fuse(mode, lexical, semantic, alpha=0.5, depth=1000)
        assert ranked, f"{mode} returned nothing"
        assert all(isinstance(d, (int, np.integer)) for d, _ in ranked)


def test_both_legs_empty_yields_no_results_rather_than_an_error():
    lexical = np.zeros(10, dtype=np.float32)
    semantic = np.zeros(0, dtype=np.float32)
    assert _fuse("bm25", lexical, semantic, alpha=0.5, depth=10) == []


def test_lexical_candidate_ids_index_correctly():
    # The dtype pin matters at the point of use, not just at construction.
    lexical = np.array([0.0, 2.0, 0.0, 5.0], dtype=np.float32)
    candidates = lexical_candidates(lexical, 4)
    assert list(lexical[candidates]) == [5.0, 2.0]


# --------------------------------------------------------------------------- #
# Streamed dense scoring must be numerically identical to the full matrix
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n_queries,block", [(1, 64), (64, 64), (65, 64),
                                             (200, 64), (200, 7), (200, 1)])
def test_streamed_dense_scores_match_the_full_matmul(n_queries: int, block: int):
    """The generator must change memory behaviour and nothing else.

    Streaming exists so the harness stops holding an n_queries × n_docs matrix;
    it is only measurement-neutral if every yielded row equals the row the full
    matmul would have produced, at every block boundary. Bit-equality is
    demanded at the shipped block size in the test below (same matmul, block
    for block); across *different* block shapes BLAS picks different kernels
    and float32 summation order shifts by ~1e-6 relative, so this test asserts
    a tolerance instead.
    """
    import types

    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((333, 32)).astype(np.float32)
    query_vecs = rng.standard_normal((n_queries, 32)).astype(np.float32)

    full = np.asarray(query_vecs @ vectors.T, dtype=np.float32)

    gen = _dense_scores(vectors, query_vecs, block=block)
    assert isinstance(gen, types.GeneratorType), \
        "streaming means a generator, not a materialised array"
    rows = list(gen)
    assert len(rows) == n_queries
    streamed = np.stack(rows)
    np.testing.assert_allclose(streamed, full, rtol=1e-5, atol=1e-5)


def test_streamed_dense_scores_default_block_is_bit_identical():
    # The shipped block size is the one the published numbers were produced
    # with, so at that size the stream must be bit-for-bit the old matrix.
    rng = np.random.default_rng(11)
    vectors = rng.standard_normal((500, 24)).astype(np.float32)
    query_vecs = rng.standard_normal((150, 24)).astype(np.float32)

    old = np.empty((150, 500), dtype=np.float32)
    for start in range(0, 150, 64):
        stop = min(start + 64, 150)
        old[start:stop] = query_vecs[start:stop] @ vectors.T

    streamed = np.stack(list(_dense_scores(vectors, query_vecs)))
    assert np.array_equal(streamed, old)


def test_streamed_dense_scores_accumulates_elapsed_time():
    rng = np.random.default_rng(3)
    vectors = rng.standard_normal((100, 8)).astype(np.float32)
    query_vecs = rng.standard_normal((10, 8)).astype(np.float32)
    elapsed = [0.0]
    list(_dense_scores(vectors, query_vecs, elapsed=elapsed))
    assert elapsed[0] > 0.0


# --------------------------------------------------------------------------- #
# Zero-vector queries: the dense leg must fall back to the lexical ranking
# --------------------------------------------------------------------------- #

def _oov_query():
    """A query with lexical matches whose embedding is the zero vector.

    35 of NFCorpus's 323 queries do this under LSA at min_df=2 — single rare
    words the vocabulary dropped. Their cosine against every document is 0.0,
    so a dense ranking would be an arbitrary tie-break over the corpus.
    """
    lexical = np.array([0.0, 3.0, 0.0, 7.0, 1.0, 0.0], dtype=np.float32)
    semantic = np.zeros(6, dtype=np.float32)
    return lexical, semantic


def test_fallback_vector_mode_returns_the_lexical_ranking():
    lexical, semantic = _oov_query()
    ranked = _fuse("vector", lexical, semantic, alpha=0.5, depth=6,
                   dense_fallback=True)
    assert [d for d, _ in ranked] == [3, 1, 4]
    assert [s for _, s in ranked] == [7.0, 3.0, 1.0]


def test_fallback_never_pads_with_unmatched_documents():
    # The fallback must go through `lexical_candidates`, not `top_k`: a zero
    # BM25 score means "not retrieved", and the dense leg falling back to the
    # lexical leg must not reintroduce the padding defect through the back door.
    lexical, semantic = _oov_query()
    for mode in ("vector", "rrf", "hybrid"):
        ranked = _fuse(mode, lexical, semantic, alpha=0.5, depth=6,
                       dense_fallback=True)
        assert {d for d, _ in ranked} == {1, 3, 4}, mode


def test_fallback_rrf_and_hybrid_reduce_to_the_lexical_order():
    lexical, semantic = _oov_query()
    want = [3, 1, 4]
    for mode in ("rrf", "hybrid"):
        ranked = _fuse(mode, lexical, semantic, alpha=0.5, depth=6,
                       dense_fallback=True)
        assert [d for d, _ in ranked] == want, mode


def test_fallback_hybrid_is_alpha_invariant():
    # With both legs identical, min-max of each over the pool is identical, so
    # the interpolation must produce the same ordering for every alpha.
    lexical, semantic = _oov_query()
    orders = {
        alpha: [d for d, _ in _weighted(lexical, semantic, alpha=alpha,
                                        depth=6, dense_fallback=True)]
        for alpha in (0.0, 0.3, 0.5, 0.8, 1.0)
    }
    assert len({tuple(o) for o in orders.values()}) == 1


def test_fallback_with_no_lexical_match_retrieves_nothing():
    # Zero vector *and* no lexical match: neither leg has any signal, and the
    # honest run is empty — not a coin flip over the corpus.
    lexical = np.zeros(6, dtype=np.float32)
    semantic = np.zeros(6, dtype=np.float32)
    for mode in ("bm25", "vector", "rrf", "hybrid"):
        assert _fuse(mode, lexical, semantic, alpha=0.5, depth=6,
                     dense_fallback=True) == [], mode


def test_no_fallback_leaves_the_dense_leg_untouched():
    # dense_fallback=False must be exactly the old behaviour.
    rng = np.random.default_rng(5)
    lexical = rng.random(50).astype(np.float32)
    semantic = rng.random(50).astype(np.float32)
    for mode in ("bm25", "vector", "rrf", "hybrid"):
        a = _fuse(mode, lexical, semantic, alpha=0.5, depth=20)
        b = _fuse(mode, lexical, semantic, alpha=0.5, depth=20,
                  dense_fallback=False)
        assert a == b, mode


def test_engine_falls_back_when_the_query_embeds_to_zero():
    """The product path: `SearchEngine.search` on an out-of-vocabulary query."""
    from strata.corpus import Chunk, Corpus
    from strata.lexical import BM25Index
    from strata.pipeline import SearchEngine

    rng = np.random.default_rng(0)
    n = 40
    vectors = rng.standard_normal((n, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    corpus = Corpus(chunks=[
        Chunk(id=i, doc_id=f"d{i}", title=f"t{i}",
              text=f"zoloft dosage note {i}" if i < 3 else f"other topic {i}",
              start_line=1, n_tokens=4)
        for i in range(n)
    ])

    class _ZeroForQueries:
        name, dim = "zero-query", 8

        def embed_documents(self, texts):
            return vectors

        def embed_queries(self, texts):
            return np.zeros((len(texts), 8), dtype=np.float32)

    bm25 = BM25Index().fit([c.indexable() for c in corpus.chunks])
    engine = SearchEngine(corpus, _ZeroForQueries(), bm25, vectors)

    for mode in ("vector", "rrf", "hybrid"):
        hits, trace = engine.search("zoloft", k=3, mode=mode)
        assert trace.dense_fallback, mode
        assert hits, mode
        # The lexical leg is the only signal; the top hits must be the chunks
        # that actually contain the query term, not an arbitrary tie-break.
        assert {h.doc_id for h in hits[:3]} <= {0, 1, 2}, mode

    # A query with real vocabulary coverage must not trip the fallback flag.
    class _Real:
        name, dim = "real", 8

        def embed_documents(self, texts):
            return vectors

        def embed_queries(self, texts):
            return vectors[:len(texts)]

    engine = SearchEngine(corpus, _Real(), bm25, vectors)
    _, trace = engine.search("zoloft", k=3, mode="rrf")
    assert not trace.dense_fallback


# --------------------------------------------------------------------------- #
# RRF is the shipped default fusion mode (measured: BEIR.md §3)
# --------------------------------------------------------------------------- #

def test_rrf_is_the_default_mode_everywhere():
    """With bge-base-en-v1.5 the oracle alpha is 0.8–0.9 on all five BEIR
    datasets, so the untuned alpha=0.5 hybrid is never the best mode, while
    parameter-free RRF lands within 0.001 of the best mode on two datasets.
    The default therefore changed from weighted fusion to RRF; this pins it
    in the engine, the CLI and the web UI so it cannot silently regress.
    """
    import inspect

    from strata import cli
    from strata.pipeline import SearchEngine
    from strata.server import PAGE

    assert inspect.signature(SearchEngine.search).parameters["mode"].default == "rrf"
    assert cli.build_parser().parse_args(["search", "q"]).mode == "rrf"
    # Weighted fusion stays fully available; only the default moved.
    assert cli.build_parser().parse_args(
        ["search", "q", "--mode", "hybrid", "--alpha", "0.8"]).alpha == 0.8
    assert 'value="rrf" selected' in PAGE


# --------------------------------------------------------------------------- #
# The ANN "not retrieved" sentinel must sort below real negative similarities
# --------------------------------------------------------------------------- #

def test_ann_miss_sentinel_sorts_below_genuine_negative_similarity():
    """Cosine over signed LSA vectors is legitimately negative.

    Filling un-retrieved documents with 0.0 makes a miss outrank every document
    the graph correctly found and scored negatively — a ranking inversion, not
    approximation error. The sentinel must be finite (so min-max normalisation
    downstream does not go NaN) and below every returned score.
    """
    import numpy as np

    from strata.corpus import Chunk, Corpus
    from strata.pipeline import SearchEngine

    rng = np.random.default_rng(0)
    n = 300
    vectors = rng.standard_normal((n, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    corpus = Corpus(chunks=[
        Chunk(id=i, doc_id=f"d{i}", title=f"t{i}", text=f"document number {i} alpha",
              start_line=1, n_tokens=4)
        for i in range(n)
    ])

    class _Fixed:
        name, dim = "fixed", 16

        def embed_documents(self, texts):
            return vectors

        def embed_queries(self, texts):
            return vectors[:1]

    from strata.ann import HNSW
    from strata.lexical import BM25Index

    bm25 = BM25Index().fit([c.indexable() for c in corpus.chunks])
    ann = HNSW(M=8, ef_construction=64).build(vectors)
    engine = SearchEngine(corpus, _Fixed(), bm25, vectors, ann)

    hits, _ = engine.search("alpha", k=10, mode="vector", candidates=20,
                            use_ann=True)
    scores = [h.vector for h in hits]

    assert all(np.isfinite(s) for s in scores), "sentinel must stay finite"
    # Nothing that the graph missed may outrank something it returned, so the
    # returned top-10 must all be real similarities rather than sentinel fill.
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)   # the query is document 0
