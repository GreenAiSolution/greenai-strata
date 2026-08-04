"""Self-contained test suite. `python3 tests/test_strata.py` — no pytest needed.

These are the invariants that, if they break, make every number in the
evaluation report meaningless. They are cheap and they run in a couple of
seconds, which is the whole argument for having them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strata.ann import ExactIndex, HNSW                     # noqa: E402
from strata.corpus import build_corpus                      # noqa: E402
from strata.embed import LSAEmbedder, l2_normalise          # noqa: E402
from strata.evaluate import build_query_set, score_rankings  # noqa: E402
from strata.fusion import reciprocal_rank_fusion, weighted_fusion  # noqa: E402
from strata.lexical import BM25Index                        # noqa: E402
from strata.pipeline import SearchEngine                    # noqa: E402
from strata.rerank import LocalCrossEncoder                 # noqa: E402
from strata.text import content_words, sentences, tokenize  # noqa: E402

DOCS = [
    "Hierarchical navigable small world graphs index high dimensional vectors.",
    "BM25 is a bag of words ranking function using term frequency and inverse document frequency.",
    "Reciprocal rank fusion merges result lists by rank rather than by score.",
    "The cat sat on the mat while the dog slept by the door.",
    "Cross encoders score a query and a document together instead of separately.",
    "Approximate nearest neighbour search trades a little recall for a lot of speed.",
]

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        failures.append(name)


# --------------------------------------------------------------------------- #

def test_tokenizer() -> None:
    tokens = tokenize("vector_store.search HNSW-index")
    check("tokenizer keeps compound tokens", "vector_store.search" in tokens)
    check("tokenizer emits parts too", "vector" in tokens and "search" in tokens)
    check("stopwords filtered from content words",
          "the" not in content_words("the quick retrieval engine"))
    text = "First sentence here with enough words to count. Second sentence also has plenty of words."
    check("sentence splitter finds both", len(sentences(text, min_words=6)) == 2,
          f"got {len(sentences(text, min_words=6))}")


def test_bm25() -> None:
    index = BM25Index().fit(DOCS)
    scores = index.score("bm25 term frequency ranking")
    check("bm25 ranks the right doc first", int(np.argmax(scores)) == 1)
    check("bm25 ignores unrelated docs", scores[3] < scores[1])

    # The invariant that matters is the IDF ordering itself, not the margin —
    # on a six-document corpus even "the" has a non-trivial IDF.
    idf_rare = float(index.idf[index.vocab["hierarchical"]])   # df = 1
    idf_common = float(index.idf[index.vocab["a"]])            # df = 3
    check("rare terms carry more idf than common ones", idf_rare > idf_common,
          f"rare={idf_rare:.2f} common={idf_common:.2f}")
    check("idf is non-negative under +0.5 smoothing", bool((index.idf >= 0).all()))

    terms = dict(index.explain("bm25 term frequency ranking", 1))
    check("explain returns contributing terms", "bm25" in terms and terms["bm25"] > 0)


def test_embedder_and_projection() -> None:
    embedder = LSAEmbedder(dim=8, min_df=1).fit(DOCS)
    doc_vectors = embedder.embed_documents(DOCS)
    check("doc matrix shape", doc_vectors.shape == (len(DOCS), embedder.dim),
          str(doc_vectors.shape))
    check("doc vectors are unit norm",
          np.allclose(np.linalg.norm(doc_vectors, axis=1), 1.0, atol=1e-4))

    # Fold-in must be consistent: re-embedding a document as a query should
    # land it nearest to itself. If this breaks, every cosine score is noise.
    query_vectors = embedder.embed_queries(DOCS)
    sims = query_vectors @ doc_vectors.T
    check("fold-in recovers each document", bool((sims.argmax(axis=1) ==
                                                  np.arange(len(DOCS))).all()),
          str(sims.argmax(axis=1)))

    unseen = embedder.embed_queries(["completely unrelated zebra vocabulary"])
    check("unseen vocabulary yields a finite vector",
          bool(np.isfinite(unseen).all()))


def test_hnsw_matches_exact() -> None:
    rng = np.random.default_rng(7)
    vectors = l2_normalise(rng.standard_normal((900, 48)).astype(np.float32))
    exact = ExactIndex(vectors)
    graph = HNSW(M=12, ef_construction=120, seed=1).build(vectors)

    queries = l2_normalise(rng.standard_normal((60, 48)).astype(np.float32))
    overlap = 0
    for query in queries:
        truth = {d for d, _ in exact.search(query, k=10)}
        approx = {d for d, _ in graph.search(query, k=10, ef=96)}
        overlap += len(truth & approx)
    recall = overlap / (len(queries) * 10)
    check("hnsw recall@10 >= 0.90 vs exact", recall >= 0.90, f"recall={recall:.3f}")

    check("hnsw is connected at layer 0", len(graph.graph[0]) == len(vectors),
          f"{len(graph.graph[0])}/{len(vectors)}")
    check("layer 0 respects the degree cap",
          max(len(v) for v in graph.graph[0].values()) <= graph.M0)


def test_fusion() -> None:
    fused = reciprocal_rank_fusion({"bm25": [5, 1, 2], "vector": [1, 5, 9]}, k=60)
    order = [s.doc_id for s in fused]
    check("rrf promotes documents both legs like", set(order[:2]) == {1, 5},
          str(order))

    lexical = np.array([0.0, 9.0, 1.0, 0.0], dtype=np.float32)
    semantic = np.array([9.0, 0.0, 1.0, 0.0], dtype=np.float32)
    pool = np.array([0, 1, 2, 3])
    lex_first = weighted_fusion(lexical, semantic, alpha=0.0, candidates=pool)
    vec_first = weighted_fusion(lexical, semantic, alpha=1.0, candidates=pool)
    check("alpha=0 is pure lexical", lex_first[0].doc_id == 1)
    check("alpha=1 is pure semantic", vec_first[0].doc_id == 0)


def test_reranker() -> None:
    encoder = LocalCrossEncoder()
    good, _ = encoder.score_pair(
        "cross encoder scores query and document together",
        "Re-ranking",
        "Cross encoders score a query and a document together instead of separately.",
    )
    bad, _ = encoder.score_pair(
        "cross encoder scores query and document together",
        "Animals",
        "The cat sat on the mat while the dog slept by the door.",
    )
    check("reranker separates relevant from irrelevant", good > bad + 0.2,
          f"good={good:.3f} bad={bad:.3f}")
    check("reranker scores stay in range", 0.0 <= good <= 1.0)


def test_claude_reranker_against_a_mock_client() -> None:
    """Exercise the judge's batching, schema handling, parsing and ordering.

    There is no API key in this environment, so the live call is *not* covered —
    what is covered is everything around it: that candidates are split into
    batches, that a schema-shaped response is parsed, that scores are scaled to
    0–1, and that ties break on retrieval order rather than arbitrarily.
    """
    from strata.rerank import ClaudeReranker

    calls: list[dict] = []

    class FakeBlock:
        type = "text"

        def __init__(self, text): self.text = text

    class FakeResponse:
        stop_reason = "end_turn"

        def __init__(self, text): self.content = [FakeBlock(text)]

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            body = kwargs["messages"][0]["content"]
            ids = [int(part.split('"')[0])
                   for part in body.split('<passage id="')[1:]]
            # deterministic stand-in for a judge: lower id == more relevant
            judgements = [{"id": i, "relevance": max(0, 10 - n), "reason": "mock"}
                          for n, i in enumerate(ids)]
            return FakeResponse(json.dumps({"judgements": judgements}))

    class FakeClient:
        messages = FakeMessages()

    reranker = ClaudeReranker(batch_size=3, client=FakeClient())
    candidates = [{"doc_id": i, "title": f"t{i}", "text": f"body {i}"}
                  for i in range(7)]
    out = reranker.rerank("a query", candidates, top_k=5)

    check("judge batches candidates", len(calls) == 3, f"{len(calls)} calls")
    check("judge respects batch_size",
          all(k["messages"][0]["content"].count("<passage") <= 3 for k in calls))
    check("judge uses a strict json schema",
          calls[0]["output_config"]["format"]["type"] == "json_schema")
    check("rubric is sent as a cached system block",
          calls[0]["system"][0]["cache_control"]["type"] == "ephemeral")
    check("scores are scaled into 0–1", all(0.0 <= r.score <= 1.0 for r in out))
    check("results are sorted by descending score",
          [r.score for r in out] == sorted((r.score for r in out), reverse=True))
    check("top_k is honoured", len(out) == 5, str(len(out)))


def test_metrics() -> None:
    metrics = score_rankings([[7, 1, 2], [3, 4, 5], [9, 8, 4]], [7, 5, 0])
    check("recall@1 counts only rank-1 hits", abs(metrics.recall_at[1] - 1 / 3) < 1e-9,
          str(metrics.recall_at))
    check("recall@5 counts hits inside the cutoff",
          abs(metrics.recall_at[5] - 2 / 3) < 1e-9)
    check("mrr averages reciprocal ranks",
          abs(metrics.mrr - (1.0 + 1 / 3) / 3) < 1e-9, f"{metrics.mrr}")


def test_evaluation_masks_the_answer() -> None:
    """The single most important correctness property of the harness."""
    corpus = build_corpus([str(Path(__file__).resolve().parents[1])],
                          min_tokens=15)
    if len(corpus) < 12:
        print("  skip  masking test (not enough local documents)")
        return
    queries, masked = build_query_set(corpus, n=8, variant="verbatim", seed=3)
    check("query set produced", len(queries) > 0)
    leaked = [
        q for q in queries
        if " ".join(q.query.split()).lower() in
        " ".join(masked[q.target].text.split()).lower()
    ]
    check("target sentence is removed from its own chunk", not leaked,
          f"{len(leaked)} leaked")

    engine = SearchEngine.build(masked, embedder=LSAEmbedder(dim=32, min_df=1),
                                use_ann=False)
    hits, _ = engine.search(queries[0].query, k=5)
    check("engine returns results on the masked corpus", len(hits) > 0)


def test_end_to_end_search() -> None:
    corpus = build_corpus([str(Path(__file__).resolve().parents[1] / "README.md")],
                          min_tokens=10)
    if not len(corpus):
        print("  skip  end-to-end (README.md not built yet)")
        return
    engine = SearchEngine.build(corpus, embedder=LSAEmbedder(dim=16, min_df=1),
                                use_ann=False)
    for mode in ("bm25", "vector", "rrf", "hybrid"):
        hits, trace = engine.search("how is relevance measured", k=3, mode=mode)
        check(f"mode {mode} returns hits", len(hits) > 0)
        check(f"mode {mode} reports timings", bool(trace.timings_ms))


if __name__ == "__main__":
    for fn in [
        test_tokenizer, test_bm25, test_embedder_and_projection,
        test_hnsw_matches_exact, test_fusion, test_reranker,
        test_claude_reranker_against_a_mock_client, test_metrics,
        test_evaluation_masks_the_answer, test_end_to_end_search,
    ]:
        print(f"\n{fn.__name__}")
        fn()
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)
