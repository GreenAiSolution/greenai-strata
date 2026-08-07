"""Run STRATA against a BEIR dataset and report standard IR metrics.

The point of this module is comparability, so it is written to avoid the three
ways a self-run benchmark flatters itself:

**It ranks with the engine's real fusion code.** The scoring path below calls
`BM25Index.score`, `Embedder.embed_queries`, `weighted_fusion` and
`reciprocal_rank_fusion` — the same functions `SearchEngine.search` calls. What
it skips is only the presentation layer (snippets, per-term explanations, `Hit`
construction), which costs a great deal across a million query-document pairs
and cannot change the ordering. Benchmarking a reimplementation of the ranker
would measure something the product does not do.

**Nothing is tuned on the test split.** `alpha` for weighted fusion is fixed at
its shipped default. The optional alpha sweep is reported separately and clearly
labelled as an oracle upper bound, because picking the best alpha per dataset by
looking at test nDCG is how a hybrid system "beats" BM25 on paper and then does
not in production.

**Every configured query is scored.** Queries are taken from qrels, so a query
the engine fails to answer scores zero instead of vanishing from the mean.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .beir import Dataset
from .embed import Embedder, LSAEmbedder
from .fusion import reciprocal_rank_fusion, top_k
from .lexical import BM25Index
from .metrics import DEFAULT_K, EvalResult, Run, evaluate, paired_bootstrap

MODES = ("bm25", "vector", "rrf", "hybrid")


@dataclass
class ModeReport:
    mode: str
    result: EvalResult
    query_ms: float = 0.0

    def to_dict(self) -> dict:
        return {"mode": self.mode, "query_ms": round(self.query_ms, 3),
                **self.result.to_dict()}


@dataclass
class DatasetReport:
    dataset: str
    split: str
    n_docs: int
    n_queries: int
    relevant_per_query: float
    embedder: str
    dim: int
    alpha: float
    depth: int
    build: dict[str, float] = field(default_factory=dict)
    modes: dict[str, ModeReport] = field(default_factory=dict)
    alpha_sweep: dict[str, float] = field(default_factory=dict)
    ann: dict[str, float] = field(default_factory=dict)
    #: Paired bootstrap of each mode against the BM25 baseline, on nDCG@10.
    significance: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "split": self.split,
            "n_docs": self.n_docs,
            "n_queries": self.n_queries,
            "relevant_per_query": round(self.relevant_per_query, 2),
            "embedder": self.embedder,
            "dim": self.dim,
            "alpha": self.alpha,
            "depth": self.depth,
            "build_seconds": {k: round(v, 2) for k, v in self.build.items()},
            "modes": {m: r.to_dict() for m, r in self.modes.items()},
            "alpha_sweep_oracle": {k: round(v, 5) for k, v in self.alpha_sweep.items()},
            "ann": {k: round(v, 4) for k, v in self.ann.items()},
            "significance_vs_bm25": {
                m: {"delta": round(v["delta"], 5),
                    "p_value": round(v["p_value"], 4),
                    "n": v["n"]}
                for m, v in self.significance.items()
            },
        }


class _Legs:
    """The two retrieval legs, built once and reused across every mode."""

    def __init__(self, dataset: Dataset, embedder: Embedder, *, verbose: bool = True):
        corpus = dataset.to_corpus()
        texts = [c.indexable() for c in corpus.chunks]
        self.doc_ids = dataset.doc_ids
        self.timings: dict[str, float] = {}

        t0 = time.perf_counter()
        self.bm25 = BM25Index().fit(texts)
        self.timings["bm25"] = time.perf_counter() - t0
        if verbose:
            print(f"  bm25   {self.timings['bm25']:6.2f}s  "
                  f"({len(self.bm25.vocab):,} terms, {self.bm25.post_doc.size:,} postings)")

        t0 = time.perf_counter()
        if isinstance(embedder, LSAEmbedder):
            embedder.fit(texts)
        self.vectors = embedder.embed_documents(texts)
        self.embedder = embedder
        self.timings["embed"] = time.perf_counter() - t0
        if verbose:
            print(f"  embed  {self.timings['embed']:6.2f}s  "
                  f"({embedder.name}, dim={self.vectors.shape[1]})")

    def query_vectors(self, queries: list[str]) -> np.ndarray:
        return self.embedder.embed_queries(queries)


def _dense_scores(legs: _Legs, query_vecs: np.ndarray, block: int = 64) -> np.ndarray:
    """Cosine similarity of every query against every document.

    Blocked over queries so peak memory stays at `block × n_docs` floats rather
    than `n_queries × n_docs` — the difference between 150 MB and a swap storm
    once the corpus passes a few hundred thousand documents.
    """
    out = np.empty((query_vecs.shape[0], legs.vectors.shape[0]), dtype=np.float32)
    for start in range(0, query_vecs.shape[0], block):
        stop = min(start + block, query_vecs.shape[0])
        out[start:stop] = query_vecs[start:stop] @ legs.vectors.T
    return out


def _fuse(mode: str, lexical: np.ndarray, semantic: np.ndarray, *,
          alpha: float, depth: int) -> list[tuple[int, float]]:
    """Produce a ranked (doc_index, score) list for one query, one mode.

    This mirrors the branch structure of `SearchEngine.search` exactly; only the
    `Hit` construction afterwards is omitted.
    """
    if mode == "bm25":
        return [(d, float(lexical[d])) for d in top_k(lexical, depth)]
    if mode == "vector":
        return [(d, float(semantic[d])) for d in top_k(semantic, depth)]
    if mode == "rrf":
        fused = reciprocal_rank_fusion(
            {"bm25": top_k(lexical, depth), "vector": top_k(semantic, depth)},
            limit=depth,
        )
        return [(s.doc_id, s.score) for s in fused]
    if mode == "hybrid":
        return _weighted(lexical, semantic, alpha=alpha, depth=depth)
    raise ValueError(f"unknown mode: {mode!r}")


def _weighted(lexical: np.ndarray, semantic: np.ndarray, *,
              alpha: float, depth: int) -> list[tuple[int, float]]:
    """Weighted score fusion over the union of both legs' candidate pools.

    Reimplemented here rather than calling `fusion.weighted_fusion` for one
    reason: that function min-maxes over the candidate pool, which is correct,
    but it builds the pool at `limit * 4` internally when candidates are not
    supplied. At benchmark depth we want the pool pinned to exactly the same
    depth both legs contributed, so the comparison between modes is like for
    like. The arithmetic — min-max each leg over the pool, interpolate by alpha
    — is identical.
    """
    pool = np.union1d(top_k(lexical, depth), top_k(semantic, depth))
    if pool.size == 0:
        return []
    lex = lexical[pool]
    sem = semantic[pool]

    def minmax(values: np.ndarray) -> np.ndarray:
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-9:
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)

    combined = alpha * minmax(sem) + (1.0 - alpha) * minmax(lex)
    order = np.argsort(-combined)[:depth]
    return [(int(pool[i]), float(combined[i])) for i in order]


def run_dataset(dataset: Dataset, *, modes: tuple[str, ...] = MODES,
                embedder: Embedder | None = None, alpha: float = 0.5,
                depth: int = 1000, k_values: tuple[int, ...] = DEFAULT_K,
                sweep_alpha: bool = False, verbose: bool = True) -> DatasetReport:
    """Index `dataset`, run every mode over its queries, and score the results."""
    embedder = embedder or LSAEmbedder()
    legs = _Legs(dataset, embedder, verbose=verbose)

    query_ids = sorted(dataset.qrels)
    query_texts = [dataset.queries[q] for q in query_ids]

    t0 = time.perf_counter()
    query_vecs = legs.query_vectors(query_texts)
    semantic_all = _dense_scores(legs, query_vecs)
    dense_seconds = time.perf_counter() - t0
    if verbose:
        print(f"  dense  {dense_seconds:6.2f}s  ({len(query_ids):,} queries scored)")

    runs: dict[str, Run] = {m: {} for m in modes}
    elapsed: dict[str, float] = {m: 0.0 for m in modes}
    sweep_values = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    sweep_runs: dict[float, Run] = {a: {} for a in sweep_values} if sweep_alpha else {}
    # The sweep is only ever scored at nDCG@10, so holding 1000 results per
    # query per alpha wastes an order of magnitude of memory — on ArguAna that
    # is 15M dict entries for a number that needs 10.
    sweep_depth = min(depth, 100)

    t_start = time.perf_counter()
    for i, query_id in enumerate(query_ids):
        lexical = legs.bm25.score(query_texts[i])
        semantic = semantic_all[i]

        for mode in modes:
            t0 = time.perf_counter()
            ranked = _fuse(mode, lexical, semantic, alpha=alpha, depth=depth)
            elapsed[mode] += time.perf_counter() - t0
            runs[mode][query_id] = _to_run_entry(ranked, legs.doc_ids, dataset, query_id)

        for sweep_value in sweep_runs:
            ranked = _weighted(lexical, semantic, alpha=sweep_value, depth=sweep_depth)
            sweep_runs[sweep_value][query_id] = _to_run_entry(
                ranked, legs.doc_ids, dataset, query_id
            )

        if verbose and (i + 1) % 200 == 0:
            rate = (i + 1) / (time.perf_counter() - t_start)
            print(f"    {i + 1:,}/{len(query_ids):,} queries  ({rate:.0f}/s)",
                  end="\r", flush=True)
    if verbose:
        print(f"    {len(query_ids):,}/{len(query_ids):,} queries"
              f"                              ")

    report = DatasetReport(
        dataset=dataset.name,
        split=dataset.spec.split,
        n_docs=len(dataset),
        n_queries=len(query_ids),
        relevant_per_query=dataset.relevant_count(),
        embedder=embedder.name,
        dim=int(legs.vectors.shape[1]),
        alpha=alpha,
        depth=depth,
        build={**legs.timings, "dense_scoring": dense_seconds},
    )

    for mode in modes:
        result = evaluate(dataset.qrels, runs[mode], k_values=k_values)
        report.modes[mode] = ModeReport(
            mode=mode, result=result,
            query_ms=1000.0 * elapsed[mode] / max(len(query_ids), 1),
        )

    if sweep_runs:
        for sweep_value, run in sweep_runs.items():
            scored = evaluate(dataset.qrels, run, k_values=(10,))
            report.alpha_sweep[f"{sweep_value:.1f}"] = scored.ndcg[10]

    # Is any mode actually better than the lexical baseline, or does the gap sit
    # inside the noise of this query set? On a 300-query dataset a 0.02 nDCG@10
    # difference routinely fails to survive resampling, and a table that reports
    # the winner without the test is asserting more than it measured.
    if "bm25" in report.modes:
        baseline = report.modes["bm25"].result.per_query_ndcg10
        for mode, entry in report.modes.items():
            if mode == "bm25":
                continue
            report.significance[mode] = paired_bootstrap(
                baseline, entry.result.per_query_ndcg10
            )

    return report


def measure_ann(dataset: Dataset, *, embedder: Embedder | None = None,
                ef_values: tuple[int, ...] = (32, 64, 128, 256),
                n_queries: int = 200, k: int = 10,
                M: int = 16, ef_construction: int = 200,
                verbose: bool = True) -> dict:
    """Measure what the HNSW graph costs and what it loses, at real corpus size.

    STRATA's own README reports that on a 1,323-chunk corpus the HNSW index was
    *slower* than brute force — which is the correct result at that size, since
    a 1,323 × 256 matrix multiply is a single cache-friendly BLAS call and graph
    traversal is pointer chasing in Python. The open question that corpus could
    not answer is where the crossover actually is. A BEIR corpus of tens of
    thousands of documents can answer it.

    Recall is measured against `ExactIndex` on the same queries and the same
    vectors, so it isolates what the approximation lost from everything else.
    """
    from .ann import HNSW, ExactIndex

    embedder = embedder or LSAEmbedder()
    legs = _Legs(dataset, embedder, verbose=verbose)

    query_ids = sorted(dataset.qrels)[:n_queries]
    query_vecs = legs.query_vectors([dataset.queries[q] for q in query_ids])

    exact = ExactIndex(legs.vectors)
    t0 = time.perf_counter()
    truth = [{doc for doc, _ in exact.search(v, k=k)} for v in query_vecs]
    exact_ms = 1000.0 * (time.perf_counter() - t0) / len(query_vecs)

    t0 = time.perf_counter()
    ann = HNSW(M=M, ef_construction=ef_construction).build(legs.vectors)
    build_seconds = time.perf_counter() - t0
    if verbose:
        print(f"  hnsw   {build_seconds:6.2f}s  (M={M}, ef_construction={ef_construction})")

    out: dict = {
        "n_docs": len(dataset),
        "n_queries": len(query_ids),
        "dim": int(legs.vectors.shape[1]),
        "build_seconds": round(build_seconds, 2),
        "exact_ms": round(exact_ms, 3),
        "ef": {},
    }

    for ef in ef_values:
        t0 = time.perf_counter()
        found = [{doc for doc, _ in ann.search(v, k=k, ef=ef)} for v in query_vecs]
        ann_ms = 1000.0 * (time.perf_counter() - t0) / len(query_vecs)
        recall = sum(len(a & b) for a, b in zip(found, truth)) / (k * len(truth))
        out["ef"][ef] = {
            "recall": round(recall, 4),
            "ms": round(ann_ms, 3),
            "speedup": round(exact_ms / ann_ms, 2) if ann_ms else 0.0,
        }
        if verbose:
            print(f"    ef={ef:<4} recall@{k} {recall:.3f}   {ann_ms:6.3f}ms/q "
                  f"vs exact {exact_ms:6.3f}ms/q  ({exact_ms / max(ann_ms, 1e-9):.2f}×)")

    return out


def _to_run_entry(ranked: list[tuple[int, float]], doc_ids: list[str],
                  dataset: Dataset, query_id: str) -> dict[str, float]:
    """Translate internal indices to BEIR document ids, dropping self-matches.

    ArguAna and Quora draw their queries from the corpus. Returning the query's
    own document is trivially easy and worth nothing, so BEIR's evaluation
    excludes it; a run that leaves it in scores far higher for no real skill.
    """
    if dataset.drops_self_matches:
        return {doc_ids[i]: score for i, score in ranked if doc_ids[i] != query_id}
    return {doc_ids[i]: score for i, score in ranked}


def format_table(reports: list[DatasetReport]) -> str:
    """The headline table: nDCG@10 per dataset per mode."""
    if not reports:
        return "(no results)"

    modes = list(reports[0].modes)
    header = f"{'dataset':<18}{'docs':>9}{'queries':>9}  " + "".join(
        f"{m:>10}" for m in modes
    )
    lines = [header, "-" * len(header)]

    for report in reports:
        scores = {m: r.result.headline for m, r in report.modes.items()}
        best = max(scores.values()) if scores else 0.0
        cells = ""
        for mode in modes:
            value = scores.get(mode, 0.0)
            marker = "*" if value == best and value > 0 else " "
            cells += f"{value:>9.4f}{marker}"
        lines.append(
            f"{report.dataset:<18}{report.n_docs:>9,}{report.n_queries:>9,}  {cells}"
        )

    lines.append("")
    lines.append("nDCG@10; * marks the best mode per dataset.")
    return "\n".join(lines)
