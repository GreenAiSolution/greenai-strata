"""The engine: index build, retrieve, fuse, re-rank — with a full trace.

Every search returns not just an ordering but the per-stage numbers that
produced it (BM25 score and its top contributing terms, cosine similarity,
fused score with each leg's share, re-ranker score and rationale). That trace is
what makes relevance debuggable instead of mystical.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .ann import ExactIndex, HNSW
from .corpus import Corpus
from .embed import Embedder, LSAEmbedder
from .fusion import Scored, reciprocal_rank_fusion, top_k, weighted_fusion
from .lexical import BM25Index
from .rerank import RerankResult


@dataclass
class Hit:
    doc_id: int
    score: float
    title: str
    doc: str
    snippet: str
    project: str = ""
    bm25: float = 0.0
    vector: float = 0.0
    fused: float = 0.0
    reranked: float | None = None
    rationale: str = ""
    terms: list[tuple[str, float]] = field(default_factory=list)
    parts: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "score": round(self.score, 5),
            "title": self.title, "doc": self.doc, "project": self.project,
            "snippet": self.snippet,
            "bm25": round(self.bm25, 4), "vector": round(self.vector, 4),
            "fused": round(self.fused, 4),
            "reranked": None if self.reranked is None else round(self.reranked, 4),
            "rationale": self.rationale,
            "terms": [[t, round(v, 3)] for t, v in self.terms],
            "parts": {k: round(v, 4) for k, v in self.parts.items()},
        }


@dataclass
class SearchTrace:
    query: str
    mode: str
    alpha: float
    candidates: int
    reranker: str | None
    timings_ms: dict[str, float] = field(default_factory=dict)
    #: True when the query embedded to the zero vector and the dense leg fell
    #: back to the lexical scores. Surfaced so a caller can see that "vector"
    #: numbers for this query are really BM25 numbers.
    dense_fallback: bool = False


class SearchEngine:
    """BM25 + dense vectors + fusion + optional re-ranking over one corpus."""

    def __init__(self, corpus: Corpus, embedder: Embedder,
                 bm25: BM25Index, vectors: np.ndarray,
                 ann: HNSW | None = None):
        self.corpus = corpus
        self.embedder = embedder
        self.bm25 = bm25
        self.vectors = vectors
        self.exact = ExactIndex(vectors)
        self.ann = ann

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, corpus: Corpus, *, embedder: Embedder | None = None,
              use_ann: bool = True, ann_M: int = 16, ann_ef: int = 200,
              verbose: bool = False) -> "SearchEngine":
        texts = [c.indexable() for c in corpus.chunks]

        t0 = time.perf_counter()
        bm25 = BM25Index().fit(texts)
        t1 = time.perf_counter()

        embedder = embedder or LSAEmbedder()
        if isinstance(embedder, LSAEmbedder):
            embedder.fit(texts)
        vectors = embedder.embed_documents(texts)
        t2 = time.perf_counter()

        ann = None
        if use_ann and len(texts) >= 256:
            def report(done: int, total: int) -> None:
                if verbose:
                    print(f"    hnsw {done}/{total}", end="\r", flush=True)

            ann = HNSW(M=ann_M, ef_construction=ann_ef).build(vectors, progress=report)
        t3 = time.perf_counter()

        if verbose:
            print(f"  bm25   {t1 - t0:6.2f}s  ({len(bm25.vocab):,} terms)")
            print(f"  embed  {t2 - t1:6.2f}s  ({embedder.name}, dim={vectors.shape[1]})")
            print(f"  hnsw   {t3 - t2:6.2f}s  ({'built' if ann else 'skipped'})")

        return cls(corpus, embedder, bm25, vectors, ann)

    # ----------------------------------------------------------------- search
    def _snippet(self, chunk, query: str, width: int = 260) -> str:
        from .text import content_words, normalise

        body = " ".join(chunk.text.split())
        lowered = normalise(body)
        best, best_pos = 0, 0
        terms = content_words(query)
        for pos in range(0, max(len(body) - width, 1), 40):
            window = lowered[pos : pos + width]
            hits = sum(1 for t in terms if t in window)
            if hits > best:
                best, best_pos = hits, pos
        out = body[best_pos : best_pos + width]
        return ("…" if best_pos else "") + out + ("…" if best_pos + width < len(body) else "")

    def search(self, query: str, *, k: int = 10, mode: str = "rrf",
               alpha: float = 0.5, candidates: int = 60,
               reranker=None, rerank_depth: int = 25,
               ef: int = 96, use_ann: bool | None = None
               ) -> tuple[list[Hit], SearchTrace]:
        """Retrieve, fuse, optionally re-rank. Returns hits plus the trace.

        The default fusion mode is RRF, and that is a measured choice, not a
        preference. On BEIR with a real encoder (bge-base-en-v1.5) the oracle
        alpha is 0.8–0.9 on all five datasets, so the shipped alpha=0.5
        weighted hybrid is never the best mode — while parameter-free RRF comes
        within 0.001 of the best mode on two datasets without tuning anything.
        Weighted fusion stays fully available (`mode="hybrid"`, `alpha=...`)
        for corpora where a tuned alpha is worth it; see BEIR.md §3.
        """
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        lexical = self.bm25.score(query)
        timings["bm25"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        query_vec = self.embedder.embed_queries([query])[0]
        dense_fallback = float(np.linalg.norm(query_vec)) <= 1e-6
        want_ann = self.ann is not None if use_ann is None else (use_ann and self.ann)
        if dense_fallback:
            # Every query term is outside the embedder's vocabulary, so the
            # query embeds to the zero vector and its cosine against every
            # document is exactly 0.0 — a "dense" ranking would be an arbitrary
            # tie-break over the whole corpus (and the ANN graph would walk to
            # nowhere). Fall back to the one signal that exists: the lexical
            # scores. Vector and hybrid modes then reduce to the BM25 ranking
            # for this query, which is honest rather than random. The trace
            # records it so the caller knows what the numbers mean.
            semantic = lexical
        elif want_ann:
            # Documents the graph did not return need a sentinel that sorts
            # *below* everything it did. Zero is the wrong choice: cosine over
            # signed LSA vectors is legitimately negative for roughly half a
            # corpus, so a zero-filled miss outranks every genuinely dissimilar
            # document the graph correctly found. That is a ranking inversion,
            # not approximation error — on a 500-document sample it changed six
            # of the hybrid top ten.
            #
            # The sentinel is finite rather than -inf on purpose: weighted
            # fusion min-maxes the leg over the candidate pool, and a single
            # -inf would make the whole normalised leg NaN.
            found = self.ann.search(query_vec, k=candidates, ef=ef)
            floor = min((sim for _, sim in found), default=0.0) - 1e-3
            semantic = np.full_like(lexical, floor)
            for doc_id, sim in found:
                semantic[doc_id] = sim
        else:
            semantic = self.exact.scores(query_vec)
        timings["vector"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        if mode == "bm25":
            ranked = [Scored(d, float(lexical[d]), {"bm25": float(lexical[d])})
                      for d in top_k(lexical, candidates)]
        elif mode == "vector":
            ranked = [Scored(d, float(semantic[d]), {"vector": float(semantic[d])})
                      for d in top_k(semantic, candidates)]
        elif mode == "rrf":
            ranked = reciprocal_rank_fusion(
                {"bm25": top_k(lexical, candidates),
                 "vector": top_k(semantic, candidates)},
                limit=candidates,
            )
        elif mode == "hybrid":
            pool = np.union1d(top_k(lexical, candidates), top_k(semantic, candidates))
            ranked = weighted_fusion(lexical, semantic, alpha=alpha,
                                     limit=candidates, candidates=pool)
        else:
            raise ValueError(f"unknown mode: {mode!r}")
        timings["fuse"] = (time.perf_counter() - t0) * 1000

        hits = [self._to_hit(s, query, lexical, semantic) for s in ranked[:max(k, rerank_depth)]]

        reranker_name = None
        if reranker is not None and hits:
            t0 = time.perf_counter()
            pool = hits[:rerank_depth]
            payload = [
                {"doc_id": h.doc_id, "title": h.title,
                 "text": self.corpus[h.doc_id].text, "vector_score": h.vector}
                for h in pool
            ]
            results: list[RerankResult] = reranker.rerank(query, payload, top_k=len(pool))
            by_id = {h.doc_id: h for h in pool}
            hits = []
            for res in results:
                hit = by_id[res.doc_id]
                hit.reranked = res.score
                hit.rationale = res.rationale
                hit.score = res.score
                hits.append(hit)
            reranker_name = getattr(reranker, "name", reranker.__class__.__name__)
            timings["rerank"] = (time.perf_counter() - t0) * 1000

        trace = SearchTrace(query=query, mode=mode, alpha=alpha,
                            candidates=len(ranked), reranker=reranker_name,
                            timings_ms={k_: round(v, 2) for k_, v in timings.items()},
                            dense_fallback=dense_fallback)
        return hits[:k], trace

    def _to_hit(self, scored: Scored, query: str, lexical, semantic) -> Hit:
        chunk = self.corpus[scored.doc_id]
        return Hit(
            doc_id=scored.doc_id,
            score=scored.score,
            title=chunk.title,
            doc=chunk.doc_id,
            project=chunk.project,
            snippet=self._snippet(chunk, query),
            bm25=float(lexical[scored.doc_id]),
            vector=float(semantic[scored.doc_id]),
            fused=scored.score,
            terms=self.bm25.explain(query, scored.doc_id),
            parts=scored.parts,
        )

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.corpus.save(directory / "corpus.json")
        np.save(directory / "vectors.npy", self.vectors)
        if isinstance(self.embedder, LSAEmbedder):
            self.embedder.save(str(directory / "embedder.npz"))
        if self.ann is not None:
            self.ann.save(directory / "hnsw.pkl")
        (directory / "meta.json").write_text(json.dumps({
            "embedder": self.embedder.name,
            "dim": int(self.vectors.shape[1]),
            "chunks": len(self.corpus),
            "has_ann": self.ann is not None,
        }, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "SearchEngine":
        directory = Path(directory)
        corpus = Corpus.load(directory / "corpus.json")
        vectors = np.load(directory / "vectors.npy")
        embedder = LSAEmbedder.load(str(directory / "embedder.npz"))
        bm25 = BM25Index().fit([c.indexable() for c in corpus.chunks])
        ann = HNSW.load(directory / "hnsw.pkl") if (directory / "hnsw.pkl").exists() else None
        return cls(corpus, embedder, bm25, vectors, ann)
