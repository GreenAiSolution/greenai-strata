"""STRATA — a layered retrieval system built to be measured, not believed.

    corpus → chunk → [BM25 | dense vectors → HNSW] → fusion → re-rank → trace

Every layer is swappable behind a small interface, and `strata eval` reports
what each one is actually worth on your own data.
"""

from .ann import ExactIndex, HNSW
from .corpus import Chunk, Corpus, build_corpus
from .embed import (
    Embedder,
    LSAEmbedder,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    VoyageEmbedder,
)
from .fusion import reciprocal_rank_fusion, weighted_fusion
from .lexical import BM25Index
from .pipeline import Hit, SearchEngine
from .rerank import ClaudeReranker, LocalCrossEncoder, default_reranker

__version__ = "0.1.0"

__all__ = [
    "BM25Index", "Chunk", "ClaudeReranker", "Corpus", "Embedder", "ExactIndex",
    "HNSW", "Hit", "LSAEmbedder", "LocalCrossEncoder", "OpenAIEmbedder",
    "SearchEngine", "SentenceTransformerEmbedder", "VoyageEmbedder",
    "build_corpus", "default_reranker", "reciprocal_rank_fusion",
    "weighted_fusion",
]
