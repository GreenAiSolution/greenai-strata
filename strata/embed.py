"""Embedding providers behind one interface.

The default provider is a **local latent-semantic embedder** — TF-IDF followed
by a randomised truncated SVD, computed in pure numpy over a hand-rolled CSR
matrix. It is genuinely semantic (it learns term co-occurrence structure, so
"retrieval" and "search" land near each other without ever sharing a token) and
it runs with no API key, no model download and no network.

It is *not* a claim to match a modern neural encoder. It is the honest floor:
swap `LSAEmbedder` for `OpenAIEmbedder`, `VoyageEmbedder` or
`SentenceTransformerEmbedder` — same three methods — and every other layer of
the stack is unchanged. The evaluation harness will tell you exactly how much
the swap bought.
"""

from __future__ import annotations

import os
from typing import Protocol, Sequence

import numpy as np

from .text import tokenize


class Embedder(Protocol):
    """The whole contract. Four members, no inheritance required."""

    name: str
    dim: int

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    np.maximum(norms, 1e-9, out=norms)
    return (matrix / norms).astype(np.float32)


# --------------------------------------------------------------------------- #
# Sparse helpers — a minimal CSR so we never pull in scipy.
# --------------------------------------------------------------------------- #

class _RankMajorPlan:
    """A reduction schedule for `sum of w[p] * dense[g[p]]` over each segment.

    Both products the SVD needs are segment sums over the postings: `A @ dense`
    sums each *row's* postings, `A.T @ dense` sums each *column's*. The obvious
    implementations are both bad. `np.add.reduceat` has to materialise the whole
    `(nnz, k)` product first — 4.5 GB on fiqa — and then runs an order of
    magnitude below memory bandwidth because the segments are short. A
    `np.bincount` per output column avoids the temporary but makes one full pass
    over every posting per column, so 268 passes.

    The trick here is to reorder the work *rank-major* instead of segment-major.
    Sort the segments by descending length; then "the segments that own a t-th
    posting" is exactly a prefix of that order, and its length `m[t]` is a
    non-increasing staircase. So the whole product is

        for t in range(longest_segment):
            acc[:m[t]] += dense[gather[t]] * weight[t][:, None]

    Every posting is touched exactly once, each step is a contiguous vectorised
    fused gather-multiply-add into a live accumulator, and there is no `(nnz, k)`
    temporary at all — the largest array is the `(segments, k)` accumulator.

    Summation order is preserved: within a segment the postings are still added
    in ascending posting order, one rank at a time. Together with the float64
    accumulator — which is what `np.bincount` used — that makes `rdot` bit-for-bit
    identical to the per-column bincount it replaces. `dot` agrees with the old
    reduceat to float32 rounding rather than exactly, because reduceat summed in
    float32 with its own internal unrolling; where they differ this one is the
    more accurate.

    The permutation is built once per matrix and reused across all five power
    iterations, which is what makes the up-front sort pay for itself.
    """

    __slots__ = ("order", "offsets", "gather", "weights", "n_out", "width")

    def __init__(self, indptr, gather_index, weights, n_out: int):
        counts = np.diff(indptr).astype(np.int64)
        self.n_out = n_out
        # Descending length. Stable so equal-length segments keep their order —
        # not required for correctness, only for reproducible plans.
        self.order = np.argsort(-counts, kind="stable")
        self.width = int(counts.max()) if counts.size else 0
        if self.width == 0:
            self.offsets = np.zeros(1, dtype=np.int64)
            self.gather = gather_index[:0]
            self.weights = weights[:0]
            return

        # m[t] = number of segments holding at least t + 1 postings, i.e. the
        # length of the active prefix at rank t. A reversed cumulative histogram
        # of the segment lengths gives the whole staircase in one pass.
        histogram = np.bincount(counts, minlength=self.width + 1)
        m = np.cumsum(histogram[::-1])[::-1][1 : self.width + 1]
        self.offsets = np.zeros(self.width + 1, dtype=np.int64)
        np.cumsum(m, out=self.offsets[1:])

        permutation = np.empty(int(self.offsets[-1]), dtype=np.int64)
        base = indptr[self.order]
        for t in range(self.width):
            permutation[self.offsets[t] : self.offsets[t + 1]] = base[: m[t]] + t
        self.gather = gather_index[permutation]
        self.weights = weights[permutation]

    def apply(self, dense: np.ndarray) -> np.ndarray:
        accumulator = np.zeros((self.n_out, dense.shape[1]), dtype=np.float64)
        gather, weights = self.gather, self.weights
        # .tolist() once: numpy scalars in a loop this hot cost more than the
        # slicing they index.
        offsets = self.offsets.tolist()
        for t in range(self.width):
            start, stop = offsets[t], offsets[t + 1]
            accumulator[: stop - start] += (
                dense[gather[start:stop]] * weights[start:stop, None]
            )
        out = np.empty((self.n_out, dense.shape[1]), dtype=np.float32)
        out[self.order] = accumulator          # undo the length sort
        return out


class _CSR:
    __slots__ = ("indptr", "indices", "_data", "shape", "_rows",
                 "_row_plan", "_column_plan")

    def __init__(self, indptr, indices, data, shape):
        self.indptr = indptr
        self.indices = indices
        self.shape = shape
        self._rows = np.repeat(
            np.arange(shape[0], dtype=np.int32), np.diff(indptr)
        )
        self._data = data
        self._row_plan: _RankMajorPlan | None = None
        self._column_plan: _RankMajorPlan | None = None

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray) -> None:
        # The plans bake the posting weights in, so reweighting the matrix (row
        # normalisation does exactly that) has to drop them.
        self._data = value
        self._row_plan = None
        self._column_plan = None

    def dot(self, dense: np.ndarray) -> np.ndarray:
        """A @ dense — sum each row's postings."""
        if self._row_plan is None:
            self._row_plan = _RankMajorPlan(
                self.indptr, self.indices, self._data, self.shape[0]
            )
        return self._row_plan.apply(dense)

    def rdot(self, dense: np.ndarray) -> np.ndarray:
        """A.T @ dense — sum each column's postings."""
        if self._column_plan is None:
            # Transpose to CSC once. A stable sort on the column ids leaves each
            # column's postings in ascending row order, which is the order the
            # old per-column bincount accumulated them in.
            order = np.argsort(self.indices, kind="stable")
            indptr = np.zeros(self.shape[1] + 1, dtype=np.int64)
            np.cumsum(
                np.bincount(self.indices, minlength=self.shape[1]),
                out=indptr[1:],
            )
            self._column_plan = _RankMajorPlan(
                indptr, self._rows[order], self._data[order], self.shape[1]
            )
        return self._column_plan.apply(dense)


def _randomised_svd(matrix: _CSR, rank: int, *, n_iter: int = 4, oversample: int = 12,
                    seed: int = 0):
    """Halko-Martinsson-Tropp randomised range finder + small dense SVD."""
    rng = np.random.default_rng(seed)
    sketch = rng.standard_normal((matrix.shape[1], rank + oversample)).astype(np.float32)
    basis, _ = np.linalg.qr(matrix.dot(sketch))
    for _ in range(n_iter):
        basis, _ = np.linalg.qr(matrix.rdot(basis))
        basis, _ = np.linalg.qr(matrix.dot(basis))
    projected = matrix.rdot(basis).T          # (rank+p, vocab)
    u_small, singular, vt = np.linalg.svd(projected, full_matrices=False)
    left = basis @ u_small
    return left[:, :rank], singular[:rank], vt[:rank]


# --------------------------------------------------------------------------- #
# Local provider
# --------------------------------------------------------------------------- #

class LSAEmbedder:
    """TF-IDF -> randomised truncated SVD. Trains on the corpus in seconds."""

    name = "lsa-local"

    def __init__(self, dim: int = 256, min_df: int = 2, max_df_ratio: float = 0.5,
                 max_features: int = 40_000, seed: int = 0):
        self.dim = dim
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.max_features = max_features
        self.seed = seed
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0, dtype=np.float32)
        self.components = np.zeros((0, 0), dtype=np.float32)  # (vocab, dim)
        self._doc_vectors: np.ndarray | None = None

    # -- internals ---------------------------------------------------------- #
    def _counts(self, texts: Sequence[str]) -> list[dict[str, int]]:
        rows = []
        for text in texts:
            counts: dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            rows.append(counts)
        return rows

    def _to_csr(self, rows: list[dict[str, int]]) -> _CSR:
        indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        indices: list[int] = []
        data: list[float] = []
        for i, counts in enumerate(rows):
            entries = []
            for term, tf in counts.items():
                term_id = self.vocab.get(term)
                if term_id is not None:
                    # sublinear tf damping — one term repeated 40 times should
                    # not dominate a 300-token chunk
                    entries.append((term_id, (1.0 + np.log(tf)) * self.idf[term_id]))
            entries.sort()
            indptr[i + 1] = indptr[i] + len(entries)
            for term_id, weight in entries:
                indices.append(term_id)
                data.append(weight)
        csr = _CSR(
            indptr,
            np.asarray(indices, dtype=np.int32),
            np.asarray(data, dtype=np.float32),
            (len(rows), len(self.vocab)),
        )
        # cosine-normalise each row so document length does not bias the SVD
        sq = np.zeros(csr.shape[0], dtype=np.float32)
        counts_per_row = np.diff(indptr)
        nonempty = np.nonzero(counts_per_row > 0)[0]
        if csr.data.size:
            sq[nonempty] = np.add.reduceat(csr.data ** 2, indptr[nonempty])
        scale = 1.0 / np.sqrt(np.maximum(sq, 1e-12))
        csr.data = csr.data * scale[csr._rows]
        return csr

    # -- public ------------------------------------------------------------- #
    def fit(self, documents: Sequence[str]) -> "LSAEmbedder":
        rows = self._counts(documents)
        n_docs = len(rows)
        df: dict[str, int] = {}
        for counts in rows:
            for term in counts:
                df[term] = df.get(term, 0) + 1

        max_df = max(int(self.max_df_ratio * n_docs), 1)
        kept = [t for t, d in df.items() if self.min_df <= d <= max_df]
        kept.sort(key=lambda t: (-df[t], t))
        kept = sorted(kept[: self.max_features])

        self.vocab = {t: i for i, t in enumerate(kept)}
        self.idf = np.array(
            [np.log((1.0 + n_docs) / (1.0 + df[t])) + 1.0 for t in kept],
            dtype=np.float32,
        )

        matrix = self._to_csr(rows)
        rank = int(min(self.dim, max(2, min(matrix.shape) - 1)))
        left, singular, vt = _randomised_svd(matrix, rank, seed=self.seed)
        self.dim = rank
        self.components = np.ascontiguousarray(vt.T)          # (vocab, dim)
        self._doc_vectors = l2_normalise(left * singular)     # (n_docs, dim)
        return self

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if self._doc_vectors is not None and len(texts) == len(self._doc_vectors):
            # fit() already produced these exactly; don't recompute
            return self._doc_vectors
        return self.embed_queries(texts)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Fold unseen text into the latent space: q_tfidf @ V."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            counts: dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            acc = np.zeros(self.dim, dtype=np.float32)
            for term, tf in counts.items():
                term_id = self.vocab.get(term)
                if term_id is None:
                    continue
                acc += (1.0 + np.log(tf)) * self.idf[term_id] * self.components[term_id]
            out[i] = acc
        return l2_normalise(out)

    # -- persistence -------------------------------------------------------- #
    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            terms=np.array(list(self.vocab.keys()), dtype=object),
            idf=self.idf,
            components=self.components,
            dim=np.array([self.dim]),
        )

    @classmethod
    def load(cls, path: str) -> "LSAEmbedder":
        blob = np.load(path, allow_pickle=True)
        obj = cls(dim=int(blob["dim"][0]))
        obj.vocab = {t: i for i, t in enumerate(blob["terms"].tolist())}
        obj.idf = blob["idf"]
        obj.components = blob["components"]
        return obj


# --------------------------------------------------------------------------- #
# API-backed providers — same three members, activate by passing them in.
# --------------------------------------------------------------------------- #

class _BatchedAPIEmbedder:
    name = "api"
    dim = 0
    batch_size = 96

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, is_query=True)


class OpenAIEmbedder(_BatchedAPIEmbedder):
    name = "openai"

    def __init__(self, model: str = "text-embedding-3-large", dim: int = 1024):
        from openai import OpenAI  # noqa: PLC0415 — optional dependency

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model
        self.dim = dim

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[i : i + self.batch_size]]
            resp = self.client.embeddings.create(
                model=self.model, input=batch, dimensions=self.dim
            )
            vectors.extend(item.embedding for item in resp.data)
        return l2_normalise(np.asarray(vectors, dtype=np.float32))


class VoyageEmbedder(_BatchedAPIEmbedder):
    """Voyage is the provider Anthropic points at for embeddings."""

    name = "voyage"

    def __init__(self, model: str = "voyage-3", dim: int = 1024):
        import voyageai  # noqa: PLC0415 — optional dependency

        self.client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
        self.model = model
        self.dim = dim

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            resp = self.client.embed(
                list(texts[i : i + self.batch_size]),
                model=self.model,
                input_type="query" if is_query else "document",
            )
            vectors.extend(resp.embeddings)
        return l2_normalise(np.asarray(vectors, dtype=np.float32))


class SentenceTransformerEmbedder(_BatchedAPIEmbedder):
    """Local neural encoder — BGE / E5 / GTE all load through this."""

    name = "sentence-transformers"

    def __init__(self, model: str = "BAAI/bge-base-en-v1.5",
                 query_prefix: str = "Represent this sentence for searching relevant passages: "):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model = SentenceTransformer(model)
        # sentence-transformers 5.x renamed this; support both rather than
        # pinning a version, since the provider is an optional extra and the
        # user's installed version is not ours to choose.
        dimension = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = int(dimension())
        self.query_prefix = query_prefix

    def _encode(self, texts: Sequence[str], *, is_query: bool) -> np.ndarray:
        payload = [self.query_prefix + t for t in texts] if is_query else list(texts)
        vectors = self.model.encode(payload, batch_size=self.batch_size,
                                    show_progress_bar=False)
        return l2_normalise(np.asarray(vectors, dtype=np.float32))
