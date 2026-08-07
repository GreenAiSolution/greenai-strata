"""The sparse products under `LSAEmbedder.fit` must not drift when made faster.

`_CSR.dot` / `_CSR.rdot` were rewritten from (reduceat over an `(nnz, k)`
temporary) and (one `np.bincount` per output column) to a single rank-major
accumulation. That is a pure performance change, so the reference
implementations are kept here verbatim and every test compares against them.

`rdot` is expected to be *bit identical*: the rank-major pass visits each
column's postings in ascending row order and accumulates in float64, which is
exactly what `np.bincount` did. `dot` is expected to agree to float32 rounding
rather than exactly, because the old path let `np.add.reduceat` sum in float32
with its own internal unrolling; the new path accumulates the same products in
float64, so where the two differ the new one is the more accurate.
"""

from __future__ import annotations

import numpy as np
import pytest

from strata.embed import LSAEmbedder, _CSR, _randomised_svd, l2_normalise


# --------------------------------------------------------------------------- #
# The implementations as they stood before the optimisation.
# --------------------------------------------------------------------------- #

def reference_dot(csr: _CSR, dense: np.ndarray) -> np.ndarray:
    out = np.zeros((csr.shape[0], dense.shape[1]), dtype=np.float32)
    if csr.data.size == 0:
        return out
    prod = dense[csr.indices] * csr.data[:, None]
    counts = np.diff(csr.indptr)
    nonempty = np.nonzero(counts > 0)[0]
    out[nonempty] = np.add.reduceat(prod, csr.indptr[nonempty], axis=0)
    return out


def reference_rdot(csr: _CSR, dense: np.ndarray) -> np.ndarray:
    out = np.zeros((csr.shape[1], dense.shape[1]), dtype=np.float32)
    gathered = dense[csr._rows]
    for j in range(dense.shape[1]):
        out[:, j] = np.bincount(
            csr.indices,
            weights=csr.data * gathered[:, j],
            minlength=csr.shape[1],
        )
    return out


def reference_randomised_svd(matrix, rank, *, n_iter=4, oversample=12, seed=0):
    rng = np.random.default_rng(seed)
    sketch = rng.standard_normal(
        (matrix.shape[1], rank + oversample)
    ).astype(np.float32)
    basis, _ = np.linalg.qr(reference_dot(matrix, sketch))
    for _ in range(n_iter):
        basis, _ = np.linalg.qr(reference_rdot(matrix, basis))
        basis, _ = np.linalg.qr(reference_dot(matrix, basis))
    projected = reference_rdot(matrix, basis).T
    u_small, singular, vt = np.linalg.svd(projected, full_matrices=False)
    left = basis @ u_small
    return left[:, :rank], singular[:rank], vt[:rank]


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def synthetic_corpus(n_docs: int = 240, seed: int = 7) -> list[str]:
    """A corpus with real latent structure, not noise.

    Documents are drawn from a handful of topics with overlapping vocabularies,
    so the SVD has something to find and the trailing singular values are not
    degenerate — which is where a reordered summation would show up worst.
    """
    rng = np.random.default_rng(seed)
    topics = [
        [f"topic{t}term{i}" for i in range(40)] + [f"shared{i}" for i in range(12)]
        for t in range(6)
    ]
    documents = []
    for d in range(n_docs):
        vocabulary = topics[d % len(topics)]
        length = int(rng.integers(25, 160))
        words = rng.choice(vocabulary, size=length, replace=True)
        documents.append(" ".join(words.tolist()))
    return documents


def build_csr(documents, **kwargs) -> tuple[LSAEmbedder, _CSR]:
    """Run the front half of `fit` and hand back the normalised TF-IDF matrix."""
    embedder = LSAEmbedder(**kwargs)
    rows = embedder._counts(documents)
    n_docs = len(rows)
    df: dict[str, int] = {}
    for counts in rows:
        for term in counts:
            df[term] = df.get(term, 0) + 1
    max_df = max(int(embedder.max_df_ratio * n_docs), 1)
    kept = [t for t, d in df.items() if embedder.min_df <= d <= max_df]
    kept.sort(key=lambda t: (-df[t], t))
    kept = sorted(kept[: embedder.max_features])
    embedder.vocab = {t: i for i, t in enumerate(kept)}
    embedder.idf = np.array(
        [np.log((1.0 + n_docs) / (1.0 + df[t])) + 1.0 for t in kept],
        dtype=np.float32,
    )
    return embedder, embedder._to_csr(rows)


# --------------------------------------------------------------------------- #
# Products
# --------------------------------------------------------------------------- #

def test_rdot_is_bit_identical_to_the_bincount_reference():
    _, matrix = build_csr(synthetic_corpus())
    dense = np.random.default_rng(0).standard_normal(
        (matrix.shape[0], 68)
    ).astype(np.float32)
    assert np.array_equal(matrix.rdot(dense), reference_rdot(matrix, dense))


def test_dot_matches_the_reduceat_reference():
    _, matrix = build_csr(synthetic_corpus())
    dense = np.random.default_rng(1).standard_normal(
        (matrix.shape[1], 68)
    ).astype(np.float32)
    fast, reference = matrix.dot(dense), reference_dot(matrix, dense)
    scale = np.abs(reference).max()
    assert np.abs(fast - reference).max() <= 1e-5 * scale


def test_products_agree_with_a_dense_matmul():
    """Both directions against ground truth, not just against the old code."""
    _, matrix = build_csr(synthetic_corpus(n_docs=60))
    dense_matrix = np.zeros(matrix.shape, dtype=np.float64)
    dense_matrix[matrix._rows, matrix.indices] = matrix.data
    rng = np.random.default_rng(2)

    right = rng.standard_normal((matrix.shape[1], 9)).astype(np.float32)
    assert np.allclose(matrix.dot(right), dense_matrix @ right, atol=1e-5)

    left = rng.standard_normal((matrix.shape[0], 9)).astype(np.float32)
    assert np.allclose(matrix.rdot(left), dense_matrix.T @ left, atol=1e-5)


@pytest.mark.parametrize("n_docs,dim", [(1, 3), (2, 3), (5, 4)])
def test_degenerate_shapes(n_docs, dim):
    """Single documents, empty rows and empty columns must not crash."""
    documents = ["alpha beta gamma", "alpha beta", "", "gamma gamma", "delta"]
    _, matrix = build_csr(documents[:n_docs], min_df=1)
    rng = np.random.default_rng(3)
    right = rng.standard_normal((matrix.shape[1], dim)).astype(np.float32)
    left = rng.standard_normal((matrix.shape[0], dim)).astype(np.float32)
    assert np.allclose(matrix.dot(right), reference_dot(matrix, right), atol=1e-6)
    assert np.array_equal(matrix.rdot(left), reference_rdot(matrix, left))


def test_empty_matrix():
    matrix = _CSR(
        np.zeros(4, dtype=np.int64),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.float32),
        (3, 5),
    )
    assert not matrix.dot(np.ones((5, 2), dtype=np.float32)).any()
    assert not matrix.rdot(np.ones((3, 2), dtype=np.float32)).any()


def test_reweighting_the_matrix_invalidates_the_cached_plans():
    """`_to_csr` mutates `data` after construction; stale plans would be wrong."""
    _, matrix = build_csr(synthetic_corpus(n_docs=40))
    dense = np.random.default_rng(4).standard_normal(
        (matrix.shape[0], 5)
    ).astype(np.float32)
    before = matrix.rdot(dense)                      # builds and caches the plan
    matrix.data = matrix.data * np.float32(2.0)      # must drop it
    assert np.array_equal(matrix.rdot(dense), reference_rdot(matrix, dense))
    assert np.allclose(matrix.rdot(dense), 2.0 * before, rtol=1e-6)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #

def test_randomised_svd_matches_the_reference():
    _, matrix = build_csr(synthetic_corpus())
    rank = 32
    left, singular, vt = _randomised_svd(matrix, rank)
    r_left, r_singular, r_vt = reference_randomised_svd(matrix, rank)

    assert np.allclose(singular, r_singular, rtol=1e-4, atol=1e-5)
    # Singular vectors are only defined up to sign; compare the projectors the
    # embeddings actually use.
    assert np.allclose(np.abs(left.T @ r_left).diagonal(), 1.0, atol=1e-3)
    assert np.allclose(np.abs(vt @ r_vt.T).diagonal(), 1.0, atol=1e-3)


def test_fit_embeddings_match_the_reference_implementation():
    """The whole point: `fit` output must be unchanged by the optimisation."""
    documents = synthetic_corpus()

    fast = LSAEmbedder(dim=32).fit(documents)

    reference = LSAEmbedder(dim=32)
    rows = reference._counts(documents)
    n_docs = len(rows)
    df: dict[str, int] = {}
    for counts in rows:
        for term in counts:
            df[term] = df.get(term, 0) + 1
    max_df = max(int(reference.max_df_ratio * n_docs), 1)
    kept = [t for t, d in df.items() if reference.min_df <= d <= max_df]
    kept.sort(key=lambda t: (-df[t], t))
    kept = sorted(kept[: reference.max_features])
    reference.vocab = {t: i for i, t in enumerate(kept)}
    reference.idf = np.array(
        [np.log((1.0 + n_docs) / (1.0 + df[t])) + 1.0 for t in kept],
        dtype=np.float32,
    )
    matrix = reference._to_csr(rows)
    rank = int(min(reference.dim, max(2, min(matrix.shape) - 1)))
    left, singular, vt = reference_randomised_svd(matrix, rank, seed=reference.seed)
    reference.dim = rank
    reference.components = np.ascontiguousarray(vt.T)
    reference._doc_vectors = l2_normalise(left * singular)

    assert fast.vocab == reference.vocab
    assert fast.dim == reference.dim

    # A singular vector pair is defined only up to a shared sign, and a 1e-7
    # nudge anywhere is enough to tip LAPACK's choice. The flip applies to the
    # left and right vector together so it cancels in every dot product the
    # pipeline takes, but it has to be undone to compare axis by axis.
    sign = np.sign((fast.components * reference.components).sum(axis=0))
    sign[sign == 0] = 1.0

    assert np.abs(fast.components * sign - reference.components).max() < 1e-4

    fast_docs, reference_docs = fast.embed_documents(documents), reference._doc_vectors
    cosine = (fast_docs * sign * reference_docs).sum(axis=1)
    assert cosine.min() > 1 - 1e-6, f"worst document cosine {cosine.min()}"

    # `shared*` terms appear in every topic and are pruned by max_df, so queries
    # are built from topic terms that survive into the vocabulary.
    queries = ["topic0term3 topic0term9", "topic2term11 topic2term12",
               "topic5term1 topic3term2 topic1term7"]
    assert all(any(t in fast.vocab for t in q.split()) for q in queries)
    fast_q, reference_q = fast.embed_queries(queries), reference.embed_queries(queries)
    assert (fast_q * sign * reference_q).sum(axis=1).min() > 1 - 1e-6

    # And the retrieval behaviour itself, which needs no sign alignment at all:
    # same scores, same neighbours, same order.
    fast_scores = fast_q @ fast_docs.T
    reference_scores = reference_q @ reference_docs.T
    assert np.abs(fast_scores - reference_scores).max() < 1e-5
    assert np.array_equal(
        np.argsort(-fast_scores, axis=1)[:, :10],
        np.argsort(-reference_scores, axis=1)[:, :10],
    )


def test_fit_is_deterministic():
    documents = synthetic_corpus(n_docs=80)
    first = LSAEmbedder(dim=16).fit(documents)
    second = LSAEmbedder(dim=16).fit(documents)
    assert np.array_equal(first._doc_vectors, second._doc_vectors)
    assert np.array_equal(first.components, second.components)
