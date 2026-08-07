"""Measure `LSAEmbedder.fit` against the implementation it replaced.

`fit` used to dominate index construction — around 12x the cost of building the
BM25 inverted index at every corpus size. Almost all of that was two sparse
products inside the randomised SVD, so this script times the products and the
whole of `fit` both ways, in one process, on real BEIR corpora.

The "before" path is a verbatim copy of the old `_CSR.dot` / `_CSR.rdot` /
`_randomised_svd`, so the comparison is like for like rather than against a
number recorded on some other machine. It also reports how far the outputs
moved, because a speedup that changes the embeddings is not a speedup.

    python scripts/bench_embed.py scifact scidocs fiqa
"""

from __future__ import annotations

import argparse
import gc
import sys
import time

import numpy as np

from strata import beir
from strata.embed import LSAEmbedder, _randomised_svd, l2_normalise


# --------------------------------------------------------------------------- #
# The pre-optimisation implementation, kept verbatim for comparison.
# --------------------------------------------------------------------------- #

def old_dot(csr, dense):
    out = np.zeros((csr.shape[0], dense.shape[1]), dtype=np.float32)
    if csr.data.size == 0:
        return out
    prod = dense[csr.indices] * csr.data[:, None]
    counts = np.diff(csr.indptr)
    nonempty = np.nonzero(counts > 0)[0]
    out[nonempty] = np.add.reduceat(prod, csr.indptr[nonempty], axis=0)
    return out


def old_rdot(csr, dense):
    out = np.zeros((csr.shape[1], dense.shape[1]), dtype=np.float32)
    gathered = dense[csr._rows]
    for j in range(dense.shape[1]):
        out[:, j] = np.bincount(
            csr.indices,
            weights=csr.data * gathered[:, j],
            minlength=csr.shape[1],
        )
    return out


def old_randomised_svd(matrix, rank, *, n_iter=4, oversample=12, seed=0):
    rng = np.random.default_rng(seed)
    sketch = rng.standard_normal((matrix.shape[1], rank + oversample)).astype(np.float32)
    basis, _ = np.linalg.qr(old_dot(matrix, sketch))
    for _ in range(n_iter):
        basis, _ = np.linalg.qr(old_rdot(matrix, basis))
        basis, _ = np.linalg.qr(old_dot(matrix, basis))
    projected = old_rdot(matrix, basis).T
    u_small, singular, vt = np.linalg.svd(projected, full_matrices=False)
    left = basis @ u_small
    return left[:, :rank], singular[:rank], vt[:rank]


# --------------------------------------------------------------------------- #

def build_matrix(embedder: LSAEmbedder, documents):
    """The front half of `fit`: vocabulary, idf and the normalised TF-IDF CSR."""
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
    return embedder._to_csr(rows)


def timed(fn):
    gc.collect()
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def run(name: str, products_only: bool) -> None:
    dataset = beir.load(name, verbose=False)
    documents = [f"{title} {text}".strip() for title, text in dataset.documents.values()]

    embedder = LSAEmbedder()
    prep, matrix = timed(lambda: build_matrix(embedder, documents))
    nnz = matrix.data.size
    rank = int(min(embedder.dim, max(2, min(matrix.shape) - 1)))
    width = rank + 12
    print(f"\n{name}: {matrix.shape[0]:,} docs x {matrix.shape[1]:,} terms, "
          f"{nnz:,} postings, rank {rank} (+12 oversample)")
    print(f"  tokenise + tf-idf + csr        {prep:7.2f}s   (unchanged, shared by both)")

    rng = np.random.default_rng(0)
    right = rng.standard_normal((matrix.shape[1], width)).astype(np.float32)
    left = rng.standard_normal((matrix.shape[0], width)).astype(np.float32)

    t_old_dot, ref_dot = timed(lambda: old_dot(matrix, right))
    t_old_rdot, ref_rdot = timed(lambda: old_rdot(matrix, left))
    del ref_dot, ref_rdot
    t_new_dot, new_d = timed(lambda: matrix.dot(right))       # includes plan build
    t_new_rdot, new_r = timed(lambda: matrix.rdot(left))      # includes transpose
    t_warm_dot, _ = timed(lambda: matrix.dot(right))
    t_warm_rdot, _ = timed(lambda: matrix.rdot(left))

    _, ref_dot = timed(lambda: old_dot(matrix, right))
    _, ref_rdot = timed(lambda: old_rdot(matrix, left))
    d_err = np.abs(new_d - ref_dot).max() / max(float(np.abs(ref_dot).max()), 1e-30)
    r_exact = np.array_equal(new_r, ref_rdot)
    del ref_dot, ref_rdot, new_d, new_r
    gc.collect()

    print(f"  dot   A @ X   before {t_old_dot:6.2f}s  after {t_new_dot:6.2f}s "
          f"(warm {t_warm_dot:5.2f}s)  {t_old_dot / t_new_dot:5.2f}x   "
          f"max rel diff {d_err:.1e}")
    print(f"  rdot  A.T @ Y before {t_old_rdot:6.2f}s  after {t_new_rdot:6.2f}s "
          f"(warm {t_warm_rdot:5.2f}s)  {t_old_rdot / t_new_rdot:5.2f}x   "
          f"bit identical: {r_exact}")

    if products_only:
        return

    t_old_svd, (ol, old_sigma, ovt) = timed(lambda: old_randomised_svd(matrix, rank, seed=0))
    gc.collect()
    t_new_svd, (nl, new_sigma, nvt) = timed(lambda: _randomised_svd(matrix, rank, seed=0))
    gc.collect()

    # A singular vector pair is only defined up to a shared sign, and a 1e-7
    # nudge is enough to tip LAPACK's choice. The flip lands on the left and
    # right vector together, so every downstream dot product is unaffected —
    # but it has to be undone before the two runs can be compared elementwise.
    sign = np.sign((ol * nl).sum(axis=0)).astype(np.float32)
    flipped = int((sign < 0).sum())
    old_docs = l2_normalise(ol * old_sigma)
    new_docs = l2_normalise((nl * sign) * new_sigma)
    old_components = np.ascontiguousarray(ovt.T)
    new_components = np.ascontiguousarray((nvt * sign[:, None]).T)
    del ol, nl
    gc.collect()

    # Documents with no in-vocabulary term embed to the exact zero vector under
    # both implementations. They are identical, but a cosine against them is 0/0,
    # so they are counted rather than averaged in.
    empty = np.diff(matrix.indptr) == 0
    assert not old_docs[empty].any() and not new_docs[empty].any()
    cosine = np.abs((old_docs[~empty] * new_docs[~empty]).sum(axis=1))
    sigma_err = np.abs(new_sigma - old_sigma).max() / float(old_sigma.max())
    print(f"  randomised svd  before {t_old_svd:7.2f}s  after {t_new_svd:7.2f}s  "
          f"{t_old_svd / t_new_svd:5.2f}x")
    print(f"  equivalence: {flipped}/{rank} singular vectors sign-flipped "
          f"(free choice, cancels downstream)")
    print(f"    singular values  max rel diff {sigma_err:.2e}")
    print(f"    components       max abs diff "
          f"{np.abs(new_components - old_components).max():.2e}")
    print(f"    doc embeddings   worst |cos(old, new)| {cosine.min():.9f} "
          f"over {cosine.size:,} docs "
          f"({int(empty.sum())} empty docs zero in both)")

    # The quantity that decides retrieval: the full query x document score matrix.
    def scored(components, doc_vectors):
        probe = LSAEmbedder()
        probe.vocab, probe.idf, probe.dim = embedder.vocab, embedder.idf, rank
        probe.components = components
        return probe.embed_queries(queries) @ doc_vectors.T

    queries = list(dataset.queries.values())
    old_scores = scored(old_components, old_docs)
    new_scores = scored(new_components, new_docs)
    identical = np.array_equal(np.argsort(-old_scores, axis=1)[:, :10],
                               np.argsort(-new_scores, axis=1)[:, :10])
    print(f"    scores ({len(queries)} queries)  max abs diff "
          f"{np.abs(old_scores - new_scores).max():.2e}, "
          f"top-10 ranking identical: {identical}")

    total_old, total_new = prep + t_old_svd, prep + t_new_svd
    print(f"  FIT TOTAL       before {total_old:7.2f}s  after {total_new:7.2f}s  "
          f"{total_old / total_new:5.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=["scifact", "scidocs", "fiqa"])
    parser.add_argument("--products-only", action="store_true",
                        help="skip the full SVD comparison; time the two products only")
    args = parser.parse_args()
    for name in args.datasets:
        run(name, args.products_only)


if __name__ == "__main__":
    sys.exit(main())
