# Redrob Hackathon — submission dossier

The call asks whether we've worked on semantic search, RAG pipelines, embedding
models, vector databases, LLM re-ranking, hybrid search, and search relevance
optimisation.

Rather than answer with a list of claims, this repo answers with a system you
can clone and run in under a minute, and — more to the point — with the
**measurements**, including the ones that came out badly.

```bash
python3 -m strata.cli --index ./index index <any directory of documents>
python3 -m strata.cli --index ./index eval -n 200
```

---

## Point-by-point

### Semantic search / RAG pipelines

The full pipeline: structure-aware chunking with heading trails and provenance
(`corpus.py`) → dual retrieval → fusion → re-ranking → a per-stage trace exposed
in the CLI, the JSON API and the web UI (`pipeline.py`, `server.py`).

The design choice worth defending: **every stage emits its own score, and the
final ranking carries the decomposition.** A result row shows its BM25 value
*and the four terms that produced it*, its cosine similarity, each leg's share
of the fused score, and the re-ranker's verdict with a one-line reason. Relevance
work is debugging work; a pipeline that only emits a final ordering cannot be
debugged.

### Embedding models

`embed.py` defines a four-member `Embedder` protocol and four implementations:

- **`LSAEmbedder`** — the default. TF-IDF with sublinear damping → randomised
  truncated SVD (Halko–Martinsson–Tropp), written in pure numpy over a
  hand-rolled CSR matrix (segment-sum via `reduceat` for `A@B`, per-column
  `bincount` for `Aᵀ@B`). Trains on 1,300 chunks in ~2s, no dependencies, no key.
- **`OpenAIEmbedder`**, **`VoyageEmbedder`**, **`SentenceTransformerEmbedder`**
  (BGE / E5 / GTE) — same three methods, lazily imported.

The point of the interface is that swapping providers is one constructor
argument and the evaluation harness then *prices* the swap per query type. The
local embedder is deliberately the floor, not a claim to parity — and the
results show exactly where it breaks (three-word queries, where fold-in has
almost nothing to project).

### Vector databases

`ann.py` ships a complete **HNSW** — multi-layer navigable small world graph,
exponential level assignment, greedy descent through upper layers, `ef`-bounded
beam search at layer 0, and the Algorithm-4 neighbour-selection heuristic that
preserves long-range links — plus an `ExactIndex` baseline, persistence, and a
harness that reports **ANN recall against exact search on the same queries**
(`evaluate.py:ann_recall`).

That last piece is the part that matters. The measured verdict on this corpus is
that HNSW is *correct* (recall → 1.000 as `ef` grows) and *the wrong choice*
(12–30× slower than a single BLAS matmul at 1,323 vectors). The repo ships the
recommendation not to enable it. Prior art in a different language: a
from-scratch HNSW in JavaScript, `~/greenai-godmode/server/vector.js`, 519 lines,
384-dim, pluggable embedder, running in production inside another project.

Adapters for Pinecone / Weaviate / Qdrant / Chroma / FAISS are not written —
they would all sit behind the same `search(query_vector, k)` shape that
`ExactIndex` and `HNSW` already implement, and the repo prefers one measured
implementation to five unmeasured wrappers.

### LLM-based ranking and re-ranking

`rerank.py`. Two implementations behind one call signature:

- `ClaudeReranker` — batched LLM judge. Candidates are grouped into a single
  call (fewer round-trips, *and* side-by-side comparison stabilises the scale
  versus isolated per-passage calls), scored against an explicit anchored 0–10
  rubric, constrained by a **strict JSON schema** via `output_config.format` so
  the result is validated structure rather than parsed prose. The rubric ships as
  a cached system block; `stop_reason: "refusal"` is handled; ties break on
  retrieval order for determinism.
- `LocalCrossEncoder` — offline pairwise scoring over interpretable features
  (IDF-weighted coverage, bigram hits, smallest-window proximity, title match,
  vector agreement).

**Disclosure, because it matters more than the feature list:** this machine has
no Anthropic credentials and no SDK, so the Claude judge's *network call is
unverified*. Its batching, schema construction, response parsing, score scaling
and ordering are covered by a mock-client test. Every number in the results is
from the local re-ranker. That distinction is stated in the README too.

### Hybrid search combining keyword and vector search

`fusion.py` implements both families — Reciprocal Rank Fusion (rank-based,
scale-free, untunable by design) and weighted score fusion with pool-local
normalisation and a tunable α — and `evaluate.py` sweeps α so the choice is a
measurement.

Measured finding: RRF *loses* to tuned weighted fusion on this corpus, and the
reason is diagnosable rather than mysterious — RRF gives both legs an equal vote,
which costs it precisely when one leg is weaker, and on three-word queries the
vector leg is much weaker.

### Information retrieval and search relevance optimisation

This is the part the repo is actually built around.

**Labels without hand-labelling.** Relevance judgements written by the person who
built the ranker are not evidence. The harness uses the **Inverse Cloze Task**:
lift a sentence out of a passage, use it as the query, and — critically —
*delete it from the index* so nothing can win by matching the query back to
itself. Three query variants stress different legs (full sentence, 45% of content
words dropped and shuffled, three highest-IDF words).

**Metrics:** Recall@1/5/10, MRR@10, nDCG@10, document-level recall@10, per-query
latency, ANN recall vs exact, and a **re-ranking ceiling** — how often the correct
passage even reached the candidate pool. That last one is the metric most RAG
demos never print, and on this corpus it is 33–55%, which means the next
improvement is retrieval, not ranking.

**The headline finding** is a dissociation nobody would have guessed from the
architecture diagram: BM25 has the best *document*-level recall (0.845) while the
vector leg has the best *chunk*-level scores (nDCG 0.279 vs 0.219). Lexical
retrieval finds the right file and the wrong section; semantic retrieval is
fuzzier about the file and sharper about the paragraph. They fail in different
directions — which is the actual justification for fusing them, and it is a
measurement rather than a slogan.

**A negative result, reported:** the feature-based re-ranker makes chunk-level
nDCG *worse* on every variant. That is the correct outcome, not a bug — its
features are lexical, and ICT deletes exactly the lexical evidence it scores on.
A re-ranker whose features duplicate the retriever's adds variance, not
information.

---

## The thing worth reading if you only read one

An early version of this harness reported **nDCG ≈ 0.80** across the board. Those
numbers were wrong.

`sentences()` returns whitespace-collapsed text, so a target sentence that
spanned a line break silently failed to match the raw chunk during masking and
was never removed. The query stayed a literal substring of its own answer. Every
system scored beautifully and the entire ablation was meaningless.

A regression test caught it. The corrected numbers are roughly a quarter of the
size, and they are the ones published in the README.

The failure mode of retrieval work is not a system that breaks loudly. It is a
system that reports 0.80 while doing nothing — and the only defence is a harness
you trust more than you trust your own results. That is what this repo is.

---

## Running the evidence yourself

```bash
python3 tests/test_strata.py                          # 33 assertions, ~6s
python3 -m strata.cli --index ./index eval -n 200     # the full ablation
python3 -m strata.cli --index ./index serve           # per-stage scores, α slider
```

Everything above runs offline with numpy as the only dependency. Nothing in the
results section requires an API key, and nothing in it is estimated.
