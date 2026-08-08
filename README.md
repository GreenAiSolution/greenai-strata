# STRATA

**A layered retrieval system built to be measured, not believed.**

```
documents → chunk → ┌─ BM25 (inverted index) ─┐
                    │                          ├─ fusion (RRF | weighted α) → re-rank → trace
                    └─ dense vectors → HNSW ───┘
```

Semantic search, hybrid scoring, an ANN index, and an LLM re-ranker — every
layer written from scratch on numpy, every layer swappable behind a small
interface, and an evaluation harness that reports what each one is actually
worth on your own corpus.

The architecture is not the interesting part; everyone's diagram looks like the
one above. The interesting part is that `strata eval` will tell you, on your
data, that your lexical leg finds the right *file* while your vector leg finds
the right *paragraph*, that your feature-based re-ranker is making things worse,
that your ANN index is 12× slower than brute force at your corpus size, and that
no re-ranker can exceed 55% recall because retrieval never handed it the right
passage.

Every one of those sentences is a measured result from this repo, below,
including the ones that are unflattering.

---

## Externally comparable results: [BEIR.md](BEIR.md)

Everything measured on your own corpus is unfalsifiable by a stranger. So STRATA
also runs on [BEIR](https://arxiv.org/abs/2104.08663) — public corpora, public
relevance judgements, metrics differentially tested against NIST's `trec_eval`.

```bash
strata beir small --sweep-alpha
```

The headline is the baseline, not the product. STRATA's hand-rolled BM25 —
its own tokeniser, its own inverted index, its own Porter stemmer, no Lucene
anywhere — reproduces published BM25 nDCG@10 to a **mean absolute deviation of
0.0066** across five datasets, none off by more than 0.0102:

| dataset  | STRATA BM25 | published (Anserini, flat) |      Δ |
|----------|------------:|---------------------------:|-------:|
| nfcorpus |      0.3187 |                      0.322 | −0.0033 |
| scifact  |      0.6840 |                      0.679 | +0.0050 |
| arguana  |      0.4064 |                      0.397 | +0.0094 |
| scidocs  |      0.1539 |                      0.149 | +0.0049 |
| fiqa     |      0.2462 |                      0.236 | +0.0102 |

Getting there meant eliminating a hypothesis rather than guessing. It was not
the BM25 parameters — matching Anserini's published k1=0.9/b=0.4 makes agreement
*worse* (0.0239). It was the analyzer: Lucene stems and we did not, and adding
Porter stemming halved the deviation (0.0128 → 0.0066).

**The dense leg reproduces too.** Swap in `bge-base-en-v1.5` and STRATA lands
within **0.0004** of BAAI's own published numbers for that model (SciFact agrees
to four decimal places). Two independent reproductions — a lexical baseline from
one set of authors, a neural one from another, different toolkits — is much
stronger evidence the shared pipeline is correct than either alone.

With a real encoder in place, **fusion beats BM25 on all five datasets, every
one at p ≤ 0.001.** With the bundled offline LSA floor it wins on only two of
five. The architecture works; the dependency-free embedder was what held it
back, and both configurations are reported rather than only the flattering one.

One finding was worth acting on, and has been: at α = 0.5 the weighted hybrid
is *never* the best mode once the encoder is good — the oracle α is 0.8–0.9
everywhere, and parameter-free RRF gets within 0.001 of the best mode without
tuning anything. **RRF is now the default fusion mode** (`strata search` and
`SearchEngine.search`); weighted fusion remains one flag away
(`--mode hybrid --alpha …`). The oracle α itself was *not* adopted as a
default — it is tuned on test labels and stays labelled as such.

Full numbers, significance tests, and the three defects an adversarial audit
found — including a wrong diagnosis this project had already published — are in
[BEIR.md](BEIR.md).

---

## Quickstart

Python 3.11+ and numpy. Nothing else — no API key, no model download, no
network.

```bash
git clone <this repo> && cd greenai-strata

# 1. Index a directory tree of documents
python3 -m strata.cli --index ./index index ~/notes ~/projects

# 2. Search it
python3 -m strata.cli --index ./index search "how does the ranker break ties"

# 3. Find out whether any of it works
python3 -m strata.cli --index ./index eval -n 200

# 4. Watch the ranking move as you drag alpha
python3 -m strata.cli --index ./index serve   # http://127.0.0.1:8105
```

```bash
python3 tests/test_strata.py     # 33 assertions, ~6 seconds, no test framework

pip install -e '.[dev]' && pytest tests/    # 221 tests including the BEIR harness
```

The BEIR metrics are differentially tested against `pytrec_eval` — the binding
around NIST's `trec_eval` that produced the published BEIR tables — so nDCG@10
here means the same quantity it means in the papers. That test skips cleanly if
`pytrec_eval` is not installed.

---

## The layers

### 1. Chunking (`corpus.py`)

Splits on markdown structure first, sliding window with overlap only inside an
over-long section, heading trail carried into the indexed text. A section titled
"Hybrid scoring" whose body never repeats the phrase stays retrievable by it.
Chunking is the most under-rated stage in a RAG pipeline and the easiest place
to lose recall you can never recover downstream.

### 2. Lexical retrieval (`lexical.py`)

Okapi BM25 over a real inverted index — three flat numpy arrays plus an offset
table, so scoring is vectorised slices rather than a dict walk. `k1` and `b` are
exposed, and `explain()` returns the per-term contribution for any (query,
document) pair. That audit trail is why the lexical leg is hand-rolled rather
than imported: when a hybrid ranker misfires you need to be able to say *"BM25
gave this chunk 7.2, and here are the four terms that produced it."*

### 3. Embeddings (`embed.py`)

The default provider is a **local latent-semantic embedder**: TF-IDF with
sublinear term damping, then a randomised truncated SVD
(Halko–Martinsson–Tropp) computed in pure numpy over a hand-rolled CSR matrix.
It learns term co-occurrence structure, so it matches passages sharing no tokens
with the query. Trains on 1,300 chunks in ~2 seconds with zero dependencies.

It is **not** a claim to match a modern neural encoder. It is the honest floor.
The interface is four members:

```python
class Embedder(Protocol):
    name: str
    dim: int
    def embed_documents(self, texts) -> np.ndarray: ...
    def embed_queries(self, texts) -> np.ndarray: ...
```

`OpenAIEmbedder`, `VoyageEmbedder` and `SentenceTransformerEmbedder` (BGE / E5 /
GTE) ship in the same file. Swap one in and re-run `strata eval` — the harness
prices the upgrade per query type instead of asking you to take it on faith.

### 4. Vector index (`ann.py`)

`ExactIndex` is brute-force cosine: one matmul, exact by definition.

`HNSW` is a full implementation of Malkov & Yashunin — multi-layer navigable
small world graph, exponential level assignment, greedy descent through upper
layers, `ef`-bounded beam search at layer 0, and the Algorithm-4 neighbour
selection heuristic (keep a candidate only if it is closer to the query than to
any already-kept neighbour), which is what preserves long-range links instead of
collapsing the graph into a cluster.

Both are kept deliberately. An approximate index is only trustworthy if you
measure what it lost, so the harness reports HNSW recall **against exact search
on the same queries**. An unmeasured ANN layer is a silent recall bug.

### 5. Fusion (`fusion.py`)

- **Reciprocal Rank Fusion** — fuses ranks, ignores scores. Scale-free, no
  tuning, hard to break. **The shipped default**, on measured evidence: with a
  real encoder the oracle α is 0.8–0.9 on all five BEIR datasets, so any fixed
  untuned α loses somewhere, while RRF lands within 0.001 of the best mode on
  two datasets with nothing to tune ([BEIR.md §3](BEIR.md)).
- **Weighted score fusion** — normalises both legs over the *candidate pool*
  (not the corpus: min-maxing across 1,300 docs where 1,290 score zero squashes
  the interesting range to nothing) and interpolates with `alpha`.

Both are implemented and the harness sweeps alpha, so the choice is a
measurement rather than a preference.

One degenerate case is handled explicitly: a query whose every term is outside
the embedder's vocabulary embeds to the zero vector, making its cosine against
every document exactly 0.0 — a "dense" ranking for it is an arbitrary
tie-break. The dense leg falls back to the lexical ranking for such queries
(35 of NFCorpus's 323 under the bundled LSA embedder), and the trace says so
rather than letting coin-flip numbers pass as measurements.

### 6. Re-ranking (`rerank.py`)

Retrieval scores query and document *independently* — one vector each — because
it has to run over the whole corpus. A re-ranker sees the pair *together* and
can model interaction, where the real relevance signal lives, but only
affordably over a few dozen candidates. That asymmetry is the entire reason two
stages exist.

- `LocalCrossEncoder` — pairwise scoring over interpretable features:
  IDF-weighted term coverage, bigram/phrase hits, a smallest-window proximity
  term, title match, vector agreement. Deterministic, offline, free.
- `ClaudeReranker` — an LLM judge. Candidates are **batched into one call** and
  scored against an explicit 0–10 anchored rubric with a **strict JSON schema**
  (`output_config.format`), so the output is validated structure, not parsed
  prose. Batching cuts round-trips *and* lets the judge see candidates side by
  side, which stabilises the scale versus one isolated call per passage. The
  rubric ships as a cached system block; `stop_reason: "refusal"` is handled.

`default_reranker()` picks whichever is actually available and **reports which
one it chose** — a demo that silently degrades from an LLM judge to a heuristic
is a demo that lies about its own numbers.

> **Disclosure:** this machine has no Anthropic credentials and no `anthropic`
> SDK installed, so **`ClaudeReranker` has not been run against the live API.**
> Its batching, schema, parsing, score scaling and tie-breaking are covered by a
> mock-client test; the network call itself is unverified. Every number in the
> results below comes from `LocalCrossEncoder`. Saying otherwise would undermine
> the one thing this repo is for.

### 7. Trace

Every search returns the per-stage numbers behind the ordering: BM25 score with
its top contributing terms, cosine similarity, fused score with each leg's
share, re-ranker score and rationale, and per-stage latency. The web UI renders
all of it with an alpha slider so you can watch the ranking move.

---

## Evaluation: labels without hand-labelling

The hard part of evaluating retrieval on your own corpus is relevance labels —
and labels written by the person who built the ranker are not evidence.

STRATA uses the **Inverse Cloze Task** (Lee et al., 2019). Take a sentence out of
a passage; the sentence becomes the query and the passage it came from is the
one correct answer. Critically, **the sentence is deleted from the index before
searching**, so no system can win by matching the query back to itself. Labels
are objective, there are hundreds of them, and no system in the ablation gets an
advantage from how they were produced.

Three query variants, because they stress different legs:

| variant | construction | models |
|---|---|---|
| `verbatim` | the sentence, markdown stripped | full-sentence natural language query |
| `hard` | stopwords removed, 45% of content words dropped, order shuffled | degraded lexical overlap |
| `keyword` | the 3 highest-IDF content words | what people actually type |

Code, tables and markup are filtered out of the query sample — a JSX fragment is
near-identical to every other JSX fragment in a corpus, so the "correct" target
is genuinely ambiguous and the ablation ends up measuring noise. The filter
affects only which sentences become *queries*; everything stays *indexed*.

**Stated limitations.** ICT queries are sentences lifted from the corpus, not
questions typed by a human — they measure passage discrimination, which is what
a retriever must do, but they are not a substitute for query logs. And on a
corpus with sibling sections (`PROGRESS.md` phases 8–16, four sales playbooks),
the single-correct-chunk label is often unfair: retrieving "Phase 13" when the
target was "Phase 10" scores zero. That is why every table below reports
**document-level recall alongside chunk-level** — and the gap between the two
turned out to be the most informative result in the whole exercise.

---

## Results

Corpus: **1,323 chunks from 186 markdown files** (working project documentation
and notes). 200 queries per variant, local LSA embedder, `LocalCrossEncoder`
re-ranking the top 25. Reproduce with:

```bash
python3 -m strata.cli --index ./index eval -n 200 --rerank local
```

### `verbatim` — full-sentence queries

| system | R@1 | R@5 | R@10 | MRR | nDCG@10 | **docR@10** |
|---|---|---|---|---|---|---|
| BM25 only | 0.070 | 0.285 | 0.400 | 0.164 | 0.219 | **0.845** |
| vectors only | **0.125** | **0.365** | **0.460** | **0.222** | **0.279** | 0.740 |
| RRF | 0.085 | 0.295 | 0.440 | 0.188 | 0.248 | 0.800 |
| hybrid α=0.3 | 0.090 | 0.315 | 0.430 | 0.190 | 0.247 | **0.845** |
| hybrid α=0.5 | 0.095 | 0.330 | 0.445 | 0.198 | 0.257 | 0.810 |
| hybrid α=0.7 | 0.105 | 0.355 | 0.450 | 0.210 | 0.267 | 0.785 |
| hybrid α=0.7 + local re-rank | 0.040 | 0.260 | 0.370 | 0.130 | 0.186 | 0.825 |

*Re-ranking headroom: the target reached the pool@25 for **55.0%** of queries.*

### `hard` — 45% of content words dropped and shuffled

| system | R@1 | R@10 | MRR | nDCG@10 | **docR@10** |
|---|---|---|---|---|---|
| BM25 only | 0.050 | 0.280 | 0.117 | 0.156 | **0.745** |
| vectors only | 0.055 | 0.320 | 0.132 | 0.177 | 0.645 |
| RRF | 0.075 | 0.290 | 0.132 | 0.169 | 0.685 |
| hybrid α=0.3 | 0.070 | 0.305 | 0.136 | 0.176 | **0.745** |
| **hybrid α=0.5** | **0.090** | **0.325** | **0.150** | **0.191** | 0.715 |
| hybrid α=0.7 | 0.075 | 0.320 | 0.142 | 0.184 | 0.680 |
| hybrid α=0.5 + local re-rank | 0.035 | 0.280 | 0.102 | 0.144 | 0.745 |

*Headroom: **46.5%**.*

### `keyword` — three high-IDF words

| system | R@1 | R@10 | MRR | nDCG@10 | **docR@10** |
|---|---|---|---|---|---|
| BM25 only | **0.065** | 0.165 | 0.090 | 0.108 | 0.580 |
| vectors only | 0.035 | 0.260 | 0.094 | **0.133** | 0.500 |
| RRF | 0.050 | 0.205 | 0.087 | 0.114 | 0.530 |
| hybrid α=0.3 | 0.060 | 0.215 | 0.096 | 0.124 | **0.610** |
| hybrid α=0.5 | 0.055 | 0.240 | **0.098** | 0.131 | 0.595 |
| hybrid α=0.7 | 0.045 | **0.250** | 0.094 | 0.131 | 0.575 |
| hybrid α=0.5 + local re-rank | 0.035 | 0.220 | 0.080 | 0.113 | 0.590 |

*Headroom: **33.5%**.*

### What the numbers actually say

**1. The lexical and semantic legs disagree about what they are good at — and
that dissociation is the case for hybrid.** On `verbatim`, BM25 has the best
*document*-level recall (0.845 vs 0.740) while the vector leg has the best
*chunk*-level scores across the board (nDCG 0.279 vs 0.219). BM25 reliably finds
the right file and then picks the wrong section inside it, because sibling
sections share the file's vocabulary. The vector leg is fuzzier about which file
but sharper about which paragraph. Neither leg is "better"; they fail in
different directions, which is exactly the condition under which fusion is worth
its complexity.

**2. Weighted fusion at α=0.3 is the best single configuration, and it is not
the one with the best headline number.** It ties BM25's document recall (0.845)
while beating it on chunk-level nDCG (0.247 vs 0.219), and it is the only system
within 0.02 nDCG of the leader on all three variants. Picking α=0.7 would score
better on `verbatim` and lose 6 points of document recall. That trade is a
product decision, and the harness is what makes it visible instead of accidental.

**3. RRF loses to tuned weighted fusion here, and the reason is diagnosable.**
RRF gives both legs an equal vote regardless of quality. On `keyword` queries,
where the LSA vector leg is weakest, an equal vote costs it: RRF scores 0.114
nDCG against weighted fusion's 0.131 at α=0.5. The tuning knob earns its keep
precisely when one leg is weaker — which is the common case in production, and
the reason to implement both rather than pick one on principle. (This table is
the weak-encoder case; the shipped default is still RRF, chosen on the BEIR
evidence with a strong encoder where a fixed untuned α is never best — both
measurements are reported, and the knob is one flag away when your corpus
rewards it.)

**4. The feature-based re-ranker makes things worse, consistently.** −0.08 nDCG
on `verbatim`, −0.05 on `hard`. This is not a bug, it is the correct result: the
`LocalCrossEncoder`'s features are lexical (term coverage, phrase hits,
proximity), and ICT deliberately deletes the query sentence from the target
passage — so it removes exactly the evidence this re-ranker scores on. A
re-ranker whose features duplicate the retriever's cannot add information; it
can only add variance. Interestingly, document-level recall is barely dented
(0.825 vs 0.785), so it is shuffling within the right file rather than throwing
the file away.

This is the strongest argument in the repo for a *semantic* judge, and the
honest place to note that the `ClaudeReranker` which would test that claim has
not been run here.

**5. The ceiling is 33–55%, and that is a retrieval problem, not a ranking
problem.** The harness reports how often the correct passage even reached the
candidate pool. No judge, however good, gets past that number. If you want a
better system on this corpus, the next move is widening the pool, better
chunking, or a stronger encoder — not a smarter re-ranker. This is the metric
most RAG demos never print, and it is the one that would have stopped a lot of
wasted re-ranker tuning.

**6. Absolute numbers are low because the label is harsh.** R@1 of 0.07–0.13
looks alarming until you look at the document column: the right *file* is in the
top 10 for 85% of full-sentence queries. Most of the chunk-level "misses" are
sibling sections of the correct document. Both numbers are reported because
either one alone tells a misleading story — and which one you should optimise
depends on whether your downstream consumer needs a paragraph or a document.

### HNSW fidelity — and why not to switch it on here

| ef | recall@10 vs exact | latency/query | exact latency/query |
|---|---|---|---|
| 32 | 0.855 – 0.993 | 0.41 ms | 0.03 ms |
| 64 | 0.874 – 0.999 | 0.60 ms | 0.03 ms |
| 128 | 0.879 – 1.000 | 0.92 ms | 0.03 ms |

The graph is correct — recall converges to 1.000 on full-sentence queries as
`ef` grows, which is what a working HNSW does. It is also **12–30× slower than
brute force at this scale, and the honest recommendation is not to use it
here.** 1,323 × 256 floats is a single BLAS matmul in 0.03 ms; a Python graph
traversal cannot compete. ANN pays once O(n·d) stops fitting in cache — typically
10⁵–10⁶ vectors, further out still for a pure-Python implementation.

Note the low end of the recall range: on short `keyword` queries HNSW recall
drops to 0.855 at ef=32 and plateaus at 0.879. Sparse query vectors land in
low-density regions of the graph where greedy descent gets stuck. Raising `ef`
does not fix it — that is a property of the embedding, not the index.

Shipping the measurement that says *don't turn this on yet* is the point. The
alternative — enabling ANN because it sounds sophisticated, and quietly paying
30× latency for 12% lost recall — is the failure mode this harness exists to
catch.

### One more result, from the test suite

An early version of the harness reported nDCG around **0.80** across the board.
Those numbers were wrong. `sentences()` returns whitespace-collapsed text, so a
target sentence spanning a line break silently failed to match the raw chunk
during masking and was never removed — leaving the query as a literal substring
of its own answer. Every system scored beautifully and the ablation was
meaningless.

The regression test that catches it (`test_evaluation_masks_the_answer`) is now
the single most important assertion in the file. The correct numbers, above, are
a quarter of the size. That is what an evaluation harness is *for*: the failure
mode of RAG work is not a system that breaks loudly, it is a system that reports
0.80 while doing nothing.

---

## Using it as a library

```python
from strata import build_corpus, SearchEngine, ClaudeReranker

corpus = build_corpus(["~/docs"], max_tokens=320, overlap=60,
                      exclude=("node_modules", "vendor"))
engine = SearchEngine.build(corpus)                 # BM25 + vectors + HNSW

hits, trace = engine.search(
    "how do we handle vocabulary mismatch",
    mode="hybrid", alpha=0.3, k=10,
    reranker=ClaudeReranker(model="claude-opus-5", batch_size=8),
    rerank_depth=25,
)

for hit in hits:
    print(f"{hit.score:.3f}  bm25={hit.bm25:.2f}  cos={hit.vector:.3f}  {hit.title}")
    print(f"        terms: {hit.terms}   why: {hit.rationale}")
```

Swapping the embedding provider is one argument:

```python
from strata import SearchEngine, VoyageEmbedder, SentenceTransformerEmbedder

engine = SearchEngine.build(corpus, embedder=VoyageEmbedder("voyage-3"))
engine = SearchEngine.build(corpus, embedder=SentenceTransformerEmbedder("BAAI/bge-base-en-v1.5"))
```

---

## Layout

```
strata/
  text.py       tokenisation, sentence splitting, stopwords
  corpus.py     structure-aware chunking with heading trails + provenance
  lexical.py    Okapi BM25 over a flat-array inverted index, with explain()
  embed.py      Embedder protocol; local LSA (TF-IDF + randomised SVD); OpenAI / Voyage / ST
  ann.py        ExactIndex (brute force) + HNSW (Malkov & Yashunin)
  fusion.py     RRF and weighted score fusion with pool-local normalisation
  rerank.py     LocalCrossEncoder (offline features) + ClaudeReranker (batched LLM judge)
  pipeline.py   SearchEngine: build → retrieve → fuse → re-rank, with a full trace
  evaluate.py   ICT query generation, chunk+document metrics, ablation, ANN recall, ceiling
  metrics.py    nDCG / Recall / MAP / MRR to trec_eval semantics + paired bootstrap
  beir.py       BEIR download, parse, and adaptation to Corpus (1 doc = 1 chunk)
  beir_eval.py  BEIR runner: all modes over one index, significance, HNSW crossover
  reference.py  published BM25 baselines, transcribed and cited per table
  server.py     stdlib web UI showing per-stage scores and an alpha slider
  cli.py        index / search / eval / beir / serve
scripts/
  ann_crossover.py  where HNSW starts beating brute force, swept over corpus size
tests/
  test_strata.py                33 assertions, no test framework required
  test_metrics.py               hand-worked metric fixtures
  test_metrics_vs_pytrec_eval.py differential test against NIST trec_eval
  test_beir_eval.py             adapter correctness + fusion-equivalence pin
  test_reference.py             guards on the transcribed published baselines
```

~3,400 lines. numpy is still the only runtime dependency; every API-backed
provider is imported lazily so the system runs fully offline by default. The
BEIR harness adds `certifi` (only to download the archives) and `pytest` plus
`pytrec_eval-terrier` as dev extras.

---

## Design notes worth arguing about

**Why hand-roll BM25 and HNSW instead of importing them?** Because the failure
mode of a hybrid system is a ranking you cannot explain, and you cannot explain
what you cannot open. `explain()` returning per-term BM25 contributions and
`ann_recall()` returning what the graph lost are only possible because the
implementations are ours. (The production sibling of this HNSW — same algorithm,
JavaScript, zero dependencies — runs inside another of my projects; this is the
Python port with the measurement harness bolted on.)

**Why normalise fusion scores over the candidate pool rather than the corpus?**
Corpus-wide min-max is dominated by the long tail of zeros, which compresses the
top candidates into a narrow band and makes alpha nearly inert.

**Why report document-level recall as well as chunk-level?** Because on a real
corpus the single-correct-chunk assumption is frequently false, and a system
that looks broken at chunk level (R@1 = 0.07) can be putting the right file in
front of the user 85% of the time. Reporting only the flattering number is
marketing; reporting only the harsh one is self-flagellation. Report both and
say which one your product needs.

**Why the ceiling metric?** Because re-ranker improvements are the easiest thing
in RAG to over-claim. If the correct passage reaches the pool 55% of the time,
the re-ranker's honest maximum is 0.55 — and a lot of reported "re-ranking
lifted our numbers" results are really "we widened the pool."

**Why a rubric and a strict schema for the LLM judge?** Free-text relevance
scoring drifts: the same passage gets a 7 in one call and a 4 in the next
depending on what it was compared against. An anchored scale, batched
side-by-side judging, and schema-validated integers remove the parsing failure
mode and most of the scale drift at once.

---

## Roadmap

- Run `ClaudeReranker` against the live API and publish the delta — the one
  claim in this repo that is currently argued rather than measured
- Query expansion / HyDE as an optional retrieval leg, measured the same way
- Learned fusion weights (logistic regression on ICT labels) vs. hand-tuned α
- Batch API path for the judge — 50% cheaper for offline evaluation runs
- Graded relevance labels, so nDCG uses more than a binary target
- A crossover benchmark for HNSW vs exact at 10⁴–10⁶ vectors
