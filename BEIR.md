# STRATA on BEIR

Everything else in this repository is measured on a corpus I built myself. That
is useful for development and worthless as evidence, because nobody can compare
it to anything. This document reports STRATA on [BEIR][beir] — public corpora,
public relevance judgements, and metrics that mean the same thing as everybody
else's.

Reproduce the whole thing with:

```bash
pip install -e '.[dev]'
strata beir small --stem --sweep-alpha --out results/beir_lsa_stemmed.json
```

Datasets download to `~/.cache/strata/beir` on first use. The five-dataset suite
takes about four minutes on an M1 Max; no GPU, no API keys, and numpy is still
the only runtime dependency.

---

## 1. The result that matters: the BM25 baseline reproduces

Before any claim about hybrid retrieval is worth reading, the harness itself has
to be trustworthy. The way to check that is to run a well-known baseline and see
whether it lands where the literature says it should.

STRATA's BM25 is hand-rolled — an inverted index on flat numpy arrays, its own
tokeniser, its own Porter stemmer, Robertson/Sparck-Jones IDF. It shares no code
with Lucene, Anserini or Elasticsearch. Against published BM25 baselines on the
same data:

| dataset  | STRATA | Thakur 2021 | Kamalloo multifield | Kamalloo **flat** | Δ vs flat |
|----------|-------:|------------:|--------------------:|------------------:|----------:|
| nfcorpus | 0.3187 |       0.325 |               0.325 |             0.322 |   −0.0033 |
| scifact  | 0.6840 |       0.665 |               0.665 |             0.679 |   +0.0050 |
| arguana  | 0.4064 |       0.315 |               0.414 |             0.397 |   +0.0094 |
| scidocs  | 0.1539 |       0.158 |               0.158 |             0.149 |   +0.0049 |
| fiqa     | 0.2462 |       0.236 |               0.236 |             0.236 |   +0.0102 |

**Mean absolute deviation from the flat reference: 0.0066.** No dataset is off
by more than 0.0102.

STRATA concatenates title and body into one indexed field, so the like-for-like
column is Kamalloo et al.'s *flat* BM25, not the multi-field variant. Sources
are transcribed per-table in [`strata/reference.py`](strata/reference.py):

- **Thakur 2021** — [BEIR][beir], NeurIPS Datasets & Benchmarks, Table 2. Elasticsearch, multi-field.
- **Kamalloo 2024** — [Resources for Brewing BEIR][brewing], SIGIR, Table 3. Pyserini/Anserini, both variants.

### Getting there took eliminating two hypotheses, not one guess

The first version of this harness reported a mean deviation of 0.0128, always in
the same direction — under the reference on most datasets. A consistent signed
bias is a clue, not noise, so it was worth chasing.

**It was not the parameters.** STRATA ships BM25 at k1=1.5, b=0.75; Anserini's
BEIR configuration is k1=0.9, b=0.4. Matching a reference implementation's
published configuration is not tuning on test, so this was a free thing to try —
and it made agreement *worse*, mean |Δ| 0.0239 against our defaults' 0.0128, at
a cost of 0.09 nDCG@10 on ArguAna alone. `scripts/bm25_params.py` sweeps the
full grid; the best oracle cell (chosen by cheating and looking at the test
labels) buys at most 0.005 on any dataset. Parameters were never the story.

**It was the analyzer.** Lucene's `EnglishAnalyzer`, which Anserini uses, runs a
Porter stem filter. STRATA's tokeniser did not stem at all, so "retrieval",
"retrieved" and "retrieves" were three unrelated terms to us and one term to
Lucene — a different posting list and a different IDF for every morphological
family in the index. Implementing Porter (1980) closed most of the gap:

| dataset  | unstemmed | stemmed |   delta |     p | \|Δ\| unstemmed | \|Δ\| stemmed |
|----------|----------:|--------:|--------:|------:|---------------:|-------------:|
| nfcorpus |    0.3063 |  0.3187 | +0.0124 | 0.011 |         0.0157 |       0.0033 |
| scifact  |    0.6613 |  0.6840 | +0.0226 | 0.012 |         0.0177 |       0.0050 |
| arguana  |    0.4204 |  0.4064 | −0.0140 | 0.008 |         0.0234 |       0.0094 |
| scidocs  |    0.1497 |  0.1539 | +0.0042 | 0.115 |         0.0007 |       0.0049 |
| fiqa     |    0.2295 |  0.2462 | +0.0167 | 0.015 |         0.0065 |       0.0102 |
| **mean** |           |         |         |       |     **0.0128** |   **0.0066** |

Retrieval improves on four of five datasets, three of them significantly, and
the deviation from the reference is roughly halved. ArguAna's nDCG@10 *drops*
while moving *toward* the reference — which is the outcome that matters when the
goal is reproducing someone else's system rather than winning. Vocabulary shrinks
about 21%; index build cost rises about 13%.

Stemming is **off by default** (`BM25Index(stem=True)`, `strata beir --stem`).
Turning it on silently would change numbers already published, and both
configurations are reported here rather than only the flattering one.

### Why ArguAna looks wrong and isn't

ArguAna is the one dataset where our number sits well above the figure most
people quote — 0.4064 against the BEIR paper's 0.315. That gap is not ours. The
published BM25 values for ArguAna span **0.099** across implementations: 0.315
in the original BEIR release, 0.414 in the Anserini multi-field reproduction,
0.397 flat. Ours sits next to the reproducible Anserini numbers, and it is the
BEIR paper's 0.315 that is the outlier. Documenting exactly this kind of
disagreement is what the Kamalloo et al. paper exists for.

That 0.099 spread is wider than almost any improvement over BM25 claimed
anywhere in the retrieval literature. "We beat BM25" is not a meaningful
sentence unless it says *which* BM25, which is why `strata beir` prints all
three reference columns and refuses to average them.

I checked the boring explanation first: ArguAna draws its queries from its own
corpus (1,298 of 1,406 query ids are also document ids), so a system returning
the query's own document scores brilliantly for no skill. BEIR excludes that and
so do we, verified in `tests/test_beir_eval.py`. Self-matching was not the cause.

---

## 1b. And the dense leg reproduces too

The same argument applies to the semantic half. Swapping `LSAEmbedder` for
`bge-base-en-v1.5` is a one-line change, and BAAI publish that model's BEIR
numbers in the machine-readable `model-index` on its official model card — so
driving somebody else's encoder correctly is a second, independently checkable
claim.

```bash
strata beir small --stem --embedder st:BAAI/bge-base-en-v1.5
```

| dataset  | STRATA | BAAI published |       Δ |
|----------|-------:|---------------:|--------:|
| nfcorpus | 0.3735 |        0.37389 | −0.0003 |
| scifact  | 0.7404 |        0.74039 | +0.0000 |
| arguana  | 0.6375 |        0.63605 | +0.0014 |
| scidocs  | 0.2172 |        0.21731 | −0.0001 |
| fiqa     | 0.4062 |        0.40646 | −0.0002 |

**Mean absolute deviation: 0.0004.** SciFact agrees to four decimal places.

Two independent reproductions — a lexical baseline to 0.0066 and a neural one to
0.0004, against references from different authors using different toolkits — is
much stronger evidence that the loader, the qrels handling, the ranking and the
metric are all correct than either would be alone. A pipeline with a bug in the
shared parts could not hit both.

---

## 2. Full results

nDCG@10, five datasets, four retrieval modes, Lucene-matched analyzer, α = 0.5
untuned. **With `bge-base-en-v1.5` as the dense leg:**

| dataset  |   docs | queries |   bm25 |     vector |        rrf | hybrid | hybrid − bm25 |
|----------|-------:|--------:|-------:|-----------:|-----------:|-------:|--------------:|
| nfcorpus |  3,633 |     323 | 0.3187 |     0.3735 | **0.3741** | 0.3728 | +0.0541 (p .000) |
| scifact  |  5,183 |     300 | 0.6840 |     0.7404 | **0.7408** | 0.7401 | +0.0561 (p .000) |
| arguana  |  8,674 |   1,406 | 0.4064 | **0.6375** |     0.5542 | 0.6156 | +0.2092 (p .000) |
| scidocs  | 25,657 |   1,000 | 0.1539 | **0.2172** |     0.1988 | 0.1920 | +0.0381 (p .000) |
| fiqa     | 57,638 |     648 | 0.2462 | **0.4062** |     0.3637 | 0.3568 | +0.1106 (p .000) |

**With the bundled LSA embedder — the offline, dependency-free floor:**

| dataset  |   bm25 | vector |    rrf |     hybrid | hybrid − bm25 |
|----------|-------:|-------:|-------:|-----------:|--------------:|
| nfcorpus | 0.3187 | 0.2939 | 0.3099 | **0.3332** | +0.0145 (p .004) |
| scifact  | **0.6840** | 0.4829 | 0.5996 | 0.6675 | −0.0164 (n.s., p .128) |
| arguana  | 0.4064 | 0.4572 | 0.4568 | **0.4708** | +0.0644 (p .000) |
| scidocs  | **0.1539** | 0.0804 | 0.1201 | 0.1465 | −0.0074 (p .010) |
| fiqa     | **0.2462** | 0.0558 | 0.1384 | 0.1977 | −0.0485 (p .000) |

Every difference is tested against the BM25 baseline with a paired bootstrap
over per-query nDCG@10 (10,000 resamples).

**A note on the NFCorpus row.** 35 of NFCorpus's 323 queries embed to the zero
vector under LSA at `min_df=2` — every term out of vocabulary — so their
"dense" ranking used to be an arbitrary tie-break over 3,633 documents. The
harness now falls back to the lexical ranking for exactly those queries (the
other four datasets have none, and their numbers are bit-identical before and
after). Re-measured, the change cut three ways and all three are reported:
`vector` rose 0.2835 → 0.2939 and `rrf` 0.3090 → 0.3099, because a real signal
replaced a coin flip; but `hybrid` *fell* 0.3348 → 0.3332, because for those
queries the old candidate pool was padded with up to a thousand arbitrary
zero-cosine documents, and with 38 relevant documents per query some of that
padding landed lucky. The old hybrid number was partly coin-flip credit; the
new one is smaller and real.

---

## 3. What the two embedders say

**The architecture is validated, and the thing it depended on was the encoder.**
With `bge-base-en-v1.5` in the dense leg, combining the two legs beats BM25 on
**all five datasets, every one at p ≤ 0.001**, by between +0.04 and +0.21
nDCG@10. That is the claim this project could not make an hour ago and can now.

**But the shipped α = 0.5 is wrong for a strong encoder, and the sweep says so
loudly.** The oracle α is 0.8 on four datasets and 0.9 on the fifth — the dense
leg deserves most of the weight, and at α = 0.5 the fixed hybrid is *never* the
best mode. Pure dense wins on three datasets and RRF on two. RRF is the
interesting one: it needs no α at all, and it is within 0.001 of the best mode
on nfcorpus and scifact while being far more robust than a guessed weight. On
the evidence here **RRF is the better default than weighted fusion at α = 0.5**,
and that change has now been made to the product rather than only to the
write-up: `SearchEngine.search`, the CLI and the web UI all default to
`mode="rrf"`. Weighted fusion is unchanged and one flag away
(`--mode hybrid --alpha …`) for corpora where an α tuned on held-out queries —
not on these — earns its keep.

Note what did *not* happen: I did not quietly adopt the oracle α. It is tuned on
test labels and stays labelled as such. The defensible move is switching the
default to a parameter-free fusion, not to a number read off the answer sheet.

**The bundled LSA embedder, by contrast, does not earn its place.** On the same
five datasets it wins on two, ties on one, and loses on two. On FiQA it scores
0.0558 against BM25's 0.2462 and drags the hybrid down five points; an oracle
allowed to cheat picks α = 0.2, and on the unstemmed index it picks **0.0** —
switch the dense leg off entirely.

That is not a surprise, and the [embedder's own docstring](strata/embed.py) says
so up front — TF-IDF plus a truncated SVD is "the honest floor", not a neural
encoder. But there is a difference between predicting that in a comment and
measuring it, and the measurement is more specific than the prediction was:

- **It loses badly where the vocabulary gap is the whole task.** FiQA is
  financial questions asked in plain language against expert answers; SciDocs is
  citation prediction. LSA learns co-occurrence within *this* corpus, so it
  cannot bridge a gap that needs knowledge from outside it.
- **It wins where the query is long and the answer is a paraphrase.** ArguAna
  queries are entire arguments and the target is the counter-argument, sharing
  topic but not phrasing. It is the one dataset where the dense leg outscores
  BM25 outright (0.4572 vs 0.4064).
- **It improves Recall@1000 even where it wrecks nDCG@10.** On FiQA the dense
  leg cannot rank at all, yet RRF still lifts Recall@1000 above the lexical leg.
  It finds documents BM25 misses and merely cannot order them — precisely the
  profile of a useful *first-stage* retriever feeding a re-ranker, and an
  argument for the re-ranking layer this repo already has rather than against
  the dense leg entirely.

The honest one-line summary: **the hybrid architecture works, and the bundled
offline embedder is the thing that was holding it back.** The engine drives a
real encoder to its published numbers and fusion then beats BM25 everywhere;
with the dependency-free floor it does not. Both configurations are reported
above rather than only the flattering one, because "works with a 440 MB model
download and a GPU" and "works with numpy alone" are different products and a
reader is entitled to know which number belongs to which.

The cost of that upgrade is not nothing: encoding FiQA's 57,638 documents took
582s on an M1 Max GPU against 35s for LSA on the CPU, plus torch and a model
download. The floor exists for a reason; it is just not the configuration to
quote a quality number from.

---

## 4. Why you can believe the numbers

A benchmark you run on yourself is worth exactly as much as its resistance to
self-flattery. Concretely:

**The metrics are differentially tested against `pytrec_eval`.** nDCG@10 has
several plausible definitions that disagree, and the BEIR tables come from
NIST's `trec_eval`. `tests/test_metrics_vs_pytrec_eval.py` fuzzes randomised
runs and qrels — graded relevance, exact score ties, unjudged documents, judged
documents missing from the run — and asserts agreement to 1e-9 on nDCG, Recall,
MAP and P at five cutoffs. `tests/test_metrics.py` separately pins the same
functions to arithmetic worked out by hand, because a test that checks a
function against itself proves nothing. The four decisions that most often go
wrong are documented in [`strata/metrics.py`](strata/metrics.py): linear rather
than exponential gain, the ideal ranking taken from the whole qrels file, AP
divided by the total relevant count, and unjudged documents scored zero rather
than skipped.

**The benchmark ranks with the engine's real fusion code**, pinned by test
across 30 parametrised cases. The single deliberate divergence — dropping
unmatched documents from the lexical candidate list — has its own test saying so.

**Nothing is tuned on the test split.** α is fixed at its shipped default. The
alpha sweep and the k1/b grid are reported separately and labelled oracle upper
bounds everywhere they appear, in the code and in the output, because choosing
the best cell per dataset by looking at test nDCG is how a hybrid system beats
BM25 on paper and then does not in production.

**Every judged query is scored.** Queries come from the qrels file, so a query
the engine answers badly counts as a low score rather than disappearing from the
mean.

**Differences are significance-tested.** With 300 queries a 0.02 nDCG@10 gap is
often noise; SciFact's hybrid deficit is exactly that (p = 0.128), and reporting
it as a loss would overstate what was measured.

**Truncation is loud.** `--max-docs` makes retrieval easier and drops relevant
documents, so the loader says the results are no longer comparable and removes
queries left with nothing relevant.

221 tests. `pytest tests/`.

### Three defects an adversarial audit found

The harness above was then handed to a separate reviewer whose only brief was to
find defects that produce *wrong numbers* rather than crashes, and to verify each
one by reproduction. It found three. All are fixed; all are recorded here rather
than quietly patched, because a page claiming rigour should show its corrections.

**A published diagnosis in this document was wrong.** An earlier revision
reported that NFCorpus HNSW recall plateaued at 0.865 regardless of search width
and attributed it to a build-time graph connectivity defect. That was incorrect.
27 of 200 NFCorpus queries are single rare words — "okra", "fenugreek",
"Zoloft" — that the LSA vocabulary drops at `min_df=2`. They embed to the zero
vector, so every document has cosine exactly 0.0, and both the "true" nearest
neighbours and the graph's are arbitrary tie-breaks over 3,633 identical scores.
173/200 = 0.865, exactly the observed ceiling. With degenerate queries excluded
the graph reaches **recall 1.000** at ef=128. The graph was never broken; the
measurement manufactured a defect. The harness now detects and reports these
queries instead of scoring against noise.

**A fix in this harness introduced a crash.** Dropping unmatched documents from
the lexical candidates can legitimately leave nothing — 25 of NFCorpus's 323
queries match no document at all — and `np.union1d` on an empty Python list
promotes to float64, which raises `IndexError` when used as an index. Hybrid
mode, a default, died on NFCorpus. Fixing it also improved RRF there from 0.2897
to 0.3090, because unmatched documents had been diluting the fusion ranking.

**The ANN path had a ranking inversion.** Documents the graph did not return
were filled with 0.0, but cosine over signed LSA vectors is negative for roughly
half a corpus, so a miss outranked every genuinely dissimilar document the graph
correctly found — six of ten hybrid results changed on a 500-document sample.
The sentinel is now a finite floor below every returned score.

The audit also independently cleared what matters most: BM25 differential-tested
against a naive reimplementation to 9.5e-07, the metrics bit-exact against
`pytrec_eval` on real NFCorpus runs, and the randomised SVD converging to within
0.6% of the optimal rank-k truncation.

---

## 5. Cost

Index build, single-threaded, M1 Max:

| dataset  |   docs | BM25 | embed (LSA) | postings |
|----------|-------:|-----:|------------:|---------:|
| nfcorpus |  3,633 | 1.1s |        4.5s |  474,043 |
| scifact  |  5,183 | 1.5s |        5.9s |  606,911 |
| arguana  |  8,674 | 1.7s |        7.5s |  892,694 |
| scidocs  | 25,657 | 6.0s |       22.5s | 2,619,588 |
| fiqa     | 57,638 | 9.1s |       35.0s | 4,570,584 |

Query latency is 0.2–2.0 ms depending on mode and corpus size.

The embedder used to be far worse. Profiling showed the randomised SVD was 76%
of `fit`, and its two sparse products 64% — `np.add.reduceat` runs well below
memory bandwidth on short segments and has to materialise an `(nnz, k)` product
first, 4.5 GB on FiQA. Replacing both with a rank-major schedule that touches
each posting once and allocates no such temporary made index builds **2.3–2.8×
faster** (FiQA 96.8s → 34.8s). `rdot` is bit-identical afterwards; `dot` differs
by ≤2.7e-07 because it now accumulates in float64 rather than float32, so where
it differs it is the more accurate one. Top-10 rankings are unchanged on every
corpus. `scripts/bench_embed.py` reproduces it.

Tokenisation is now the largest remaining single cost.

---

## 6. Where HNSW starts beating brute force

The README reports an unflattering result: on a 1,323-chunk corpus the
hand-rolled HNSW index was *slower* than exact search. That is correct at that
size, but one corpus is one point on the x-axis and cannot say where the curves
cross. BEIR provides the rest of the axis.

```bash
python scripts/ann_crossover.py
```

Recall@10 is measured against `ExactIndex` on the same vectors and the same
queries. Speedup is exact-search latency divided by HNSW latency; above 1.00×
the graph is winning.

| dataset  |   docs | build | exact ms | ef=32 recall / speedup | ef=256 recall / speedup |
|----------|-------:|------:|---------:|-----------------------:|------------------------:|
| nfcorpus |  3,633 |   24s |    0.065 |     0.984 / **0.12×** |          1.000 / 0.02× |
| scifact  |  5,183 |   36s |    0.091 |     0.988 / **0.16×** |          1.000 / 0.03× |
| arguana  |  8,674 |   66s |    0.238 |     1.000 / **0.49×** |          1.000 / 0.08× |
| scidocs  | 25,657 |  233s |    0.829 |     0.966 / **1.39×** |          0.999 / 0.24× |
| fiqa     | 57,638 |  704s |    1.557 |     0.905 / **1.66×** |          0.994 / 0.36× |

**The crossover sits between 8,674 and 25,657 documents, and only at ef=32.** At
ef=64 the graph roughly breaks even on the largest corpus; at ef≥128 exact
search wins everywhere, including at 57,638 documents. The reason is not
mysterious: exact search is one contiguous `(n × 256) @ (256,)` matmul that
numpy hands to BLAS, while HNSW traversal is per-node Python with heap
operations, so the graph must eliminate a very large fraction of the corpus
before it pays for its own interpreter overhead.

**And the crossover is the wrong question, because the build never amortises.**
FiQA's graph costs 704 seconds to save 0.62 ms per query at ef=32 — about
**1.1 million queries to break even**, and 9.5% of recall as the price. SciDocs
works out at almost exactly the same figure. On corpora this size, on this
stack, the honest recommendation is `use_ann=False`; the graph earns its place
only when exact search stops fitting the latency budget at all, which is well
past where these datasets end.

**These are the corrected numbers.** An earlier revision of this document
reported NFCorpus recall plateauing at 0.865 no matter how wide the beam, and
diagnosed a build-time graph connectivity defect. That diagnosis was wrong — see
§4. NFCorpus is the only dataset in the suite with degenerate queries (27 of
200; every other dataset has none), which is exactly why it was the only one
that appeared to plateau. Excluding them, its recall is 0.984 → 1.000 and the
graph is doing its job perfectly. FiQA's 9.5% loss at ef=32 is by contrast real
approximation error, and stands.

---

## 7. Limitations

- **Five datasets, not eighteen.** The suite covers 3.6k–57.6k documents. The
  million-document BEIR datasets are registered in `strata/beir.py` but have not
  been run. The harness-side blocker is gone — `_dense_scores` now streams
  cosine rows block by block instead of materialising the full
  `n_queries × n_docs` matrix (peak memory 64 × n_docs, verified numerically
  identical at the shipped block size) — but embedding a million documents with
  the bundled LSA encoder is untested and unmeasured, so no claim is made until
  it runs. Claiming a BEIR average from a five-dataset subset would be
  misleading, so there is no average row in this document.
- **The dense leg is a local LSA embedder, by design.** These are not
  competitive dense-retrieval numbers and are not offered as such. They are the
  floor a real encoder should be measured against.
- **Out-of-vocabulary queries have no dense representation at all.** A query
  whose every term the embedder's vocabulary dropped embeds to the zero vector
  and cannot be ranked densely. The engine and the harness now fall back to the
  lexical ranking for exactly those queries (35 of 323 on NFCorpus; zero on the
  other four datasets) instead of scoring an arbitrary tie-break — see the note
  under the LSA table in §2 for what that measurably changed, including the
  number it made *worse*. The underlying limitation stands: the bundled LSA
  embedder simply has nothing to say about a term it never learned, and the
  fallback makes the failure honest rather than making it disappear.
- **The Claude re-ranker is still unrun.** `ClaudeReranker` has no measured
  numbers anywhere in this repo, here included, because this machine has no
  Anthropic API key. Every re-ranking figure comes from the offline
  `LocalCrossEncoder`.

[beir]: https://arxiv.org/abs/2104.08663
[brewing]: https://dl.acm.org/doi/10.1145/3626772.3657862
