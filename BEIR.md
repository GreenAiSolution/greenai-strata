# STRATA on BEIR

Everything else in this repository is measured on a corpus I built myself. That
is useful for development and worthless as evidence, because nobody can compare
it to anything. This document reports STRATA on [BEIR][beir] — public corpora,
public relevance judgements, and metrics that mean the same thing as everybody
else's.

Reproduce the whole thing with:

```bash
pip install -e '.[dev]'
strata beir small --sweep-alpha --out results/beir_lsa.json
```

Datasets download to `~/.cache/strata/beir` on first use. The five-dataset suite
takes about seven minutes end to end on an M1 Max; no GPU, no API keys, and
numpy is still the only runtime dependency.

---

## 1. The result that matters: the BM25 baseline reproduces

Before any claim about hybrid retrieval is worth reading, the harness itself has
to be trustworthy. The way to check that is to run a well-known baseline and see
whether it lands where the literature says it should.

STRATA's BM25 is hand-rolled — an inverted index on flat numpy arrays, its own
tokeniser, Robertson/Sparck-Jones IDF, roughly 110 lines in
[`strata/lexical.py`](strata/lexical.py). It shares no code with Lucene,
Anserini or Elasticsearch. Here is what it scores against published BM25
baselines on the same data:

| dataset  | STRATA | Thakur 2021 | Kamalloo multifield | Kamalloo **flat** | Δ vs flat |
|----------|-------:|------------:|--------------------:|------------------:|----------:|
| nfcorpus | 0.3101 |       0.325 |               0.325 |             0.322 |   −0.0119 |
| scifact  | 0.6613 |       0.665 |               0.665 |             0.679 |   −0.0177 |
| arguana  | 0.4204 |       0.315 |               0.414 |             0.397 |   +0.0234 |
| scidocs  | 0.1497 |       0.158 |               0.158 |             0.149 |   +0.0007 |
| fiqa     | 0.2295 |       0.236 |               0.236 |             0.236 |   −0.0065 |

**Mean absolute deviation from the flat reference: 0.0120.**

STRATA concatenates title and body into one indexed field, so the like-for-like
column is Kamalloo et al.'s *flat* BM25, not the multi-field variant. Sources
are transcribed in [`strata/reference.py`](strata/reference.py):

- **Thakur 2021** — [BEIR][beir], NeurIPS Datasets & Benchmarks, Table 2. Elasticsearch, multi-field.
- **Kamalloo 2024** — [Resources for Brewing BEIR][brewing], SIGIR, Table 3. Pyserini/Anserini, both variants.

### Why ArguAna looks wrong and isn't

ArguAna is the one dataset where our number sits well above the figure most
people quote — 0.4204 against the BEIR paper's 0.315. That gap is not ours. The
published BM25 values for ArguAna span **0.099** across implementations: 0.315
in the original BEIR release, 0.414 in the Anserini multi-field reproduction,
0.397 flat. Our 0.4204 sits next to the reproducible Anserini numbers, and it is
the BEIR paper's 0.315 that is the outlier. Documenting exactly this kind of
disagreement is what the Kamalloo et al. paper exists for.

This matters beyond one dataset. That 0.099 spread is larger than almost any
improvement over BM25 claimed anywhere in the retrieval literature. "We beat
BM25" is not a meaningful sentence unless it says *which* BM25, which is why
`strata beir` prints all three reference columns and refuses to average them.

I also checked the boring explanation first: ArguAna draws its queries from its
own corpus (1,298 of 1,406 query ids are also document ids), so a system that
returns the query's own document scores brilliantly for no skill. BEIR excludes
that case and so do we — `Dataset.drops_self_matches` is set for ArguAna and
Quora, and it is verified in `tests/test_beir_eval.py`. The self-match path was
not the cause.

---

## 2. Full results

nDCG@10, five datasets, four retrieval modes. `α = 0.5`, untuned.

| dataset  |   docs | queries |   bm25 | vector |    rrf | hybrid |
|----------|-------:|--------:|-------:|-------:|-------:|-------:|
| nfcorpus |  3,633 |     323 | 0.3101 | 0.2835 | 0.2897 | **0.3283** |
| scifact  |  5,183 |     300 | **0.6613** | 0.4829 | 0.5770 | 0.6413 |
| arguana  |  8,674 |   1,406 | 0.4204 | 0.4572 | 0.4568 | **0.4675** |
| scidocs  | 25,657 |   1,000 | **0.1497** | 0.0804 | 0.1166 | 0.1421 |
| fiqa     | 57,638 |     648 | **0.2295** | 0.0558 | 0.1232 | 0.1806 |

Each mode is tested against the BM25 baseline with a paired bootstrap over
per-query nDCG@10 (10,000 resamples):

| dataset  | hybrid − bm25 | p | verdict |
|----------|--------------:|--:|---------|
| nfcorpus | +0.0182 | 0.000 | hybrid genuinely better |
| scifact  | −0.0200 | 0.073 | **no significant difference** |
| arguana  | +0.0471 | 0.000 | hybrid genuinely better |
| scidocs  | −0.0075 | 0.009 | hybrid genuinely worse |
| fiqa     | −0.0489 | 0.000 | hybrid genuinely worse |

---

## 3. What this says about the local embedder — the unflattering part

The headline finding is negative, and it is the most useful thing here.

**The bundled LSA embedder does not earn its place on most of these datasets.**
Hybrid fusion beats BM25 on two of five, ties on one, and loses on two. On FiQA
the dense leg scores 0.0558 against BM25's 0.2295 and drags the hybrid down by
almost five points. The alpha sweep makes the same point from the other side: on
FiQA the best possible α is **0.0**, meaning that even an oracle allowed to tune
on the test set would choose to switch the dense leg off entirely. On SciDocs
the oracle picks α = 0.1.

This is not a surprise, and the [embedder's own docstring](strata/embed.py) says
so up front — TF-IDF plus a truncated SVD is "the honest floor", not a neural
encoder. But there is a difference between predicting that in a comment and
measuring it on five public datasets, and the measurement is more specific than
the prediction was. The pattern in where it fails:

- **It loses badly where the vocabulary gap is the whole task.** FiQA is
  financial questions asked in plain language against expert answers; SciDocs is
  citation prediction. LSA learns co-occurrence within *this* corpus, so it
  cannot bridge a gap that requires knowledge from outside it.
- **It wins where the query is long and the answer is a paraphrase.** ArguAna
  queries are entire arguments and the target is the counter-argument, which
  shares topic but not phrasing. This is the one case where the dense leg
  outscores BM25 outright (0.4572 vs 0.4204).
- **Recall@1000 improves almost everywhere, even where nDCG@10 falls.** On FiQA
  the dense leg is terrible at ranking (nDCG@10 0.0558) but RRF still lifts
  Recall@1000 from 0.7246 to 0.7456. It finds documents BM25 misses; it just
  cannot order them. That is precisely the profile of a useful *first-stage*
  retriever feeding a re-ranker, and it is an argument for the re-ranking layer
  this repo already has rather than against the dense leg entirely.

The honest one-line summary: **on this benchmark STRATA is a strong BM25
implementation with a hybrid layer that helps on 2 of 5 datasets and hurts on 2.**
Swapping `LSAEmbedder` for a neural encoder is a one-line change and the harness
will say exactly what it bought — that is the experiment to run next, and until
it is run I am not claiming the hybrid architecture is validated.

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

**The benchmark ranks with the engine's real fusion code.** It would be easy to
accidentally benchmark a cleaner reimplementation of the ranker instead of the
thing the product runs. `tests/test_beir_eval.py` pins the benchmark's weighted
fusion to `strata.fusion.weighted_fusion` output across 30 parametrised cases
including the sparse-BM25 score distribution.

**Nothing is tuned on the test split.** α is fixed at its shipped default of
0.5. The alpha sweep is reported separately and labelled an oracle upper bound
everywhere it appears, in the code and in the output, because choosing the best
α per dataset by looking at test nDCG is how a hybrid system beats BM25 on paper
and then does not in production.

**Every judged query is scored.** Queries come from the qrels file, so a query
the engine answers badly counts as a low score rather than disappearing from the
mean.

**Differences are significance-tested.** With 300 queries a 0.02 nDCG@10 gap is
often noise; SciFact's hybrid deficit is exactly that (p = 0.073), and reporting
it as a loss would overstate what was measured.

**Truncation is loud.** `--max-docs` makes retrieval easier and drops relevant
documents, so the loader prints that the results are no longer comparable and
removes queries left with no relevant documents.

98 tests cover this. `pytest tests/`.

---

## 5. Cost

Index build, single-threaded, M1 Max:

| dataset  |   docs | BM25 | embed (LSA) | postings |
|----------|-------:|-----:|------------:|---------:|
| nfcorpus |  3,633 | 0.9s |       10.9s |  498,138 |
| scifact  |  5,183 | 1.2s |       15.1s |  658,984 |
| arguana  |  8,674 | 1.5s |       20.1s |  932,004 |
| scidocs  | 25,657 | 5.1s |       54.6s | 2,758,284 |
| fiqa     | 57,638 | 7.9s |       96.2s | 4,895,002 |

Query latency is 0.2–2.0 ms depending on mode and corpus size. Both build stages
are close to linear in postings. The randomised SVD in the LSA embedder is the
bottleneck at every size — it is roughly 12× the cost of building the inverted
index — and its inner loop is the per-column `bincount` in `_CSR.rdot`, which is
the first thing to optimise if these corpora get larger.

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
queries, so it isolates what the approximation lost. Speedup is exact-search
latency divided by HNSW latency; above 1.00× the graph is winning.

| dataset  |   docs | build | exact ms | ef=32 recall / speedup | ef=256 recall / speedup |
|----------|-------:|------:|---------:|-----------------------:|------------------------:|
| nfcorpus |  3,633 |  24s |    0.067 |     0.852 / **0.12×** |          0.865 / 0.03× |
| scifact  |  5,183 |  36s |    0.097 |     0.988 / **0.18×** |          1.000 / 0.04× |
| arguana  |  8,674 |  64s |    0.197 |     1.000 / **0.40×** |          1.000 / 0.06× |
| scidocs  | 25,657 | 233s |    0.793 |     0.966 / **1.37×** |          0.999 / 0.23× |
| fiqa     | 57,638 | 714s |    1.569 |     0.905 / **1.85×** |          0.994 / 0.36× |

**The crossover sits between 8,674 and 25,657 documents, and only at ef=32.**
At ef=64 the graph roughly breaks even on the largest corpus; at ef≥128 exact
search wins everywhere, including at 57,638 documents. The reason is not
mysterious: exact search is one contiguous `(n × 256) @ (256,)` matmul that
numpy hands to BLAS, while HNSW traversal is per-node Python with heap
operations, so the graph has to eliminate a very large fraction of the corpus
before it can pay for its own interpreter overhead.

**And the crossover is the wrong question anyway, because the build never
amortises.** Building the FiQA graph costs 714 seconds to save 0.72 ms per
query at ef=32 — that is **990,000 queries to break even**, and you pay 9.5% of
your recall for it. SciDocs works out at almost exactly the same figure. On a
corpus of this size, on this stack, the honest recommendation is
`use_ann=False`; the graph earns its place only when the corpus is large enough
that exact search stops fitting the latency budget at all, which is well past
where these datasets end.

One anomaly worth flagging rather than smoothing over: NFCorpus recall
**plateaus at 0.865 and will not improve with search width** — ef=256 buys
0.013 over ef=32, where every other dataset reaches ≥0.99. Widening the beam
cannot fix it, which points at build-time graph connectivity (nodes that are
unreachable from the entry point) rather than at the search. NFCorpus documents
are short medical abstracts with heavy vocabulary overlap, so the LSA vectors
are unusually clustered and the neighbour-selection heuristic may be pruning the
long-range links that keep the graph navigable. That is a real defect in the
index, it is visible only because recall is measured against exact search, and
it is unfixed.

---

## 7. Limitations

- **Five datasets, not eighteen.** The suite covers 3.6k–57.6k documents. The
  million-document BEIR datasets (HotpotQA, FEVER, MS MARCO, Climate-FEVER,
  DBPedia) are registered in `strata/beir.py` but have not been run; the pure-
  numpy LSA embedder would need real work to get there, and claiming a BEIR
  average from a five-dataset subset would be misleading. There is no average
  row in this document for that reason.
- **The dense leg is a local LSA embedder, by design.** These are not
  competitive dense-retrieval numbers and are not offered as such. They are the
  floor that a real encoder should be measured against.
- **The Claude re-ranker is still unrun.** `ClaudeReranker` has no measured
  numbers anywhere in this repo, here included, because this machine has no
  Anthropic API key. Every re-ranking figure in the repo comes from the offline
  `LocalCrossEncoder`.
- **BM25 parameters are the defaults** (k1=1.5, b=0.75) and were not tuned per
  dataset. Anserini's BEIR configuration uses k1=0.9, b=0.4, which is part of
  why our numbers sit slightly below the flat reference on three of five
  datasets.

[beir]: https://arxiv.org/abs/2104.08663
[brewing]: https://dl.acm.org/doi/10.1145/3626772.3657862
