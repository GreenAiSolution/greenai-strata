"""Does Porter stemming close STRATA's remaining gap to published BM25?

Two hypotheses were available for why our BM25 sits ~0.013 under the published
flat reference. The parameters have been eliminated — `scripts/bm25_params.py`
shows Anserini's k1=0.9/b=0.4 makes the agreement *worse* (mean |Δ| 0.0239 vs
our defaults' 0.0128). That leaves the analyzer: Lucene's `EnglishAnalyzer` runs
a Porter stem filter and STRATA's tokeniser does not stem at all.

This measures it. Both configurations are run over the identical corpus, query
set and metric, so the only difference is the analyzer, and each dataset gets a
paired bootstrap so a small mean shift is not mistaken for an effect.

    python scripts/stemming.py --out results/stemming.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from strata import beir
from strata.lexical import BM25Index
from strata.metrics import evaluate, paired_bootstrap
from strata.reference import PRIMARY, published

DATASETS = ("nfcorpus", "scifact", "arguana", "scidocs", "fiqa")


def run(index: BM25Index, dataset, depth: int = 1000):
    """Score the whole query set, keeping only documents BM25 actually matched."""
    run_dict: dict[str, dict[str, float]] = {}
    for query_id in sorted(dataset.qrels):
        scores = index.score(dataset.queries[query_id])
        top = scores.argsort()[::-1][:depth]
        entry = {}
        for i in top:
            if scores[i] <= 0.0:
                continue
            doc_id = dataset.doc_ids[i]
            if dataset.drops_self_matches and doc_id == query_id:
                continue
            entry[doc_id] = float(scores[i])
        run_dict[query_id] = entry
    return evaluate(dataset.qrels, run_dict, k_values=(10, 100))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--out", default="results/stemming.json")
    args = parser.parse_args()

    results = []
    for name in args.datasets:
        print(f"\n{name}")
        dataset = beir.load(name)
        texts = [c.indexable() for c in dataset.to_corpus().chunks]

        t0 = time.perf_counter()
        plain_index = BM25Index().fit(texts)
        plain_build = time.perf_counter() - t0

        t0 = time.perf_counter()
        stem_index = BM25Index(stem=True).fit(texts)
        stem_build = time.perf_counter() - t0

        plain = run(plain_index, dataset)
        stemmed = run(stem_index, dataset)
        reference = published(name)
        test = paired_bootstrap(plain.per_query_ndcg10, stemmed.per_query_ndcg10)

        print(f"  vocabulary  {len(plain_index.vocab):>8,} -> {len(stem_index.vocab):>8,} "
              f"({100 * (1 - len(stem_index.vocab) / len(plain_index.vocab)):.1f}% smaller)")
        print(f"  build       {plain_build:>8.2f}s -> {stem_build:>8.2f}s")
        print(f"  nDCG@10     {plain.headline:>8.4f} -> {stemmed.headline:>8.4f}   "
              f"({test['delta']:+.4f}, p={test['p_value']:.3f})")
        print(f"  Recall@100  {plain.recall[100]:>8.4f} -> {stemmed.recall[100]:>8.4f}")
        if reference is not None:
            print(f"  vs published {reference:.3f}:  plain {plain.headline - reference:+.4f}"
                  f"   stemmed {stemmed.headline - reference:+.4f}")

        results.append({
            "dataset": name,
            "vocab_plain": len(plain_index.vocab),
            "vocab_stemmed": len(stem_index.vocab),
            "build_plain_s": round(plain_build, 2),
            "build_stemmed_s": round(stem_build, 2),
            "ndcg10_plain": round(plain.headline, 5),
            "ndcg10_stemmed": round(stemmed.headline, 5),
            "recall100_plain": round(plain.recall[100], 5),
            "recall100_stemmed": round(stemmed.recall[100], 5),
            "published": reference,
            "significance": {"delta": round(test["delta"], 5),
                             "p_value": round(test["p_value"], 4)},
        })

    print("\n" + "=" * 88)
    print(f"{'dataset':<12}{'plain':>9}{'stemmed':>9}{'delta':>9}{'p':>8}"
          f"{'published':>11}{'Δ plain':>10}{'Δ stem':>10}")
    print("-" * 88)
    sum_plain = sum_stem = 0.0
    counted = 0
    for row in results:
        line = (f"{row['dataset']:<12}{row['ndcg10_plain']:>9.4f}"
                f"{row['ndcg10_stemmed']:>9.4f}{row['significance']['delta']:>+9.4f}"
                f"{row['significance']['p_value']:>8.3f}")
        reference = row["published"]
        if reference is not None:
            d_plain = row["ndcg10_plain"] - reference
            d_stem = row["ndcg10_stemmed"] - reference
            sum_plain += abs(d_plain)
            sum_stem += abs(d_stem)
            counted += 1
            line += f"{reference:>11.3f}{d_plain:>+10.4f}{d_stem:>+10.4f}"
        print(line)
    if counted:
        print("-" * 88)
        print(f"{'mean |Δ|':<12}{'':>9}{'':>9}{'':>9}{'':>8}{'':>11}"
              f"{sum_plain / counted:>10.4f}{sum_stem / counted:>10.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
