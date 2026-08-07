"""How much of the gap to published BM25 is just k1 and b?

STRATA ships BM25 at k1=1.5, b=0.75 — the textbook Robertson defaults. The
Anserini configuration that produced the published BEIR baselines uses k1=0.9,
b=0.4. Our numbers sit slightly under the published flat reference on three of
five datasets, and the obvious question is whether that is the parameters rather
than the implementation.

This script answers it. Two things are worth being careful about:

**Matching a reference configuration is not tuning on test.** Running at
Anserini's published k1=0.9/b=0.4 makes the comparison like-for-like; it uses no
information from the test labels. Choosing the best cell of the sweep per
dataset *would* use that information, so the sweep is reported separately and
labelled an oracle bound, exactly as the alpha sweep is.

**The sweep is nearly free, which is why it is worth doing properly.** k1 and b
appear only in `BM25Index.score`, never in `fit` — the postings, document
lengths and IDF are all parameter-independent. So the index is built once per
dataset and the whole grid is swept by mutating two floats, rather than
rebuilding 30 indexes.

    python scripts/bm25_params.py --out results/bm25_params.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strata import beir
from strata.lexical import BM25Index
from strata.metrics import evaluate, paired_bootstrap
from strata.reference import PRIMARY, published

DATASETS = ("nfcorpus", "scifact", "arguana", "scidocs", "fiqa")

#: STRATA's shipped default, and the configuration Anserini used for BEIR.
STRATA_DEFAULT = (1.5, 0.75)
ANSERINI_BEIR = (0.9, 0.4)

K1_GRID = (0.6, 0.9, 1.2, 1.5, 1.8)
B_GRID = (0.3, 0.4, 0.6, 0.75, 0.9)


def score_at(index: BM25Index, dataset, k1: float, b: float, depth: int = 1000):
    """Score the whole query set at one (k1, b), reusing the built index."""
    index.k1, index.b = k1, b
    run: dict[str, dict[str, float]] = {}
    for query_id in sorted(dataset.qrels):
        scores = index.score(dataset.queries[query_id])
        top = scores.argsort()[::-1][:depth]
        entry = {}
        for i in top:
            # Zero score means no query term occurs in the document, so it was
            # never retrieved — see the note in beir_eval._fuse.
            if scores[i] <= 0.0:
                continue
            doc_id = dataset.doc_ids[i]
            if dataset.drops_self_matches and doc_id == query_id:
                continue
            entry[doc_id] = float(scores[i])
        run[query_id] = entry
    return evaluate(dataset.qrels, run, k_values=(10,))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--out", default="results/bm25_params.json")
    args = parser.parse_args()

    results = []
    for name in args.datasets:
        print(f"\n{name}")
        dataset = beir.load(name)
        texts = [c.indexable() for c in dataset.to_corpus().chunks]
        index = BM25Index().fit(texts)

        default = score_at(index, dataset, *STRATA_DEFAULT)
        anserini = score_at(index, dataset, *ANSERINI_BEIR)
        reference = published(name)

        print(f"  k1={STRATA_DEFAULT[0]} b={STRATA_DEFAULT[1]} (strata default) "
              f"nDCG@10 {default.headline:.4f}")
        print(f"  k1={ANSERINI_BEIR[0]} b={ANSERINI_BEIR[1]} (anserini BEIR)    "
              f"nDCG@10 {anserini.headline:.4f}")
        if reference is not None:
            print(f"  published ({PRIMARY}) {reference:.3f}   "
                  f"Δ default {default.headline - reference:+.4f}   "
                  f"Δ anserini {anserini.headline - reference:+.4f}")

        test = paired_bootstrap(default.per_query_ndcg10, anserini.per_query_ndcg10)
        print(f"  anserini − default: {test['delta']:+.4f}  p={test['p_value']:.3f}")

        grid = {}
        best = (None, -1.0)
        for k1 in K1_GRID:
            for b in B_GRID:
                value = score_at(index, dataset, k1, b).headline
                grid[f"k1={k1},b={b}"] = round(value, 5)
                if value > best[1]:
                    best = ((k1, b), value)
        print(f"  ORACLE best cell (tuned on test, not a fair result): "
              f"k1={best[0][0]} b={best[0][1]} → {best[1]:.4f}")

        results.append({
            "dataset": name,
            "n_docs": len(dataset),
            "strata_default": {"k1": STRATA_DEFAULT[0], "b": STRATA_DEFAULT[1],
                               "ndcg10": round(default.headline, 5)},
            "anserini_beir": {"k1": ANSERINI_BEIR[0], "b": ANSERINI_BEIR[1],
                              "ndcg10": round(anserini.headline, 5)},
            "published": reference,
            "significance_anserini_vs_default": {
                "delta": round(test["delta"], 5),
                "p_value": round(test["p_value"], 4),
            },
            "oracle_best": {"k1": best[0][0], "b": best[0][1],
                            "ndcg10": round(best[1], 5)},
            "grid": grid,
        })

    print("\n" + "=" * 82)
    print(f"{'dataset':<12}{'default':>10}{'anserini':>10}{'published':>11}"
          f"{'Δ default':>11}{'Δ anserini':>12}{'oracle':>9}")
    print("-" * 82)
    sum_default = sum_anserini = 0.0
    counted = 0
    for row in results:
        reference = row["published"]
        line = (f"{row['dataset']:<12}{row['strata_default']['ndcg10']:>10.4f}"
                f"{row['anserini_beir']['ndcg10']:>10.4f}")
        if reference is not None:
            d_default = row["strata_default"]["ndcg10"] - reference
            d_anserini = row["anserini_beir"]["ndcg10"] - reference
            sum_default += abs(d_default)
            sum_anserini += abs(d_anserini)
            counted += 1
            line += f"{reference:>11.3f}{d_default:>+11.4f}{d_anserini:>+12.4f}"
        else:
            line += f"{'—':>11}{'—':>11}{'—':>12}"
        line += f"{row['oracle_best']['ndcg10']:>9.4f}"
        print(line)

    if counted:
        print("-" * 82)
        print(f"{'mean |Δ|':<12}{'':>10}{'':>10}{'':>11}"
              f"{sum_default / counted:>11.4f}{sum_anserini / counted:>12.4f}")
    print("\nnDCG@10. 'oracle' is the best grid cell chosen by looking at the test")
    print("labels — an upper bound for context, never a reportable result.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
