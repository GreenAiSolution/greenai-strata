"""Where does the HNSW graph start beating brute force?

STRATA's README reports an unflattering result: on a 1,323-chunk corpus the
hand-rolled HNSW index was *slower* than exact search. That is the correct
outcome at that size — 1,323 × 256 floats is one cache-resident matrix multiply
that numpy hands to BLAS, while graph traversal is pointer chasing in Python
with per-node overhead. What that corpus could not show is where the curves
cross, because it only had one point on the x-axis.

This script sweeps real BEIR corpora from 3.6k to 57.6k documents and reports
recall@10 against exact search alongside latency, so the crossover is measured
rather than assumed.

    python scripts/ann_crossover.py --out results/ann_crossover.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from strata import beir
from strata.beir_eval import measure_ann

DEFAULT_DATASETS = ("nfcorpus", "scifact", "arguana", "scidocs", "fiqa")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--out", default="results/ann_crossover.json")
    args = parser.parse_args()

    results = []
    for name in args.datasets:
        print(f"\n{name}")
        dataset = beir.load(name)
        measurement = measure_ann(dataset, n_queries=args.queries)
        measurement["dataset"] = name
        results.append(measurement)

    print("\n" + "=" * 78)
    print(f"{'dataset':<12}{'docs':>9}{'build s':>9}{'exact ms':>10}"
          f"{'ef':>6}{'recall@10':>11}{'ann ms':>9}{'speedup':>9}")
    print("-" * 78)
    for row in results:
        for ef, stats in row["ef"].items():
            print(f"{row['dataset']:<12}{row['n_docs']:>9,}"
                  f"{row['build_seconds']:>9.1f}{row['exact_ms']:>10.3f}"
                  f"{ef:>6}{stats['recall']:>11.3f}{stats['ms']:>9.3f}"
                  f"{stats['speedup']:>8.2f}×")
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
