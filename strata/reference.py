"""Published BM25 baselines on BEIR, transcribed from the source papers.

A benchmark number is only meaningful next to somebody else's number, so these
tables exist to let `strata beir` print its result beside the published one. Two
rules govern this file:

**Every value is transcribed from a specific table in a specific paper**, cited
below, not recalled or averaged. If a number is not in one of those tables it is
not in here.

**There is no single "the BM25 number".** BM25 on BEIR varies by implementation
by more than most papers' claimed improvements, which is the entire subject of
Kamalloo et al. (2024). The clearest case is ArguAna: the original BEIR release
reports 0.315, while the Pyserini/Anserini reproduction reports 0.414 — a gap of
ten points on the same dataset with the same metric. Reporting a system as
"beating BM25" without saying *which* BM25 is therefore close to meaningless,
so this module keeps the variants separate and never averages them.

Sources
-------
THAKUR
    Thakur, Reimers, Rücklé, Srivastava, Gurevych. "BEIR: A Heterogeneous
    Benchmark for Zero-shot Evaluation of Information Retrieval Models."
    NeurIPS 2021 Datasets and Benchmarks. Table 2, BM25 column.
    The BM25 baseline there is Elasticsearch with multi-field indexing.

KAMALLOO_*
    Kamalloo, Thakur, Lassance, Ma, Yang, Lin. "Resources for Brewing BEIR:
    Reproducible Reference Models and Statistical Analyses." SIGIR 2024.
    Table 3, BM25 columns. Implemented in Pyserini on Anserini/Lucene.
    - `multifield` indexes title and body as separate weighted fields
      (the default in the original BEIR release).
    - `flat` concatenates title and body into one field.

Which column applies to STRATA
------------------------------
`Chunk.indexable()` concatenates the title and the body into a single string and
indexes that, so STRATA is a **flat** BM25. `KAMALLOO_FLAT` is the like-for-like
comparison; the other two are included for context, not as the target.
"""

from __future__ import annotations

#: Thakur et al. 2021, Table 2 — the original BEIR paper's BM25 row.
THAKUR_2021: dict[str, float] = {
    "msmarco": 0.228,
    "trec-covid": 0.656,
    "bioasq": 0.465,
    "nfcorpus": 0.325,
    "nq": 0.329,
    "hotpotqa": 0.603,
    "fiqa": 0.236,
    "signal1m": 0.330,
    "trec-news": 0.398,
    "robust04": 0.408,
    "arguana": 0.315,
    "webis-touche2020": 0.367,
    "cqadupstack": 0.299,
    "quora": 0.789,
    "dbpedia-entity": 0.313,
    "scidocs": 0.158,
    "fever": 0.753,
    "climate-fever": 0.213,
    "scifact": 0.665,
}

#: Kamalloo et al. 2024, Table 3, "multifield" column.
KAMALLOO_MULTIFIELD: dict[str, float] = {
    "trec-covid": 0.656,
    "bioasq": 0.465,
    "nfcorpus": 0.325,
    "nq": 0.329,
    "hotpotqa": 0.603,
    "fiqa": 0.236,
    "signal1m": 0.330,
    "trec-news": 0.398,
    "robust04": 0.407,
    "arguana": 0.414,
    "webis-touche2020": 0.367,
    "cqadupstack": 0.299,
    "quora": 0.789,
    "dbpedia-entity": 0.313,
    "scidocs": 0.158,
    "fever": 0.753,
    "climate-fever": 0.213,
    "scifact": 0.665,
}

#: Kamalloo et al. 2024, Table 3, "flat" column — title and body concatenated.
#: This is the configuration STRATA matches.
KAMALLOO_FLAT: dict[str, float] = {
    "trec-covid": 0.595,
    "bioasq": 0.522,
    "nfcorpus": 0.322,
    "nq": 0.305,
    "hotpotqa": 0.633,
    "fiqa": 0.236,
    "signal1m": 0.330,
    "trec-news": 0.395,
    "robust04": 0.407,
    "arguana": 0.397,
    "webis-touche2020": 0.442,
    "cqadupstack": 0.302,
    "quora": 0.789,
    "dbpedia-entity": 0.318,
    "scidocs": 0.149,
    "fever": 0.651,
    "climate-fever": 0.165,
    "scifact": 0.679,
}

BASELINES: dict[str, dict[str, float]] = {
    "thakur2021": THAKUR_2021,
    "kamalloo-multifield": KAMALLOO_MULTIFIELD,
    "kamalloo-flat": KAMALLOO_FLAT,
}

#: The like-for-like reference for STRATA's flat index.
PRIMARY = "kamalloo-flat"

CITATIONS = {
    "thakur2021": "Thakur et al. 2021, BEIR (NeurIPS D&B), Table 2 — Elasticsearch multi-field",
    "kamalloo-multifield": "Kamalloo et al. 2024, SIGIR, Table 3 — Anserini multifield",
    "kamalloo-flat": "Kamalloo et al. 2024, SIGIR, Table 3 — Anserini flat (title+body concatenated)",
}


def published(dataset: str, source: str = PRIMARY) -> float | None:
    """Published BM25 nDCG@10 for a dataset, or None if it was never reported."""
    return BASELINES.get(source, {}).get(dataset)


def spread(dataset: str) -> float | None:
    """How far apart the published BM25 numbers are for one dataset.

    Useful as a sanity bar when reading a reproduction: a gap to the reference
    that is smaller than the disagreement *between* published references is not
    evidence of a bug. On ArguAna that spread is 0.099, which is larger than
    most reported improvements over BM25 anywhere in the literature.
    """
    values = [table[dataset] for table in BASELINES.values() if dataset in table]
    if len(values) < 2:
        return None
    return max(values) - min(values)


def comparison_table(measured: dict[str, float]) -> str:
    """Render measured BM25 scores against every published reference."""
    sources = list(BASELINES)
    header = (f"{'dataset':<18}{'strata':>9}" +
              "".join(f"{s.replace('kamalloo-', 'kam-'):>18}" for s in sources) +
              f"{'Δ vs flat':>11}")
    lines = [header, "-" * len(header)]

    deltas = []
    for dataset, value in measured.items():
        row = f"{dataset:<18}{value:>9.4f}"
        for source in sources:
            reference = published(dataset, source)
            row += f"{reference:>18.3f}" if reference is not None else f"{'—':>18}"
        primary = published(dataset, PRIMARY)
        if primary is not None:
            delta = value - primary
            deltas.append(abs(delta))
            row += f"{delta:>+11.4f}"
        else:
            row += f"{'—':>11}"
        lines.append(row)

    if deltas:
        lines.append("-" * len(header))
        mean_absolute = sum(deltas) / len(deltas)
        lines.append(f"{'mean |Δ| vs flat':<18}{mean_absolute:>9.4f}")

    lines.append("")
    lines.append("BM25 nDCG@10. STRATA concatenates title+body, so 'kam-flat' is the")
    lines.append("like-for-like reference; the other columns show how much published")
    lines.append("BM25 baselines disagree with each other on the same data.")
    for source in sources:
        lines.append(f"  {source:<22} {CITATIONS[source]}")
    return "\n".join(lines)
