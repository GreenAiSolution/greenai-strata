"""BEIR dataset loading — download, parse, and adapt to STRATA's corpus type.

BEIR (Thakur et al., 2021) is the standard zero-shot retrieval benchmark: a set
of corpora with human relevance judgements, so a retrieval system can be scored
against numbers other people have published rather than against a query set its
own author invented.

Three things in here are correctness-critical, and each one silently inflates
results if you get it wrong:

**One document, one retrieval unit.** STRATA normally chunks documents on
markdown headings, which is the right thing for a documentation corpus. It is
the wrong thing here: BEIR's relevance judgements are at the document level, so
the retrieval unit must be the document. Chunking and then scoring the chunk ids
against document-level qrels would score most hits as misses; chunking and then
pooling would be a different (easier) task than the published one. So the
adapter maps each BEIR document to exactly one `Chunk`.

**Self-retrieval must be excluded.** In ArguAna the queries *are* documents from
the corpus — the task is finding the counter-argument to an argument. A system
that returns the query itself at rank 1 scores near-perfectly while doing
nothing useful, so BEIR's own evaluation drops the case `doc_id == query_id`.
`Dataset.drops_self_matches` carries that flag and the runner honours it.

**Only the queries with judgements count.** `queries.jsonl` holds every query
across every split; the split is defined by which query ids appear in
`qrels/{split}.tsv`. Evaluating against all of them would score the system on
train queries with no labels and average in a pile of zeros.

Datasets are cached under `~/.cache/strata/beir` and only downloaded once.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import Chunk, Corpus

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DEFAULT_CACHE = Path.home() / ".cache" / "strata" / "beir"


@dataclass(frozen=True)
class DatasetSpec:
    """Static facts about a BEIR dataset, used for planning and sanity checks."""

    name: str
    split: str = "test"
    #: Approximate corpus size, for choosing what will fit and predicting runtime.
    n_docs: int = 0
    n_queries: int = 0
    #: True where queries are drawn from the corpus itself (ArguAna, Quora).
    drops_self_matches: bool = False
    note: str = ""


# Sizes are from the BEIR paper's dataset table and are used only for progress
# reporting and for `--max-docs` planning — never for scoring.
REGISTRY: dict[str, DatasetSpec] = {
    "nfcorpus": DatasetSpec("nfcorpus", n_docs=3_633, n_queries=323,
                            note="Medical IR. Small, graded relevance 0-2."),
    "scifact": DatasetSpec("scifact", n_docs=5_183, n_queries=300,
                           note="Scientific claim verification. Sparse qrels, ~1.1 relevant per query."),
    "arguana": DatasetSpec("arguana", n_docs=8_674, n_queries=1_406,
                           drops_self_matches=True,
                           note="Counter-argument retrieval. Queries are corpus documents."),
    "scidocs": DatasetSpec("scidocs", n_docs=25_657, n_queries=1_000,
                           note="Citation prediction. ~4.9 relevant per query, graded."),
    "fiqa": DatasetSpec("fiqa", n_docs=57_638, n_queries=648,
                        note="Financial QA. Vocabulary mismatch is severe here."),
    "trec-covid": DatasetSpec("trec-covid", n_docs=171_332, n_queries=50,
                              note="COVID literature. Deep graded qrels, ~494 judged per query."),
    "webis-touche2020": DatasetSpec("webis-touche2020", n_docs=382_545, n_queries=49,
                                    note="Argument retrieval. Long documents."),
    "quora": DatasetSpec("quora", n_docs=522_931, n_queries=10_000,
                         drops_self_matches=True,
                         note="Duplicate question retrieval. Very short documents."),
    "dbpedia-entity": DatasetSpec("dbpedia-entity", n_docs=4_635_922, n_queries=400,
                                  note="Entity retrieval. Large — expect a long index build."),
    "climate-fever": DatasetSpec("climate-fever", n_docs=5_416_593, n_queries=1_535,
                                 note="Climate claim verification. Large."),
    "hotpotqa": DatasetSpec("hotpotqa", n_docs=5_233_329, n_queries=7_405,
                            note="Multi-hop QA. Large."),
    "nq": DatasetSpec("nq", n_docs=2_681_468, n_queries=3_452,
                      note="Natural Questions. Large."),
    "fever": DatasetSpec("fever", n_docs=5_416_568, n_queries=6_666,
                         note="Fact verification. Large."),
    "msmarco": DatasetSpec("msmarco", split="dev", n_docs=8_841_823, n_queries=6_980,
                           note="Passage ranking. Uses the dev split; test qrels are not public."),
}

#: The subset that fits comfortably in 32 GB with a pure-numpy stack, ordered by
#: corpus size. This is the default table.
SMALL_SUITE = ("nfcorpus", "scifact", "arguana", "scidocs", "fiqa")


@dataclass
class Dataset:
    """A loaded BEIR dataset, plus the mapping back to STRATA's integer ids."""

    spec: DatasetSpec
    #: {doc_id: (title, text)} in the order they were read.
    documents: dict[str, tuple[str, str]] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)
    #: STRATA indexes by position; these translate back and forth.
    doc_ids: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def drops_self_matches(self) -> bool:
        return self.spec.drops_self_matches

    def __len__(self) -> int:
        return len(self.doc_ids)

    def relevant_count(self) -> float:
        """Mean number of relevant documents per query — context for the scores.

        A dataset averaging 1.1 relevant documents per query (SciFact) and one
        averaging 494 (TREC-COVID) produce nDCG@10 values that are not remotely
        comparable to each other, which is exactly why BEIR is reported
        per-dataset rather than as a single average.
        """
        if not self.qrels:
            return 0.0
        return sum(sum(1 for v in rel.values() if v > 0)
                   for rel in self.qrels.values()) / len(self.qrels)

    def to_corpus(self) -> Corpus:
        """One BEIR document becomes exactly one STRATA Chunk.

        The title goes in `Chunk.title`, which `indexable()` prepends to the
        body — matching BEIR's own convention of indexing `title + " " + text`.
        """
        chunks = [
            Chunk(
                id=i,
                doc_id=doc_id,
                title=self.documents[doc_id][0],
                text=self.documents[doc_id][1],
                start_line=1,
                n_tokens=0,          # filled lazily; nothing in the eval path reads it
                project=self.spec.name,
            )
            for i, doc_id in enumerate(self.doc_ids)
        ]
        return Corpus(chunks=chunks, root=f"beir://{self.spec.name}")


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def _ssl_context():
    """Build an SSL context that actually has root certificates.

    The python.org macOS builds ship without wiring into the system keychain, so
    a stock `urlopen` on HTTPS fails with CERTIFICATE_VERIFY_FAILED even though
    `curl` on the same machine is fine. Prefer `certifi`'s bundle when it is
    installed and fall back to the default context otherwise — never disable
    verification, which is the fix most StackOverflow answers reach for.
    """
    import ssl

    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _report(done: int, total: int, label: str) -> None:
    # Carriage-return progress is for a terminal. Piped into a log it produces
    # one enormous line, which is worse than no progress at all.
    if not sys.stderr.isatty():
        return
    if total <= 0:
        sys.stderr.write(f"\r  {label}: {done / 1e6:.1f} MB")
    else:
        pct = 100.0 * done / total
        sys.stderr.write(f"\r  {label}: {pct:5.1f}%  ({done / 1e6:.1f}/{total / 1e6:.1f} MB)")
    sys.stderr.flush()


def _fetch_with_retries(url: str, destination: Path, *, attempts: int = 5) -> None:
    """Download `url` to `destination`, resuming and retrying on failure.

    The BEIR host is slow and, from some networks, intermittently fails DNS
    resolution partway through a multi-dataset fetch. Without retries a whole
    benchmark run dies on a blip forty minutes in. Partial downloads resume via
    a Range request rather than starting over, because the larger corpora are
    hundreds of megabytes at a few hundred KB/s.
    """
    import time

    partial = destination.with_suffix(destination.suffix + ".part")
    delay = 2.0

    for attempt in range(1, attempts + 1):
        resume_from = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url)
        if resume_from:
            request.add_header("Range", f"bytes={resume_from}-")

        try:
            with urllib.request.urlopen(request, context=_ssl_context(),  # noqa: S310
                                        timeout=60) as response:
                # A server that ignores the Range header replies 200 and sends
                # the whole file; appending to the partial would corrupt it.
                if resume_from and response.status != 206:
                    resume_from = 0
                total = int(response.headers.get("Content-Length", 0)) + resume_from
                done = resume_from
                mode = "ab" if resume_from else "wb"
                with partial.open(mode) as fh:
                    while chunk := response.read(1 << 20):
                        fh.write(chunk)
                        done += len(chunk)
                        _report(done, total, "download")
            partial.replace(destination)
            return
        except Exception as exc:  # noqa: BLE001 — retry on anything transient
            if attempt == attempts:
                partial.unlink(missing_ok=True)
                raise
            sys.stderr.write(
                f"\n  attempt {attempt}/{attempts} failed ({type(exc).__name__}: {exc}); "
                f"retrying in {delay:.0f}s\n"
            )
            time.sleep(delay)
            delay = min(delay * 2, 30.0)


def download(name: str, cache_dir: Path | str = DEFAULT_CACHE, *,
             force: bool = False) -> Path:
    """Fetch and extract a BEIR dataset, returning its directory.

    Extraction goes to a temporary directory that is only moved into place once
    it is complete, so an interrupted download can never leave a half-extracted
    dataset that later looks valid and silently evaluates on partial data.
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown BEIR dataset {name!r}; known: {', '.join(sorted(REGISTRY))}")

    cache_dir = Path(cache_dir).expanduser()
    target = cache_dir / name
    if target.exists() and not force:
        return target

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"{name}.zip"
    url = BEIR_URL.format(name=name)

    print(f"  downloading {name} from {url}", file=sys.stderr)
    _fetch_with_retries(url, archive, attempts=5)
    sys.stderr.write("\n")

    staging = cache_dir / f".{name}.extracting"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(staging)

    # The archive contains a single top-level directory named after the dataset.
    extracted = staging / name
    if not extracted.exists():
        candidates = [p for p in staging.iterdir() if p.is_dir()]
        if len(candidates) != 1:
            raise RuntimeError(f"unexpected archive layout for {name}: {candidates}")
        extracted = candidates[0]

    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(extracted), str(target))
    shutil.rmtree(staging, ignore_errors=True)
    archive.unlink(missing_ok=True)
    return target


# --------------------------------------------------------------------------- #
# Parse
# --------------------------------------------------------------------------- #

def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse a BEIR qrels TSV: `query-id`, `corpus-id`, `score` with a header."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        header = next(reader, None)
        if header is None:
            raise ValueError(f"{path} is empty")
        if header[:3] != ["query-id", "corpus-id", "score"]:
            # Some mirrors ship the file without a header row; don't lose that line.
            fh.seek(0)
            reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if len(row) < 3:
                continue
            query_id, doc_id, score = row[0], row[1], row[2]
            try:
                value = int(float(score))
            except ValueError:
                continue
            qrels.setdefault(query_id, {})[doc_id] = value
    return qrels


def load(name: str, *, split: str | None = None,
         cache_dir: Path | str = DEFAULT_CACHE,
         max_docs: int | None = None,
         verbose: bool = True) -> Dataset:
    """Download if needed, then load a BEIR dataset into memory.

    `max_docs` truncates the corpus for smoke tests. It is deliberately noisy:
    a truncated corpus makes retrieval easier (fewer distractors) *and* drops
    relevant documents, so the resulting numbers are not comparable to anything
    and the loader says so loudly rather than quietly returning a nice score.
    """
    spec = REGISTRY[name]
    split = split or spec.split
    directory = download(name, cache_dir)

    qrels_path = directory / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        available = sorted(p.stem for p in (directory / "qrels").glob("*.tsv"))
        raise FileNotFoundError(
            f"{name} has no '{split}' qrels; available splits: {', '.join(available)}"
        )
    qrels = _read_qrels(qrels_path)

    documents: dict[str, tuple[str, str]] = {}
    doc_ids: list[str] = []
    truncated = False
    for record in _read_jsonl(directory / "corpus.jsonl"):
        doc_id = str(record["_id"])
        documents[doc_id] = (record.get("title") or "", record.get("text") or "")
        doc_ids.append(doc_id)
        if max_docs is not None and len(doc_ids) >= max_docs:
            truncated = True
            break

    # Only queries with judgements in this split are part of the evaluation.
    wanted = set(qrels)
    queries = {
        str(r["_id"]): r["text"]
        for r in _read_jsonl(directory / "queries.jsonl")
        if str(r["_id"]) in wanted
    }

    # A query whose judged documents were all truncated away cannot be scored
    # fairly; drop the qrels entries pointing at documents we did not load, and
    # then drop any query left with nothing relevant.
    if truncated:
        present = set(documents)
        qrels = {
            qid: {d: v for d, v in rel.items() if d in present}
            for qid, rel in qrels.items()
        }
        qrels = {qid: rel for qid, rel in qrels.items()
                 if any(v > 0 for v in rel.values())}
        queries = {q: t for q, t in queries.items() if q in qrels}

    missing = wanted - set(queries) if not truncated else set()
    if missing:
        # Judged query ids with no text: a real corruption, not a normal case.
        raise ValueError(
            f"{name}/{split}: {len(missing)} judged queries missing from queries.jsonl "
            f"(e.g. {sorted(missing)[:3]})"
        )

    dataset = Dataset(spec=spec, documents=documents, queries=queries,
                      qrels=qrels, doc_ids=doc_ids)

    if verbose:
        print(f"  {name}/{split}: {len(doc_ids):,} docs, {len(queries):,} queries, "
              f"{dataset.relevant_count():.1f} relevant/query", file=sys.stderr)
        if truncated:
            print(f"  !! corpus truncated to {max_docs:,} docs — these numbers are "
                  f"NOT comparable to published BEIR results", file=sys.stderr)

    return dataset
