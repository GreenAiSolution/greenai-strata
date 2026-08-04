"""Command line: `index`, `search`, `eval`, `serve`."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .corpus import build_corpus
from .embed import LSAEmbedder
from .evaluate import ann_recall, build_query_set, run_ablation
from .pipeline import SearchEngine
from .rerank import ClaudeReranker, LocalCrossEncoder

DEFAULT_INDEX = Path.home() / ".strata" / "index"

BOLD, DIM, CYAN, GREEN, YELLOW, RESET = (
    "\033[1m", "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"
)


def _make_embedder(name: str):
    if name == "lsa":
        return LSAEmbedder()
    if name == "openai":
        from .embed import OpenAIEmbedder
        return OpenAIEmbedder()
    if name == "voyage":
        from .embed import VoyageEmbedder
        return VoyageEmbedder()
    if name.startswith("st:"):
        from .embed import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(name.split(":", 1)[1])
    raise SystemExit(f"unknown embedder: {name}")


def _make_reranker(name: str, engine: SearchEngine | None = None):
    if name in ("none", ""):
        return None
    if name == "local":
        idf = {}
        if engine is not None:
            idf = {t: float(engine.bm25.idf[i]) for t, i in engine.bm25.vocab.items()}
        return LocalCrossEncoder(idf=idf)
    if name == "claude":
        return ClaudeReranker()
    raise SystemExit(f"unknown reranker: {name}")


# --------------------------------------------------------------------------- #

def cmd_index(args: argparse.Namespace) -> int:
    print(f"{BOLD}Building corpus{RESET} from {', '.join(args.roots)}")
    t0 = time.perf_counter()
    corpus = build_corpus(args.roots, max_tokens=args.chunk_tokens,
                          overlap=args.overlap, max_files=args.max_files,
                          exclude=tuple(args.exclude))
    if not len(corpus):
        print("no documents found — check the paths and --ext filter")
        return 1
    docs = len({c.doc_id for c in corpus.chunks})
    print(f"  {len(corpus):,} chunks from {docs:,} files "
          f"({time.perf_counter() - t0:.2f}s)")

    print(f"{BOLD}Indexing{RESET}")
    engine = SearchEngine.build(corpus, embedder=_make_embedder(args.embedder),
                                use_ann=not args.no_ann, verbose=True)
    engine.save(args.index)
    print(f"{GREEN}saved{RESET} → {args.index}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    engine = SearchEngine.load(args.index)
    reranker = _make_reranker(args.rerank, engine)
    hits, trace = engine.search(
        args.query, k=args.k, mode=args.mode, alpha=args.alpha,
        candidates=args.candidates, reranker=reranker,
        rerank_depth=args.rerank_depth,
    )

    if args.json:
        print(json.dumps({"trace": trace.__dict__,
                          "hits": [h.to_dict() for h in hits]}, indent=2))
        return 0

    stages = "  ".join(f"{k} {v:.1f}ms" for k, v in trace.timings_ms.items())
    print(f"{DIM}{trace.mode}"
          + (f" α={trace.alpha:g}" if trace.mode == "hybrid" else "")
          + (f" + {trace.reranker}" if trace.reranker else "")
          + f"  ·  {stages}{RESET}\n")

    for rank, hit in enumerate(hits, start=1):
        print(f"{BOLD}{rank:2}. {hit.title}{RESET}")
        print(f"    {DIM}{hit.doc}#{hit.doc_id}{RESET}")
        bar = "█" * max(1, int(round(hit.score * 24)))
        detail = f"bm25 {hit.bm25:6.2f}  cos {hit.vector:5.3f}"
        if hit.reranked is not None:
            detail += f"  rerank {hit.reranked:5.3f}"
        print(f"    {CYAN}{bar}{RESET} {hit.score:.4f}   {DIM}{detail}{RESET}")
        if hit.terms:
            terms = " ".join(f"{t}:{v:.1f}" for t, v in hit.terms[:5])
            print(f"    {DIM}terms  {terms}{RESET}")
        if hit.rationale:
            print(f"    {YELLOW}why{RESET}    {hit.rationale}")
        print(f"    {hit.snippet}\n")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .corpus import Corpus

    corpus = Corpus.load(Path(args.index) / "corpus.json")
    print(f"{BOLD}Evaluation{RESET}  ·  inverse cloze task, "
          f"{args.n} queries per variant, corpus of {len(corpus):,} chunks\n")

    for variant in args.variants:
        queries, masked = build_query_set(corpus, n=args.n, variant=variant,
                                          seed=args.seed)
        print(f"{BOLD}{variant}{RESET} — {len(queries)} queries "
              f"(target sentence removed from the index)")
        engine = SearchEngine.build(masked, embedder=_make_embedder(args.embedder),
                                    use_ann=not args.no_ann)
        reranker = _make_reranker(args.rerank, engine)
        if reranker is not None and getattr(reranker, "requires_network", False):
            queries = queries[: args.rerank_queries]
            print(f"  {DIM}re-ranking limited to {len(queries)} queries "
                  f"(network judge){RESET}")

        ceilings: dict[int, float] = {}
        table = run_ablation(engine, queries, alphas=tuple(args.alphas),
                             reranker=reranker, rerank_depth=args.rerank_depth,
                             ceilings=ceilings)
        width = max(len(k) for k in table) + 2
        print(f"  {'system'.ljust(width)}{'R@1':>6}{'R@5':>7}{'R@10':>7}"
              f"{'MRR':>7}{'nDCG':>7}{'docR@10':>9}{'ms/q':>9}")
        best = max(table.values(), key=lambda m: m.ndcg)
        for name, metrics in table.items():
            mark = f"{GREEN} ←{RESET}" if metrics is best else ""
            print(f"  {name.ljust(width)}  {metrics.row()}{mark}")
        for depth, value in ceilings.items():
            print(f"  {DIM}re-ranking headroom: target reached the pool@{depth} "
                  f"for {value:.1%} of queries — that is the hard ceiling on "
                  f"anything the judge can do{RESET}")
        print()

        if engine.ann is not None:
            recalls = ann_recall(engine, queries[:100], k=10)
            print(f"  {DIM}HNSW fidelity vs exact search (recall@10 of the "
                  f"true nearest neighbours){RESET}")
            for ef, stats in recalls.items():
                print(f"    ef={ef:<4} recall {stats['recall']:.3f}   "
                      f"{stats['latency_ms']:.2f}ms/q vs exact "
                      f"{stats['exact_latency_ms']:.2f}ms/q "
                      f"({stats['speedup']:.1f}×)")
            print()
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .server import serve

    serve(args.index, host=args.host, port=args.port)
    return 0


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="strata",
        description="Hybrid retrieval: BM25 + vectors + fusion + re-ranking, "
                    "with an evaluation harness that keeps it honest.",
    )
    parser.add_argument("--index", default=str(DEFAULT_INDEX),
                        help="index directory (default: ~/.strata/index)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="build an index from a directory tree")
    p.add_argument("roots", nargs="+")
    p.add_argument("--embedder", default="lsa",
                   help="lsa | openai | voyage | st:<model-name>")
    p.add_argument("--chunk-tokens", type=int, default=320)
    p.add_argument("--overlap", type=int, default=60)
    p.add_argument("--max-files", type=int, default=None)
    p.add_argument("--exclude", nargs="*", default=[],
                   help="skip paths containing any of these substrings")
    p.add_argument("--no-ann", action="store_true", help="skip the HNSW graph")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("search", help="query the index")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=8)
    p.add_argument("--mode", default="hybrid",
                   choices=["bm25", "vector", "rrf", "hybrid"])
    p.add_argument("--alpha", type=float, default=0.5,
                   help="hybrid weight: 1.0 = pure vector, 0.0 = pure BM25")
    p.add_argument("--candidates", type=int, default=60)
    p.add_argument("--rerank", default="none", choices=["none", "local", "claude"])
    p.add_argument("--rerank-depth", type=int, default=25)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("eval", help="run the ablation + ANN fidelity report")
    p.add_argument("-n", type=int, default=250, help="queries per variant")
    p.add_argument("--variants", nargs="+",
                   default=["verbatim", "hard", "keyword"],
                   choices=["verbatim", "hard", "keyword"])
    p.add_argument("--alphas", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--embedder", default="lsa")
    p.add_argument("--rerank", default="local", choices=["none", "local", "claude"])
    p.add_argument("--rerank-depth", type=int, default=25)
    p.add_argument("--rerank-queries", type=int, default=40,
                   help="cap when the re-ranker calls a network judge")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ann", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("serve", help="browser UI showing per-stage scores")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8105)
    p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
