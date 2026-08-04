"""Stage 2: re-ranking the candidate pool.

Retrieval and re-ranking optimise different things. Retrieval must be cheap
across the whole corpus, so it scores query and document *independently*
(one vector each, one dot product). A re-ranker sees the pair together and can
model interaction — which is where the real relevance signal lives — but costs
too much to run over more than a few dozen candidates.

Two implementations:

* `LocalCrossEncoder` — a feature-based pairwise scorer (IDF-weighted term
  coverage, bigram/phrase hits, proximity, title match, vector agreement).
  Deterministic, offline, free. This is the fallback that makes the demo run
  anywhere, and it is an honest cross-encoder *shape* even though it is not a
  trained transformer.
* `ClaudeReranker` — an LLM judge. Candidates are batched into one call and
  scored against an explicit rubric with a strict JSON schema, so the output is
  validated structure rather than parsed prose.

Both return `(doc_id, score, rationale)` so the pipeline can show its work.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .text import content_words, tokenize


@dataclass
class RerankResult:
    doc_id: int
    score: float
    rationale: str = ""


# --------------------------------------------------------------------------- #
# Offline feature-based reranker
# --------------------------------------------------------------------------- #

class LocalCrossEncoder:
    """Pairwise scorer over interpretable features. No model, no network."""

    name = "local-cross-encoder"
    requires_network = False

    WEIGHTS = {
        "coverage": 0.34,     # how much of the query's information is present
        "phrase": 0.22,       # contiguous bigram hits — strong precision signal
        "proximity": 0.14,    # query terms appearing close together
        "title": 0.12,        # heading match
        "vector": 0.18,       # agreement with the semantic leg
    }

    def __init__(self, idf: dict[str, float] | None = None):
        self.idf = idf or {}

    def _idf(self, term: str) -> float:
        return self.idf.get(term, 3.0)

    def _coverage(self, query_terms: list[str], doc_terms: set[str]) -> float:
        if not query_terms:
            return 0.0
        total = sum(self._idf(t) for t in query_terms)
        hit = sum(self._idf(t) for t in query_terms if t in doc_terms)
        return hit / total if total else 0.0

    @staticmethod
    def _bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
        return set(zip(tokens, tokens[1:]))

    @staticmethod
    def _proximity(query_terms: list[str], doc_tokens: list[str]) -> float:
        """Smallest window containing the most distinct query terms."""
        wanted = set(query_terms)
        positions = [(i, t) for i, t in enumerate(doc_tokens) if t in wanted]
        if len(wanted) < 2 or len(positions) < 2:
            return 0.0
        best, seen, left = math.inf, {}, 0
        for right, (pos, term) in enumerate(positions):
            seen[term] = pos
            while len(seen) == len(wanted):
                span = pos - positions[left][0] + 1
                best = min(best, span)
                out_term = positions[left][1]
                if seen.get(out_term) == positions[left][0]:
                    seen.pop(out_term)
                left += 1
        if best is math.inf:
            covered = len({t for _, t in positions})
            return 0.35 * covered / len(wanted)
        return float(np.clip(len(wanted) / best, 0.0, 1.0))

    def score_pair(self, query: str, title: str, body: str,
                   vector_score: float = 0.0) -> tuple[float, str]:
        query_terms = content_words(query)
        doc_tokens = tokenize(body)
        doc_terms = set(doc_tokens)
        title_terms = set(tokenize(title))

        features = {
            "coverage": self._coverage(query_terms, doc_terms),
            "phrase": (
                len(self._bigrams(query_terms) & self._bigrams(doc_tokens))
                / max(len(self._bigrams(query_terms)), 1)
            ),
            "proximity": self._proximity(query_terms, doc_tokens),
            "title": (
                len([t for t in query_terms if t in title_terms])
                / max(len(query_terms), 1)
            ),
            "vector": float(np.clip(vector_score, 0.0, 1.0)),
        }
        score = sum(self.WEIGHTS[k] * v for k, v in features.items())
        top = sorted(features.items(), key=lambda kv: -self.WEIGHTS[kv[0]] * kv[1])[:2]
        rationale = ", ".join(f"{k}={v:.2f}" for k, v in top)
        return float(score), rationale

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10
               ) -> list[RerankResult]:
        out = []
        for cand in candidates:
            score, why = self.score_pair(
                query, cand.get("title", ""), cand.get("text", ""),
                cand.get("vector_score", 0.0),
            )
            out.append(RerankResult(cand["doc_id"], score, why))
        out.sort(key=lambda r: -r.score)
        return out[:top_k]


# --------------------------------------------------------------------------- #
# LLM judge
# --------------------------------------------------------------------------- #

RUBRIC = """You are a relevance judge in a search system. Score how well each \
passage answers the user's query.

Scale (use the whole range):
  0  — unrelated to the query
  3  — same broad topic, does not address the query
  6  — partially answers, or answers a neighbouring question
  8  — directly answers the query
 10  — directly and completely answers it, with the specifics the query asks for

Judge only what the passage actually contains. A passage that merely mentions \
the query's keywords without addressing the question scores low; a passage that \
answers the question using different vocabulary scores high. Ignore passage \
length and writing quality. Score every passage independently — do not grade on \
a curve."""

SCHEMA = {
    "type": "object",
    "properties": {
        "judgements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "relevance": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "relevance", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["judgements"],
    "additionalProperties": False,
}


class ClaudeReranker:
    """LLM re-ranking with a strict output schema and candidate batching.

    Batching matters twice over: it cuts round-trips, and it lets the judge see
    candidates side by side, which measurably stabilises the scale compared with
    one isolated call per passage.
    """

    name = "claude-reranker"
    requires_network = True

    def __init__(self, model: str = "claude-opus-5", batch_size: int = 8,
                 max_chars: int = 1200, effort: str = "low",
                 client=None):
        self.model = model
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.effort = effort
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic  # noqa: PLC0415 — optional dependency

            # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
            # `ant auth login` profile — do not pass a key explicitly.
            self._client = anthropic.Anthropic()
        return self._client

    def _judge_batch(self, query: str, batch: list[dict]) -> list[RerankResult]:
        passages = "\n\n".join(
            f"<passage id=\"{c['doc_id']}\" source=\"{c.get('title', '')}\">\n"
            f"{c.get('text', '')[: self.max_chars]}\n</passage>"
            for c in batch
        )
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=[{"type": "text", "text": RUBRIC,
                     "cache_control": {"type": "ephemeral"}}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": SCHEMA},
            },
            messages=[{
                "role": "user",
                "content": f"<query>{query}</query>\n\n{passages}\n\n"
                           f"Return one judgement per passage id.",
            }],
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("reranker request was declined by safety classifiers")
        text = next(b.text for b in response.content if b.type == "text")
        payload = json.loads(text)
        return [
            RerankResult(int(j["id"]), float(j["relevance"]) / 10.0, j["reason"])
            for j in payload["judgements"]
        ]

    def rerank(self, query: str, candidates: list[dict], top_k: int = 10
               ) -> list[RerankResult]:
        results: list[RerankResult] = []
        for i in range(0, len(candidates), self.batch_size):
            results.extend(self._judge_batch(query, candidates[i : i + self.batch_size]))
        # Stable tie-break on the incoming retrieval order keeps the ranking
        # deterministic when the judge assigns the same integer to several docs.
        order = {c["doc_id"]: i for i, c in enumerate(candidates)}
        results.sort(key=lambda r: (-r.score, order.get(r.doc_id, 10**6)))
        return results[:top_k]


def default_reranker(idf: dict[str, float] | None = None, prefer_llm: bool = True):
    """Pick the best re-ranker actually available in this environment.

    Reports honestly which one it chose — a demo that silently degrades from an
    LLM judge to a heuristic is a demo that lies about its own numbers.
    """
    if prefer_llm and (os.environ.get("ANTHROPIC_API_KEY")
                       or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        try:
            import anthropic  # noqa: F401,PLC0415

            return ClaudeReranker()
        except ImportError:
            pass
    return LocalCrossEncoder(idf=idf)
