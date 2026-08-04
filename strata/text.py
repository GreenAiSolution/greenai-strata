"""Tokenisation and light linguistic normalisation.

Deliberately dependency-free: the whole point of STRATA is that every scoring
number can be traced back to code you can read, not to a black-box library.
"""

from __future__ import annotations

import re
import unicodedata

# A small, standard English stoplist. Kept short on purpose — BM25's IDF term
# already suppresses most high-frequency words, so an aggressive stoplist mostly
# just destroys phrase signal ("the who", "to be").
STOPWORDS = frozenset("""
a an and are as at be been but by for from has have he her his i if in into is
it its of on or our ours she that the their them then there these they this to
was were what when where which who will with you your
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[`])|\n{2,}")


def normalise(text: str) -> str:
    """NFKD-fold, lowercase, and collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """Split into index terms.

    Keeps `snake_case`, `kebab-case` and `dotted.paths` as single tokens *and*
    emits their parts, because in a technical corpus both are legitimate query
    forms ("vector_store" and "vector store" should both hit).
    """
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(normalise(text)):
        token = match.group(0)
        if drop_stopwords and token in STOPWORDS:
            continue
        tokens.append(token)
        if any(sep in token for sep in "._-"):
            for part in re.split(r"[._-]", token):
                if len(part) > 1 and not (drop_stopwords and part in STOPWORDS):
                    tokens.append(part)
    return tokens


def sentences(text: str, *, min_words: int = 8) -> list[str]:
    """Rough sentence split, used to build the evaluation query set."""
    out = []
    for raw in _SENTENCE_RE.split(text):
        candidate = " ".join(raw.split())
        if candidate.count(" ") + 1 >= min_words:
            out.append(candidate)
    return out


def content_words(text: str) -> list[str]:
    """Query terms with stopwords removed — the 'hard' evaluation variant."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 2]
