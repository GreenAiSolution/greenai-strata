"""The Porter stemmer — the last untested hypothesis for our gap to Lucene.

STRATA's BM25 reproduces published BM25 on BEIR to a mean absolute deviation of
about 0.013, consistently *under* the reference on most datasets. Two candidate
explanations were available, and the parameters have now been eliminated:
running at Anserini's BEIR configuration (k1=0.9, b=0.4) makes the agreement
worse, not better (mean |Δ| 0.0239 against our defaults' 0.0128) — see
`scripts/bm25_params.py`.

That leaves the analyzer. Lucene's `EnglishAnalyzer`, which Anserini uses, runs
a `PorterStemFilter`; STRATA's tokeniser does not stem at all. So "retrieval",
"retrieved" and "retrieves" are three unrelated terms to us and one term to
Lucene, which changes both the postings and every IDF in the index.

This module implements Porter (1980) so the hypothesis can be measured instead
of argued about. It is **opt-in and off by default** — `BM25Index(stem=True)` —
because turning it on silently would change every number already published in
`BEIR.md`.

Reference: M.F. Porter, "An algorithm for suffix stripping", Program 14(3),
1980, pp. 130-137. The step numbering below follows the paper.
"""

from __future__ import annotations

import re

_VOWELS = frozenset("aeiou")

# Step 2 and 3 are pure suffix -> replacement tables gated on the measure of the
# stem. Kept as ordered tuples rather than dicts because Porter requires the
# *longest* matching suffix to win and insertion order encodes that.
_STEP2 = (
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
)

_STEP3 = (
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
)

_STEP4 = (
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
)


def _is_consonant(word: str, i: int) -> bool:
    letter = word[i]
    if letter in _VOWELS:
        return False
    # 'y' is a consonant unless preceded by one, so "toy" ends in a consonant
    # but "happy" ends in a vowel.
    if letter == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's `m`: the number of vowel-consonant sequences in the stem."""
    count = 0
    previous_was_vowel = False
    for i in range(len(stem)):
        if _is_consonant(stem, i):
            if previous_was_vowel:
                count += 1
            previous_was_vowel = False
        else:
            previous_was_vowel = True
    return count


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    return (len(word) >= 2 and word[-1] == word[-2]
            and _is_consonant(word, len(word) - 1))


def _cvc(word: str) -> bool:
    """True if the word ends consonant-vowel-consonant, last not in w/x/y.

    This is the condition for restoring a silent 'e' — "hop" -> "hope".
    """
    if len(word) < 3:
        return False
    if not (_is_consonant(word, len(word) - 1)
            and not _is_consonant(word, len(word) - 2)
            and _is_consonant(word, len(word) - 3)):
        return False
    return word[-1] not in "wxy"


def stem(word: str) -> str:
    """Reduce an English word to its Porter stem.

    Words of two letters or fewer are returned unchanged, per the algorithm.
    """
    if len(word) <= 2:
        return word

    # ---- Step 1a: plurals -------------------------------------------------- #
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # ---- Step 1b: past tense and gerunds ----------------------------------- #
    second_pass = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _has_vowel(word[:-2]):
        word = word[:-2]
        second_pass = True
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        word = word[:-3]
        second_pass = True

    if second_pass:
        # Tidy up the stem left behind by removing -ed / -ing.
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and not word.endswith(("l", "s", "z")):
            word = word[:-1]
        elif _measure(word) == 1 and _cvc(word):
            word += "e"

    # ---- Step 1c: terminal y ------------------------------------------------ #
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    # ---- Step 2 and 3: derivational suffixes -------------------------------- #
    for suffix, replacement in _STEP2:
        if word.endswith(suffix):
            if _measure(word[: -len(suffix)]) > 0:
                word = word[: -len(suffix)] + replacement
            break

    for suffix, replacement in _STEP3:
        if word.endswith(suffix):
            if _measure(word[: -len(suffix)]) > 0:
                word = word[: -len(suffix)] + replacement
            break

    # ---- Step 4: strip the suffix entirely when the stem is long enough ----- #
    for suffix in sorted(_STEP4, key=len, reverse=True):
        if word.endswith(suffix):
            base = word[: -len(suffix)]
            if _measure(base) > 1:
                # -ion only goes when the stem ends in s or t ("adoption", not
                # "lion").
                if suffix == "ion" and not base.endswith(("s", "t")):
                    break
                word = base
            break

    # ---- Step 5: tidy the ending -------------------------------------------- #
    if word.endswith("e"):
        base = word[:-1]
        measure = _measure(base)
        if measure > 1 or (measure == 1 and not _cvc(base)):
            word = base
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]

    return word


_CACHE: dict[str, str] = {}


def stem_cached(word: str) -> str:
    """Memoised `stem`.

    Indexing a BEIR corpus stems several million tokens drawn from a vocabulary
    of tens of thousands, so the hit rate is extreme and the cache turns the
    stemmer from a visible cost into a rounding error.
    """
    cached = _CACHE.get(word)
    if cached is None:
        cached = _CACHE[word] = stem(word)
    return cached
