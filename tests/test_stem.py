"""Porter stemmer fixtures, taken from the algorithm's published examples.

The pairs below are from Porter's 1980 paper and the reference vocabulary that
ships with it. They are worth pinning precisely because a stemmer that is
*almost* right is worse than none: it conflates words it should not, and the
damage shows up as a diffuse drop in retrieval quality that is very hard to
trace back to its cause.
"""

from __future__ import annotations

import pytest

from strata.lexical import BM25Index
from strata.stem import _cvc, _measure, stem, stem_cached


# --------------------------------------------------------------------------- #
# The worked examples from the paper, step by step
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("word,expected", [
    # Step 1a — plurals
    ("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"),
    ("caress", "caress"), ("cats", "cat"),
    # Step 1b — past tense and gerunds
    ("feed", "feed"), ("agreed", "agre"), ("plastered", "plaster"),
    ("bled", "bled"), ("motoring", "motor"), ("sing", "sing"),
    # Step 1b cleanup — the -at/-bl/-iz and double-consonant rules
    ("conflated", "conflat"), ("troubled", "troubl"), ("sized", "size"),
    ("hopping", "hop"), ("tanned", "tan"), ("falling", "fall"),
    ("hissing", "hiss"), ("fizzed", "fizz"), ("failing", "fail"),
    ("filing", "file"),
    # Step 1c — terminal y
    ("happy", "happi"), ("sky", "sky"),
])
def test_paper_examples(word: str, expected: str):
    assert stem(word) == expected


@pytest.mark.parametrize("word,expected", [
    # Step 2
    ("relational", "relat"), ("conditional", "condit"), ("rational", "ration"),
    ("valenci", "valenc"), ("hesitanci", "hesit"), ("digitizer", "digit"),
    ("conformabli", "conform"), ("radicalli", "radic"), ("differentli", "differ"),
    ("vileli", "vile"), ("analogousli", "analog"), ("vietnamization", "vietnam"),
    ("predication", "predic"), ("operator", "oper"), ("feudalism", "feudal"),
    ("decisiveness", "decis"), ("hopefulness", "hope"), ("callousness", "callous"),
    ("formaliti", "formal"), ("sensitiviti", "sensit"), ("sensibiliti", "sensibl"),
])
def test_step2_derivational_suffixes(word: str, expected: str):
    assert stem(word) == expected


@pytest.mark.parametrize("word,expected", [
    # Step 3
    ("triplicate", "triplic"), ("formative", "form"), ("formalize", "formal"),
    ("electriciti", "electr"), ("electrical", "electr"), ("hopeful", "hope"),
    ("goodness", "good"),
])
def test_step3_derivational_suffixes(word: str, expected: str):
    assert stem(word) == expected


@pytest.mark.parametrize("word,expected", [
    # Step 4 — the suffix goes entirely when the stem is long enough
    ("revival", "reviv"), ("allowance", "allow"), ("inference", "infer"),
    ("airliner", "airlin"), ("gyroscopic", "gyroscop"), ("adjustable", "adjust"),
    ("defensible", "defens"), ("irritant", "irrit"), ("replacement", "replac"),
    ("adjustment", "adjust"), ("dependent", "depend"), ("adoption", "adopt"),
    ("homologou", "homolog"), ("communism", "commun"), ("activate", "activ"),
    ("angulariti", "angular"), ("homologous", "homolog"), ("effective", "effect"),
    ("bowdlerize", "bowdler"),
])
def test_step4_suffix_removal(word: str, expected: str):
    assert stem(word) == expected


@pytest.mark.parametrize("word,expected", [
    # Step 5 — tidying the ending
    ("probate", "probat"), ("rate", "rate"), ("cease", "ceas"),
    ("controll", "control"), ("roll", "roll"),
])
def test_step5_final_tidy(word: str, expected: str):
    assert stem(word) == expected


def test_ion_only_strips_after_s_or_t():
    # "adoption" -> "adopt" but a short stem like "lion" must survive intact,
    # which is the rule people most often drop when hand-porting the algorithm.
    assert stem("adoption") == "adopt"
    assert stem("lion") == "lion"


def test_short_words_are_untouched():
    for word in ("a", "an", "be", "is", "of"):
        assert stem(word) == word


def test_measure_counts_vowel_consonant_sequences():
    # From the paper: m=0 for "tree", m=1 for "trouble", m=2 for "troubles".
    assert _measure("tr") == 0
    assert _measure("tree") == 0
    assert _measure("trouble") == 1
    assert _measure("troubles") == 2
    assert _measure("private") == 2


def test_cvc_detects_the_silent_e_condition():
    assert _cvc("hop")
    assert _cvc("fil")
    assert not _cvc("fail")      # ends vowel-vowel-consonant
    assert not _cvc("snow")      # excluded final w
    assert not _cvc("box")       # excluded final x


def test_stemming_is_idempotent():
    # Stemming an already-stemmed term must not change it again, or repeated
    # indexing of the same corpus would drift.
    for word in ("relational", "conditional", "hopefulness", "adjustable",
                 "digitizer", "sensitiviti", "happy", "running"):
        once = stem(word)
        assert stem(once) == once, word


def test_cache_agrees_with_the_uncached_path():
    for word in ("relational", "ponies", "happy", "adoption", "controll", "sky"):
        assert stem_cached(word) == stem(word)


# --------------------------------------------------------------------------- #
# Integration with the index
# --------------------------------------------------------------------------- #

DOCS = [
    "the retrieval system retrieves documents",
    "we are running fast and he runs faster",
    "completely unrelated content about gardening",
]


def test_stemming_is_off_by_default():
    # Turning this on silently would change every number already published.
    assert BM25Index().stem is False
    plain = BM25Index().fit(DOCS)
    assert plain.score("retrieving")[0] == 0.0


def test_stemming_bridges_morphological_variants():
    stemmed = BM25Index(stem=True).fit(DOCS)
    # "retrieving" never appears in the corpus, but shares a stem with both
    # "retrieval" and "retrieves".
    assert stemmed.score("retrieving")[0] > 0.0
    assert stemmed.score("run")[1] > 0.0


def test_index_and_query_use_the_same_analyzer():
    # An index built with stemming must stem queries too. If they diverge the
    # vocabulary silently stops matching and it looks like poor relevance
    # rather than like a bug.
    stemmed = BM25Index(stem=True).fit(DOCS)
    assert any(term != token for term, token in zip(stemmed.vocab, DOCS[0].split()))
    assert stemmed.score("retrieval").argmax() == 0
    assert stemmed.score("retrieves").argmax() == 0


def test_stemming_shrinks_the_vocabulary():
    plain = BM25Index().fit(DOCS)
    stemmed = BM25Index(stem=True).fit(DOCS)
    assert len(stemmed.vocab) < len(plain.vocab)


def test_explain_still_works_under_stemming():
    stemmed = BM25Index(stem=True).fit(DOCS)
    terms = stemmed.explain("retrieval documents", 0)
    assert terms
    assert all(isinstance(value, float) for _, value in terms)
