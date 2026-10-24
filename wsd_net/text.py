"""Turning a sentence into a bag of context lemmas.

Deliberately simple and dependency-free: a regex tokenizer, a stopword list,
and WordNet's own lemmatiser (``morphy``), which maps *cranes* -> *crane* and
*swung* -> *swing* using WordNet's exception lists. Using WordNet's lemmatiser
means context words and gloss words are normalised the same way, which is what
gloss overlap needs.
"""

from __future__ import annotations

import re

from .wordnet import wn

TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*")

STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each either else
etc even ever every few for from further get gets got had has have having he her here hers
herself him himself his how however i if in into is it its itself just let like made make
many may me might more most much must my myself no nor not now of off often on once one
only or other our ours ourselves out over own per put rather same she should since so some
such than that the their theirs them themselves then there these they this those though
through thus to too under until up upon us use used using usually very was way we well were
what when where whether which while who whom whose why will with within without would yet
you your yours yourself yourselves especially something someone anything various e g esp
""".split())


def tokenize(text: str) -> list[str]:
    return [t.lower().strip("'-") for t in TOKEN.findall(text)]


def lemma(token: str, pos: str | None = None) -> str:
    """WordNet's morphy, trying the given POS first, then any POS."""
    for p in ([pos] if pos else []) + ["n", "v", "a", "r"]:
        base = wn.morphy(token, p)
        if base:
            return base
    return token


def content_lemmas(text: str) -> list[str]:
    out = []
    for tok in tokenize(text):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        base = lemma(tok)
        if base not in STOPWORDS:
            out.append(base)
    return out


def find_target(sentence: str, word: str) -> int | None:
    """Token index of the first occurrence of ``word`` (any inflection)."""
    word = word.lower()
    for i, tok in enumerate(tokenize(sentence)):
        if tok == word or lemma(tok, "n") == word or lemma(tok, "v") == word:
            return i
    return None


def context_bag(sentence: str, word: str) -> list[str]:
    """Content lemmas of the sentence, without the target word itself."""
    target = word.lower()
    return [t for t in content_lemmas(sentence) if t != target]
