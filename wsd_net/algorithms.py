"""Word sense disambiguation algorithms.

Five knowledge-based methods plus a random baseline, all with the same
signature: ``method(sentence, word, pos) -> Prediction``. Every prediction
carries a score for *every* candidate sense and, for the Lesk family, the
overlapping words that produced the score - so the interface can show why a
sense won, not just which one.

Methods
-------
random            - uniform over the word's senses (expected accuracy, not a sample).
mfs               - most frequent sense: WordNet's first synset (SemCor order).
nltk_lesk         - NLTK's ``nltk.wsd.lesk``: original Lesk overlap between the raw
                    sentence tokens and each sense's definition.
simplified_lesk   - Lesk (1986) as simplified by Kilgarriff & Rosenzweig (2000):
                    overlap between the *lemmatised, stopword-free* context and
                    the sense's definition + examples + synonyms.
extended_lesk     - in the spirit of Banerjee & Pedersen (2003): the signature also
                    includes the glosses of every synset one WordNet relation away
                    (hypernyms, hyponyms, meronyms, holonyms ...). Own-gloss overlaps
                    count double.
wup_context       - a taxonomy method: each sense is scored by how close it sits, in
                    WordNet's is-a hierarchy (Wu-Palmer similarity), to the nouns
                    around it.

Ties are broken in favour of the more frequent sense, and when every sense
scores zero the method backs off to the most frequent sense; ``backoff`` says
when that happened, because a "correct" back-off answer is the baseline's
success, not the method's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from nltk.wsd import lesk as nltk_lesk_impl

from .config import OWN_GLOSS_WEIGHT, RELATED_GLOSS_WEIGHT
from .text import content_lemmas, context_bag, tokenize
from .wordnet import related_synsets, senses, wn


@dataclass
class Prediction:
    method: str
    synset: object | None
    scores: dict[str, float] = field(default_factory=dict)     # synset name -> score
    overlaps: dict[str, list[str]] = field(default_factory=dict)
    backoff: bool = False

    @property
    def name(self) -> str | None:
        return self.synset.name() if self.synset is not None else None


def _pick(method: str, candidates: list, scores: dict, overlaps: dict | None = None) -> Prediction:
    """Highest score wins; ties go to the earlier (more frequent) sense."""
    if not candidates:
        return Prediction(method, None)
    best = max(range(len(candidates)), key=lambda i: (scores[candidates[i].name()], -i))
    backoff = all(v == 0 for v in scores.values())
    return Prediction(method, candidates[best], scores, overlaps or {}, backoff)


# --- signatures ------------------------------------------------------------------

@lru_cache(maxsize=8192)
def own_signature(name: str) -> frozenset[str]:
    s = wn.synset(name)
    text = " ".join([s.definition(), *s.examples(), *(l.name().replace("_", " ") for l in s.lemmas())])
    return frozenset(content_lemmas(text))


@lru_cache(maxsize=8192)
def related_signature(name: str) -> frozenset[str]:
    words: set[str] = set()
    for rel in related_synsets(wn.synset(name)):
        words |= own_signature(rel.name())
    return frozenset(words)


# --- methods -----------------------------------------------------------------------

def mfs(sentence: str, word: str, pos: str | None = "n") -> Prediction:
    cands = senses(word, pos)
    return _pick("mfs", cands, {s.name(): float(len(cands) - i) for i, s in enumerate(cands)})


def nltk_lesk(sentence: str, word: str, pos: str | None = "n") -> Prediction:
    cands = senses(word, pos)
    context = set(tokenize(sentence))
    scores = {s.name(): float(len(context & set(s.definition().split()))) for s in cands}
    chosen = nltk_lesk_impl(tokenize(sentence), word, pos)
    pred = Prediction("nltk_lesk", chosen, scores,
                      backoff=all(v == 0 for v in scores.values()))
    return pred


def simplified_lesk(sentence: str, word: str, pos: str | None = "n") -> Prediction:
    cands = senses(word, pos)
    context = set(context_bag(sentence, word))
    scores, overlaps = {}, {}
    for s in cands:
        hit = sorted(context & (own_signature(s.name()) - {word}))
        scores[s.name()] = float(len(hit))
        overlaps[s.name()] = hit
    return _pick("simplified_lesk", cands, scores, overlaps)


def extended_lesk(sentence: str, word: str, pos: str | None = "n") -> Prediction:
    cands = senses(word, pos)
    context = set(context_bag(sentence, word))
    scores, overlaps = {}, {}
    for s in cands:
        own = context & (own_signature(s.name()) - {word})
        rel = (context & (related_signature(s.name()) - {word})) - own
        scores[s.name()] = OWN_GLOSS_WEIGHT * len(own) + RELATED_GLOSS_WEIGHT * len(rel)
        overlaps[s.name()] = sorted(own) + [f"{w}*" for w in sorted(rel)]
    return _pick("extended_lesk", cands, scores, overlaps)


@lru_cache(maxsize=4096)
def _noun_senses(lemma: str) -> tuple:
    return tuple(wn.synsets(lemma, "n"))


def wup_context(sentence: str, word: str, pos: str | None = "n") -> Prediction:
    cands = senses(word, pos)
    context_nouns = [w for w in set(context_bag(sentence, word)) if _noun_senses(w)]
    scores, overlaps = {}, {}
    for s in cands:
        total, best_words = 0.0, []
        for c in context_nouns:
            best = max(((s.wup_similarity(cs) or 0.0), cs.name()) for cs in _noun_senses(c))
            total += best[0]
            if best[0] >= 0.6:
                best_words.append(f"{c} ({best[0]:.2f})")
        scores[s.name()] = round(total / len(context_nouns), 4) if context_nouns else 0.0
        overlaps[s.name()] = best_words
    return _pick("wup_context", cands, scores, overlaps)


METHOD_FUNCS = {
    "mfs": mfs,
    "nltk_lesk": nltk_lesk,
    "simplified_lesk": simplified_lesk,
    "extended_lesk": extended_lesk,
    "wup_context": wup_context,
}

METHOD_LABELS = {
    "random": "Random (expected)",
    "mfs": "Most frequent sense",
    "nltk_lesk": "NLTK Lesk",
    "simplified_lesk": "Simplified Lesk",
    "extended_lesk": "Extended Lesk",
    "wup_context": "Wu-Palmer context",
}


def disambiguate(sentence: str, word: str, pos: str | None = "n") -> dict[str, Prediction]:
    return {name: fn(sentence, word, pos) for name, fn in METHOD_FUNCS.items()}
