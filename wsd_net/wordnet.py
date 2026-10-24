"""WordNet access: loading, the sense inventory, and lexical relations.

WordNet 3.0 is bundled in ``data/nltk_data/corpora/wordnet.zip`` and NLTK reads
it straight from the zip, so the project works offline on a fresh clone. If
the bundle is missing, :func:`ensure_wordnet` downloads it once into the same
folder.

Two facts about WordNet that the algorithms rely on:

* **Sense order is frequency order.** A word's synsets are listed by how often
  each sense was tagged in the SemCor corpus, so ``synsets(w)[0]`` is the
  most-frequent-sense baseline. ``Lemma.count()`` exposes those counts.
* **Relations link synsets, not words** - hypernymy (*is-a*), meronymy
  (*part-of*), holonymy (*has-part*) and so on connect meanings. Antonymy is
  the exception: it links lemmas, which is why :func:`relations` reads it
  from the lemmas.
"""

from __future__ import annotations

from functools import lru_cache

import nltk

from .config import NLTK_DIR

if str(NLTK_DIR) not in nltk.data.path:
    nltk.data.path.insert(0, str(NLTK_DIR))

from nltk.corpus import wordnet as wn  # noqa: E402

POS_NAMES = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective (satellite)", "r": "adverb"}


def ensure_wordnet() -> str:
    """Make sure WordNet loads; download it into the project if needed."""
    try:
        wn.ensure_loaded()
    except LookupError:
        NLTK_DIR.mkdir(parents=True, exist_ok=True)
        nltk.download("wordnet", download_dir=str(NLTK_DIR), quiet=True)
        wn.ensure_loaded()
    return wn.get_version()


def senses(word: str, pos: str | None = None) -> list:
    return wn.synsets(word.replace(" ", "_"), pos=pos)


def sense_count(synset) -> int:
    """How many times this sense was tagged in SemCor (summed over its lemmas)."""
    return sum(lemma.count() for lemma in synset.lemmas())


def relations(synset) -> dict[str, list]:
    rel = {
        "hypernyms (is a)": synset.hypernyms() + synset.instance_hypernyms(),
        "hyponyms (kinds of)": synset.hyponyms() + synset.instance_hyponyms(),
        "part meronyms (has part)": synset.part_meronyms(),
        "member meronyms (has member)": synset.member_meronyms(),
        "substance meronyms (made of)": synset.substance_meronyms(),
        "part holonyms (part of)": synset.part_holonyms(),
        "member holonyms (member of)": synset.member_holonyms(),
        "substance holonyms (substance of)": synset.substance_holonyms(),
        "also see": synset.also_sees(),
        "similar to": synset.similar_tos(),
        "entailments": synset.entailments(),
    }
    antonyms = sorted({a.synset() for lemma in synset.lemmas() for a in lemma.antonyms()},
                      key=lambda s: s.name())
    rel["antonyms"] = antonyms
    return {k: v for k, v in rel.items() if v}


def related_synsets(synset) -> list:
    """Every synset one relation away - the material Extended Lesk reads."""
    out = []
    for group in relations(synset).values():
        out.extend(group)
    return out


@lru_cache(maxsize=4096)
def hypernym_path(name: str) -> list[str]:
    """The first hypernym path from the root (entity.n.01) down to the synset."""
    paths = wn.synset(name).hypernym_paths()
    return [s.name() for s in min(paths, key=len)] if paths else [name]


def describe(synset) -> dict:
    return {
        "synset": synset.name(),
        "pos": POS_NAMES.get(synset.pos(), synset.pos()),
        "lemmas": ", ".join(l.name().replace("_", " ") for l in synset.lemmas()),
        "definition": synset.definition(),
        "examples": " | ".join(synset.examples()),
        "semcor_count": sense_count(synset),
        "hypernym": ", ".join(h.name() for h in synset.hypernyms()[:2]),
        "lexname": synset.lexname(),
    }


def similarity(word_a: str, word_b: str, pos: str = "n") -> dict | None:
    """Best-matching sense pair of two words and three similarity measures."""
    best = None
    for a in senses(word_a, pos):
        for b in senses(word_b, pos):
            wup = a.wup_similarity(b) or 0.0
            if best is None or wup > best[2]:
                best = (a, b, wup)
    if best is None:
        return None
    a, b, wup = best
    lcs = a.lowest_common_hypernyms(b)
    return {"sense_a": a, "sense_b": b, "wup": wup,
            "path": a.path_similarity(b) or 0.0,
            "lch": a.lch_similarity(b) if a.pos() == b.pos() else None,
            "lcs": lcs[0] if lcs else None}
