"""Self-check for WSD__Net.

    python scripts/check.py

Checks that WordNet loads from the bundled zip, that the text utilities
behave, that every algorithm returns a valid sense for every gold sentence,
and a handful of disambiguations whose answer is not in doubt.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsd_net.algorithms import METHOD_FUNCS, extended_lesk, simplified_lesk  # noqa: E402
from wsd_net.config import DATA_DIR  # noqa: E402
from wsd_net.evaluate import load_gold  # noqa: E402
from wsd_net.text import context_bag, find_target, lemma  # noqa: E402
from wsd_net.wordnet import ensure_wordnet, relations, senses, similarity, wn  # noqa: E402

RULE = "-" * 64
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print(RULE + "\n1. WORDNET\n" + RULE)
check("WordNet 3.0 loads", ensure_wordnet() == "3.0")
check("senses ordered by frequency", senses("plant", "n")[0].name() == "plant.n.01")
check("hypernym relation", "canine.n.02" in {h.name() for h in wn.synset("dog.n.01").hypernyms()})
check("antonyms read from lemmas", "antonyms" in relations(wn.synset("good.a.01")))
check("meronyms", any("meronym" in k for k in relations(wn.synset("car.n.01"))))
sim = similarity("car", "bicycle")
check("car-bicycle similarity is high", sim is not None and sim["wup"] >= 0.75, f"wup {sim['wup']:.2f}")

print(RULE + "\n2. TEXT\n" + RULE)
check("plural lemmatised", lemma("cranes") == "crane")
check("irregular verb lemmatised", lemma("swung", "v") == "swing")
check("target found through inflection", find_target("Flocks of cranes fly south", "crane") == 2)
check("target removed from context", "bank" not in context_bag("the bank of the river", "bank"))
check("stopwords removed", "the" not in context_bag("the bank of the river", "bank"))

print(RULE + "\n3. EVERY METHOD ON EVERY GOLD SENTENCE\n" + RULE)
items = load_gold() + load_gold(DATA_DIR / "gold" / "semcor_sample.json")
for name, fn in METHOD_FUNCS.items():
    valid = sum(1 for it in items
                if (p := fn(it["sentence"], it["word"], "n")).synset is not None
                and p.synset in senses(it["word"], "n"))
    check(f"{name}: a valid sense for all {len(items)} sentences", valid == len(items), f"{valid}")

print(RULE + "\n4. CLEAR CASES\n" + RULE)
cases = [
    (simplified_lesk, "I deposited money at the bank and asked about a loan.", "bank",
     {"depository_financial_institution.n.01", "bank.n.09"}),
    (extended_lesk, "The cat chased a mouse into a hole in the wall.", "mouse", {"mouse.n.01"}),
    (extended_lesk, "Move the mouse to point the cursor at the icon on the computer screen.", "mouse",
     {"mouse.n.04"}),
    (extended_lesk, "The tall crane lifted the heavy load at the construction site.", "crane", {"crane.n.04"}),
]
for fn, sentence, word, gold in cases:
    p = fn(sentence, word, "n")
    check(f"{fn.__name__}: {word} in '{sentence[:40]}...'", p.name in gold, p.name or "-")

print(RULE)
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
