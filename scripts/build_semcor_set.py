"""Build data/gold/semcor_sample.json - naturally occurring test sentences.

    python scripts/build_semcor_set.py

The hand-written gold set is balanced and was written by the developer, so it
can flatter gloss-overlap methods. This script draws real sentences from
SemCor (Miller et al., 1993; the Brown corpus hand-tagged with WordNet senses)
in which one of the ten target nouns was tagged with one of its noun senses -
an independent check written by nobody involved in this project.

You do NOT need to run it - the output is committed. It needs NLTK's SemCor:
    python -m nltk.downloader semcor
Reading SemCor takes about a minute.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsd_net.config import DATA_DIR  # noqa: E402
from wsd_net.wordnet import wn  # noqa: E402

TARGETS = ["bank", "plant", "bass", "crane", "mouse", "bat", "spring", "star", "cell", "court"]
PER_WORD = 20
SEED = 7


def main() -> None:
    from nltk.corpus import semcor
    from nltk.tree import Tree

    noun_senses = {w: {s.name() for s in wn.synsets(w, "n")} for w in TARGETS}
    found = defaultdict(list)
    for sent in semcor.tagged_sents(tag="sem"):
        words, hits = [], []
        for chunk in sent:
            if isinstance(chunk, Tree):
                text = " ".join(chunk.leaves())
                label = chunk.label()
                if hasattr(label, "synset"):
                    lemma = label.name().lower()
                    synset = label.synset().name()
                    if lemma in noun_senses and synset in noun_senses[lemma]:
                        hits.append((lemma, synset, text))
                words.append(text)
            else:
                words.append(" ".join(chunk) if isinstance(chunk, list) else str(chunk))
        for lemma, synset, surface in hits:
            found[lemma].append({"word": lemma, "surface": surface, "gold": [synset],
                                 "sentence": " ".join(words)})
    rng = random.Random(SEED)
    items = []
    for w in TARGETS:
        pool = found.get(w, [])
        sample = rng.sample(pool, min(PER_WORD, len(pool)))
        for i, item in enumerate(sample, 1):
            items.append({"id": f"semcor-{w}-{i}", **item,
                          "sense_group": wn.synset(item["gold"][0]).definition()[:40]})
        print(f"  {w:<7} {len(pool):>4} tagged noun instances, {len(sample)} sampled")
    out = DATA_DIR / "gold" / "semcor_sample.json"
    out.write_text(json.dumps({"description": "Random sample of SemCor 3.0 sentences in which a target "
                               "noun is sense-tagged; gold is the single SemCor synset (fine-grained).",
                               "items": items}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(items)} items written to {out.relative_to(DATA_DIR.parent)}")


if __name__ == "__main__":
    main()
