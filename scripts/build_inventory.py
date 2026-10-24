"""Write outputs/sense_inventory.csv - the sense inventory of the ten target nouns.

    python scripts/build_inventory.py

One row per noun sense: synset id, synonyms, definition, examples, how often it
was tagged in SemCor, its hypernym and its WordNet lexicographer file.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsd_net.config import OUTPUT_DIR  # noqa: E402
from wsd_net.evaluate import load_gold  # noqa: E402
from wsd_net.wordnet import describe, ensure_wordnet, senses  # noqa: E402

ensure_wordnet()
words = list(dict.fromkeys(item["word"] for item in load_gold()))
rows = []
for word in words:
    for rank, s in enumerate(senses(word, "n"), 1):
        rows.append({"word": word, "rank": rank, **describe(s)})
OUTPUT_DIR.mkdir(exist_ok=True)
with (OUTPUT_DIR / "sense_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} noun senses for {len(words)} words -> outputs/sense_inventory.csv")
for word in words:
    n = sum(1 for r in rows if r["word"] == word)
    print(f"  {word:<7} {n:>2} senses")
