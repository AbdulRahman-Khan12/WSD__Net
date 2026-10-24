"""Evaluate every method on both test sets and write outputs/.

    python scripts/evaluate.py

Writes
  outputs/predictions.csv      one row per sentence x method (the annotation output)
  outputs/summary.json         accuracy / decided / accuracy-when-decided per method and set
  outputs/per_word.csv         accuracy per target word, method and set
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wsd_net.algorithms import METHOD_LABELS  # noqa: E402
from wsd_net.config import DATA_DIR, METHODS, OUTPUT_DIR  # noqa: E402
from wsd_net.evaluate import load_gold, per_word, run, summarise  # noqa: E402
from wsd_net.wordnet import ensure_wordnet  # noqa: E402

print(f"WordNet {ensure_wordnet()}")
OUTPUT_DIR.mkdir(exist_ok=True)
sets = {"handwritten": DATA_DIR / "gold" / "sense_tagged.json",
        "semcor": DATA_DIR / "gold" / "semcor_sample.json"}
summary, all_rows, word_rows = {}, [], []
for name, path in sets.items():
    t = time.perf_counter()
    items = load_gold(path)
    rows = run(items)
    summary[name] = {"items": len(items), "methods": summarise(rows)}
    for r in rows:
        all_rows.append({"set": name, **r})
    for word, accs in per_word(rows).items():
        word_rows.append({"set": name, "word": word, **{m: accs.get(m) for m in METHODS}})
    print(f"\n{name} set: {len(items)} sentences ({time.perf_counter() - t:.1f}s)")
    print(f"  {'method':<22}{'accuracy':>9}{'decided':>9}{'acc|decided':>13}")
    for m, v in summary[name]["methods"].items():
        awd = "-" if v["accuracy_when_decided"] is None else f"{v['accuracy_when_decided']:.3f}"
        print(f"  {METHOD_LABELS[m]:<22}{v['accuracy']:>9.3f}{v['decided']:>9.2f}{awd:>13}")

with (OUTPUT_DIR / "predictions.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
    w.writeheader()
    w.writerows(all_rows)
with (OUTPUT_DIR / "per_word.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(word_rows[0]))
    w.writeheader()
    w.writerows(word_rows)
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("\nwrote outputs/predictions.csv, outputs/per_word.csv, outputs/summary.json")
