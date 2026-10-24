"""Scoring the methods on the hand-annotated sentences.

For each item a prediction is correct if it is any synset in the item's gold
list. Besides plain accuracy the report gives, per method:

* **decided** - the share of items where the method had any evidence at all
  (some sense scored above zero), i.e. did not fall back to the first sense;
* **accuracy when decided** - how good the method is on its own evidence.

A method can look respectable only because its back-off is the strong
most-frequent-sense baseline; these two numbers separate the two.

The random baseline is reported as its *expected* accuracy (share of the
word's senses that are gold), not as one noisy random draw.
"""

from __future__ import annotations

import json
from collections import defaultdict

from .algorithms import METHOD_FUNCS
from .config import GOLD_FILE, METHODS
from .wordnet import senses


def load_gold(path=GOLD_FILE) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["items"]


def run(items: list[dict] | None = None) -> list[dict]:
    """One row per (item, method)."""
    items = items or load_gold()
    rows = []
    for it in items:
        cands = senses(it["word"], "n")
        gold = set(it["gold"])
        rows.append({"id": it["id"], "word": it["word"], "sense_group": it["sense_group"],
                     "method": "random", "predicted": "", "correct": len(gold & {s.name() for s in cands}) / len(cands),
                     "backoff": False, "n_senses": len(cands), "sentence": it["sentence"], "gold": " ".join(it["gold"])})
        for name, fn in METHOD_FUNCS.items():
            p = fn(it["sentence"], it["word"], "n")
            rows.append({"id": it["id"], "word": it["word"], "sense_group": it["sense_group"],
                         "method": name, "predicted": p.name or "", "correct": float(p.name in gold),
                         "backoff": p.backoff, "n_senses": len(cands), "sentence": it["sentence"],
                         "gold": " ".join(it["gold"])})
    return rows


def summarise(rows: list[dict]) -> dict:
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    out = {}
    for m in METHODS:
        rs = by_method[m]
        decided = [r for r in rs if not r["backoff"]]
        out[m] = {
            "accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4),
            "decided": round(len(decided) / len(rs), 4),
            "accuracy_when_decided": round(sum(r["correct"] for r in decided) / len(decided), 4) if decided else None,
        }
    return out


def per_word(rows: list[dict]) -> dict[str, dict[str, float]]:
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["word"]][r["method"]].append(r["correct"])
    return {w: {m: round(sum(v) / len(v), 3) for m, v in ms.items()} for w, ms in acc.items()}


def per_sense_group(rows: list[dict]) -> dict[str, dict[str, float]]:
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[f"{r['word']}: {r['sense_group']}"][r["method"]].append(r["correct"])
    return {g: {m: round(sum(v) / len(v), 3) for m, v in ms.items()} for g, ms in acc.items()}
