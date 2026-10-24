"""Central configuration for WSD__Net."""

from __future__ import annotations

from pathlib import Path

PROJECT_NAME = "WSD__Net"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
NLTK_DIR = DATA_DIR / "nltk_data"            # WordNet 3.0 is bundled here (corpora/wordnet.zip)
GOLD_FILE = DATA_DIR / "gold" / "sense_tagged.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Extended Lesk: an overlap with the sense's *own* gloss counts more than an
# overlap with the gloss of a related sense (a hypernym, a part ...), because
# it is more direct evidence.
OWN_GLOSS_WEIGHT = 2.0
RELATED_GLOSS_WEIGHT = 1.0

# The methods compared, in the order they are reported.
METHODS = ("random", "mfs", "nltk_lesk", "simplified_lesk", "extended_lesk", "wup_context")
