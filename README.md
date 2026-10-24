# WSD__Net

Word sense disambiguation with **WordNet**: given a sentence and an ambiguous
word, decide which of the word's WordNet senses is meant. Five
knowledge-based methods are compared side by side, each showing the evidence
it used, and evaluated on two test sets that disagree in an instructive way.

Built for **AML23702 Advance Natural Language Processing, Lab Exercise 5**.

**Runs offline in one command on a fresh clone** — WordNet 3.0 is bundled in
the repository, so nothing is downloaded at run time.

---

## Pipeline

```
sentence + target word
   │
   ├── text ─────────── regex tokens → stopwords out → WordNet morphy lemmas
   │                    (cranes → crane, swung → swing)
   ├── sense inventory ─ WordNet synsets of the target, in SemCor frequency order
   │
   ├── methods ─────┬── Most frequent sense      first synset (the baseline to beat)
   │                ├── NLTK Lesk                raw word overlap with the definition
   │                ├── Simplified Lesk          lemma overlap with definition + examples + synonyms
   │                ├── Extended Lesk            + glosses of hypernyms, hyponyms, meronyms, holonyms …
   │                └── Wu-Palmer context        closeness in the is-a hierarchy to nearby nouns
   │
   └── output ────────── chosen sense, the score of every sense, the overlapping words
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

Reproduce the outputs and run the self-check:

```bash
python scripts/build_inventory.py   # outputs/sense_inventory.csv
python scripts/evaluate.py          # outputs/predictions.csv, per_word.csv, summary.json
python scripts/check.py             # self-check
```

## The app

| Tab | Shows |
|---|---|
| Disambiguate | any sentence, any word in it: each method's chosen sense, every sense's score, and the overlapping words behind each score |
| Sense inventory | all senses of a word with definitions, examples, SemCor counts; every lexical relation of a sense; its hypernym path to *entity* |
| Lexical relations & similarity | Wu-Palmer, path and Leacock-Chodorow similarity between two words, with their lowest common hypernym |
| Evaluation | accuracy on both test sets, accuracy per word, and every prediction with errors filterable |

## Test sets

| Set | Size | What it is |
|---|---|---|
| **Hand-written** | 60 sentences | 10 ambiguous nouns × 6 sentences, 2–3 clearly different meanings each, equal numbers per meaning (`data/gold/sense_tagged.json`) |
| **SemCor sample** | 131 sentences | real Brown-corpus sentences sense-tagged by WordNet's annotators, randomly sampled for the same nouns (`data/gold/semcor_sample.json`) |

The ten nouns are *bank, plant, bass, crane, mouse, bat, spring, star, cell,
court* — 68 noun senses in all (`outputs/sense_inventory.csv`). Because WordNet
splits some meanings very finely (three separate "fish" senses of *bass*), a
hand-written item lists every synset that is an acceptable answer. SemCor items
have the single synset its annotators chose.

## Results

| Method | Hand-written | SemCor sample |
|---|---|---|
| Random (expected) | 0.256 | 0.159 |
| Most frequent sense | 0.417 | **0.702** |
| NLTK Lesk | 0.467 | 0.282 |
| Simplified Lesk | 0.783 | 0.573 |
| Extended Lesk | **0.900** | 0.641 |
| Wu-Palmer context | 0.467 | 0.267 |

**The two sets disagree, and the disagreement is the main finding.**

- On the **balanced, hand-written** set the first sense is right only for the
  meaning listed first, so MFS sits near 0.42 and gloss overlap wins clearly:
  Extended Lesk reaches 0.90.
- On **natural text** one sense dominates (the legal *court*, the season
  *spring*, the rodent *mouse*), so MFS scores 0.70 and **no knowledge-based
  method beats it** — the classic result in WSD. Part of the gap is structural:
  WordNet's sense order was derived from SemCor counts, so on SemCor the
  baseline is partly evaluated on its own training data.
- **Extended Lesk is the best method on both sets.** Reading related glosses
  lifts the share of sentences with any evidence from 60% to 87% (hand-written)
  and from 37% to 73% (SemCor).
- **Where Lesk has evidence it is very accurate on clean sentences (0.94) but
  not on real ones (0.58)**: real sentences are long, and incidental words
  overlap with the wrong gloss.
- **NLTK's Lesk** never removes stopwords, so *the, of, a* overlap with every
  definition: it almost always "decides", mostly on noise.
- **Wu-Palmer context** is weak because every context noun is itself
  ambiguous; taking the best of its senses lets almost any target sense find a
  close match.

Extended Lesk's six errors on the hand-written set show both failure modes
(all in `outputs/predictions.csv`):

- **A spurious overlap.** *The power plant burns coal …* → the botanical
  sense, because that sense's own definition says a plant lacks "the **power**
  of locomotion". *A new assembly plant that will employ two thousand workers*
  → botanical again, on the single word *two* in a related gloss.
- **No overlap at all.** *The crane spread its wings*, *Call me on my cell* and
  *Use the mouse and keyboard* share no word with any gloss, so the method
  falls back to the first sense — the writer Stephen Crane, "any small
  compartment" and the rodent.

## Layout

```
WSD__Net/
├── app.py                  Streamlit interface
├── wsd_net/
│   ├── config.py           paths, Extended Lesk weights, method list
│   ├── wordnet.py          loading (bundled zip), inventory, relations, similarity
│   ├── text.py             tokenise, stopwords, morphy lemmas, context bag
│   ├── algorithms.py       MFS, NLTK Lesk, Simplified / Extended Lesk, Wu-Palmer
│   └── evaluate.py         accuracy, decided rate, accuracy when decided
├── data/
│   ├── nltk_data/corpora/wordnet.zip   WordNet 3.0 (Princeton licence, see NOTICE.md)
│   └── gold/               sense_tagged.json (60), semcor_sample.json (131)
├── outputs/                sense_inventory.csv, predictions.csv, per_word.csv, summary.json
└── scripts/
    ├── evaluate.py         both test sets -> outputs/
    ├── build_inventory.py  sense inventory -> outputs/
    ├── build_semcor_set.py samples SemCor (not needed - output committed)
    └── check.py            self-check
```

## Design notes

**Why report "decided".** When no sense overlaps the context, a Lesk method
falls back to the first sense — so part of its accuracy is really the
baseline's. Reporting how often a method found any evidence, and its accuracy
on those sentences alone, separates the two.

**Why acceptable-sense lists.** WordNet is famously fine-grained: human
annotators agree on fine senses only about 70% of the time. Scoring *bass*
(fish) as wrong because the system picked *sea bass* rather than *freshwater
bass* would measure WordNet's granularity, not the method.

**Why a second, natural test set.** The hand-written sentences were written by
the developer and are balanced across meanings, which flatters methods that
read context and penalises the frequency baseline. SemCor sentences were
tagged by WordNet's own annotators and follow real sense frequencies; together
the two sets bracket the truth.

**Why WordNet's lemmatiser.** Lesk matches context words against gloss words;
lemmatising both with the same tool (morphy) is what lets *lifted* in the
context match *lifts* in the gloss.

**Why bundle WordNet.** `nltk.download()` at start-up is the most common way
these projects fail on a fresh machine. The 10 MB zip removes that failure
mode; the licence permits redistribution with its notice (see `NOTICE.md`).

## Limitations

- The target's part of speech is chosen by the user (default noun); there is
  no POS tagger.
- Lesk overlap is unweighted: a match on *water* counts the same as a match
  on *thing*.
- Ten nouns is a small sample; verbs and adjectives, which are harder, are
  not evaluated.
- A natural next step is a hybrid that adds a sense-frequency prior to the
  Extended Lesk score, combining the two strengths seen above.
