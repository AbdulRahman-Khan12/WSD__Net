"""WSD__Net - Streamlit interface.

Run with:  streamlit run app.py

Three questions, one tab each: *what can this word mean?* (the sense
inventory and its lexical relations), *which meaning is it here?* (five
disambiguation methods side by side, with the evidence each one used), and
*how good are the methods?* (accuracy on two test sets).
"""

from __future__ import annotations

import html
import json

import altair as alt
import pandas as pd
import streamlit as st

from wsd_net import __version__
from wsd_net.algorithms import METHOD_LABELS, disambiguate
from wsd_net.config import METHODS, OUTPUT_DIR
from wsd_net.text import content_lemmas, find_target, lemma, tokenize
from wsd_net.wordnet import POS_NAMES, describe, ensure_wordnet, hypernym_path, relations, senses, similarity, wn

st.set_page_config(page_title="WSD__Net", page_icon="❦", layout="wide")
st.markdown(
    """
    <style>
      .sent { font-size: 1.12rem; line-height: 1.7; }
      .sent mark { background: #FDE68A; padding: 0 0.2em; border-radius: 3px; font-weight: 600; }
      .sense { border-left: 3px solid #B45309; background: #FBF7F1; padding: 0.5rem 0.8rem; margin: 0.3rem 0; }
      .sense.win { border-left-color: #15803D; background: #F0FDF4; }
      .meta { font-family: ui-monospace, Menlo, monospace; font-size: 0.76rem; color: #5B6573; }
      .chip { display: inline-block; font-family: ui-monospace, Menlo, monospace; font-size: 0.74rem;
              padding: 0.05rem 0.4rem; margin: 0.1rem; border-radius: 3px; background: #E7EEF7; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading WordNet ...")
def boot() -> str:
    return ensure_wordnet()


version = boot()

with st.sidebar:
    st.markdown("### WSD__Net")
    st.caption(f"Word sense disambiguation with WordNet {version}")
    st.markdown("**Methods**")
    st.markdown(
        "- **MFS** — first WordNet sense (most frequent in SemCor)\n"
        "- **NLTK Lesk** — raw overlap with the definition\n"
        "- **Simplified Lesk** — lemmatised overlap with definition, examples, synonyms\n"
        "- **Extended Lesk** — also the glosses of related senses (hypernyms, parts ...)\n"
        "- **Wu-Palmer context** — closeness in the is-a hierarchy to the nouns nearby")
    st.caption(f"v{__version__}")

tab_wsd, tab_inv, tab_sim, tab_eval = st.tabs(
    ["Disambiguate", "Sense inventory", "Lexical relations & similarity", "Evaluation"])

EXAMPLES = {
    "He sat on the bank of the river and watched the boats.": "bank",
    "The bank approved her loan and raised the interest rate.": "bank",
    "The crane lifted steel beams to the top of the tower.": "crane",
    "Click the mouse to move the cursor across the screen.": "mouse",
    "The plant needs sunlight, water and good soil to grow.": "plant",
    "The judge told the court that the trial would continue.": "court",
}

# --- disambiguate ---------------------------------------------------------------
with tab_wsd:
    pick = st.selectbox("Example", ["(type your own)"] + list(EXAMPLES))
    default = pick if pick != "(type your own)" else "The fisherman caught a bass in the lake."
    sentence = st.text_input("Sentence", default)
    tokens = tokenize(sentence)
    candidates = [t for t in dict.fromkeys(tokens) if senses(lemma(t))]
    if not candidates:
        st.warning("No word in this sentence is in WordNet.")
    else:
        guess = EXAMPLES.get(pick, "bass")
        idx = next((i for i, t in enumerate(candidates) if lemma(t) == guess), 0)
        c1, c2 = st.columns([2, 1])
        token = c1.selectbox("Word to disambiguate", candidates, index=idx)
        pos_label = c2.selectbox("Part of speech", ["noun", "verb", "adjective", "adverb", "any"])
        pos = {"noun": "n", "verb": "v", "adjective": "a", "adverb": "r", "any": None}[pos_label]
        word = lemma(token, pos)
        cands = senses(word, pos)
        if not cands:
            st.warning(f"'{word}' has no {pos_label} senses in WordNet - try another part of speech.")
        else:
            marked = " ".join(f"<mark>{html.escape(t)}</mark>" if i == find_target(sentence, word) else html.escape(t)
                              for i, t in enumerate(sentence.split()))
            st.markdown(f"<div class='sent'>{marked}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='meta'>target lemma: {word} · {len(cands)} {pos_label} senses · "
                        f"context lemmas: {', '.join(t for t in content_lemmas(sentence) if t != word)}</div>",
                        unsafe_allow_html=True)
            preds = disambiguate(sentence, word, pos)
            st.markdown("#### What each method chose")
            cols = st.columns(len(preds))
            for col, (name, p) in zip(cols, preds.items()):
                with col:
                    st.markdown(f"**{METHOD_LABELS[name]}**")
                    if p.synset is None:
                        st.caption("no answer")
                        continue
                    st.markdown(f"`{p.name}`")
                    st.caption(p.synset.definition())
                    if p.backoff and name != "mfs":
                        st.markdown("<span class='chip'>no evidence → first sense</span>", unsafe_allow_html=True)
            st.markdown("#### Score of every sense")
            table = []
            for rank, s in enumerate(cands, 1):
                row = {"#": rank, "synset": s.name(), "definition": s.definition(), "SemCor count": describe(s)["semcor_count"]}
                for name, p in preds.items():
                    if name != "mfs":
                        row[METHOD_LABELS[name]] = p.scores.get(s.name(), 0.0)
                table.append(row)
            st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
            st.markdown("#### Why: the overlapping words")
            st.caption("Extended Lesk marks words found only in a *related* sense's gloss with *. "
                       "Wu-Palmer lists context nouns with similarity ≥ 0.6.")
            for name in ("simplified_lesk", "extended_lesk", "wup_context"):
                p = preds[name]
                with st.expander(METHOD_LABELS[name], expanded=(name == "extended_lesk")):
                    for s in cands:
                        words = p.overlaps.get(s.name(), [])
                        win = "win" if p.synset is not None and s.name() == p.name else ""
                        chips = "".join(f"<span class='chip'>{html.escape(w)}</span>" for w in words) or \
                            "<span class='meta'>no overlap</span>"
                        st.markdown(f"<div class='sense {win}'><b>{s.name()}</b> — score {p.scores.get(s.name(), 0):g}"
                                    f"<br>{chips}</div>", unsafe_allow_html=True)


# --- inventory -----------------------------------------------------------------
with tab_inv:
    c1, c2 = st.columns([2, 1])
    w = c1.text_input("Word", "bank").strip().lower()
    p_label = c2.selectbox("POS", ["all", "noun", "verb", "adjective", "adverb"], key="inv_pos")
    p = {"all": None, "noun": "n", "verb": "v", "adjective": "a", "adverb": "r"}[p_label]
    ss = senses(w, p)
    if not ss:
        st.warning("Not in WordNet.")
    else:
        st.markdown(f"**{len(ss)} senses** of *{w}* — ordered by SemCor frequency, so #1 is the MFS baseline.")
        frame = pd.DataFrame([{"#": i, **describe(s)} for i, s in enumerate(ss, 1)])
        st.dataframe(frame, hide_index=True, width="stretch")
        chosen = st.selectbox("Explore the relations of", [s.name() for s in ss])
        syn = wn.synset(chosen)
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"<div class='sense win'><b>{chosen}</b> ({POS_NAMES.get(syn.pos())})<br>"
                        f"{html.escape(syn.definition())}</div>", unsafe_allow_html=True)
            for rel, group in relations(syn).items():
                st.markdown(f"**{rel}**: " + ", ".join(f"`{g.name()}`" for g in group[:12]) +
                            (f" … (+{len(group) - 12})" if len(group) > 12 else ""))
        with right:
            path = hypernym_path(chosen)
            dot = ["digraph H { rankdir=TB; bgcolor=transparent; node [shape=box, style=filled, "
                   "fontname=Helvetica, fontsize=10, fillcolor=\"#EEF2F7\"];"]
            for a, b in zip(path, path[1:]):
                dot.append(f'"{a}" -> "{b}";')
            dot.append(f'"{chosen}" [fillcolor="#FDE68A"];')
            dot.append("}")
            st.markdown("**Hypernym path** (root → sense)")
            st.graphviz_chart("\n".join(dot))


# --- similarity -------------------------------------------------------------------
with tab_sim:
    st.markdown("#### How related are two words?")
    st.caption("The sense pair with the highest Wu-Palmer similarity is chosen, then three WordNet measures are reported.")
    c1, c2 = st.columns(2)
    a = c1.text_input("First noun", "car")
    b = c2.text_input("Second noun", "bicycle")
    res = similarity(a.strip().lower(), b.strip().lower())
    if res is None:
        st.warning("One of the words has no noun senses.")
    else:
        m = st.columns(3)
        m[0].metric("Wu-Palmer", f"{res['wup']:.3f}")
        m[1].metric("Path", f"{res['path']:.3f}")
        m[2].metric("Leacock-Chodorow", f"{res['lch']:.3f}" if res["lch"] is not None else "-")
        st.markdown(f"Closest senses: `{res['sense_a'].name()}` ({res['sense_a'].definition()}) and "
                    f"`{res['sense_b'].name()}` ({res['sense_b'].definition()}).")
        if res["lcs"] is not None:
            st.markdown(f"Lowest common hypernym: **`{res['lcs'].name()}`** — {res['lcs'].definition()}")
    st.markdown("#### Word pairs")
    pairs = [("car", "bicycle"), ("dog", "cat"), ("coffee", "tea"), ("bank", "river"), ("doctor", "hospital"),
             ("king", "queen"), ("computer", "mouse"), ("star", "sun")]
    st.dataframe(pd.DataFrame([{"pair": f"{x} – {y}", "wup": round(r["wup"], 3), "path": round(r["path"], 3),
                                "common hypernym": r["lcs"].name() if r["lcs"] else ""}
                               for x, y in pairs if (r := similarity(x, y))]), hide_index=True, width="stretch")


# --- evaluation ------------------------------------------------------------------
@st.cache_data
def evaluation():
    summary = json.loads((OUTPUT_DIR / "summary.json").read_text()) if (OUTPUT_DIR / "summary.json").exists() else None
    preds = pd.read_csv(OUTPUT_DIR / "predictions.csv") if (OUTPUT_DIR / "predictions.csv").exists() else None
    return summary, preds


with tab_eval:
    summary, preds = evaluation()
    if summary is None:
        st.info("Run `python scripts/evaluate.py` to produce the evaluation files.")
    else:
        set_label = st.radio("Test set", ["Hand-written (60, balanced)", "SemCor sample (131, natural)"], horizontal=True)
        key = "handwritten" if set_label.startswith("Hand") else "semcor"
        st.caption("Hand-written: 10 ambiguous nouns × 6 sentences, two or three meanings each, equal numbers per meaning. "
                   "SemCor: real Brown-corpus sentences tagged by WordNet's own annotators, where the first sense dominates.")
        frame = pd.DataFrame([{"method": METHOD_LABELS[m], **v} for m, v in summary[key]["methods"].items()])
        bars = alt.Chart(frame).mark_bar(color="#B45309").encode(
            x=alt.X("accuracy:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("method:N", sort=[METHOD_LABELS[m] for m in METHODS], title=None),
            tooltip=["method", "accuracy", "decided", "accuracy_when_decided"])
        st.altair_chart(bars.properties(height=230), width="stretch")
        st.dataframe(frame, hide_index=True, width="stretch")
        st.caption("*decided* = share of sentences where the method found any evidence; otherwise it falls back "
                   "to the first sense. *accuracy_when_decided* is the method's accuracy on its own evidence.")
        sub = preds[preds.set == key]
        heat = sub.groupby(["word", "method"])["correct"].mean().reset_index()
        heat["method"] = heat["method"].map(METHOD_LABELS)
        st.markdown("**Accuracy per word**")
        st.altair_chart(alt.Chart(heat).mark_rect().encode(
            x=alt.X("method:N", sort=[METHOD_LABELS[m] for m in METHODS], title=None), y=alt.Y("word:N", title=None),
            color=alt.Color("correct:Q", scale=alt.Scale(domain=[0, 1], scheme="greens"), title="accuracy"),
            tooltip=["word", "method", alt.Tooltip("correct:Q", format=".2f")]).properties(height=300) +
            alt.Chart(heat).mark_text(fontSize=10).encode(
                x=alt.X("method:N", sort=[METHOD_LABELS[m] for m in METHODS]), y="word:N",
                text=alt.Text("correct:Q", format=".2f")), width="stretch")
        st.markdown("**Every prediction**")
        method = st.selectbox("Method", [m for m in METHODS if m != "random"], format_func=METHOD_LABELS.get, index=3)
        only_err = st.checkbox("Only errors")
        show = sub[sub.method == method][["id", "sentence", "gold", "predicted", "correct", "backoff"]]
        if only_err:
            show = show[show.correct == 0]
        st.dataframe(show, hide_index=True, width="stretch")
