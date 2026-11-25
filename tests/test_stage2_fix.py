"""Regression tests for the Windows/spaCy findings: parser misses on gerund-like nouns, concept filtering, relation evidence,
and coverage of lecturer-defined technical terms. A stub parser reproduces the reported behaviour without needing spaCy."""
import re
from contextlib import contextmanager
from pathlib import Path

from evaluation import metrics as M
from evaluation import run_eval as E
from lecturenotes import nlp
from lecturenotes.pipeline import process

SAMPLE = (Path(__file__).parent.parent / "samples" / "sample_lecture.txt").read_text(encoding="utf-8")
REPORTED = ["Overfitting happens when a model memorizes the training data instead of learning general patterns.",
            "Regularization reduces overfitting by penalizing large weights.",
            "The most important way to detect overfitting is to monitor the validation set during training."]


class _Tok:
    def __init__(self, text, idx):
        self.text, self.idx, self.lower_, self.is_stop, self.pos_ = text, idx, text.lower(), False, "NOUN"

    def __len__(self):
        return len(self.text)


class _Doc:
    """Parser stub: noun chunks for every phrase EXCEPT single '-ing' words (a tagger that labels them VERB/VBG)."""
    def __init__(self, sentence):
        self.noun_chunks = []
        for m in nlp._rule_mentions(sentence, frozenset()):
            if " " not in m.surface and m.surface.lower().endswith("ing"):
                continue
            self.noun_chunks.append([_Tok(t.group(), m.start + t.start()) for t in nlp.TOKEN.finditer(m.surface) if t.group()[0].isalpha()])


class _StubNLP:
    def pipe(self, texts):
        return (_Doc(t) for t in texts)


@contextmanager
def parser_stub():
    old = (nlp._NLP, nlp._TRIED)
    nlp._NLP, nlp._TRIED = _StubNLP(), True
    try:
        yield
    finally:
        nlp._NLP, nlp._TRIED = old


def _keys(sentence):
    return {m.key for s in nlp.analyze(sentence + " Padding sentence number one. Padding sentence number two.") for m in s.mentions}


# ---------------------------------------------------------------- gerund-like nouns the parser tags as verbs
def test_stub_parser_really_misses_the_gerund_noun_chunks():
    with parser_stub():
        chunks = [[t.text for t in c] for s in REPORTED for c in _Doc(s).noun_chunks]
    assert ["Overfitting"] not in chunks and ["overfitting"] not in chunks  # reproduces the reported parser behaviour


def test_noun_the_parser_missed_is_still_a_mention_in_every_reported_sentence():
    with parser_stub():
        for sentence in REPORTED:
            assert "overfitting" in _keys(sentence), sentence


def test_verbal_gerund_phrases_are_not_turned_into_terms():
    with parser_stub():
        keys = _keys(REPORTED[1]) | _keys(REPORTED[0])
    assert not any("penaliz" in k for k in keys)


def test_parser_chunks_win_over_rule_phrases_and_rules_only_fill_gaps():
    P = [nlp.Mention("training data", "training data", 10, 23)]
    R = [nlp.Mention("data", "data", 19, 23), nlp.Mention("overfitting", "overfitting", 30, 41)]
    merged = nlp._merge_mentions(P, R)
    assert [m.key for m in merged] == ["training data", "overfitting"]


# ---------------------------------------------------------------- concept filtering and relation evidence with parser gaps
def test_concept_and_stated_relation_survive_parser_gaps_with_evidence():
    with parser_stub():
        r = process(SAMPLE)
    names = {c["name"].lower() for c in r["concepts"]}
    assert "overfitting" in names
    rel = [x for x in r["relations"] if (x["source"], x["label"], x["target"]) == ("regularization", "reduces", "overfitting")]
    assert rel and any("Regularization reduces overfitting" in e["text"] for e in rel[0]["evidence"])
    cmap = {c["id"]: c for c in r["concepts"]}
    for x in (x for x in r["relations"] if x["kind"] == "pattern"):
        assert any(M.mentions(e["text"], cmap[x["source"]]) and M.mentions(e["text"], cmap[x["target"]]) for e in x["evidence"])
    defined = next(c for c in r["concepts"] if c["name"].lower() == "overfitting")
    assert defined["definition"] and "memorizes" in defined["definition"]


# ---------------------------------------------------------------- technical-term coverage in the notes
def test_every_term_the_lecturer_defines_gets_its_definition_in_the_notes():
    for sp, g, text in E.load("dev") and [("dev", g, t) for g, t in E.load("dev")]:
        for vname, vtext in E.variants(g, text):
            r = process(vtext)
            defined = {c["evidence"][0]["sentence"]: c["name"] for c in r["concepts"] if c["definition"]}
            shown = {i for n in r["notes"] for b in n["bullets"] for i in b["sentences"]}
            missing = [name for i, name in defined.items() if i not in shown]
            assert not missing, (g["name"], vname, missing)


def test_parallel_definitions_are_not_crowded_out_by_the_bullet_budget():
    text = ("A catalyst is a substance that speeds up a reaction. A catalyst lowers the activation energy of a reaction. "
            "Reaction rate depends on temperature. Reaction rate increases with concentration. Temperature changes the reaction rate. "
            "Concentration changes the reaction rate. The reaction rate matters for every catalyst. "
            "An inhibitor is a substance that slows down a reaction. An enzyme is a protein that speeds up a reaction. "
            "A buffer is a solution that resists changes in acidity. A solvent is a liquid that dissolves a solute. "
            "Reaction rate depends on the catalyst and the temperature.")
    r = process(text)
    shown = " ".join(b["text"].lower() for n in r["notes"] for b in n["bullets"])
    for term in ("inhibitor", "enzyme", "buffer", "solvent"):
        assert term in shown, term
