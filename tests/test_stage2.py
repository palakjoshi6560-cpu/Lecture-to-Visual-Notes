"""Stage 2 tests: general mechanisms checked on toy texts from unrelated subjects (no lecture-specific content)."""
from lecturenotes import nlp
from lecturenotes.pipeline import process
from lecturenotes.relations_parse import parse_relations


def names(text, **kw):
    return {c["name"].lower() for c in process(text, **kw)["concepts"]}


def pad(text):  # the pipeline needs a few sentences; this filler adds no concepts
    return text + " The notes continue here. Another sentence follows now. One more remark ends it."


def test_compound_terms_with_common_head_nouns_stay_whole():
    n = names(pad("A public key encrypts a message. The private key decrypts the message. Public key cryptography needs both a public key "
                  "and a private key. Each sender keeps the private key secret."))
    assert {"public key", "private key"} <= n and "public" not in n


def test_enumerated_ordinals_are_kept_but_stray_ones_are_trimmed():
    n = names(pad("The first law of motion describes inertia. The second law of motion relates force to acceleration. "
                  "The first law of motion is also called the law of inertia. The second law of motion uses mass."))
    assert {"first law", "second law"} <= n
    S = nlp.analyze("A common activation function helps training. The activation function shapes the output. "
                    "Each activation function is differentiable. Another sentence closes the lecture here.")
    assert "activation function" in {m.key for s in S for m in s.mentions} and "common activation function" not in {m.key for s in S for m in s.mentions}


def test_numeral_names_recurring_in_the_text_are_kept():
    assert "two factor authentication" in names(pad("Two factor authentication adds a second check. Two factor authentication reduces fraud. "
                                                    "Banks require two factor authentication for payments."))


def test_verbs_do_not_leak_into_terms_in_any_domain():
    S = nlp.analyze("Prices shifted after the war. Inflation shifted consumer behavior. The central bank can raise rates. "
                    "A central bank describes policy goals. The central bank follows inflation data.")
    keys = {m.key for s in S for m in s.mentions}
    assert "central bank" in keys and not any(w in k for k in keys for w in ("shifted", "raise", "describes", "follows"))


def test_verbs_are_induced_from_context_without_a_subject_lexicon():
    texts = ["The sensor can calibrate itself.", "The sensor will calibrate the probe.", "A probe measures heat.",
             "The calibration process takes time.", "We can process the signal.", "The process ends."]
    verbs = nlp.induce_verbs(texts)
    assert "calibrate" in verbs and "calibrates" in verbs and "process" not in verbs  # 'process' also follows a determiner: noun evidence


def test_open_vocabulary_relations_have_direction_label_and_evidence():
    text = ("The central bank raises interest rates. The central bank watches inflation closely. Interest rates affect inflation. "
            "Higher interest rates reduce inflation. Inflation hurts consumers. The central bank sets interest rates every month.")
    r = process(text)
    rels = {(x["source"], x["label"], x["target"]) for x in r["relations"] if x["kind"] == "pattern"}
    assert ("central bank", "raises", "interest rate") in rels
    assert ("interest rate", "controls", "inflation") in rels
    ev = next(x for x in r["relations"] if (x["source"], x["target"]) == ("central bank", "interest rate"))["evidence"]
    assert any("raises interest rates" in e["text"] for e in ev)


def test_relative_clause_after_a_prepositional_phrase_is_not_attached_to_the_wrong_noun():
    text = ("A gene is a segment in one chromosome that encodes a protein. A chromosome carries many genes. "
            "Each protein folds into a shape. A gene determines the protein. The chromosome is copied before division.")
    r = process(text)
    assert not any(x["source"] == "chromosome" and x["target"] == "protein" and x["kind"] == "pattern" for x in r["relations"])


def test_terms_the_lecture_defines_survive_concept_pruning():
    text = ("A photon is a quantum of light. Waves carry energy. Waves have a frequency. The frequency sets the energy. "
            "Energy moves through waves. Frequency and energy matter. Waves bend near edges. Edges split the waves.")
    assert "photon" in names(text, max_concepts=4)


def test_metadata_header_and_direct_address_are_excluded_but_preserved():
    text = ("LECTURE NOTES WEEK THREE\nCourse: Macroeconomics\nInstructor: Dr Rao\n\n"
            "Inflation measures how fast prices rise. Inflation reduces what money can buy. Wages respond to inflation slowly. "
            "You should check what you learned about wages. Prices matter to every household.")
    r = process(text)
    flag = {s["text"]: s["excluded"] for s in r["transcript"]}
    assert flag["LECTURE NOTES WEEK THREE"] == "metadata" and flag["Course: Macroeconomics"] == "metadata"
    assert flag["You should check what you learned about wages."] == "instructional"
    assert r["raw_transcript"] == text and "course" not in {c["name"].lower() for c in r["concepts"]}


# ---- optional parse-based relations: logic verified with stub tokens only (real spaCy not available in the dev sandbox)
class Tok:
    def __init__(self, idx, text, pos="NOUN", dep="", lemma=None):
        self.idx, self.text, self.pos_, self.dep_, self.lemma_ = idx, text, pos, dep, lemma or text.lower()
        self.children, self.conjuncts = [], []


def _sent(doc, mentions):
    s = nlp.Sentence(0, "x")
    s.doc = doc
    s.resolved = [(k, nlp.Mention(k, k, a, b)) for k, a, b in mentions]
    return s


def test_parse_relations_active_passive_and_negation_with_stub_tokens():
    # "Mitochondria produce ATP"
    sub, verb, obj = Tok(0, "Mitochondria", dep="nsubj"), Tok(13, "produce", "VERB", "ROOT", "produce"), Tok(21, "ATP", dep="dobj")
    verb.children = [sub, obj]
    r = parse_relations([_sent([sub, verb, obj], [("mitochondrion", 0, 12), ("atp", 21, 24)])], {"mitochondrion": 1, "atp": 1})
    assert [(x["source"], x["label"], x["target"]) for x in r] == [("mitochondrion", "produce", "atp")]
    # "ATP is produced by mitochondria" -> direction is flipped to the actor
    s2, aux, v2, by, ag = Tok(0, "ATP", dep="nsubjpass"), Tok(4, "is", "AUX", "auxpass", "be"), Tok(7, "produced", "VERB", "ROOT", "produce"), Tok(16, "by", "ADP", "agent"), Tok(19, "mitochondria", dep="pobj")
    by.children, v2.children = [ag], [s2, aux, by]
    r2 = parse_relations([_sent([s2, aux, v2, by, ag], [("atp", 0, 3), ("mitochondrion", 19, 31)])], {"mitochondrion": 1, "atp": 1})
    assert [(x["source"], x["label"], x["target"]) for x in r2] == [("mitochondrion", "produce", "atp")]
    # negation is skipped
    neg = Tok(8, "not", "PART", "neg")
    verb.children = [sub, neg, obj]
    assert parse_relations([_sent([sub, neg, verb, obj], [("mitochondrion", 0, 12), ("atp", 21, 24)])], {"mitochondrion": 1, "atp": 1}) == []
