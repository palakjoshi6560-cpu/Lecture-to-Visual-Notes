from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from lecturenotes import nlp
from lecturenotes.pipeline import process
from lecturenotes.topics import segment_topics

SAMPLE = (Path(__file__).parent.parent / "samples" / "sample_lecture.txt").read_text(encoding="utf-8")
R = process(SAMPLE)
NAMES = {c["name"].lower() for c in R["concepts"]}


# ---- nlp -------------------------------------------------------------
def test_sentence_split_keeps_abbreviations_and_removes_fillers():
    out = nlp.split_sentences("Okay, um, loss is a number, e.g. 0.3 here. Next we update the weights.")
    assert len(out) == 2 and "e.g. 0.3" in out[0][0] and "um" not in out[0][0].lower().split()


def test_paragraphs_are_tracked():
    s = nlp.analyze("First idea is simple enough. Second idea follows it.\n\nNew paragraph starts right here. And it goes on.")
    assert [x.para for x in s] == [0, 0, 1, 1]


def test_concept_key_singularises_last_word():
    assert nlp.concept_key(["Neural", "Networks"]) == "neural network"
    assert nlp.concept_key(["loss"]) == "loss"
    assert nlp.concept_key(["non", "-", "key", "fields"]) == "non key field"


# ---- concepts, relations, scoring ------------------------------------
def test_key_concepts_found():
    for name in ["neural network", "gradient descent", "backpropagation", "overfitting", "learning rate", "loss function"]:
        assert name in NAMES, name


def test_acronym_is_merged_into_full_term():
    assert any("SGD" in c["aliases"] and c["name"].lower() == "stochastic gradient descent" for c in R["concepts"])
    assert "sgd" not in NAMES


def test_definition_extracted():
    gd = next(c for c in R["concepts"] if c["name"] == "gradient descent")
    assert "optimization algorithm" in gd["definition"]


def test_relationships_have_evidence():
    triples = {(r["source"], r["label"], r["target"]) for r in R["relations"]}
    assert ("backpropagation", "computes", "gradient") in triples
    assert ("regularization", "reduces", "overfitting") in triples
    for r in R["relations"]:
        assert r["evidence"] and all(e["text"] for e in r["evidence"])


def test_importance_scores_valid_and_sorted():
    scores = [c["importance"] for c in R["concepts"]]
    assert all(0 <= s <= 1 for s in scores) and scores == sorted(scores, reverse=True)
    assert R["concepts"][0]["name"].lower() in {"neural network", "backpropagation", "gradient descent"}


def test_concept_evidence_mentions_concept():
    for c in R["concepts"]:
        assert c["evidence"]
        for e in c["evidence"]:
            low = e["text"].lower()
            assert any(w[:4] in low for w in c["name"].lower().split()) or any(a.lower() in low for a in c["aliases"])


# ---- topics, notes, quick revision -----------------------------------
def test_topic_segmentation_splits_distinct_blocks():
    a = " ".join(f"Photosynthesis converts sunlight into chemical energy in chloroplasts, step {i}." for i in range(6))
    b = " ".join(f"Medieval castles used thick stone walls and moats for defence, siege {i}." for i in range(6))
    sents = nlp.analyze(a + " " + b)
    topics = segment_topics(sents)
    assert len(set(topics)) >= 2 and topics == sorted(topics)


def test_notes_are_topic_wise_concise_and_non_repetitive():
    assert 3 <= len(R["notes"]) <= 6
    bullets = [b["text"] for n in R["notes"] for b in n["bullets"]]
    assert len(bullets) == len(set(bullets))
    for n in R["notes"]:
        assert 1 <= len(n["bullets"]) <= 9
    X = TfidfVectorizer(stop_words="english").fit_transform(bullets)
    sim = (X @ X.T).toarray()
    for i in range(len(bullets)):
        for j in range(i + 1, len(bullets)):
            assert sim[i, j] < 0.7, (bullets[i], bullets[j])


def test_quick_revision_is_short_and_unique():
    q = R["quick_revision"]
    assert 3 <= len(q["terms"]) <= 14 and len(q["relationships"]) <= 8
    assert len({t["term"] for t in q["terms"]}) == len(q["terms"])
    assert all(len(t["definition"].split()) <= 30 for t in q["terms"])


def test_distinct_term_definitions_and_prerequisite_relations_are_preserved():
    text = """A table is in first normal form when every field contains one value.
    A table is in second normal form when it is in first normal form and each non-key field depends on the whole key.
    Third normal form removes transitive dependencies from a relation.
    Atomicity means all operations succeed together or none do.
    Durability means committed changes remain saved after a failure.
    Isolation means concurrent transactions do not expose partial changes.
    A candidate key can be selected as the primary key.
    A foreign key refers to the primary key of another relation.
    These properties support reliable data updates and recovery."""
    result = process(text)
    concepts = {c["id"]: c for c in result["concepts"]}
    assert concepts["first normal form"]["definition"]
    assert concepts["second normal form"]["definition"]
    assert concepts["third normal form"]["definition"]
    quick_terms = {x["term"] for x in result["quick_revision"]["terms"]}
    assert {"first normal form", "second normal form", "third normal form"} <= quick_terms
    edges = {(r["source"], r["label"], r["target"]) for r in result["graph"]["edges"]}
    assert ("second normal form", "requires", "first normal form") in edges
    assert not any(label == "means" for _, label, _ in edges)
    relations = {(r["source"], r["label"], r["target"]) for r in result["relations"]}
    assert ("foreign key", "refers to", "primary key") in relations
    assert any(source == "candidate key" and label == "selected as" and target == "primary key"
               for source, label, target in relations)


def test_explicit_refers_to_relation_survives_prepositional_context_guard():
    from lecturenotes.extract import extract_relations
    from lecturenotes.nlp import Mention, Sentence

    text = "Within database systems foreign key refers to primary key."
    source_start = text.index("foreign key")
    target_start = text.index("primary key")
    sent = Sentence(0, text)
    sent.resolved = [
        ("foreign key", Mention("foreign key", "foreign key", source_start, source_start + 11, False)),
        ("primary key", Mention("primary key", "primary key", target_start, target_start + 11, False)),
    ]
    concepts = {"foreign key": object(), "primary key": object()}
    relations = extract_relations([sent], concepts)
    triples = {(r["source"], r["label"], r["target"]) for r in relations}
    assert ("foreign key", "refers to", "primary key") in triples



def test_refers_to_relation_is_recovered_from_raw_text_when_parser_boundaries_differ():
    from lecturenotes.extract import build_concepts, extract_relations
    from lecturenotes import nlp

    sent_list = nlp.analyze("A foreign key refers to the primary key of another relation.")
    candidates = build_concepts(sent_list, min_count=1)
    assert "foreign key" in candidates and "primary key" in candidates

    # Simulate a parser model that does not return these two terms as adjacent
    # noun chunks. The sentence text is still authoritative evidence.
    for sent in sent_list:
        sent.resolved = [(key, mention) for key, mention in sent.resolved
                         if key not in {"foreign key", "primary key"}]

    relations = extract_relations(sent_list, candidates)
    triples = {(r["source"], r["label"], r["target"]) for r in relations}
    assert ("foreign key", "refers to", "primary key") in triples

def test_audio_segments_carry_timestamps():
    segs = [dict(text=t, start=i * 5.0, end=i * 5 + 5.0) for i, t in enumerate(
        ["Gradient descent is an optimization algorithm that minimizes the loss function.",
         "The learning rate controls the step size of gradient descent.",
         "A large learning rate makes the loss function diverge.",
         "Gradient descent updates every weight using the loss function gradient."])]
    r = process(segments=segs, source="audio")
    assert r["meta"]["source"] == "audio"
    assert all(t["time"] is not None for t in r["transcript"])
    assert r["transcript"][1]["time"] == "00:05"


def test_audio_segments_without_punctuation_are_grouped_into_sentence_chunks():
    phrases = [
        "Gradient descent is an optimization algorithm that minimizes the loss function",
        "The learning rate controls the step size of gradient descent",
        "A large learning rate makes the loss function diverge",
        "Gradient descent updates every weight using the loss function gradient",
        "Backpropagation computes gradients by applying the chain rule",
        "The optimizer uses these gradients to adjust model parameters",
        "Regularization reduces overfitting by penalizing large weights",
        "The validation set measures performance on unseen examples",
    ]
    segs = [dict(text=value, start=i * 4.0, end=i * 4.0 + 3.5) for i, value in enumerate(phrases)]
    sents = nlp.analyze(segments=segs)
    assert len(sents) >= 4
    assert all(s.time is not None for s in sents)
    assert sents[0].time == 0.0


def test_too_short_input_rejected():
    try:
        process(text="Only one sentence here.")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_concept_pruning_does_not_evict_an_already_selected_recurring_concept():
    from types import SimpleNamespace
    from lecturenotes.pipeline import _select_concept_keys

    specs = {
        "top": (10.0, 2, 2),
        "filler_a": (9.0, 1, 1),
        "filler_b": (8.0, 1, 1),
        "filler_c": (7.0, 1, 1),
        "filler_d": (6.0, 1, 1),
        "filler_e": (5.0, 1, 1),
        "filler_f": (4.0, 1, 1),
        "already_selected_recurring": (3.0, 5, 4),
        "higher_recurrence_candidate": (2.0, 9, 8),
    }
    concepts = {
        key: SimpleNamespace(importance=score, definition=None, count=count, sentences=list(range(n_sentences)))
        for key, (score, count, n_sentences) in specs.items()
    }
    selected = _select_concept_keys(concepts, max_concepts=8)

    assert len(selected) == 8
    assert "already_selected_recurring" in selected
    assert "higher_recurrence_candidate" in selected


def test_spacy_noun_chunk_drops_mistagged_terminal_relation_verb():
    from types import SimpleNamespace
    from lecturenotes.nlp import _spacy_mentions

    text = "A foreign key refers to the primary key of another relation."
    start = text.index("foreign key")
    class FakeToken:
        def __init__(self, text, lower_, idx, pos_):
            self.text, self.lower_, self.idx, self.pos_ = text, lower_, idx, pos_
            self.is_stop = False

        def __len__(self):
            return len(self.text)

    tokens = [
        FakeToken("foreign", "foreign", start, "ADJ"),
        FakeToken("key", "key", start + 8, "NOUN"),
        # Simulate a parser/model version that incorrectly tags "refers" as NOUN.
        FakeToken("refers", "refers", start + 12, "NOUN"),
    ]
    doc = SimpleNamespace(noun_chunks=[tokens])
    mentions = _spacy_mentions(doc, text)
    assert [(m.key, m.surface) for m in mentions] == [("foreign key", "foreign key")]



def test_quick_revision_prioritizes_explicit_acid_definitions():
    text = """A database stores related records in tables.
    A transaction has four important properties.
    Atomicity means all operations happen or none do.
    Consistency preserves database rules and constraints.
    Isolation controls how concurrent transactions affect one another.
    Durability means committed changes remain saved after a failure.
    A table contains rows and columns.
    Each row represents one record and each column represents an attribute.
    A primary key uniquely identifies a record.
    A foreign key refers to a primary key in another table."""
    result = process(text)
    terms = {item["term"].lower() for item in result["quick_revision"]["terms"]}
    assert {"atomicity", "consistency", "isolation", "durability"} <= terms


def test_specific_phrases_rank_above_structural_single_word_terms():
    text = """A table contains rows and columns.
    Each row contains a record and each column contains a field.
    A key identifies a row in a table.
    A foreign key refers to a primary key in another table.
    A candidate key can be selected as the primary key.
    A composite key uses two or more columns together.
    Normalization reduces redundant data in a relation.
    Transaction properties protect reliable updates."""
    result = process(text)
    score = {item["id"]: item["importance"] for item in result["concepts"]}
    assert score["foreign key"] > score["row"]
    assert score["composite key"] > score["column"]


def test_coordinated_columns_are_both_components_of_a_composite_key():
    text = """A student ID and course ID together may form a composite key.
    A composite key uses two or more columns together.
    A candidate key uniquely identifies a record.
    A foreign key refers to a primary key in another table."""
    result = process(text)
    triples = {(item["source"], item["label"], item["target"]) for item in result["relations"]}
    assert ("student id", "forms part of", "composite key") in triples
    assert ("course id", "forms part of", "composite key") in triples
    quick = {(item["source"], item["label"], item["target"]) for item in result["quick_revision"]["relationships"]}
    assert ("student ID", "forms part of", "composite key") in quick
    assert ("course ID", "forms part of", "composite key") in quick
    assert not any(source == "course id" and target == "composite key" and label == "form"
                   for source, label, target in triples)


def test_generic_assessment_word_does_not_become_a_map_concept():
    text = """A database is an organized collection of related data.
    An examination leads to the database.
    A transaction is a logical unit of database work.
    Atomicity means all operations happen or none do.
    Durability means committed changes remain saved after a failure."""
    result = process(text)
    concepts = {item["id"] for item in result["concepts"]}
    graph_ids = {item["id"] for item in result["graph"]["nodes"]}
    assert "examination" not in concepts
    assert "examination" not in graph_ids


def test_shortlist_preserves_recurring_defined_single_word_concepts():
    from types import SimpleNamespace
    from lecturenotes.extract import Concept
    from lecturenotes.pipeline import _select_concept_keys

    concepts = {
        "long phrase alpha": Concept("long phrase alpha", "Long Phrase Alpha", 1, [0], definition="an explicitly stated definition", importance=0.99),
        "long phrase beta": Concept("long phrase beta", "Long Phrase Beta", 1, [1], definition="another explicitly stated definition", importance=0.98),
        "semaphore": Concept("semaphore", "semaphore", 2, [2, 3], definition="a synchronization primitive", importance=0.25),
        "other phrase": Concept("other phrase", "Other Phrase", 1, [4], definition=None, importance=0.5),
    }
    kept = _select_concept_keys(concepts, max_concepts=4)
    assert "semaphore" in kept


def test_quick_revision_deduplicates_repeated_definitions_but_keeps_distinct_relations():
    from types import SimpleNamespace
    from lecturenotes.extract import Concept
    from lecturenotes.notes import build_quick_revision

    definition = "refers to a complete system of related information stored and managed for reliable use"
    concepts = {
        "concept alpha": Concept("concept alpha", "Concept Alpha", 1, [0], definition=definition, definition_sentence=0, importance=0.9),
        "concept beta": Concept("concept beta", "Concept Beta", 1, [1], definition=definition, definition_sentence=1, importance=0.8),
    }
    sents = [
        SimpleNamespace(idx=0, text=f"Concept Alpha {definition}.", excluded=None, kept=[], heading=None, time=None),
        SimpleNamespace(idx=1, text=f"Concept Beta {definition}.", excluded=None, kept=[], heading=None, time=None),
    ]
    result = build_quick_revision(sents, concepts, [], lambda _s, _c: 1.0, set(), n_terms=8, n_rel=4)
    assert len(result["terms"]) == 1
    assert not result["relationships"]


def test_automatic_topic_title_avoids_generic_single_word_concept():
    text = """A primary key uniquely identifies each record in a table.
    A primary key must contain a unique value for every row.
    The primary key connects each row with related records.
    A key is important for identifying a record.
    A key can be used to find a row quickly.
    A table stores records and each record contains attributes.
    The table structure helps organize related data.
    Records are stored in the table for later retrieval."""
    result = process(text)
    titles = [topic["title"].lower() for topic in result["topics"]]
    assert "key" not in titles


def test_os_lecture_signposts_produce_coherent_major_topics_and_coverage():
    fixture = Path(__file__).parent / "fixtures" / "os_lecture.txt"
    result = process(fixture.read_text(encoding="utf-8"))
    titles = [topic["title"].lower() for topic in result["topics"]]
    assert titles == ["operating system", "process", "cpu scheduling", "memory management"]
    assert [topic["first_sentence"] for topic in result["topics"]] == [0, 11, 24, 33]
    bullet_text = " ".join(b["text"].lower() for note in result["notes"] for b in note["bullets"])
    for required in ["process control block", "ready process", "context switch", "round robin", "time quantum", "paging", "page fault"]:
        assert required in bullet_text, required


def test_dbms_lecture_keeps_subject_sections_and_core_content():
    fixture = Path(__file__).parent / "fixtures" / "dbms_lecture.txt"
    result = process(fixture.read_text(encoding="utf-8"))
    assert 4 <= result["meta"]["topics"] <= 6
    bullet_text = " ".join(b["text"].lower() for note in result["notes"] for b in note["bullets"])
    for required in ["database", "foreign key", "composite key", "normalization", "sql", "transaction", "atomicity"]:
        assert required in bullet_text, required
