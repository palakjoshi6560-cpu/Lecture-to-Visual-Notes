from lecturenotes import nlp
from lecturenotes.pipeline import process
from lecturenotes.topics import segment_topics


def test_missing_asr_punctuation_recovers_clear_clause_boundaries():
    text = (
        "A student table contains student ID, name, email address, and department ID "
        "The table schema describes its columns and the kinds of values they can store. "
        "This connection helps maintain referential integrity"
        " A candidate key is a minimal set of attributes that can uniquely identify a record. "
        "If only one operation succeeds, the records become inconsistent A transaction groups these operations together."
    )
    pieces = [item[0] for item in nlp.split_sentences(text)]
    assert any(x.startswith("A student table contains") for x in pieces)
    assert any(x.startswith("The table schema describes") for x in pieces)
    assert any(x.startswith("This connection helps") for x in pieces)
    assert any(x.startswith("A transaction groups") for x in pieces)
    # Coordinated attributes must not be split into separate fragments.
    assert any("student ID, name, email address, and department ID" in x for x in pieces)


def test_normal_form_is_kept_as_a_complete_technical_phrase():
    sents = nlp.analyze(
        "First normal form requires each field to contain a single value. "
        "Second normal form requires first normal form and removes partial dependencies. "
        "Third normal form requires second normal form and removes transitive dependencies."
    )
    found = {m.key for sent in sents for m in sent.mentions}
    assert {"first normal form", "second normal form", "third normal form"} <= found


def test_dbms_topics_keep_keys_normalization_sql_and_transaction_properties_together():
    text = """We will cover databases, relational tables, keys, normalization, structured query language and transactions.
    A database is an organized collection of related data. A database management system is software used to create, read, update, and manage data.
    In a relational database, information is arranged in tables. A table contains rows and columns. Each row represents one record and each column represents an attribute.
    A student table might contain student ID, name, email address, and department ID. The table schema describes its columns and the kinds of values they can store.
    Keys help identify records and connect tables. A primary key uniquely identifies each row. A foreign key refers to a key in another table, creating a relationship between the two tables.
    A candidate key is a minimal set of attributes that can uniquely identify a record. One candidate key is chosen as the primary key. Other candidate keys are called alternate keys. A composite key uses two or more columns together.
    In an enrollment table, student ID and course ID may form a composite key. Next, let us discuss normalization.
    Normalization organizes tables to reduce unnecessary duplication and prevent insertion, update, and deletion anomalies. First normal form requires each field to contain a single value rather than a repeating list.
    Second normal form requires first normal form and removes partial dependencies on part of a composite candidate key. Third normal form requires second normal form and removes transitive dependencies of non-key attributes.
    For instance, instead of repeating a department name for every student, store departments in a separate department table and reference it using department ID. This reduces duplicated data and makes updates more consistent.
    When related information is needed from separate tables, SQL joins can combine the matching rows. Structured query language, or SQL is used to query and modify relational data.
    The select statement retrieves rows, insert adds a record, update changes existing values, and delete removes records. A where clause filters rows according to a condition, while ORDER BY sorts the results.
    An inner join returns rows with matching values in both tables. A left join returns every row from the left table and includes matching information from the right table when available. Joins commonly use primary key and foreign key relationships.
    A transaction is a logical unit of database work. Imagine transferring money between two bank accounts. One operation subtracts money from the sender, and another adds it to the receiver.
    If only one operation succeeds, the records become inconsistent. A transaction groups these operations so they can be handled together. Reliable transactions follow four important properties: atomicity, consistency, isolation, and durability.
    Atomicity means all operations in the transaction happen, or none of them do. Consistency means the transaction preserves database rules and constraints.
    Isolation controls how concurrent transactions affect one another. Durability means that after a transaction commits, its changes persist according to the database's recovery guarantees.
    Commit confirms a transaction while rollback cancels uncommitted changes. Database systems also use locks and other concurrency control techniques to coordinate simultaneous access and protect shared data.
    To summarize, a database stores organized information. A database management system controls access to it. Relational tables organize records, keys connect related tables.
    Normalization reduces redundancy, SQL queries and changes data, and transactions protect reliable updates. These concepts work together. Good table design makes data consistent.
    Keys establish relationships, joins retrieve related information, and transaction properties help preserve correctness when failures or concurrent operations occur."""
    result = process(text)
    titles = [topic["title"].lower() for topic in result["topics"]]
    assert "key types" in titles
    assert "normalization" in titles
    assert "structured query language" in titles
    assert "transaction" in titles
    assert "summary" in titles
    assert "primary key" not in titles
    assert "isolation" not in titles
    assert "third normal form" in {c["id"] for c in result["concepts"]}
    assert any("third normal form" in b["text"].lower() for n in result["notes"] for b in n["bullets"])
    transaction_note = next(n for n in result["notes"] if n["title"] == "Transaction")
    transaction_bullets = [b["text"].lower() for b in transaction_note["bullets"]]
    assert any("imagine transferring money" in b for b in transaction_bullets)
    assert any("subtracts money" in b for b in transaction_bullets)
    context_edges = {(e["source"], e["label"], e["target"]) for e in result["relations"] if e["kind"] == "context"}
    for property_name in {"atomicity", "consistency", "isolation", "durability"}:
        assert ("transaction", "has property", property_name) in context_edges
    graph_edges = {(e["source"], e["label"], e["target"]) for e in result["graph"]["edges"]}
    assert all(("transaction", "has property", name) in graph_edges
               for name in {"atomicity", "consistency", "isolation", "durability"})
    relation_edges = {(r["source"], r["label"], r["target"]) for r in result["relations"]}
    assert ("student id", "forms part of", "composite key") in relation_edges
    assert ("course id", "forms part of", "composite key") in relation_edges
    assert ("enrollment table", "forms part of", "composite key") not in relation_edges
    assert ("candidate key", "selected as", "primary key") in {
        (edge["source"], edge["label"], edge["target"]) for edge in result["graph"]["edges"]
    }
    graph_nodes = {node["id"] for node in result["graph"]["nodes"]}
    assert {"first normal form", "second normal form", "third normal form"} <= graph_nodes
    assert any(node["topic"] == next(t["id"] for t in result["topics"]
                                      if t["title"].lower() == "structured query language")
               for node in result["graph"]["nodes"])
    revision_terms = {item["term"].lower() for item in result["quick_revision"]["terms"]}
    assert "transaction" in revision_terms
    assert "structured query language" in revision_terms
    revision_relations = {item["text"].lower() for item in result["quick_revision"]["relationships"]}
    assert all(f"transaction has property {name}" in revision_relations
               for name in {"atomicity", "consistency", "isolation", "durability"})


def test_dbms_context_pack_corrects_observed_asr_variants_without_global_rewrites():
    from lecturenotes.terminology import correct_segments

    segments = [
        {"start": 0.0, "end": 4.0,
         "text": "A database management system is software. mammals include mySQL, postcurSQL, and Oracle database."},
        {"start": 4.0, "end": 8.0,
         "text": "Godtable design is important. Unkey, for instance, store departments and Reference M using department ID."},
        {"start": 8.0, "end": 12.0,
         "text": "The select statement retrieves rows, insert ads or record, update changes values."},
    ]
    corrected, audit = correct_segments(segments)
    joined = " ".join(item["text"] for item in corrected)
    assert "DBMS includes MySQL, PostgreSQL" in joined
    assert "Good table design" in joined
    assert "for instance" in joined and "Unkey" not in joined
    assert "reference it using department ID" in joined
    assert "insert adds a record" in joined
    assert len(audit) >= 6

    unrelated, unrelated_audit = correct_segments([
        {"start": 0.0, "end": 2.0, "text": "Mammals include several species."}
    ])
    assert unrelated[0]["text"] == "Mammals include several species."
    assert unrelated_audit == []


def test_dbms_audio_segment_path_keeps_topics_and_corrects_context_terms():
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "dbms_lecture.txt"
    lines = [line.strip() for line in fixture.read_text(encoding="utf-8").splitlines() if line.strip()]
    segments = [
        {"start": i * 4.0, "end": (i + 1) * 4.0, "text": line}
        for i, line in enumerate(lines)
    ]
    result = process(segments=segments, source="audio")
    transcript_text = " ".join(item["text"] for item in result["transcript"])
    assert "DBMS includes MySQL, PostgreSQL" in transcript_text
    assert "Good table design" in transcript_text
    assert "reference it using department ID" in transcript_text
    titles = [topic["title"].lower() for topic in result["topics"]]
    assert titles == ["database management system", "key types", "normalization",
                      "structured query language", "transaction", "summary"]
    assert all(item["time"] is not None for item in result["transcript"])
    graph_edges = {(edge["source"], edge["label"], edge["target"]) for edge in result["graph"]["edges"]}
    assert all(("transaction", "has property", name) in graph_edges
               for name in {"atomicity", "consistency", "isolation", "durability"})


def test_lowercase_discourse_markers_start_new_note_sentences():
    pieces = [item[0] for item in nlp.split_sentences(
        "Third normal form removes transitive dependencies of non-key attributes for instance "
        "instead of repeating a department name, store departments separately."
    )]
    assert len(pieces) == 2
    assert pieces[1].lower().startswith("for instance")


def test_definition_heading_can_be_found_after_intro_sentences():
    from lecturenotes.notes import TOPIC_DEFINITION_TITLE

    text = "A database is an organized collection of related data. " \
           "A database management system is software used to create, read, update, and manage data."
    match = TOPIC_DEFINITION_TITLE.match(text.split(". ", 1)[1])
    assert match is not None
    assert match.group("term").lower() == "database management system"


def test_asr_capitalization_in_example_continuation_is_repaired():
    from lecturenotes.notes import compress

    text = compress(
        "For instance, instead of repeating a department name for every student "
        "Store departments in a separate department table and reference it using department ID."
    )
    assert "student, store departments" in text.lower()


def test_dbms_audio_cleanup_preserves_boundaries_and_canonical_terms():
    from lecturenotes.terminology import correct_segments
    from lecturenotes.nlp import split_sentences
    from lecturenotes.notes import compress

    segments = [{"start": 0.0, "end": 20.0, "text": (
        "A database management system is software. animation results are stored. "
        "Third normal form removes transitive dependencies of an key attribute on a key. "
        "Isolation controls how concurrent transactions affect one another Durability means "
        "committed changes persist after failure."
    )}]
    corrected, _ = correct_segments(segments)
    text = corrected[0]["text"]
    assert "examination results" in text
    assert "non-key attributes" in text
    pieces = [x[0] for x in split_sentences(text)]
    assert any(x.startswith("Durability means") for x in pieces)
    assert not any("Isolation controls" in x and "Durability means" in x for x in pieces)
    assert compress("A first sentence. And a second sentence.") == "A first sentence. And a second sentence."


def test_leading_determiner_does_not_change_canonical_concept_id():
    from lecturenotes.nlp import concept_key
    assert concept_key(["one", "candidate", "key"]) == "candidate key"
