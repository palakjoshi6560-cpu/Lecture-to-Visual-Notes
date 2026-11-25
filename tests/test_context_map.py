from lecturenotes.pipeline import process


def test_requires_relation_uses_canonical_label_and_direction():
    text = """A first normal form table stores each field as one value.
    A second normal form table requires first normal form.
    Third normal form requires second normal form and removes transitive dependencies.
    A candidate key can be selected as the primary key.
    A foreign key refers to the primary key of another relation."""
    result = process(text)
    triples = {(r["source"], r["label"], r["target"]) for r in result["relations"] if r["kind"] == "pattern"}
    assert ("third normal form", "requires", "second normal form") in triples
    assert ("foreign key", "refers to", "primary key") in triples
    assert not any(label == "means" for _, label, _ in triples)


def test_adjacent_introduced_list_is_connected_with_two_sentence_evidence():
    text = """A database stores related records.
    A transaction is a logical unit of database work.
    Reliable transactions follow four important properties.
    Atomicity, consistency, isolation, and durability.
    Atomicity means all operations in the transaction happen or none do.
    Consistency preserves database rules and constraints.
    Isolation controls how concurrent transactions affect one another.
    Durability means committed changes persist after failure."""
    result = process(text)
    context = [r for r in result["relations"] if r["kind"] == "context"]
    for term in ("atomicity", "consistency", "isolation", "durability"):
        rel = next(r for r in context if r["target"] == term)
        assert rel["label"] in {"includes", "has property"}
        assert len(rel["evidence"]) == 2
        assert "properties" in rel["evidence"][0]["text"].lower()
        assert term in rel["evidence"][1]["text"].lower()
    graph_ids = {n["id"] for n in result["graph"]["nodes"]}
    assert {"atomicity", "consistency", "isolation", "durability"} <= graph_ids
    assert all(e["kind"] in {"pattern", "context"} for e in result["graph"]["edges"])


def test_definition_body_does_not_create_a_false_means_edge():
    text = """A database stores related records.
    Consistency means the transaction preserves database rules and constraints.
    Atomicity means all operations in the transaction happen or none do.
    Isolation controls concurrent transactions.
    Durability means committed changes persist after failure."""
    result = process(text)
    triples = {(r["source"], r["label"], r["target"]) for r in result["relations"]}
    assert not any(label == "means" for _, label, _ in triples)
    assert ("consistency", "means", "transaction") not in triples


def test_single_mention_prerequisite_pair_is_kept_together():
    text = """A candidate key uniquely identifies a record in a relation.
    Normalization reduces redundant values in a data model.
    Third normal form requires second normal form and removes transitive dependencies.
    A relation stores attributes in rows and columns.
    Foreign keys connect related tables in a data model."""
    result = process(text)
    terms = {c["id"] for c in result["concepts"]}
    assert {"third normal form", "second normal form"} <= terms
    triples = {(r["source"], r["label"], r["target"]) for r in result["relations"] if r["kind"] == "pattern"}
    assert ("third normal form", "requires", "second normal form") in triples



def test_graph_keeps_a_high_signal_one_off_technical_concept():
    from lecturenotes.graph import select_graph

    concepts = [
        {"id": "cell", "name": "cell", "importance": 0.72, "topic": 0, "mentions": 3, "definition": None},
        {"id": "mitochondrion", "name": "mitochondrion", "importance": 0.68, "topic": 0, "mentions": 1, "definition": None},
    ]
    relations = [{"source": "mitochondrion", "target": "cell", "label": "part of", "kind": "pattern", "weight": 1.0, "evidence": [{"sentence": 0, "text": "Mitochondrion is part of cell."}]}]
    graph = select_graph(concepts, relations, max_nodes=4)
    assert {node["id"] for node in graph["nodes"]} == {"cell", "mitochondrion"}
