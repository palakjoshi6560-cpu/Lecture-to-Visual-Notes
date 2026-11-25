from pathlib import Path

from lecturenotes.extract import extract_relations
from lecturenotes.nlp import Mention, Sentence, split_sentences


def test_example_lists_are_not_mislabelled_as_containment():
    text = "Examples of operating systems include Windows and Linux."
    sentence = Sentence(0, text)
    sentence.resolved = [
        ("operating system", Mention("operating system", "operating systems", 12, 29)),
        ("windows", Mention("windows", "Windows", 38, 45)),
        ("linux", Mention("linux", "Linux", 50, 55)),
    ]
    relations = extract_relations([sentence], {"operating system": 1, "windows": 1, "linux": 1})
    triples = {(r["source"], r["label"], r["target"]) for r in relations}
    assert ("operating system", "has example", "windows") in triples
    assert not any(label == "contains" and target == "windows" for _, label, target in triples)


def test_definition_descriptor_does_not_link_cpu_scheduling_to_os_process_concept():
    text = "CPU scheduling is the process of selecting a ready process for execution."
    sentence = Sentence(0, text)
    sentence.resolved = [
        ("cpu scheduling", Mention("cpu scheduling", "CPU scheduling", 0, 14)),
        ("process", Mention("process", "process", 22, 29)),
        ("ready process", Mention("ready process", "ready process", 45, 58)),
    ]
    relations = extract_relations([sentence], {"cpu scheduling": 1, "process": 1, "ready process": 1})
    assert not any(r["source"] == "cpu scheduling" and r["target"] == "process" for r in relations)


def test_explanatory_examples_are_split_from_definition_sentences():
    pieces = [text for text, _, _ in split_sentences(
        "A transaction is a logical unit of database work Imagine transferring money between two accounts. "
        "One operation subtracts money from the sender. Whereas another operation adds money to the receiver."
    )]
    assert any("logical unit of database work" in p for p in pieces)
    assert any(p.startswith("Imagine transferring") for p in pieces)
    assert any(p.startswith("Whereas another operation") for p in pieces)


def test_notes_highlighting_is_not_yellow_background_marking():
    js = Path("static/app.js").read_text(encoding="utf-8")
    css = Path("static/style.css").read_text(encoding="utf-8")
    assert '<mark>$1</mark>' not in js
    assert 'class="concept-term"' in js
    assert ".concept-term{" in css
    assert "background:transparent" in css
