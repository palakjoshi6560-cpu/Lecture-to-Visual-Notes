"""Regression tests on a real 50-minute lecture transcript with '===== Heading =====' sections.

IMPORTANT: Keep the expensive full-lecture pipeline call inside a module-scoped
fixture. Running it at import time blocks pytest collection (pytest appears to
hang at ``collecting ...`` and reports zero collected tests if interrupted).
"""
import re
from pathlib import Path

import pytest

from lecturenotes import nlp
from lecturenotes.pipeline import process

TEXT = (Path(__file__).parent.parent / "samples" / "operating_systems_lecture.txt").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def lecture_result():
    """Process the long OS lecture once, after collection, and reuse its result."""
    return process(TEXT)


def test_headings_are_parsed_not_merged_into_sentences(lecture_result):
    result = lecture_result
    assert not any("=====" in s["text"] for s in result["transcript"])
    secs = nlp.split_sections(TEXT)
    assert [t for t, _, _ in secs][0] is None and "Memory Management" in [t for t, _, _ in secs]


def test_topics_are_consolidated_into_major_topics(lecture_result):
    result = lecture_result
    titles = [t["title"] for t in result["topics"]]
    assert 8 <= len(titles) <= 14, titles
    assert "CPU Scheduling" in titles and "Memory Management" in titles
    assert not any(t.startswith(("Extended", "Topic ")) for t in titles)
    cpu = next(t for t in result["topics"] if t["title"] == "CPU Scheduling")
    assert "Extended Classroom Discussion: Scheduling Example" in cpu["sections"]  # sub-section merged by content


def test_metadata_and_filler_are_excluded_from_notes_but_kept_in_transcript(lecture_result):
    result = lecture_result
    bullets = [b["text"] for n in result["notes"] for b in n["bullets"]]
    flagged = {s["text"]: s["excluded"] for s in result["transcript"] if s["excluded"]}
    assert any(t.startswith("LECTURE TRANSCRIPT") and k == "metadata" for t, k in flagged.items())
    assert flagged.get("Good morning everyone.") == "instructional"
    bad = re.compile(r"lecture transcript|subject:|duration|=====|let us|good morning|by the end of|designed for testing|draw the", re.I)
    assert not [b for b in bullets if bad.search(b)]
    assert all(not c["evidence"][0]["excluded"] for c in result["concepts"])
    assert result["raw_transcript"] == TEXT  # original transcript preserved verbatim


def test_irrelevant_and_duplicate_concepts_removed(lecture_result):
    concepts = [c["name"].lower() for c in lecture_result["concepts"]]
    for junk in ["system", "data", "resource", "mechanism", "operation", "execution", "user", "goal", "instead", "request", "decide"]:
        assert junk not in concepts, junk
    assert not ("context switch" in concepts and "context switching" in concepts)
    assert "running process" not in concepts
    for must in ["process", "operating system", "virtual memory", "page table", "deadlock", "semaphore", "translation lookaside buffer"]:
        assert must in concepts, must


def test_notes_have_no_near_duplicates_and_keep_technical_detail(lecture_result):
    from sklearn.feature_extraction.text import TfidfVectorizer
    bullets = [b["text"] for n in lecture_result["notes"] for b in n["bullets"]]

    print("\nPROCESS BULLET DETAILS:")
    for note in lecture_result["notes"]:
        for b in note["bullets"]:
            if "process" in b["text"].lower():
                print({
                    "topic": note["title"],
                    "text": b["text"],
                    "sentence": b["sentence"],
                    "concepts": b["concepts"]
                })
    X = TfidfVectorizer(stop_words="english").fit_transform(bullets)
    S = (X @ X.T).toarray()
    pairs = [
        (S[i, j], bullets[i], bullets[j])
        for i in range(len(bullets))
        for j in range(i + 1, len(bullets))
    ]

    score, first, second = max(pairs, key=lambda x: x[0])

    print("\nMOST SIMILAR BULLETS:")
    print("Similarity:", score)
    print("Bullet 1:", first)
    print("Bullet 2:", second)
    assert max(S[i, j] for i in range(len(bullets)) for j in range(i + 1, len(bullets))) < 0.6
    text = " ".join(bullets).lower()
    for detail in ["external fragmentation", "internal fragmentation", "page table", "mutex", "context switch"]:
        assert detail in text, detail


def test_quick_revision_definitions_are_real_definitions(lecture_result):
    definitions = {t["term"]: t["definition"] for t in lecture_result["quick_revision"]["terms"]}
    assert "system software" in definitions["operating system"]
    assert all(not t["definition"].lower().startswith(("memory needs", "the operating system assigns"))
               for t in lecture_result["quick_revision"]["terms"])


def test_relationships_are_source_supported(lecture_result):
    pats = [r for r in lecture_result["relations"] if r["kind"] == "pattern"]
    assert len(pats) >= 15
    stem = lambda name: [w[:4] for w in name.lower().split()]
    names = {c["id"]: c["name"] for c in lecture_result["concepts"]}
    for r in pats:
        ev = " ".join(e["text"] for e in r["evidence"]).lower()
        assert r["evidence"] and any(s in ev for s in stem(names[r["source"]])) and any(s in ev for s in stem(names[r["target"]])), r
    triples = {(r["source"], r["label"], r["target"]) for r in pats}
    assert ("process", "contains", "thread") in triples
    assert not any(r["source"] == "kernel mode" and r["target"] == "operating system" for r in pats)


def test_concept_map_uses_only_stated_relationships(lecture_result):
    graph = lecture_result["graph"]
    ids = {n["id"] for n in graph["nodes"]}
    assert 6 <= len(ids) <= 14 and graph["edges"]
    assert all(e["kind"] in {"pattern", "context"} and e["evidence"] for e in graph["edges"])
    assert ids == {x for e in graph["edges"] for x in (e["source"], e["target"])}  # no isolated nodes
