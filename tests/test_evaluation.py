"""Domain-neutral evaluation tests: metric correctness, dataset validity, cross-domain invariants, and a
hardcoding ratchet. They assert structure and honesty of the output, NOT subject-specific content."""
import re
from pathlib import Path

from evaluation import metrics as M
from evaluation import run_eval as E
from lecturenotes.pipeline import process

ROOT = Path(__file__).parent.parent
LECTURES = [(sp, g, t) for sp in ("dev", "heldout") for g, t in E.load(sp)]
RUNS = {}


def run(sp, name, vname, vtext):
    key = (sp, name, vname)
    if key not in RUNS:
        RUNS[key] = process(vtext)
    return RUNS[key]


# ------------------------------------------------------------------ metric unit tests
def test_term_normalisation_and_matching():
    assert M.same(M.norm("Neural Networks"), M.norm("neural network"))
    assert M.same(M.norm("type I error"), M.norm("type error"))  # tolerant of dropped one-letter tokens
    assert not M.same(M.norm("sample mean"), M.norm("sample"))


def test_segmentation_metrics_on_known_cases():
    ref = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert M.pk(ref, ref, 2) == 0 and M.window_diff(ref, ref, 2) == 0
    assert M.pk(ref, [0] * 9, 2) > 0.3 and M.window_diff(ref, [0] * 9, 2) > 0.3
    seg = M.reference_segments(6, [1, 4])  # text before the first gold start joins the first topic
    assert seg[0] == seg[3] != seg[4] == seg[5]


def test_relation_scores_detect_direction_and_missing():
    concepts = [dict(id="a", name="alpha", aliases=[], topic=0, definition=None, evidence=[]),
                dict(id="b", name="beta", aliases=[], topic=0, definition=None, evidence=[])]
    rel = lambda s, t: dict(source=s, target=t, label="x", kind="pattern", evidence=[dict(text="alpha acts on beta")])
    gold = dict(concepts=[dict(term="alpha", aliases=[]), dict(term="beta", aliases=[])], relations=[dict(source="alpha", target="beta")])
    fwd = M.relation_scores(dict(concepts=concepts, relations=[rel("a", "b")]), gold)
    rev = M.relation_scores(dict(concepts=concepts, relations=[rel("b", "a")]), gold)
    none = M.relation_scores(dict(concepts=concepts, relations=[]), gold)
    assert (fwd["relation_recall"], fwd["relation_direction"]) == (1.0, 1.0)
    assert (rev["relation_recall"], rev["relation_direction"]) == (1.0, 0.0)
    assert none["relation_recall"] == 0.0 and none["relation_direction"] is None and fwd["relation_evidence_rate"] == 1.0


# ------------------------------------------------------------------ dataset validity
def test_dataset_is_diverse_and_split_cleanly():
    dev = {g["name"] for sp, g, _ in LECTURES if sp == "dev"}
    held = {g["name"] for sp, g, _ in LECTURES if sp == "heldout"}
    assert not dev & held and len(held) >= 3 and len(dev) >= 3
    assert len({g["domain"] for _, g, _ in LECTURES}) >= 5
    assert any(g["has_headings"] for _, g, _ in LECTURES) and any(not g["has_headings"] for _, g, _ in LECTURES)
    assert any(g["noisy"] for sp, g, _ in LECTURES if sp == "heldout")
    assert any(sp == "heldout" and not g["has_headings"] for sp, g, _ in LECTURES)


def test_gold_annotations_are_grounded_in_the_text():
    def present(text, phrase):
        t = M.stems(text)
        return all(M.stem(w) in t for w in re.findall(r"[a-z0-9]+", phrase.lower()))
    for sp, g, text in LECTURES:
        assert g.get("annotator") and g["concepts"], g["name"]
        for c in g["concepts"]:
            assert any(present(text, x) for x in [c["term"], *c["aliases"]]), (g["name"], c["term"])
        for d in g.get("definitions", []):
            assert any(present(text, x) for x in [d["term"]]) and all(present(text, k) for k in d["keywords"]), (g["name"], d["term"])
        for r in g.get("relations", []):
            assert present(text, r["source"]) and present(text, r["target"]), (g["name"], r)
        for p in g.get("topic_starts", []):
            assert re.sub(r"\s+", " ", text).lower().find(p.lower()) >= 0, (g["name"], p)


def test_every_lecture_produces_metrics_for_all_variants():
    for sp, g, text in LECTURES:
        for vname, vtext in E.variants(g, text):
            row, res = E.evaluate_one(g, text, vname, vtext)
            RUNS[(sp, g["name"], vname)] = res
            assert row["concept_recall_all"] is not None and 0 <= row["qr_topic_coverage"] <= 1


# ------------------------------------------------------------------ cross-domain invariants (hold for ANY lecture)
def _all_runs():
    for sp, g, text in LECTURES:
        for vname, vtext in E.variants(g, text):
            yield g["name"], vname, vtext, run(sp, g["name"], vname, vtext)


def test_outputs_are_supported_by_the_source_text():
    for name, vname, vtext, r in _all_runs():
        ctx = f"{name}/{vname}"
        assert r["raw_transcript"] == vtext, ctx
        ev = M.evidence_scores(r)
        assert ev["excluded_leaks"] == 0 and ev["concept_evidence_rate"] == 1.0, (ctx, ev)
        assert len({c["id"] for c in r["concepts"]}) == len(r["concepts"]), ctx
        cmap = {c["id"]: c for c in r["concepts"]}
        for rel in (x for x in r["relations"] if x["kind"] == "pattern"):
            assert any(M.mentions(e["text"], cmap[rel["source"]]) and M.mentions(e["text"], cmap[rel["target"]]) for e in rel["evidence"]), (ctx, rel["source"], rel["label"], rel["target"])
        for rel in (x for x in r["relations"] if x["kind"] == "context"):
            combined = " ".join(e["text"] for e in rel["evidence"])
            assert len(rel["evidence"]) >= 1 and M.mentions(combined, cmap[rel["source"]]) and M.mentions(combined, cmap[rel["target"]]), (ctx, rel["source"], rel["label"], rel["target"])
        for c in r["concepts"]:
            if c["definition"]:
                assert any(re.sub(r"\W+", " ", c["definition"].lower()) in re.sub(r"\W+", " ", e["text"].lower()) for e in c["evidence"]), (ctx, c["name"])


def test_structure_is_consistent_for_any_lecture():
    for name, vname, vtext, r in _all_runs():
        ctx = f"{name}/{vname}"
        n_topics = len({t["topic"] for t in r["transcript"]})
        assert sorted({t["topic"] for t in r["transcript"]}) == list(range(n_topics)) == [t["id"] for t in r["topics"]], ctx
        assert all(n["bullets"] for n in r["notes"]), ctx
        assert all(not r["transcript"][b["sentence"]]["excluded"] for n in r["notes"] for b in n["bullets"]), ctx
        names = {c["name"] for c in r["concepts"]}
        assert all(t["term"] in names for t in r["quick_revision"]["terms"]), ctx
        assert all(e["kind"] in {"pattern", "context"} and e["evidence"] for e in r["graph"]["edges"]), ctx


# Runs whose revision sheet still contains near-duplicate items (found by this metric on the Stage-1 baseline).
# A NEW offender fails the test; an offender that gets fixed must be removed from the set (ratchet).
KNOWN_REDUNDANT_REVISION = set()  # none left after Stage 2


def test_quick_revision_repetition_ratchet():
    offenders = set()
    for name, vname, vtext, r in _all_runs():
        red = M.revision_scores(r)["qr_internal_redundancy"]
        if red is not None and red >= 0.8:
            offenders.add((name, vname))
    assert offenders == KNOWN_REDUNDANT_REVISION, (sorted(offenders), "update KNOWN_REDUNDANT_REVISION")


# ------------------------------------------------------------------ hardcoding ratchet
FORBIDDEN = ["classroom-style", "transcript designed", "lecture-to-visual-notes", "lecturer explains", "three-step", "mental map", "final recap",
             "at the center", "so when you see", "then identify", "finally, state", "kernel", "semaphore", "mutex", "deadlock", "scheduler", "paging",
             "thread", "context switch", "page table", "virtual memory", "file system", "operating system", "tcp", "dns", "photosynthesis",
             "database", "transaction", "hypothesis", "running", "waiting", "blocked", "terminated", "suspended", "executing"]
# Known lecture-specific content already in the source (found by audit). New entries are NOT allowed; entries must be
# removed from this list when the corresponding code is made domain-neutral (the second test enforces that).
KNOWN_DEBT = set()  # Stage 2 removed all of it; keep this empty


def _violations():
    return {(p.name, t) for p in (ROOT / "lecturenotes").glob("*.py") for t in FORBIDDEN if t in p.read_text(encoding="utf-8").lower()}


def test_no_new_lecture_specific_terms_in_source():
    new = _violations() - KNOWN_DEBT
    assert not new, f"new subject/lecture-specific hardcoding: {sorted(new)}"


def test_known_debt_list_is_current():
    stale = KNOWN_DEBT - _violations()
    assert not stale, f"remove resolved entries from KNOWN_DEBT: {sorted(stale)}"
