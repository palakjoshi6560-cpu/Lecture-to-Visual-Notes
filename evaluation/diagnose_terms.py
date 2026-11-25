"""Explain what the pipeline does with specific terms in YOUR environment (e.g. with spaCy on Windows):

    python -m evaluation.diagnose_terms samples/operating_systems_lecture.txt "external fragmentation" "page table"

For each term: which sentences contain a mention (and how the active parser/backend segmented it), whether it became a concept
(rank, importance, definition) and whether the notes contain its sentence. Paste the output when reporting an extraction problem."""
import sys

from lecturenotes import nlp
from lecturenotes.extract import build_concepts
from lecturenotes.pipeline import process


def main(path, *terms):
    text = open(path, encoding="utf-8").read()
    sents = nlp.analyze(text)
    res = process(text)
    shown = {b["sentence"] for n in res["notes"] for b in n["bullets"]}
    print("backend:", nlp.backend(), "| relation mode:", res["meta"]["relation_mode"], "| concepts:", len(res["concepts"]))
    build_concepts(sents)
    for term in terms:
        key = nlp.concept_key(term.replace("-", " ").split())
        print(f"\n== {term!r} (key {key!r})")
        for s in sents:
            if any(m.key == key for m in s.mentions):
                print(f"  sentence {s.idx} [{'excluded: ' + s.excluded if s.excluded else 'in notes' if s.idx in shown else 'not in notes'}] mentions={[m.key for m in s.mentions]}")
                print(f"     {s.text[:110]}")
            elif term.lower() in s.text.lower():
                print(f"  sentence {s.idx} CONTAINS THE TEXT BUT NO MENTION [{s.excluded or ''}] mentions={[m.key for m in s.mentions]}")
        hit = next((i for i, c in enumerate(res["concepts"]) if c["id"] == key), None)
        print("  concept:", "NO (not extracted or pruned)" if hit is None else f"yes, rank {hit + 1}/{len(res['concepts'])}, importance {res['concepts'][hit]['importance']}, definition: {res['concepts'][hit]['definition']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
