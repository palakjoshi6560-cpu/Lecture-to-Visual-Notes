"""Optional dependency-parse relation extraction (spaCy). OPT-IN: set LN_RELATIONS=parse.

Subject-verb-object, passive-agent (swapped to active) and copula relations are read from the parse, so the direction does not depend
on word order and recipients are not confused with objects. NOTE: verified only with stub tokens in the development sandbox (spaCy
could not be installed there); verify with the real model before relying on it (see evaluation/README.md)."""
from __future__ import annotations

OBJ = {"dobj", "obj", "attr", "dative", "oprd"}


def _conj(tok):
    yield tok
    yield from getattr(tok, "conjuncts", [])


def parse_relations(sents, concepts):
    """Same output format as extract.extract_relations (kind='pattern', sentence evidence); needs Sentence.doc."""
    acc = {}
    for s in sents:
        doc = getattr(s, "doc", None)
        if doc is None:
            continue
        spans = [(m.start, m.end, k) for k, m in s.resolved if k in concepts]

        def concept_of(tok):
            return next((k for a, b, k in spans if a <= tok.idx < b), None)

        for v in doc:
            if v.pos_ not in ("VERB", "AUX"):
                continue
            kids = list(v.children)
            if any(c.dep_ == "neg" for c in kids):
                continue
            subjects = [x for c in kids if c.dep_ in ("nsubj", "nsubjpass") for x in _conj(c)]
            passive = any(c.dep_ == "nsubjpass" for c in kids)
            lemma = v.lemma_.lower()
            triples = []  # (source token, label, target token)
            for c in kids:
                if passive and c.dep_ == "agent":  # "ATP is produced by mitochondria" -> mitochondria produce ATP
                    triples += [(a, lemma, subj) for p in c.children if p.dep_ == "pobj" for a in _conj(p) for subj in subjects]
                elif not passive and c.dep_ in OBJ:
                    label = "is a" if lemma == "be" else lemma
                    triples += [(subj, label, o) for o in _conj(c) for subj in subjects]
                elif not passive and c.dep_ == "prep":
                    triples += [(subj, f"{lemma} {c.text.lower()}", o) for p in c.children if p.dep_ == "pobj" for o in _conj(p) for subj in subjects]
            for a, label, b in triples:
                ka, kb = concept_of(a), concept_of(b)
                if not ka or not kb or ka == kb:
                    continue
                r = acc.setdefault((ka, kb, label), dict(source=ka, target=kb, label=label, kind="pattern", weight=0.0, evidence=[]))
                r["weight"] += 1.0
                if s.idx not in r["evidence"]:
                    r["evidence"].append(s.idx)
    return list(acc.values())
