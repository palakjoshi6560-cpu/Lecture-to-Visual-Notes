"""End-to-end pipeline: transcript/segments -> JSON-ready result dict."""
from __future__ import annotations

import os

from . import nlp
from .extract import build_concepts, extract_relations, score_concepts
from .graph import select_graph
from .notes import build_notes, build_quick_revision, fmt_time
from .topics import heading_topics, segment_topics
from .terminology import correct_segments


def _select_concept_keys(concepts, max_concepts):
    """Select concepts without evicting definitions or recurring concepts already in the shortlist."""
    ranked = sorted(concepts, key=lambda k: -concepts[k].importance)
    # Explicitly defined, recurring concepts receive priority in the
    # protected definition reserve so long noun phrases cannot crowd them
    # out of the capped concept list.
    definition_candidates = [k for k in ranked if concepts[k].definition]
    # Prefer explicit multi-word definitions: these are usually the specific
    # technical concepts a lecture is teaching, across any subject. Keep all
    # such concepts available to notes; the graph can still apply its own cap.
    definition_candidates.sort(
        key=lambda k: (
            0 if len(k.split()) > 1 else 1,
            -concepts[k].importance,
        )
    )
    # Every explicitly defined concept is strong source-grounded evidence,
    # regardless of whether its name is one word or a longer phrase. Do not
    # let a phrase-only reserve evict defined single-word concepts.
    definition_keep = definition_candidates
    keep = list(definition_keep)
    keep += [k for k in ranked if k not in keep][: max_concepts - len(keep)]

    recurring = [
        k for k in ranked
        if k not in keep
        and concepts[k].count >= 5
        and len(concepts[k].sentences) >= 4
    ]
    recurring.sort(
        key=lambda k: (len(concepts[k].sentences), concepts[k].count, concepts[k].importance),
        reverse=True,
    )
    reserve = min(len(recurring), max(1, max_concepts // 6))

    protected = set(definition_keep)
    protected.update(
        k for k in keep
        if concepts[k].count >= 5 and len(concepts[k].sentences) >= 4
    )
    for k in recurring[:reserve]:
        replace = next((candidate for candidate in reversed(keep) if candidate not in protected), None)
        if replace is None:
            break
        keep.remove(replace)
        keep.append(k)
        protected.add(k)
    return keep


def process(text=None, segments=None, max_concepts=None, map_nodes=14, source="text"):
    original_segments = segments
    corrected_segments, terminology_corrections = correct_segments(segments) if segments is not None else (None, [])
    sents = nlp.analyze(text, corrected_segments if segments is not None else None)
    if len(sents) < 4:
        raise ValueError("The lecture is too short. Provide at least 4 sentences.")
    concepts = build_concepts(sents)
    if not concepts:
        raise ValueError("No concepts could be extracted from this text.")
    headed = heading_topics(sents)
    topic_of, titles, members = headed if headed else (segment_topics(sents), None, None)
    relations = extract_relations(sents, concepts)
    mode = "parse" if os.environ.get("LN_RELATIONS", "rules") == "parse" and nlp.backend() == "spacy" else "rules"
    if mode == "parse":  # opt-in, unverified against real spaCy in the dev sandbox: parse relations first, rule relations fill the rest
        from .relations_parse import parse_relations
        parsed = parse_relations(sents, concepts)
        seen = {(r["source"], r["target"], r["label"]) for r in parsed}
        relations = parsed + [r for r in relations if (r["source"], r["target"], r["label"]) not in seen]
    score_concepts(concepts, relations, topic_of)
    max_concepts = max_concepts or min(60, max(30, 5 * len(set(topic_of)) + 10))
    keep = _select_concept_keys(concepts, max_concepts)
    ranked = sorted(concepts, key=lambda k: -concepts[k].importance)
    for k in ranked:
        if k not in keep:
            del concepts[k]
    for s in sents:
        s.kept = [x for x in s.kept if x[0] in concepts]
    relations = [r for r in relations if r["source"] in concepts and r["target"] in concepts]
    topics, notes, base, used = build_notes(sents, concepts, topic_of, relations, titles)
    for t, m in zip(topics, members or []):
        t["sections"] = m
    quick = build_quick_revision(sents, concepts, relations, base, used, n_terms=min(14, max(8, len(topics))))

    def ev(i):
        return dict(sentence=i, text=sents[i].text, time=fmt_time(sents[i].time), heading=sents[i].heading, excluded=sents[i].excluded)

    out_concepts = []
    for c in sorted(concepts.values(), key=lambda c: -c.importance):
        ids = ([c.definition_sentence] if c.definition_sentence is not None else []) + [i for i in c.sentences if i != c.definition_sentence]
        out_concepts.append(dict(id=c.key, name=c.name, aliases=c.aliases, importance=c.importance, components=c.components,
                                 topic=c.topic, mentions=c.count, definition=c.definition,
                                 evidence=[ev(i) for i in ids[:4]]))
    out_relations = [dict(source=r["source"], target=r["target"], label=r["label"], kind=r["kind"], weight=round(r["weight"], 2),
                          evidence=[ev(i) for i in r["evidence"][:2]])
                     for r in sorted(relations, key=lambda r: -r["weight"])]
    graph = select_graph(out_concepts, out_relations, map_nodes)
    return dict(
        meta=dict(source=source, relation_mode=mode, excluded_sentences=sum(1 for s in sents if s.excluded), nlp_backend=nlp.backend(), sentences=len(sents), topics=len(topics),
                  concepts=len(out_concepts), relations=len(out_relations), terminology_corrections=terminology_corrections),
        topics=topics, notes=notes, quick_revision=quick, concepts=out_concepts, relations=out_relations, graph=graph,
        transcript=[dict(ev(s.idx), topic=topic_of[s.idx]) for s in sents],
        raw_transcript=text if text is not None else " ".join(g["text"] for g in original_segments),
    )
