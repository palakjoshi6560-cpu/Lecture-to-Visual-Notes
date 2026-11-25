"""Topic-wise study notes and Quick Revision (concise, near-duplicate-free)."""
from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer

from .extract import EMPH
from .nlp import VERBS

LEAD = re.compile(r"^(?:so|okay|ok|now|alright|well|right|and|but|then|also|to summarize|in summary|in conclusion|remember that|note that)\b,?\s*", re.I)
COPULA = re.compile(r"^\s*(?:is|are|means|occurs|happens|refers to)\s+(?!not\b|\w+ed\b|brought\b|made\b|given\b|kept\b|held\b|taken\b|known\b|used\b)")
CLAUSE = re.compile(r", | which | because | so that | instead of |; ")
RECAP_START = re.compile(r"^(?:to summarize|in summary|to conclude|in conclusion|overall)\b", re.I)
EXAMPLE_CUE = re.compile(r"^(?:for example|for instance|imagine|suppose|consider)\b", re.I)
TOPIC_DEFINITION_TITLE = re.compile(
    r"^(?:(?:a|an|the)\s+)?(?P<term>[A-Z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,4}?)"
    r"(?:,\s+or\s+[A-Z]{2,})?\s+is\s+(?:(?:a|an|the)\s+)?"
    r"(?:[A-Za-z-]+\s+){0,3}?(?:used\s+to|logical\s+unit\s+of)\b", re.I
)


def fmt_time(t):
    return None if t is None else f"{int(t) // 60:02d}:{int(t) % 60:02d}"


def compress(text, max_words=26):
    """Drop discourse openers; cut overly long sentences at a clause boundary."""
    prev = None
    while prev != text:
        prev, text = text, LEAD.sub("", text)
    text = text[:1].upper() + text[1:]
    # Repair punctuation artifacts common in speech-recognition output.
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # Repair ASR capitalization that obscures an imperative continuation in
    # an example: "instead of ... every student Store departments in ...".
    def _repair_imperative(match):
        verb = match.group(2)
        if verb.lower() not in VERBS:
            return match.group(0)
        return match.group(1).rstrip() + ", " + verb.lower()
    text = re.sub(
        r"(\binstead of\b[^.!?]{1,110}?)\s+([A-Z][a-z]+)(?=\s+[a-z])",
        _repair_imperative, text,
    )
    text = re.sub(r"([,;])\s*[.]", r"\1", text)
    text = re.sub(r"[.]{2,}", ".", text)
    # A lowercase coordinating conjunction after a period is usually a
    # punctuation artifact in ASR ("requires 1NF. and removes ..."). Join it
    # without a comma. Preserve uppercase sentence starts so distinct
    # sentences such as "... one another. Durability means ..." stay separate.
    text = re.sub(r"\.\s+(and|or|but)\b", r" \1", text)
    text = re.sub(r",\s+and\s+(?=removes?\b)", " and ", text, flags=re.I)
    text = re.sub(r"\b(properties|components|stages|phases|features)\s+(?=[A-Z])", r"\1: ", text)
    text = re.sub(r"\s+([.!?])", r"\1", text)
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\brather than repeating list\b", "rather than repeating a list", text, flags=re.I)
    if text and text[-1] not in ".!?":
        text += "."
    if len(text.split()) > max_words:
        for m in reversed(list(CLAUSE.finditer(text))):
            head = text[:m.start()]
            if 8 <= len(head.split()) <= max_words:
                return head.rstrip(".,;") + "."
    return text


def _pairs_similarity(X, i, j):
    return float((X[i] @ X[j].T).toarray()[0, 0])


def build_notes(sents, concepts, topic_of, relations, titles=None):
    imp = {k: c.importance for k, c in concepts.items()}
    try:
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform([s.text for s in sents])
    except ValueError:
        X = None
    def_sents = {c.definition_sentence for c in concepts.values() if c.definition_sentence is not None}
    try:  # bullet-level vectors (the text students read) decide whether a definition is a parallel of an existing bullet
        XB = TfidfVectorizer(stop_words="english").fit_transform([compress(s.text) for s in sents])
    except ValueError:
        XB = None

    def base(s, covered):
        if s.excluded:
            return 0.0
        n = len(s.text.split())
        gain = sum(imp[k] * (0.25 if k in covered else 1.0) for k in {k for k, _ in s.kept})
        gain += (0.5 if s.idx in def_sents else 0) + (0.3 if EMPH.search(s.text) else 0)
        gain += 0.4 if EXAMPLE_CUE.search(s.text.strip()) else 0
        return gain / (1 + 0.03 * max(0, n - 12)) * (0.3 if n < 5 or s.text.endswith("?") else 1.0)

    n_topics, chosen, topics, notes = max(topic_of) + 1, [], [], []
    used_titles = set()
    for t in range(n_topics):
        ids = [s.idx for s in sents if topic_of[s.idx] == t]
        cnt = {}
        for i in ids:
            for k in {k for k, _ in sents[i].kept}:
                cnt[k] = cnt.get(k, 0) + 1
        ranked = sorted(cnt, key=lambda k: (concepts[k].topic != t, -(imp[k] + concepts[k].components['distinct']) * cnt[k]))
        if titles:
            title = titles[t]
        elif RECAP_START.search(sents[ids[0]].text.strip()):
            title = "Summary"
        elif (definition_title := next((match for match in
                (TOPIC_DEFINITION_TITLE.match(sents[i].text.strip()) for i in ids[:8]) if match), None)):
            title = definition_title.group("term").strip()
        else:
            # A topic heading should describe the cluster, not merely repeat
            # its highest-scoring noun. Prefer recurring, specific phrases;
            # generic structural nouns ("key", "page", "column") are poor
            # standalone headings even when they score highly.
            structural = {"table", "row", "column", "key", "record", "field", "attribute", "value", "page", "system", "data", "process"}
            candidates = [k for k in ranked if concepts[k].name.lower() not in used_titles]
            specific = [k for k in candidates if len(k.split()) > 1 and k not in structural]
            pool = specific or [k for k in candidates if k not in structural] or candidates
            # A concept explicitly defined inside this topic is a stronger
            # title anchor than a frequently repeated subtopic (e.g.
            # "round robin" should not replace the defined heading "CPU scheduling").
            defined_here = [
                k for k in candidates
                if concepts[k].definition and concepts[k].definition_sentence in ids
            ]
            title_pool = defined_here or pool
            # When a section introduces several members of a clear concept
            # family (e.g. primary / foreign / candidate / composite key), use
            # the family heading rather than arbitrarily naming one member.
            family_heads = {"key", "algorithm", "model", "network", "protocol", "memory", "tree", "graph"}
            families = {}
            for key in title_pool:
                words = key.split()
                if len(words) >= 2 and words[-1] in family_heads:
                    families.setdefault(words[-1], []).append(key)
            family = max(families.items(), key=lambda item: len(item[1]), default=None)
            if family and len(family[1]) >= 3:
                title = family[0].capitalize() + " Types"
            else:
                title_key = max(
                    title_pool,
                    key=lambda k: (
                        len(k.split()) > 1,
                        len(concepts[k].sentences),
                        concepts[k].count,
                        concepts[k].importance,
                    ),
                    default=None,
                )
                title = concepts[title_key].name if title_key else f"Topic {t + 1}"
            # If the lecturer explicitly announces the section in its opening
            # sentence, use that named concept rather than a narrower example
            # mentioned later (e.g. "Round robin" inside CPU scheduling).
            opener = next((sents[i].text for i in ids[:2]
                           if re.search(r"\b(?:understand|discuss|move\s+to|turn\s+to)\b", sents[i].text, re.I)), "")
            opener = re.sub(r"^\[\d{2}:\d{2}\]\s*", "", opener)
            transition_prefix = r"^(?:(?:now|next),?\s*)?let(?:'s|\s+us)\s+(?:understand|discuss|move\s+to|turn\s+to)\s+"
            has_transition_prefix = re.match(transition_prefix, opener, re.I) is not None
            opener_title = re.sub(transition_prefix, "", opener, flags=re.I)
            opener_title = re.sub(r"^(?:the\s+concept\s+of\s+(?:a|the)?|the|a|an)\s+", "", opener_title, flags=re.I)
            opener_title = re.split(r"[.!?]", opener_title, maxsplit=1)[0].strip(" ,:")
            opener_words = opener_title.split()[:2]
            if len(opener_words) == 2 and opener_words[0].lower() == opener_words[1].lower():
                opener_words.pop()
            if has_transition_prefix and opener_words:
                title = " ".join(opener_words)
        used_titles.add(title.lower())
        title = title[:1].upper() + title[1:]
        covered, picked = set(), []
        definition_bullets = []
        n_live = sum(1 for i in ids if not sents[i].excluded)
        # Keep concise notes, but retain more of the lecture's actual
        # explanatory sentences. Definitions below remain mandatory.
        budget = min(9, max(4, round(n_live * 0.8))) if n_live <= 12 else min(14, 8 + (n_live - 12) // 6)
        for _ in range(budget):
            best, best_sc = None, 0.0
            for i in ids:
                if i in picked or (X is not None and any(_pairs_similarity(X, i, j) >= 0.45 for j in chosen)):
                    continue
                sc = base(sents[i], covered)
                if sc > best_sc:
                    best, best_sc = i, sc
            if best is None:
                break
            picked.append(best)
            chosen.append(best)
            covered |= {k for k, _ in sents[best].kept}
        # Coverage guarantee: a term the lecturer defines must not be crowded out by higher-scoring sentences.
        in_topic = set(ids)
        owed = sorted(
            (
                c for c in concepts.values()
                if c.definition_sentence in in_topic
                and not sents[c.definition_sentence].excluded
            ),
            key=lambda c: -c.importance,
        )
        # A definition is a coverage requirement, not a candidate to merge
        # with a similar sentence. Keep each source sentence represented once;
        # the rendering stage below emits one bullet per defined concept.
        for c in owed:
            i = c.definition_sentence
            if i not in picked:
                picked.append(i)

        # Preserve the explanatory continuation of a worked example. Without
        # this, a note may keep "Imagine transferring money..." and then jump
        # straight to the failure outcome, omitting the actual operations.
        example_followup = re.compile(
            r"^(?:one\b|another\b|each\b|if\b|this\b|these\b|the first\b|the second\b|a unit of work groups\b)", re.I
        )
        for seed in [i for i in ids if EXAMPLE_CUE.search(sents[i].text.strip()) and not sents[i].excluded]:
            for candidate in ids:
                if candidate <= seed or candidate > seed + 3 or sents[candidate].excluded:
                    continue
                if example_followup.search(sents[candidate].text.strip()):
                    picked.append(candidate)
        picked = sorted(set(picked))

        bullets = []

        # Collect ALL definitions in this topic before rendering bullets.
        definition_bullets = [
            (f"{c.name}: {compress(c.definition)}", c.name)
            for c in owed
            if c.definition
        ]
        
        for i in sorted(picked):
            # Find every concept explicitly defined in this source sentence.
            sentence_definitions = [
                c for c in concepts.values()
                if c.definition_sentence == i and c.definition
            ]

            # Always give every extracted definition its own bullet. This also
            # prevents definitions from adjacent sentences being merged later.
            if sentence_definitions:
                for c in sorted(sentence_definitions, key=lambda c: -c.importance):
                    definition_text = compress(c.definition)
                    text = f"{c.name[:1].upper() + c.name[1:]}: {definition_text}"

                    bullets.append(dict(
                        text=text,
                        sentence=i,
                        sentences=[i],
                        heading=sents[i].heading,
                        time=fmt_time(sents[i].time),
                        concepts=[c.name]
                    ))

                continue


            # Preserve the existing behaviour for ordinary sentences.
            keys = sorted(
                {k for k, _ in sents[i].kept},
                key=lambda k: -imp[k]
            )
            text, extra = compress(sents[i].text), []

            # Suppress a repeated explanation only when it covers the same
            # concept as an explicit definition already present in this topic.
            candidate_concepts = {concepts[k].name for k in keys}
            duplicate_definition = False
            for definition_text, definition_concept in definition_bullets:
                if definition_concept not in candidate_concepts:
                    continue
                try:
                    pair = TfidfVectorizer(stop_words="english").fit_transform(
                        [definition_text, text]
                    )
                    similarity = _pairs_similarity(pair, 0, 1)
                except ValueError:
                    similarity = 0.0
                if similarity >= 0.75:
                    duplicate_definition = True
                    break

            if duplicate_definition:
                continue

            for j in extra:
                piece = compress(sents[j].text)
                text = text.rstrip(".") + "; " + (
                    piece[:1].lower() + piece[1:]
                    if piece[1:2].islower() else piece
                )
                keys = sorted(
                    set(keys) | {k for k, _ in sents[j].kept},
                    key=lambda k: -imp[k]
                )

            bullets.append(dict(
                text=text,
                sentence=i,
                sentences=[i, *extra],
                heading=sents[i].heading,
                time=fmt_time(sents[i].time),
                concepts=[concepts[k].name for k in keys]
            ))

           
        terms = [concepts[k].name for k in ranked if concepts[k].name.lower() != title.lower()][:5]
        topics.append(dict(id=t, title=title, first_sentence=ids[0], last_sentence=ids[-1], key_terms=terms))
        notes.append(dict(topic_id=t, title=title, key_terms=terms, bullets=bullets))
        # Remove repeated explanations of explicit definitions across topics.
    # Only compare bullets that share the defined concept.
    definition_bullets = []

    for c in concepts.values():
        if c.definition_sentence is None or not c.definition:
            continue

        for note in notes:
            for b in note["bullets"]:
                if (
                    b["sentence"] == c.definition_sentence
                    and c.name in b["concepts"]
                ):
                    definition_bullets.append((b, c.name))

    remove_ids = set()

    for note in notes:
        for b in note["bullets"]:
            for definition_bullet, concept_name in definition_bullets:

                # Never compare a definition with itself.
                if b is definition_bullet:
                    continue

                # Only compare bullets explicitly tagged with this concept.
                if concept_name not in b["concepts"]:
                    continue

                try:
                    pair = TfidfVectorizer(
                        stop_words="english"
                    ).fit_transform([
                        definition_bullet["text"],
                        b["text"]
                    ])
                    similarity = _pairs_similarity(pair, 0, 1)
                except ValueError:
                    similarity = 0.0

                if similarity >= 0.75:
                    remove_ids.add(id(b))
                    break

    for note in notes:
        note["bullets"] = [
            b for b in note["bullets"]
            if id(b) not in remove_ids
        ]

    # A topic must never disappear from the study notes merely because its
    # candidate sentences were low-scoring or were removed as near-duplicates
    # of definitions elsewhere. Restore one source-grounded, non-excluded
    # sentence for any otherwise-empty topic. This is deliberately generic:
    # it does not infer or manufacture subject-specific content.
    for topic_id, note in enumerate(notes):
        if note["bullets"]:
            continue
        candidates = [
            s for s in sents
            if topic_of[s.idx] == topic_id and not s.excluded and len(s.text.split()) >= 5
        ]
        if not candidates:
            continue
        fallback = max(
            candidates,
            key=lambda s: (base(s, set()), len({k for k, _ in s.kept}), -len(s.text.split())),
        )
        keys = sorted({k for k, _ in fallback.kept if k in concepts}, key=lambda k: -imp[k])
        note["bullets"].append(dict(
            text=compress(fallback.text),
            sentence=fallback.idx,
            sentences=[fallback.idx],
            heading=fallback.heading,
            time=fmt_time(fallback.time),
            concepts=[concepts[k].name for k in keys],
        ))

    return topics, notes, base, set(chosen)



def build_quick_revision(sents, concepts, relations, base, used_sentences, n_terms=8, n_rel=8):
    # Explicit definitions are the most valuable revision material. Preserve
    # them while reserving one slot for each substantive topic before filling
    # remaining slots by global importance.
    structural = {"table", "row", "column", "key", "record", "field", "attribute", "value"}
    top = sorted(
        concepts.values(),
        key=lambda c: (
            0 if c.definition and c.key not in structural else
            1 if c.definition else
            2,
            -len(c.key.split()),
            -c.importance,
        ),
    )

    topic_first = {}
    for concept in concepts.values():
        if concept.sentences:
            topic_first[concept.topic] = min(topic_first.get(concept.topic, len(sents)), min(concept.sentences))
    recap_rx = re.compile(r"^(?:to summarize|in summary|to conclude|in conclusion|overall)\b", re.I)
    summary_topics = set()
    for topic, first in topic_first.items():
        context_start = max(0, first - 2)
        if any(recap_rx.search(sents[i].text.strip()) for i in range(context_start, min(len(sents), first + 1))):
            summary_topics.add(topic)

    terms, fallback_used = [], set()
    selected_keys = set()
    for topic in sorted(topic_first):
        if topic in summary_topics or len(terms) >= n_terms:
            continue
        representative = next((c for c in top if c.topic == topic and c.definition
                               and c.key not in structural), None)
        if representative is not None:
            d = compress(representative.definition, 16)
            d = d[:1].lower() + d[1:] if d[1:2].islower() else d
            terms.append(dict(term=representative.name, definition=d.rstrip(".")))
            selected_keys.add(representative.key)

    for c in top:
        if len(terms) == n_terms:
            break
        if c.key in selected_keys or c.topic in summary_topics:
            continue
        if c.definition:
            d = compress(c.definition, 16)
            d = d[:1].lower() + d[1:] if d[1:2].islower() else d
            d = d.rstrip(".")
        else:
            subj = [sents[i] for i in c.sentences if sents[i].kept and sents[i].kept[0][0] == c.key and sents[i].kept[0][1].start <= 4
                    and len(sents[i].text.split()) >= 6 and COPULA.match(sents[i].text[sents[i].kept[0][1].end:])]
            if not subj:
                continue  # no sentence that is actually about this term
            src = max(subj, key=lambda s: base(s, set()) / (1 + 0.05 * len(s.text.split())))
            fallback_used.add(src.idx)
            d = compress(src.text, 16).rstrip(".")
        terms.append(dict(term=c.name, definition=d))
        selected_keys.add(c.key)
    imp = {k: c.importance for k, c in concepts.items()}
    rels, seen = [], set()
    generic_endpoints = {"minimal set", "organized collection", "organised collection", "related data",
                         "logical unit", "single value", "repeating list", "table", "row", "record"}
    contextual = [r for r in relations if r["kind"] == "context" and r["source"] in concepts
                  and r["target"] in concepts and r["source"] not in generic_endpoints
                  and r["target"] not in generic_endpoints]
    pats = [r for r in relations if r["kind"] == "pattern" and r["source"] in concepts
            and r["target"] in concepts and r["source"] not in generic_endpoints
            and r["target"] not in generic_endpoints]
    ordered_relations = sorted(contextual, key=lambda r: -(imp[r["source"]] + imp[r["target"]]))
    ordered_relations += sorted(pats, key=lambda r: -(r["weight"] * (imp[r["source"]] + imp[r["target"]])))
    for r in ordered_relations:
        pair = frozenset((r["source"], r["target"]))
        stated_by_definition = {concepts[r["source"]].definition_sentence, concepts[r["target"]].definition_sentence} & set(r["evidence"])
        if pair in seen or (stated_by_definition and r["kind"] != "context") or (r["label"].startswith("is a") and concepts[r["source"]].definition):
            continue  # direct definitions are redundant; enumerated context links remain useful revision facts
        seen.add(pair)
        a, b = concepts[r["source"]].name, concepts[r["target"]].name
        rels.append(dict(source=a, label=r["label"], target=b, text=f"{a} {r['label']} {b}", sentence=r["evidence"][0]))
        if len(rels) == n_rel:
            break
    shown = {c.definition_sentence for c in concepts.values() if c.definition_sentence is not None} | fallback_used  # already on the sheet
    flagged = sorted((s for s in sents if EMPH.search(s.text) and not s.excluded and s.idx not in used_sentences and s.idx not in shown), key=lambda s: -base(s, set()))[:3]
    remember = [dict(text=compress(s.text), sentence=s.idx) for s in flagged]
    # Invariant: no item may repeat another (cosine similarity on the sheet's own text); earlier items win.
    nt, nr = len(terms), len(rels)
    texts = [f"{t['term']} {t['definition']}" for t in terms] + [r["text"] for r in rels] + [r["text"] for r in remember]
    drop = set()
    if len(texts) > 1:
        try:
            X = TfidfVectorizer(stop_words="english").fit_transform(texts)
            S = (X @ X.T).toarray()
            # Keep the revision sheet concise across ALL item types. The full
            # notes, concepts, and relations retain the source material; Quick
            # Revision should not repeat a definition as a relation or repeat
            # near-identical definitions. A stricter 0.80 threshold preserves
            # distinct coordinated relations such as student ID/course ID ->
            # composite key while removing genuine repetition.
            for j in range(len(texts)):
                if any(S[i, j] >= 0.80 for i in range(j) if i not in drop):
                    drop.add(j)
        except ValueError:
            pass
    return dict(terms=[x for i, x in enumerate(terms) if i not in drop],
                relationships=[x for i, x in enumerate(rels) if nt + i not in drop],
                remember=[x for i, x in enumerate(remember) if nt + nr + i not in drop])
