"""Topic segmentation: TextTiling-style valleys in TF-IDF cosine similarity."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def segment_topics(sents, window=3, min_len=4):
    """Return stable topic IDs, prioritizing explicit topic introductions.

    TF-IDF valleys are useful fallback evidence, but they can split a coherent
    section when an explanation is immediately followed by its associated
    properties. Explicit lecturer transitions, first-use topic
    definitions, and recap markers take precedence when they form a viable
    sequence of sections.
    """
    n = len(sents)
    if n < 2 * min_len:
        return [0] * n

    clean_text = [re.sub(r"^\[\d{2}:\d{2}\]\s*", "", x.text.strip()) for x in sents]
    strong = [i for i, text in enumerate(clean_text)
              if i >= min_len and n - i >= min_len and TOPIC_TRANSITION.search(text)]

    # A first-use definitional sentence can introduce a new major subject.
    # Keep this pattern deliberately narrow: broad "X is ..." matching would
    # fragment lectures into one topic per technical term.
    definition_intro = re.compile(
        r"^(?:(?:a|an|the)\s+)?(?P<term>[A-Z][A-Za-z-]*(?:\s+[A-Za-z][A-Za-z-]*){0,4}?)"
        r"(?:,\s+or\s+(?P<acronym>[A-Z]{2,}))?\s+is\s+(?:used\s+to|a\s+logical\s+unit\s+of)\b", re.I
    )
    definition_anchors = []
    for i, text in enumerate(clean_text):
        if i < min_len or n - i < min_len:
            continue
        definition_match = definition_intro.match(text)
        if definition_match:
            anchor = i
            acronym = definition_match.group("acronym")
            if acronym:
                acronym_rx = re.compile(r"\b" + re.escape(acronym) + r"\b", re.I)
                earlier_mentions = [j for j in range(max(min_len, i - 2), i)
                                    if acronym_rx.search(clean_text[j])]
                if earlier_mentions:
                    anchor = earlier_mentions[0]
            definition_anchors.append(anchor)

    recap_rx = re.compile(r"^(?:to summarize|in summary|to conclude|in conclusion|overall)\b", re.I)
    recap_anchors = [i for i, text in enumerate(clean_text)
                     if i >= min_len and n - i >= min_len and recap_rx.search(text)]

    # When the lecturer opens with an agenda/list of topics, use the first
    # substantive sentence that starts explaining one of those listed items.
    # This creates a boundary before a section such as "Keys help identify..."
    # without hard-coding any subject vocabulary.
    overview_items = []
    overview_rx = re.compile(
        r"^(?:we\s+will\s+(?:cover|discuss|study|learn)|today\s+we\s+will\s+(?:cover|discuss)|"
        r"topics\s+include|this\s+lecture\s+covers?)\s+(?P<items>.+)$", re.I
    )
    for text in clean_text[:min(4, n)]:
        match = overview_rx.match(text)
        if match:
            raw_items = re.split(r",|\band\b|\bor\b", match.group("items"), flags=re.I)
            overview_items = [x.strip(" .;:").lower() for x in raw_items if x.strip(" .;:")]
            break
    overview_anchors = []
    for term in overview_items:
        words = term.split()
        if not words:
            continue
        if words[-1].endswith("s") and len(words[-1]) > 3:
            words[-1] = words[-1][:-1]
        term_pattern = r"\s+".join(re.escape(w) for w in words) + (r"s?" if words[-1].isalpha() else "")
        start_rx = re.compile(r"^(?:(?:a|an|the)\s+)?" + term_pattern + r"\b", re.I)
        first_match = next((i for i, text in enumerate(clean_text[1:], start=1)
                            if start_rx.search(text)), None)
        if first_match is not None and first_match >= min_len and n - first_match >= min_len:
            overview_anchors.append(first_match)

    anchors = sorted(set(strong + definition_anchors + recap_anchors + overview_anchors))
    accepted = []
    for cut in anchors:
        if cut < min_len or n - cut < min_len:
            continue
        if all(abs(cut - prior) >= min_len for prior in accepted):
            accepted.append(cut)

    # If there are at least two grounded anchors, use them rather than adding
    # noisy lexical cuts inside the explicitly signposted lecture structure.
    if len(accepted) >= 2:
        return [sum(1 for cut in accepted if cut <= i) for i in range(n)]

    try:
        X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform([s.text for s in sents])
    except ValueError:
        return [0] * n
    sims = []
    for g in range(1, n):
        a = np.asarray(X[max(0, g - window):g].mean(axis=0)).ravel()
        b = np.asarray(X[g:g + window].mean(axis=0)).ravel()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        sims.append(float(a @ b / d) if d else 0.0)

    def depth(i):
        left = i
        while left > 0 and sims[left - 1] >= sims[left]:
            left -= 1
        right = i
        while right < len(sims) - 1 and sims[right + 1] >= sims[right]:
            right += 1
        return (sims[left] - sims[i]) + (sims[right] - sims[i])

    depths = np.array([depth(i) for i in range(len(sims))])
    threshold = max(0.05, depths.mean() + 0.25 * depths.std())
    max_topics = max(2, min(n // 6, 12))
    cuts = []
    for i in np.argsort(-depths):
        gap = int(i) + 1
        if depths[i] < threshold or len(cuts) >= max_topics - 1:
            break
        if gap < min_len or n - gap < min_len or any(abs(gap - c) < min_len for c in cuts):
            continue
        cuts.append(gap)

    paragraph_starts = {i for i in range(1, n) if sents[i].para != sents[i - 1].para}
    snapped = set()
    for gap in cuts:
        near = [p for p in paragraph_starts if abs(p - gap) <= 2]
        snapped.add(min(near, key=lambda p: abs(p - gap)) if near else gap)

    # A sentence introducing a set of properties/components is a continuation
    # of the current subject, not a fresh topic heading.
    continuation = re.compile(
        r"^(?:(?:reliable\s+)?[\w -]{0,35}\b(?:follow|has|have|includes?|consists? of)\b"
        r".{0,55}\b(?:properties|characteristics|components|principles|stages|phases)\b|"
        r"(?:for example|for instance|this connection|this reduces|these concepts|these properties)\b)", re.I
    )
    accepted_cuts = list(accepted)
    for gap in sorted(snapped):
        if gap < min_len or n - gap < min_len:
            continue
        if continuation.search(clean_text[gap]):
            continue
        if all(abs(gap - prior) >= min_len for prior in accepted_cuts):
            accepted_cuts.append(gap)
    accepted_cuts.sort()
    return [sum(1 for cut in accepted_cuts if cut <= i) for i in range(n)]


# Strong lecturer signposts are more reliable topic boundaries than noisy
# sentence-level lexical valleys. Match the sentence that STARTS the new section.
TOPIC_TRANSITION = re.compile(
    r"^(?:now\s*,?\s*let(?:'s|\s+us)\s+(?:understand|move\s+to|turn\s+to|look\s+at|discuss)|"
    r"next\s*,?\s*let(?:'s|\s+us)\s+(?:discuss|look\s+at|move\s+to|turn\s+to)|"
    r"moving\s+on\s+to|let(?:'s|\s+us)\s+move\s+on\s+to)\b", re.I
)

RECAP = re.compile(r"recap|summary|review|revision|conclusion", re.I)
OVERVIEW = re.compile(r"introduction|overview|goals|agenda|together", re.I)  # catch-all sections: never a merge target


def heading_topics(sents, min_sim=0.06):
    """Consolidate '===== Heading =====' sections into major topics.

    Sections that share a 'Prefix: Title' prefix (e.g. 'Extended Classroom Discussion: X') are sub-sections and are
    merged into the most similar major section (TF-IDF cosine); all recap/review sections merge into the first recap.
    Returns (topic_of, titles, members) or None when the text has fewer than 3 headings."""
    heads = {}
    for s in sents:
        if s.section >= 0:
            heads.setdefault(s.section, s.heading)
    if len(heads) < 3:
        return None
    prefix = Counter(h.split(":")[0].strip().lower() for h in heads.values() if ":" in h)
    kind, first_recap = {}, None
    for sec, h in heads.items():
        if RECAP.search(h):
            kind[sec] = "major" if first_recap is None else "recap"
            first_recap = sec if first_recap is None else first_recap
        elif ":" in h and prefix[h.split(":")[0].strip().lower()] >= 2:
            kind[sec] = "sub"
        else:
            kind[sec] = "major"
    docs = {sec: h + " " + " ".join(s.text for s in sents if s.section == sec and not s.excluded) for sec, h in heads.items()}
    order = list(heads)
    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform([docs[k] for k in order])
    sim = (X @ X.T).toarray()
    target = {}
    for i, sec in enumerate(order):
        if kind[sec] == "major":
            target[sec] = sec
        elif kind[sec] == "recap":
            target[sec] = first_recap
        else:
            majors = [j for j, o in enumerate(order) if kind[o] == "major" and o != first_recap and not OVERVIEW.search(heads[o])]
            best = max(majors, key=lambda j: sim[i, j]) if majors else None
            target[sec] = order[best] if best is not None and sim[i, best] >= min_sim else sec  # no match: own topic
    tops = [sec for sec in order if target[sec] == sec]
    tid = {sec: t for t, sec in enumerate(tops)}

    def title(sec):
        h = heads[sec]
        if ":" in h:
            a, b = (p.strip() for p in h.split(":", 1))
            return b if kind[sec] == "sub" else a
        return h

    members = [[heads[s] for s in order if target[s] == sec] for sec in tops]
    topic_of = [tid[target[s.section]] if s.section >= 0 else 0 for s in sents]
    return topic_of, [title(sec) for sec in tops], members
