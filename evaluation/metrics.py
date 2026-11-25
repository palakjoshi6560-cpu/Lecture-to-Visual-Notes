"""Domain-neutral quality metrics for lecture notes. Every function takes the result dict from
lecturenotes.pipeline.process plus (optionally) a gold annotation, and returns plain numbers (None = not measurable)."""
from __future__ import annotations

import math
import re
from statistics import mean

from lecturenotes.nlp import concept_key
from sklearn.feature_extraction.text import TfidfVectorizer


def norm(term):
    words = re.findall(r"[A-Za-z0-9/]+", term.replace("-", " "))
    return concept_key(words) if words else ""


def same(a, b):
    """Two normalised terms match if equal, or their word sets overlap (Jaccard >= 2/3)."""
    if a == b:
        return True
    x, y = set(a.split()), set(b.split())
    return bool(x and y) and len(x & y) / len(x | y) >= 2 / 3


def names(c):
    return [c["name"], *c.get("aliases", [])]


def hit(c, term_and_aliases):
    return any(same(norm(a), norm(b)) for a in names(c) for b in term_and_aliases)


def _gold_terms(g):
    return [g["term"], *g.get("aliases", [])]


def stem(w):
    """Crude plural-tolerant stem used only for matching evidence text to concept names."""
    return (w[:-1] if w.endswith("s") and len(w) > 3 else w)[:4]


def stems(text):
    return {stem(w) for w in re.findall(r"[a-z0-9]+", text.lower())}


def mentions(text, c):
    """True if every word (4-letter stem) of the concept's name, or of one alias, occurs in text."""
    t = stems(text)
    return any(all(stem(w) in t for w in re.findall(r"[a-z0-9]+", n.lower())) for n in names(c))


def _mean(xs):
    xs = list(xs)
    return mean(xs) if xs else None


# ---------------------------------------------------------------- concepts
def concept_scores(result, gold, k=None):
    G, S = gold["concepts"], result["concepts"]
    k = k or max(10, len(G))
    top = S[:k]
    rec_all = _mean(any(hit(c, _gold_terms(g)) for c in S) for g in G)
    rec_k = _mean(any(hit(c, _gold_terms(g)) for c in top) for g in G)
    prec_k = _mean(any(hit(c, _gold_terms(g)) for g in G) for c in top)  # lower bound: gold is not exhaustive
    f1 = 2 * prec_k * rec_k / (prec_k + rec_k) if prec_k and rec_k else 0.0
    return dict(concept_recall_all=rec_all, concept_recall_at_k=rec_k, concept_precision_at_k_lb=prec_k, concept_f1_at_k=f1,
                single_word_share_top_k=_mean(" " not in c["id"] for c in top))


# ---------------------------------------------------------------- definitions
def definition_scores(result, gold):
    S = result["concepts"]
    out = dict(definition_coverage=None, definition_correct=None)
    if gold.get("definitions"):
        found, ok = [], []
        for d in gold["definitions"]:
            c = next((c for c in S if hit(c, [d["term"]])), None)
            has = bool(c and c["definition"])
            found.append(has)
            if has:
                ds = stems(c["definition"])
                ok.append(all(all(stem(w) in ds for w in re.findall(r"[a-z0-9]+", k.lower())) for k in d["keywords"][:2]) or
                          _mean(all(stem(w) in ds for w in re.findall(r"[a-z0-9]+", k.lower())) for k in d["keywords"]) >= 0.5)
        out = dict(definition_coverage=_mean(found), definition_correct=_mean(ok))
    defined = [c for c in S if c["definition"]]
    norm_def = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
    out["definition_support_rate"] = _mean(any(norm_def(c["definition"]) in norm_def(e["text"]) for e in c["evidence"]) for c in defined)
    out["definitions_emitted"] = len(defined)
    return out


# ---------------------------------------------------------------- relations
def relation_scores(result, gold):
    cmap = {c["id"]: c for c in result["concepts"]}
    pats = [r for r in result["relations"] if r["kind"] == "pattern"]
    out = dict(relations_emitted=len(pats))
    out["relation_evidence_rate"] = _mean(
        any(mentions(e["text"], cmap[r["source"]]) and mentions(e["text"], cmap[r["target"]]) for e in r["evidence"]) for r in pats)
    G = gold.get("relations") or []
    if not G:
        return dict(out, relation_recall=None, relation_direction=None, relation_precision_judged_lb=None)
    found, direction = [], []
    for g in G:
        fwd = any(hit(cmap[r["source"]], [g["source"]]) and hit(cmap[r["target"]], [g["target"]]) for r in pats)
        rev = any(hit(cmap[r["source"]], [g["target"]]) and hit(cmap[r["target"]], [g["source"]]) for r in pats)
        found.append(fwd or rev)
        if fwd or rev:
            direction.append(fwd)
    gold_c = [_gold_terms(g) for g in gold["concepts"]]
    judged = [r for r in pats if any(hit(cmap[r["source"]], t) for t in gold_c) and any(hit(cmap[r["target"]], t) for t in gold_c)]
    good = [r for r in judged if any((hit(cmap[r["source"]], [g["source"]]) and hit(cmap[r["target"]], [g["target"]])) or
                                     (hit(cmap[r["source"]], [g["target"]]) and hit(cmap[r["target"]], [g["source"]])) for g in G)]
    return dict(out, relation_recall=_mean(found), relation_direction=_mean(direction),
                relation_precision_judged_lb=(len(good) / len(judged) if judged else None))


# ---------------------------------------------------------------- topics
def pk(ref, hyp, k):
    n = len(ref)
    return _mean((ref[i] == ref[i + k]) != (hyp[i] == hyp[i + k]) for i in range(n - k))


def window_diff(ref, hyp, k):
    n = len(ref)
    bnd = lambda s, i: sum(s[j] != s[j + 1] for j in range(i, i + k))
    return _mean(bnd(ref, i) != bnd(hyp, i) for i in range(n - k))


def reference_segments(n, starts):
    starts = sorted({0, *starts[1:]}) if starts else [0]  # text before the first gold start belongs to the first topic
    return [sum(1 for s in starts if s <= i) for i in range(n)]


def topic_scores(result, starts):
    if not starts:
        return dict(topic_pk=None, topic_windowdiff=None, topics_predicted=len(result["topics"]), topics_gold=None)
    n = len(result["transcript"])
    ref = reference_segments(n, starts)
    hyp = [t["topic"] for t in result["transcript"]]
    seg_len = n / max(1, len(set(ref)))
    k = max(2, round(seg_len / 2))
    return dict(topic_pk=pk(ref, hyp, k), topic_windowdiff=window_diff(ref, hyp, k),
                topics_predicted=len(set(hyp)), topics_gold=len(set(ref)))


# ---------------------------------------------------------------- revision sheet, notes, evidence hygiene
def revision_scores(result):
    qr, T = result["quick_revision"], result["transcript"]
    ct = {c["name"]: c["topic"] for c in result["concepts"]}
    items = [(ct.get(t["term"]), f'{t["term"]} {t["definition"]}') for t in qr["terms"]]
    items += [(ct.get(r["source"]), r["text"]) for r in qr["relationships"]]
    items += [(T[x["sentence"]]["topic"], x["text"]) for x in qr["remember"]]
    n_topics = len({t["topic"] for t in T})
    counts = [sum(1 for tp, _ in items if tp == t) for t in sorted({t["topic"] for t in T})]
    covered = sum(c > 0 for c in counts) / max(1, n_topics)
    total = sum(counts) or 1
    ent = -sum(c / total * math.log(c / total) for c in counts if c) / math.log(n_topics) if n_topics > 1 else 1.0
    texts = [t for _, t in items]
    bullets = [b["text"] for n in result["notes"] for b in n["bullets"]]
    red = overlap = None
    if len(texts) > 1:
        X = TfidfVectorizer(stop_words="english").fit_transform(texts + bullets)
        S = (X @ X.T).toarray()
        m = len(texts)
        red = max(S[i, j] for i in range(m) for j in range(i + 1, m))
        overlap = max(S[i, m + j] for i in range(m) for j in range(len(bullets))) if bullets else None
    words = lambda xs: sum(len(x.split()) for x in xs)
    live = [s["text"] for s in T if not s["excluded"]]
    return dict(qr_items=len(items), qr_topic_coverage=covered, qr_topic_balance=ent, qr_internal_redundancy=red,
                qr_notes_overlap=overlap, notes_compression=words(bullets) / max(1, words(live)))


def evidence_scores(result):
    T = result["transcript"]
    cs = result["concepts"]
    leak = sum(e["excluded"] is not None for c in cs for e in c["evidence"])
    leak += sum(T[b["sentence"]]["excluded"] is not None for n in result["notes"] for b in n["bullets"])
    leak += sum(T[x["sentence"]]["excluded"] is not None for x in result["quick_revision"]["remember"])
    return dict(concept_evidence_rate=_mean(any(mentions(e["text"], c) for e in c["evidence"]) for c in cs), excluded_leaks=leak)
