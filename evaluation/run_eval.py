"""Run the pipeline on the evaluation lectures and report domain-neutral quality metrics.

    python -m evaluation.run_eval --split dev [--out evaluation/results/name.json] [--rating-sheet sheet.csv]
    python -m evaluation.run_eval --split heldout        # logged in results/heldout_ledger.txt: do NOT tune against it
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from pathlib import Path

from lecturenotes import nlp
from lecturenotes.pipeline import process

from . import metrics as M

DATA = Path(__file__).parent / "data"
HEADING_LINE = re.compile(r"^[ \t]*={2,}.*$", re.M)


def load(split):
    out = []
    for g in sorted((DATA / split).glob("*.gold.json")):
        gold = json.loads(g.read_text(encoding="utf-8"))
        out.append((gold, (DATA / split / (gold["name"] + ".txt")).read_text(encoding="utf-8")))
    return out


def variants(gold, text):
    if gold["has_headings"]:
        stripped = re.sub(r"\n\s*\n+", " ", HEADING_LINE.sub("", text)).strip()
        return [("headed", text), ("stripped", stripped)]
    return [("plain", text)]


def topic_starts(gold, original_text, variant_name, transcript):
    """Sentence indices (in this variant's transcript) where gold topics begin."""
    texts = [t["text"] for t in transcript]
    if gold.get("topic_starts"):
        idx = []
        for p in gold["topic_starts"]:
            hit = next((i for i, t in enumerate(texts) if t.lower().startswith(p.lower())), None)
            if hit is None:
                raise ValueError(f"gold topic start not found: {p!r}")
            idx.append(hit)
        return sorted(idx)
    if gold.get("topics_from_headings"):
        if variant_name == "headed" and gold.get("merge_subsections"):
            return None  # the system intentionally merges sub-sections; raw headings are not a fair reference
        first = {}
        for s in nlp.analyze(original_text):
            if s.section >= 0:
                first.setdefault(s.section, s.text)
        pos = {t: i for i, t in enumerate(texts)}
        return sorted(pos[t] for t in first.values() if t in pos)
    return None


def evaluate_one(gold, text, vname, vtext):
    r = process(vtext)
    row = dict(lecture=gold["name"], variant=vname, domain=gold["domain"], sentences=r["meta"]["sentences"], concepts=r["meta"]["concepts"])
    for part in (M.concept_scores(r, gold), M.definition_scores(r, gold), M.relation_scores(r, gold),
                 M.topic_scores(r, topic_starts(gold, text, vname, r["transcript"])), M.revision_scores(r), M.evidence_scores(r)):
        row.update(part)
    return row, r


COLS = [("lecture", "lecture", 17), ("variant", "variant", 8), ("concept_recall_at_k", "cR@k", 5), ("concept_precision_at_k_lb", "cP@k", 5),
        ("definition_coverage", "dCov", 5), ("definition_correct", "dOK", 5), ("relation_recall", "rRec", 5), ("relation_direction", "rDir", 5),
        ("relation_evidence_rate", "rEvid", 5), ("relations_emitted", "rN", 3), ("topic_pk", "Pk", 5), ("topic_windowdiff", "WD", 5),
        ("topics_predicted", "tP", 3), ("topics_gold", "tG", 3), ("qr_topic_coverage", "qrCov", 5), ("qr_topic_balance", "qrBal", 5),
        ("qr_internal_redundancy", "qrRed", 5), ("concept_evidence_rate", "cEvid", 5), ("excluded_leaks", "leak", 4)]


def fmt(v, w):
    s = "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    return s.rjust(w) if w < 10 or s == "-" else s.ljust(w)


def table(rows):
    lines = [" ".join(h.rjust(w) if w < 10 else h.ljust(w) for _, h, w in COLS)]
    lines += [" ".join(fmt(r.get(k), w) for k, _, w in COLS) for r in rows]
    nums = [k for k, _, _ in COLS[2:]]
    mean = {k: M._mean(r[k] for r in rows if isinstance(r.get(k), (int, float))) for k in nums}
    lines.append(" ".join(fmt(("MEAN" if k == "lecture" else "" if k == "variant" else mean.get(k)), w) for k, _, w in COLS))
    return "\n".join(lines), mean


def rating_sheet(path, runs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lecture", "kind", "item", "evidence", "is_correct(1/0/uncertain)", "usefulness(1-5)", "comment"])
        for (name, vname), r in runs.items():
            cm = {c["id"]: c["name"] for c in r["concepts"]}
            for c in r["concepts"][:15]:
                w.writerow([name, "concept", c["name"], (c["evidence"] or [{"text": ""}])[0]["text"], "", "", ""])
                if c["definition"]:
                    w.writerow([name, "definition", f'{c["name"]}: {c["definition"]}', c["evidence"][0]["text"], "", "", ""])
            for x in r["relations"]:
                if x["kind"] == "pattern":
                    w.writerow([name, "relation", f'{cm[x["source"]]} --{x["label"]}--> {cm[x["target"]]}', x["evidence"][0]["text"], "", "", ""])
            for t in r["quick_revision"]["terms"]:
                w.writerow([name, "revision_term", f'{t["term"]}: {t["definition"]}', "", "", "", ""])
            for t in r["quick_revision"]["remember"]:
                w.writerow([name, "revision_remember", t["text"], "", "", "", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "heldout", "all"], default="dev")
    ap.add_argument("--out")
    ap.add_argument("--rating-sheet")
    a = ap.parse_args()
    splits = ["dev", "heldout"] if a.split == "all" else [a.split]
    all_rows, runs = {}, {}
    for sp in splits:
        rows = []
        for gold, text in load(sp):
            for vname, vtext in variants(gold, text):
                row, res = evaluate_one(gold, text, vname, vtext)
                rows.append(row)
                runs[(gold["name"], vname)] = res
        print(f"\n=== {sp} ({len(rows)} runs) ===")
        t, mean = table(rows)
        print(t)
        all_rows[sp] = dict(rows=rows, mean=mean)
        if sp == "heldout":
            ledger = Path(__file__).parent / "results" / "heldout_ledger.txt"
            ledger.parent.mkdir(exist_ok=True)
            with open(ledger, "a") as f:
                f.write(f"{datetime.datetime.now().isoformat(timespec='seconds')}\theld-out evaluation run\n")
            print(f"(held-out run logged in {ledger}; repeated runs while tuning invalidate the held-out result)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(all_rows, indent=1, default=float), encoding="utf-8")
    if a.rating_sheet:
        rating_sheet(a.rating_sheet, runs)


if __name__ == "__main__":
    main()
