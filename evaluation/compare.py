"""Compare two saved evaluation result files: python -m evaluation.compare before.json after.json [dev|heldout]"""
import json
import sys

KEYS = ["concept_recall_at_k", "concept_precision_at_k_lb", "definition_coverage", "definition_correct", "relation_recall", "relation_direction",
        "relation_evidence_rate", "relations_emitted", "topic_pk", "qr_topic_coverage", "qr_topic_balance", "qr_internal_redundancy", "excluded_leaks"]


def main(a, b, split="dev"):
    A, B = (json.load(open(p))[split] for p in (a, b))
    print(f"{'metric (macro-mean over runs)':32s}{'before':>9s}{'after':>9s}{'change':>9s}")
    for k in KEYS:
        x, y = A["mean"].get(k), B["mean"].get(k)
        d = "" if x is None or y is None else f"{y - x:+.2f}"
        f = lambda v: "-" if v is None else f"{v:.2f}"
        print(f"{k:32s}{f(x):>9s}{f(y):>9s}{d:>9s}")
    print("\nper run (concept recall@k / stated relations / relation recall):")
    for ra, rb in zip(A["rows"], B["rows"]):
        g = lambda r, k: "-" if r.get(k) is None else (f"{r[k]:.2f}" if isinstance(r[k], float) else str(r[k]))
        print(f"  {ra['lecture']:18s}{ra['variant']:9s} {g(ra,'concept_recall_at_k')}->{g(rb,'concept_recall_at_k')}   rels {g(ra,'relations_emitted')}->{g(rb,'relations_emitted')}   relRecall {g(ra,'relation_recall')}->{g(rb,'relation_recall')}")


if __name__ == "__main__":
    main(*sys.argv[1:])
