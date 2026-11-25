# Evaluation

`python -m evaluation.run_eval --split dev|heldout|all [--out results/x.json] [--rating-sheet results/x.csv]`

**Protocol**
- Tune and debug ONLY against `data/dev/`. `data/heldout/` is for reporting; every held-out run is appended to `results/heldout_ledger.txt`. Repeated held-out runs while changing the system invalidate the result; replace the held-out lectures with fresh ones before a final claim.
- Gold files (`*.gold.json`) are DRAFTS written by the assistant next to the lecture text and have not been reviewed by a human. Review/extend them (and add real lectures) before quoting any number in a report. The OS lecture has concept gold only (topic boundaries come from its headings, used for the headings-stripped variant).
- Precision numbers marked `_lb` are lower bounds: gold lists are not exhaustive, so valid extra items count as errors. Use the rating sheet (`--rating-sheet`) for human judgement of correctness and usefulness.

**Metrics**: concept recall/precision@k, definition coverage/correctness/support, relation recall, direction accuracy, evidence support, topic Pk/WindowDiff, revision topic coverage/balance/redundancy, excluded-sentence leaks.
Headed lectures are scored twice: as written and with headings removed (to simulate raw transcripts).
