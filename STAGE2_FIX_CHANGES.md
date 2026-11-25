# Stage 2 fix: findings from the Windows pytest/spaCy run (3 failures out of 52)

| Failure | Root cause (general) | Fix |
|---|---|---|
| `test_key_concepts_found` (overfitting) and `test_relationships_have_evidence` (regularization reduces overfitting) | The spaCy path trusted `doc.noun_chunks` alone. A tagger can label a gerund-like noun as VERB/VBG (`xcomp`), so no chunk exists and the term never becomes a mention, concept or relation endpoint. | `nlp._merge_mentions`: parser chunks win; rule-based phrases are added only where the parser found nothing. Verbal gerunds ("penalizing large weights") still do not become terms. |
| `test_notes_have_no_near_duplicates_and_keep_technical_detail` (external fragmentation) | **Hypothesis, not confirmed on spaCy** (no spaCy output was available): per-topic bullet budgets let higher-scoring sentences crowd out a term the lecturer defines. Reproduced the weakness on the rules backend: 4 of 12 OS topics had a defined term with no bullet. | Coverage guarantee in `notes.build_notes`: the definition sentence of every defined term is included (max 4 extra per topic). Parallel definitions of different terms are merged into one bullet with both sources (`sentences` field), so the notes stay non-repetitive. |

Also fixed while testing: a variable-name collision I introduced (`t`) was caught by the structural-invariant test before release.

## Changed files (vs the Stage 2 ZIP)
- `lecturenotes/nlp.py`: `_merge_mentions`; spaCy branch of `analyze` merges parser chunks with rule phrases.
- `lecturenotes/notes.py`: bullet-level vectors, definition coverage guarantee, parallel-definition merge, bullets carry `sentences`.
- `tests/test_stage2_fix.py` (new, 7 tests), `evaluation/diagnose_terms.py` (new), `STAGE2_FIX_CHANGES.md` (new), `README.md` (pointer), `evaluation/results/stage2fix_dev.json` (new).

## Please verify on Windows
1. `pytest` (expect 59 passed). If `test_notes_have_no_near_duplicates_and_keep_technical_detail` still fails, run
   `python -m evaluation.diagnose_terms samples/operating_systems_lecture.txt "external fragmentation" "internal fragmentation"` and send the output:
   it shows how spaCy segmented each sentence, whether the term became a concept (rank/definition) and whether a note covers it.
2. The held-out lectures were NOT re-evaluated in this fix.
