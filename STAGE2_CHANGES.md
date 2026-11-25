# Stage 2 changes (domain-independent extraction quality)

Built on a separate copy of the Stage 1 project. UI (`templates/`, `static/`), `app.py`, `graph.py`, `transcribe.py` and the exports are unchanged.

## Source changes
- `lecturenotes/nlp.py`: removed lecture-file-specific phrases and OS state words; metadata/header lines detected by structure (ALL-CAPS titles, `Key: value`);
  generic pedagogical-directive and second-person filler rules; per-document verb induction (`induce_verbs`); general-English verb list;
  `refine_mentions` (edge adjectives/numerals trimmed unless the phrase recurs or is an enumeration, e.g. "primary key", "first/second law");
  past-tense verbs end a phrase; short title lines are kept in the transcript (flagged).
- `lecturenotes/extract.py`: relations are formed between ADJACENT noun phrases (all mentions, not only kept concepts); open-vocabulary labels
  from the verbs actually found; precision rules (prepositional-object subjects, relative clauses, verb chains, agent-less passives, "provides X with"
  recipients, demonstratives); concepts tied to an established concept by a stated relation are admitted even if mentioned once;
  OS-specific `STATE_MODS` removed; `GENERIC` reduced to vague superordinate nouns; `EMPH` no longer matches the noun "key".
- `lecturenotes/notes.py`: Quick Revision never repeats itself (similarity filter), skips relations/remember lines already stated by a shown definition.
- `lecturenotes/pipeline.py`: lecturer-defined terms are pruned last; optional `LN_RELATIONS=parse`; `relation_mode` in meta.
- `lecturenotes/relations_parse.py` (new, OPT-IN): dependency-parse relations. Stub-tested only; see "Needs verification".
- `evaluation/`: `compare.py`, `check_spacy.py` (new); `metrics.py` plural-tolerant stemming.
- Tests: new `tests/test_stage2.py` (10 tests, toy texts from unrelated subjects); ratchets in `tests/test_evaluation.py` now empty (no lecture-specific terms remain in `lecturenotes/`).

## Needs verification on your Windows computer (not possible in the development sandbox)
1. Genuine `pytest` (not installable in the sandbox; results were obtained with a stand-in runner): `pip install -r requirements.txt` then `pytest` (expect 52 passed).
2. spaCy path: `python -m spacy download en_core_web_sm`, then `python -m evaluation.check_spacy` (compares relation modes on dev lectures). All reported numbers use the rule-based fallback.
3. Audio upload with faster-whisper and the web UI in a browser (unchanged code, but not exercised here).
4. Do not enable `LN_RELATIONS=parse` as default until check_spacy shows it is no worse than `rules`.
