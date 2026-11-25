# Extraction hotfix after combined content/map patch

This patch updates only `lecturenotes/extract.py`.

Fixes:
- Extract definitions when a concept is the complement in wording such as “a table is in first normal form when ...”.
- Preserve concise predicate descriptions such as “third normal form removes transitive dependencies ...”.
- Recognize sentence-initial prepositional adjuncts (“within database systems ...”) without treating the subject as a prepositional object.
- Recover explicit concept-to-concept relations from original sentence text when parser mention boundaries omit an endpoint; prefer longest overlapping concept phrases and allow plural surface forms.
- Recognize the grammatical relation “selected as”.
- Recognize a prerequisite clause of the form “X when it is in Y and ...” as X requires Y.
- Keep definition statements out of binary `means` edges.

Validation in the patch environment (LN_BACKEND=rules):
- `pytest -q tests/test_pipeline.py tests/test_terminology.py`: 25 passed.
- `pytest -q tests/test_context_map.py`: 4 passed.
- A targeted DBMS regression check confirmed:
  - definitions for first, second, and third normal form are retained;
  - second normal form -> requires -> first normal form;
  - foreign key -> refers to -> primary key;
  - candidate key -> selected as -> primary key;
  - no misleading `means` graph edges.

The OS lecture test module exceeded the available execution window in this patch environment, so it is not claimed as re-verified here. No other project files are replaced by this patch.
