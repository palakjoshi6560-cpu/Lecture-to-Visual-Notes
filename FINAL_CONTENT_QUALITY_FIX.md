# Consolidated content-quality and concept-map update

This patch consolidates the remaining improvements without adding a subject-specific ranking model.

## Changes

- Concept ranking now gives a measured preference to specific multi-word terms and tempers common structural singleton nouns, while retaining those nouns as supporting concepts.
- Generic assessment/discourse words such as `exam` and `examination` are not promoted as standalone concepts.
- The NLP phrase cleanup trims trailing `together`/`jointly` so coordinated subjects are not absorbed into a concept name.
- Explicit action-style definitions now include useful generic predicates such as `preserves`, `uses`, and `identifies`.
- Coordinated constructions such as “student ID and course ID together form a composite key” produce one evidence-backed `forms part of` relation for each component. They no longer imply that only the last component forms the key.
- Quick Revision prioritizes explicit definitions before inferred fallback terms, so the four ACID definitions are retained when present and fit within the existing term budget.
- Quick Revision no longer removes distinct source-supported relations merely because their wording is similar (for example, two separate composite-key components).
- Concept-map selection downranks weak, low-salience one-off nodes, but retains a one-off technical term when its importance score is high.

## Verification performed in this environment

- `tests/test_pipeline.py`, `tests/test_terminology.py`, `tests/test_context_map.py`, `tests/test_stage2.py`: **44 passed**.
- Parser regression subset from `tests/test_stage2_fix.py`: **5 passed**.
- Anti-hardcoding ratchet tests: **2 passed**.
- PNG and SVG concept-map rendering: both succeeded.

The complete test suite was not verified here: Flask is not installed in this execution environment, and the large operating-systems lecture evaluation exceeded the available runtime. Run the complete suite in the project's `.venv-audio` on Windows before treating the patch as fully validated.
