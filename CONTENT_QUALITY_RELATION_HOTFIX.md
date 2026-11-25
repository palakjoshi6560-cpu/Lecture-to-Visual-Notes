# Content quality relation hotfix

This hotfix is for the content-quality patch already applied to the project.

## Files included
- `lecturenotes/extract.py`: recovers an explicitly spoken `X refers to Y` relation when conservative noun-chunk/prepositional guards disagree. The fallback requires adjacent extracted concept mentions and permits only determiners/relative pronouns around the explicit cue, so it does not infer unsupported relations.
- `lecturenotes/nlp.py`: collapses whitespace after normalizing spaCy-split hyphenated tokens, keeping concept IDs stable (for example, `non-key field` becomes `non key field`).
- `tests/test_pipeline.py`: adds regression coverage for the referential relation and hyphen normalization.

## Apply
Extract this ZIP into the existing project root with overwrite enabled. It does not contain or replace the virtual environment, recordings, app, or UI files.

## Verification
Focused tests: `pytest -q tests\\test_pipeline.py tests\\test_terminology.py` — 23 passed in the development test environment.
