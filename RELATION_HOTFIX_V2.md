# Relation hotfix v2

This patch replaces `lecturenotes/extract.py` and `tests/test_pipeline.py`.

It adds a parser-independent recovery pass for explicit `X refers to Y` relations. The pass matches complete known concept phrases in the original sentence, so the result no longer depends on spaCy returning the two endpoints as adjacent noun chunks. It also prefers longer phrases over nested shorter concepts (for example, `foreign key` over `key`).

Verification in the patch environment:
- `pytest -q tests/test_pipeline.py tests/test_terminology.py`: 24 passed.
- `python -m compileall -q lecturenotes tests/test_pipeline.py`: passed.

The full suite could not be collected in this environment because Flask is not installed. An additional selected multi-module run exceeded the sandbox execution window, so no full-suite pass is claimed.
