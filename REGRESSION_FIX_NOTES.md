# OS + DBMS regression improvements

## Changes
- Topic segmentation now gives priority to explicit lecturer signposts (for example, transitions such as "Next, let us discuss ...") when they are present.
- A lexical cut immediately after a signpost is suppressed to avoid creating a one-sentence topic.
- Topic titles can use the named concept from a clear section-opening signpost, rather than a later example/subtopic.
- Notes selection retains a larger share of useful source sentences while keeping excluded instructional/filler sentences out.
- Added OS and DBMS transcript fixtures and regression tests for topic boundaries and coverage of core lecture content.

## Verification
- `pytest tests/test_pipeline.py -q`: 29 passed.
- Full test collection is blocked in this environment because Flask is not installed (`ModuleNotFoundError: flask`).
- A broader non-Flask test run exceeded the execution time limit; it is not reported as passing.

The fixtures are sentence-level text derived from the supplied lecture outputs. They test downstream segmentation and notes quality, not real-world speech recognition accuracy.
