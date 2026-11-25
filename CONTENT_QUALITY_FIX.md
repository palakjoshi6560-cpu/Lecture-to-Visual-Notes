# Content quality patch

Extract this ZIP over the existing `lecture-visual-notes` project folder.

## Changes
- Importance scoring now uses distinct-sentence frequency and gives more weight to explicit definitions, meaningful multiword concepts, and evidence-backed outgoing relationships. Raw repetition and graph centrality have less influence.
- Definitions expressed as `X is in Y form when ...` are attached to the named state, not to the surrounding noun. Subject-led factual statements such as `Third Normal Form removes ...` are retained as concise revision facts.
- Definition cues such as `means` are not emitted as concept-to-concept graph links.
- Explicit patterns such as `refers to`, `selected as`, and nested `requires` relationships are represented directionally.
- Quick Revision no longer deletes distinct concept definitions simply because related technical terms reuse similar words. Its term budget is now scaled by topic count (10–14 terms).
- Added a regression test for normal-form definitions, prerequisite direction, and key relationships.

## Verification in the patch workspace
- Focused pipeline + terminology tests: 22 passed.
- Additional regression + anti-hardcoding checks: 3 passed.
- Python compilation: passed.
- The full evaluation suite was attempted but exceeded the available execution window; no full-suite pass is claimed here.
