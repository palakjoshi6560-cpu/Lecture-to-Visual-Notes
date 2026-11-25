# Quality improvement pass (2026-11-23)

This pass targets lecture-narration noise that leaked into extracted concepts and concept maps.

Changes:
- Added a conservative generic-phrase filter for common lecture-production artefacts (e.g. "previous lecture", "open parenthesis", "presentation", "please").
- Prevented these narration artefacts from becoming concept-map nodes.
- Added narration words to the topic-title low-signal filter so they are less likely to become section headings.
- Kept the original timestamped transcript unchanged; filtering affects extracted concepts/notes/maps only.

Verification in the build environment:
- `python -m compileall -q lecturenotes` passed.
- `python -m pytest -q tests/test_content_quality_regressions.py tests/test_context_map.py`: 20 passed.
- A broader OS/stage2 test batch exceeded the available execution window; it is not claimed as passed.

This is a targeted improvement, not a claim that all real-world lecture outputs are now perfect. Re-test the same YouTube lecture and inspect notes, terms, and map.
