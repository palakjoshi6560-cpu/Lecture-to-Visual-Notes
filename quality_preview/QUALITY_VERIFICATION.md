# DBMS quality verification (23 November 2026)

This preview is regenerated from the audio-derived transcript already bundled in the project archive. It is not a fresh Whisper transcription.

## Changes in this update
- Recover lowercase discourse markers (for example, `for instance`) as note-sentence boundaries when ASR omits punctuation.
- Detect a defined topic heading from the first eight sentences in a section, rather than only the first three, so introductory definitions are less likely to inherit a later subtopic title.
- Repair an ASR capitalization artifact in example continuations such as `... every student Store departments ...` to `... every student, store departments ...`.
- Regenerate the DBMS JSON preview and concept-map PNG/SVG from the updated pipeline.

## Verified
- `python -m compileall -q lecturenotes`: passed.
- `tests/test_content_quality_regressions.py`: 8 passed.
- Generated DBMS preview: 61 transcript sentences, 6 topics, 40 concepts, 30 extracted relations.
- Concept map: 13 nodes and 9 source-supported edges.

## Not claimed as verified
- Full project suite was not completed in this build environment. Three app/audio test modules could not be collected because Flask is not installed here, and a non-app subset exceeded the execution time limit.
- A fresh Whisper transcription was not run. This preview reuses the bundled audio-derived transcript.
- OS audio output still needs its own review.

The map intentionally includes only relationships with source evidence. It does not invent SQL links where the current extractor has not established a supported relation.
