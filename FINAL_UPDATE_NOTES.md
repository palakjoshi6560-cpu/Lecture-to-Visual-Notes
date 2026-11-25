# Final Consolidated Update Notes

This release consolidates the audio, terminology, concept extraction, concept-map, notes, and regression fixes into one project snapshot.

## Changes

- Replaced repeated per-concept/per-sentence regex scans in relation recovery with a prebuilt token trie and indexed phrase matching.
- Cached TF-IDF similarity computations during note selection; the similarity cache remains sparse for very long transcripts to avoid quadratic dense-memory growth.
- Preserved every explicitly defined concept even when the lecture contains more definitions than the normal concept shortlist size.
- Preserved named items in explicitly introduced lists, including a one-off list member such as `consistency`, and surfaced its source-grounded membership in Quick Revision when no separate definition was spoken.
- Kept prerequisite endpoints together in concept maps when the source explicitly states a `requires` relationship.
- Preserved complete coordinated components for composite keys and removed a preceding container phrase from being misidentified as a component.
- Kept context-aware terminology corrections in separate vocabulary packs; the original ASR transcript remains available.
- Added regression coverage for list members, prerequisite map selection, composite-key components, definitions, terminology, and cross-domain extraction.

## Verification in the build environment

- 81 tests passed in the available non-Flask suite (`test_audio.py`, `test_audio_real.py`, and `test_graph_and_app.py` were excluded because Flask/faster-whisper are not installed in this build environment).
- DBMS and Operating Systems text-pipeline smoke checks passed.
- PNG and SVG graph rendering smoke checks passed for both subjects.
- Composite-key smoke check confirmed `student ID` and `course ID` are components and `enrollment table` is not.
- OS text benchmark: 7,119 words / 559 sentences processed in about 0.3–0.4 seconds on a warm run using the rule-based NLP backend. This measures post-transcription text processing, not Whisper audio transcription; timing will vary by machine and by spaCy backend.

## Limitations

- The build environment did not have Flask, faster-whisper, or the spaCy English model installed, so the Flask UI and real audio/Whisper integration tests were not executed here. Run the full test suite in the project's Windows virtual environment to verify those integrations.
- Text-pipeline smoke checks use the rule-based NLP fallback in this build environment. The local Windows installation may use spaCy and produce slightly different extraction boundaries.
- ASR output must still be reviewed for subject-specific speech-recognition errors. Context-aware terminology corrections are intentionally conservative and do not replace source transcript text.
