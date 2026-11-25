# Final quality update — 23 November 2026

This update improves the existing lecture-to-notes pipeline without adding subject-specific rules for the sample lecture.

## Pipeline changes

- Topic headings can now be derived from explicit lecture transitions, including transitions wrapped in phrases such as “On this note…”.
- Enumerations with a shared concept family can produce a family heading (for example, “Learning Types”) rather than a heading taken from a nearby example noun.
- Topic segmentation uses a more conservative lexical-boundary threshold and avoids cuts at example continuations and dependent sentences.
- Event-style definitions retain their condition clause (for example, “A page fault occurs when…”), instead of emitting an incomplete “Occurs” definition.
- Concept-map selection normalizes leading quantifiers before edge selection, retains source-defined concepts where appropriate, and filters weak one-off/proper-name endpoints. Map edges continue to require source evidence.
- Signposting and classroom instructions are excluded from study notes while the original transcript remains available.

## Link input

- Keeps Transcript and File or audio modes and adds Lecture link mode.
- Uses yt-dlp-supported extractors, tries English/Hindi captions first, and falls back to accessible audio with the existing faster-whisper pipeline.
- Private, restricted, DRM-protected, unsupported, or inaccessible links may fail; the feature does not bypass access controls.

## Verification performed in this build environment

- Python syntax compilation: passed for `app.py`, `lecturenotes/*.py`, and the link-input test module.
- Focused pipeline/regression tests: **40 passed, 2 deselected**.
- OS topic-structure regression: **1 passed**.
- Domain-neutral source audit: **2 passed**.
- Additional new regression tests: event-definition preservation, topic transition/family headings, and graph-node grounding are included in the 40-test focused run.

The full long-lecture audio/link end-to-end workflow was not run in this build environment. Flask and yt-dlp were unavailable here and package installation was blocked by network access, so the app-level link test module and real remote-provider extraction still need to be run on the user's machine. The existing link test fixture was corrected to contain real WebVTT newlines.
