# Final-submission quality patch — 22 November 2026

This is a focused update to the existing lecture-to-visual-notes project. It does not replace the Flask application or rebuild the pipeline.

## Changes made

### 1. Sentence boundaries and ASR transcript cleanup
- Recover clear sentence boundaries when Whisper omits punctuation before a capitalized clause subject and finite predicate.
- Respect lowercase sentence starts after punctuation when the ASR has lost capitalization.
- Avoid splitting a clause after an erroneous full stop when the next lowercase word is a coordinating continuation (`and`, `or`, `but`).
- Preserve acronym tokens such as `ID`, `SQL`, and `TLB` when recovering phrase boundaries.
- Split capitalized list members from a preceding list-introducer phrase (for example, `properties Atomicity`) without splitting uppercase acronyms.

### 2. Topic segmentation and headings
- Use explicit lecture transitions, first-use definition statements, and recap cues as stronger topic boundaries than noisy TF-IDF valleys.
- Use the opening lecture agenda to locate the first substantive explanation of a listed topic.
- When a definition introduces an acronym, move the topic boundary back to the first nearby use of that acronym (for example, `SQL joins`).
- Avoid adding lexical cuts inside a coherent explicitly signposted sequence.
- Generate family headings for groups such as primary key, foreign key, candidate key, and composite key (`Key Types`).
- Label a recap section `Summary` instead of promoting a random noun phrase as its heading.

### 3. Notes and definitions
- Preserve full `first normal form`, `second normal form`, and `third normal form` phrases; `form` is treated as a noun in this ordinal-normal-form construction.
- Extract functional definitions such as `Structured query language is used to query and modify relational data`.
- Preserve explanatory continuations of worked examples, so an example is not separated from the operations/outcome that explain it.
- Keep explanatory examples such as `Imagine transferring money...` rather than discarding them as classroom instructions.
- Remove the leading `that` from definitions phrased as `Durability means that ...`.
- Make Quick Revision reserve a defined term for each substantive topic before filling remaining slots by importance; recap-only sections do not consume term slots.
- Include source-supported contextual list relationships (such as all four ACID properties) in Quick Revision relationships.

### 4. Relationships and concept map
- Recover explicitly introduced lists stated in one sentence as well as across adjacent sentences; this captures all four ACID properties.
- Normalize a modifier-bearing list parent to an already extracted canonical concept (`reliable transaction` -> `transaction`).
- Avoid treating an enclosing location phrase (`in an enrollment table`) as a component of a coordinated subject (`student ID` and `course ID` form a composite key).
- Recognize `chosen as` as the same directional relationship as `selected as`.
- Select map content by topic and preserve complete contextual groups, coordinated components, and prerequisite chains.
- Exclude generic definition descriptors such as `minimal set` and standalone structural nodes such as `table`/`row` from consuming map space.

### 5. Context-gated DBMS terminology pack
Added corrections for observed DBMS ASR variants, including `mammals include` -> `DBMS includes`, `postcurSQL` -> `PostgreSQL`, `Godtable` -> `Good table`, `Reference M` -> `reference it`, and `insert ads or record` -> `insert adds a record`. These corrections run only when the DBMS context pack is activated; they are not global replacements.

## Verification performed in this workspace

- Focused pipeline, context-map, Stage 2, terminology, observed-ASR, and new quality regression tests: **61 passed**.
- Python compilation: passed.
- Audio-segment path regression: passed; it verifies the DBMS context pack, topic order, timestamps, all four ACID map edges, and Quick Revision coverage.
- DBMS preview artifacts were generated from the existing DBMS fixture treated as timed transcript segments. This is a fixture-based test, **not a fresh Whisper transcription**.

## Remaining verification before final submission

- Run the complete suite in the user's Windows virtual environment (`python -m pytest -q`). The full suite was not run in this workspace because Flask is not installed here and the long OS/evaluation tests exceeded the available run window.
- Test a fresh recording through the real faster-whisper decoder and compare the transcript with the audio. Terminology rules can correct known context-supported variants but cannot reliably reconstruct every unclear word.
- Human-review the evaluation gold annotations before quoting evaluation metrics. The evaluation README identifies the current gold files as drafts.
- Inspect DBMS and at least one non-DBMS lecture in the actual Flask UI before replacing the final-submission project folder.

## Packaging

The final ZIP contains one project root only. The duplicate nested `lecture-visual-notes/lecture-visual-notes/` copy, Python caches, and pytest caches are excluded. No virtual environment is included.
