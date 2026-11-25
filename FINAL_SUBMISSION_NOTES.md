# Final submission notes

## Project scope

This is a subject-independent lecture-to-visual-notes application. It accepts transcript text and audio, creates topic-wise notes, a quick-revision sheet, and an evidence-linked concept map. The included DBMS and Operating Systems recordings are demonstration/evaluation inputs, not fixed subject requirements.

## Quality and UI updates in this build

- Lecture announcements and instructional signposting are excluded from study notes while remaining in the source transcript.
- Explanatory transitions such as `Imagine`, `Suppose`, `Whereas`, and `For instance` are treated as candidate sentence boundaries when ASR punctuation is missing.
- Concept-map example statements are distinguished from containment statements (`examples of X include Y` is represented as `has example`).
- Generic descriptor phrases such as `X is the process of ...` are prevented from becoming misleading links to a separate concept named `process`.
- Important terms are displayed in bold blue text rather than yellow marker highlighting.
- Subject-specific terminology corrections remain optional packs; the general extraction and topic pipeline is not restricted to DBMS or OS.

## Verification performed for this build

The following targeted checks passed in the build environment:

- 23 targeted tests covering content regressions, concept-map selection, signposting, subject-hardcoding checks, example relations, and UI highlighting.
- Python compilation for `app.py`, `lecturenotes/`, and `evaluation/`.
- JavaScript syntax check for `static/app.js`.
- ZIP archive integrity check.

The full test suite and real Whisper audio transcription were not rerun in this build environment. Run the full suite on the submission laptop before claiming complete verification. Audio recognition quality can vary with recording quality and technical vocabulary; review the recognised transcript before using the generated notes.

## Final local checks (Windows PowerShell)

```powershell
python -m pip check
python -m pytest -q
python app.py
```

Open `http://127.0.0.1:5000`. Test both text input and audio upload. For screenshots, use one clean sample lecture per subject and capture the generated notes, quick revision, concept map, and evidence view.
