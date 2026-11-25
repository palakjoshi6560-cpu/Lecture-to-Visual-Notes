# Terminology correction update

- OS vocabulary pack now activates from process/OS context without requiring the exact phrase
  "process control block" to have been transcribed correctly.
- Added context-scoped corrections for `threats` -> `threads`, `PC`/`PCV` -> `PCB`,
  `contacts switching` -> `Context switching`, and `shortage of first selects` ->
  `Shortest Job First`.
- Phrase correction can span adjacent Whisper segments. The correction is assigned to the
  segment where the phrase begins; segment timestamps and extra fields are retained.
- Corrections remain restricted to an activated terminology pack and are recorded in
  `meta.terminology_corrections`. Original audio transcript text remains available as
  `raw_transcript`.

Validation in this build environment:
- `tests/test_terminology.py`: 8 passed.
- `compileall` for `lecturenotes`: passed.
- Full suite could not be collected because Flask is not installed in this build environment.
  This is an environment limitation, not a passing full-suite result.
