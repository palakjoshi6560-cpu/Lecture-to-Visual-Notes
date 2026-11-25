# Review and fixes included in this build

## Changes

1. **Audio transcription decoding**
   - The previous wrapper forced `beam_size=1` (greedy decoding).
   - The default is now `beam_size=5`, configurable from 1 to 10 with `WHISPER_BEAM_SIZE`.
   - `WHISPER_INITIAL_PROMPT` can supply a short subject glossary/acronym list to guide recognition. The prompt is capped at 1,000 characters.
   - `small.en` is recommended for English technical lectures when the extra processing time is acceptable.

2. **Concept shortlist protection**
   - Recurring concepts that had already made the shortlist are now protected from being evicted when another recurring concept is reserved.
   - Added a regression test that reproduces the cutoff case.

## Verification performed in this review environment

- `pytest -q tests/test_pipeline.py tests/test_stage2.py`: **25 passed**.
- `python -m compileall -q lecturenotes tests`: passed.
- Isolated transcription-wrapper check with a stub Whisper engine: confirmed beam size 5 and the optional technical glossary are passed to faster-whisper.

## Verification limitation

This review environment does not have Flask or faster-whisper installed and has no network access to install them. Therefore, the full Flask/audio test suite and actual speech recognition on the included recordings were **not** re-run here. The changes improve decoding configuration and concept pruning, but they cannot guarantee Whisper will correctly recognise every technical term. Always review the generated transcript.
