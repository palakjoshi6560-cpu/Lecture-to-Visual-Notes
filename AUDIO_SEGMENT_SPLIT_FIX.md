# Audio segment splitting hotfix

## What this fixes

Some faster-whisper output contains many timed text segments but very little sentence-ending punctuation. The old sentence splitter joined those phrases into one sentence, and `process()` rejected the transcript as too short before notes or the concept map could be built.

## Files in this hotfix

- `lecturenotes/nlp.py`: groups adjacent ASR segments into sentence-sized chunks only when normal punctuation splitting finds fewer than four sentences. Original segment offsets are retained so timestamps remain associated with the correct chunks.
- `tests/test_pipeline.py`: adds regression coverage for audio segments without punctuation.

## Apply

Extract this ZIP into the project root with overwrite enabled. It does not contain or replace the virtual environment, recordings, Git metadata, or other project files.
