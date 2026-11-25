# Audio input: what it does, how to verify it on Windows, and how to demo it

## What the feature does
Upload a lecture recording (**.wav .mp3 .m4a .flac .ogg .webm .mp4 .aac**, up to 200 MB, up to 180 minutes) in the "File or audio" tab.
The recording is transcribed on your own computer with **faster-whisper** (no API key, no cloud), the timed transcript goes through the same
pipeline as pasted text, and you get topic-wise notes (with a time chip showing when each point was said), key concepts and definitions,
Quick Revision, a concept map, and PNG / SVG / JSON / Transcript downloads. Pasted or uploaded text works exactly as before.

## What has and has not been verified
| | Status |
|---|---|
| Upload validation, error messages, temp-file cleanup, timestamps, hand-over to the notes pipeline, exports | Automated tests with the speech-to-text call **replaced** (`test_mock_*`) |
| Our faster-whisper wrapper (settings, segment cleaning, errors, status, one model load) | Automated tests with a **fake** faster_whisper module (`test_stubengine_*`) |
| **Real speech-to-text** (Whisper actually hearing speech) | **NOT verified in the development sandbox** (no faster-whisper, no internet). Verify with steps 3-5 below. |

## Windows steps (about 15 minutes, internet needed once)
1. Open PowerShell in the project folder and install:
   ```
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. Download the speech model once (about 150 MB for `base`; `tiny.en` is smaller and faster):
   ```
   $env:WHISPER_MODEL = "small.en"      # slower than base.en, generally better for technical English
   $env:WHISPER_BEAM_SIZE = "5"          # better search than greedy decoding; default is 5
   # Optional: give Whisper the key terms for this lecture (change them for each subject)
   $env:WHISPER_INITIAL_PROMPT = "DBMS, Atomicity, Consistency, Isolation, Durability, 1NF, 2NF, 3NF, SQL"
   python -m lecturenotes.transcribe --prepare
   ```
3. Make demo audio with the Windows voice (or record your own with Voice Recorder, which saves .m4a):
   ```
   powershell -ExecutionPolicy Bypass -File samples\make_demo_audio.ps1
   powershell -ExecutionPolicy Bypass -File samples\make_demo_audio.ps1 -InFile evaluation\data\dev\biology_plain.txt -Name demo_biology
   ```
4. Check the speech recognition outside the web app (prints time taken, language, first segments, word recall against the text, and a pipeline summary):
   ```
   python -m lecturenotes.transcribe samples\demo_audio\demo_biology.wav --reference samples\demo_audio\demo_biology.txt
   ```
   Expect a word recall around 0.9 or higher for synthetic speech. Please send me the output if it is lower than 0.7.
5. Run the real test (it must say `passed`, not `skipped`):
   ```
   pytest tests\test_audio_real.py -v -s
   pytest -q
   ```
6. Start the app: `python app.py`, open http://127.0.0.1:5000.

## 5-minute demo for your guide
1. Click **File or audio**: the hint under the file box says whether transcription is ready ("model: base.en") or that the model downloads first.
2. Upload `samples\demo_audio\demo_biology.wav` (a non-computer-science subject) and click **Generate notes**. Say that it is transcribed locally.
3. Show the summary line under the tabs (duration, language, number of segments) and **Study notes**: click a time chip to show the exact sentence.
4. Show **Quick Revision**, **Concept map** (then PNG/SVG/JSON), and **Transcript** (timestamped text of what the recogniser heard).
5. Upload `demo_lecture.wav` (machine learning), then paste text in the **Transcript** tab to show that both input types use the same pipeline.
Tip: for a live demo use 1-3 minutes of audio. A long lecture takes several minutes (transcription plus note generation).

## Settings (environment variables)
`WHISPER_MODEL` (tiny.en, base.en, small.en, small, medium, or a local folder), `WHISPER_BEAM_SIZE` (1-10, default 5), `WHISPER_INITIAL_PROMPT` (optional technical vocabulary), `WHISPER_DEVICE` (cpu), `WHISPER_COMPUTE` (int8), `WHISPER_LANGUAGE`
(default: auto-detect; non-English audio shows a warning because the notes pipeline is built for English), `LN_MAX_AUDIO_MINUTES` (180), `LN_MAX_UPLOAD_MB` (200).

## Troubleshooting
- *"faster-whisper could not be loaded"*: install the Microsoft Visual C++ Redistributable (x64), then reinstall: `pip install --force-reinstall faster-whisper`.
- *Model download fails or hangs*: you need internet for the first run. Hugging Face symlink warnings on Windows are harmless (`$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"`).
- *Very slow*: use `tiny.en` or `base.en`, close other programs, try a shorter file.
- *Wrong or garbled technical terms*: try `small.en`, keep `WHISPER_BEAM_SIZE=5`, and set `WHISPER_INITIAL_PROMPT` to a short glossary for that lecture. The prompt guides the recogniser; always review the Transcript because it cannot guarantee correct spelling.
