# Lecture to Visual Notes

A local-first application that turns lecture transcripts or audio recordings into topic-wise study notes, a quick-revision sheet, and an evidence-backed concept map.

It is designed to help students move from a raw lecture to structured revision material while keeping source-sentence references visible.

## Features

- **Two input modes:** paste a transcript or upload `.txt`, `.md`, or audio (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.webm`, `.mp4`, `.aac`).
- **Local speech recognition:** audio is transcribed with `faster-whisper`; no paid transcription API is required. The first model run may need an internet connection to download model files.
- **Topic-wise notes:** extracts concepts, definitions, key terms, and concise supporting points.
- **Quick Revision:** collects important terms, definitions, and supported relationships.
- **Concept map:** renders PNG and SVG maps from extracted relationships.
- **Evidence traceability:** note bullets and concepts link back to source sentences; audio notes include timestamps.
- **Exports:** download notes and evidence as JSON, the map as PNG/SVG, and the recognised transcript as text.
- **Subject-neutral pipeline:** general NLP/extraction logic is separate from optional subject-specific terminology packs.

## How it works

1. **Input and transcription** — accepts text directly or converts audio into timestamped segments using `faster-whisper`.
2. **Preprocessing and NLP** — sentence segmentation and phrase/concept detection use spaCy when available, with a built-in rules fallback.
3. **Concept extraction** — identifies concepts, definitions, importance signals, and candidate relationships.
4. **Topic segmentation** — uses explicit lecture headings when present; otherwise applies TF-IDF-based segmentation.
5. **Notes and revision** — selects evidence-backed sentences and builds topic-wise notes and a compact revision view.
6. **Visualisation** — NetworkX and Matplotlib render the concept map; Flask serves the interface and downloads.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Recommended for better phrase detection
python -m spacy download en_core_web_sm

python app.py
```

Open `http://127.0.0.1:5000`.

If PowerShell blocks virtual-environment activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\\Scripts\\Activate.ps1
```

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # recommended, optional
python app.py
```

## Audio setup and verification

Audio processing uses the CPU by default and downloads the selected Whisper model on first use. See [AUDIO_DEMO.md](AUDIO_DEMO.md) for model settings, a Windows demo workflow, and the distinction between mocked audio tests and real speech-recognition verification.

Useful settings:

| Variable | Default | Purpose |
|---|---:|---|
| `WHISPER_MODEL` | `base` | Whisper model name or local model directory |
| `WHISPER_DEVICE` | `cpu` | Inference device |
| `WHISPER_COMPUTE` | `int8` | Compute type |
| `WHISPER_BEAM_SIZE` | `5` | Beam-search width (1–10) |
| `WHISPER_LANGUAGE` | auto | Optional language override |
| `WHISPER_INITIAL_PROMPT` | empty | Optional lecture-specific vocabulary |
| `LN_MAX_UPLOAD_MB` | `200` | Maximum uploaded file size |
| `LN_MAX_AUDIO_MINUTES` | `180` | Maximum audio duration |

For English technical lectures, `base.en` or `small.en` may be useful model choices. Review the recognised transcript because speech recognition can still mishear technical terms.

## Tests

```bash
pytest -q
```

The automated suite covers text processing, concept extraction, topic segmentation, terminology corrections, graph generation, Flask routes, and mocked/stubbed audio workflows. Real Whisper tests require `faster-whisper`, a downloaded model, and the demo audio; check `AUDIO_DEMO.md` and ensure the real-audio test is **passed**, not skipped, before claiming end-to-end audio verification.

Evaluation datasets and the evaluation protocol are in [`evaluation/`](evaluation/). Gold annotations are drafts and should be human-reviewed before using evaluation scores as research claims.

## Project structure

```text
lecture-visual-notes/
├── app.py
├── lecturenotes/              # NLP, extraction, topics, notes, graph, transcription
│   └── terminology_packs/     # Optional subject-specific correction rules
├── templates/                 # Flask HTML
├── static/                    # CSS and JavaScript
├── samples/                   # Sample lecture text and demo recordings
├── evaluation/                # Dev/held-out datasets and evaluation scripts
├── tests/                     # Automated tests
├── requirements.txt
└── README.md
```

## Current limitations

- The notes pipeline is designed for English; other detected languages may produce poor results.
- Relationship extraction recognises supported linguistic patterns; unusual phrasing can be missed or misinterpreted.
- Heavy background noise, accents, and technical vocabulary can reduce transcription quality.
- Results are kept in memory (latest 25); restarting the server clears them.
- Audio processing runs synchronously and can take several minutes for long recordings.
- This is a study aid, not a substitute for checking the original lecture or course material.

## Resume description

**AI-Based Lecture-to-Visual Notes System** — Built a Flask-based application that converts lecture transcripts and audio into topic-wise study notes, quick-revision summaries, and concept maps. Implemented NLP-based concept/definition extraction, TF-IDF topic segmentation, local speech-to-text using faster-whisper, evidence-linked outputs, and PNG/SVG/JSON exports.

Only claim real-audio end-to-end verification after the real Whisper test has passed on your machine.
