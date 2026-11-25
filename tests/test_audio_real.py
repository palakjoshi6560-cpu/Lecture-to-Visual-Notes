"""REAL speech-to-text tests. They are SKIPPED unless (1) faster-whisper is installed and (2) demo audio exists.

Create demo audio on Windows with:  powershell -ExecutionPolicy Bypass -File samples\\make_demo_audio.ps1
(or put your own  samples/demo_audio/<name>.wav  next to  <name>.txt  containing the words that are spoken).
The first run downloads the Whisper model (internet needed)."""
import importlib.util
import io
import unittest
from pathlib import Path

from app import create_app
from lecturenotes import transcribe as asr
from lecturenotes.pipeline import process

ROOT = Path(__file__).parent.parent


def _pairs():
    d = ROOT / "samples" / "demo_audio"
    return [(a, a.with_suffix(".txt")) for a in sorted(d.glob("*")) if a.suffix.lower() in asr.AUDIO_EXTS and a.with_suffix(".txt").exists()] if d.exists() else []


def test_real_speech_is_transcribed_and_becomes_notes_and_a_concept_map():
    if importlib.util.find_spec("faster_whisper") is None:
        raise unittest.SkipTest("faster-whisper is not installed")
    pairs = _pairs()
    if not pairs:
        raise unittest.SkipTest("no demo audio in samples/demo_audio (see AUDIO_DEMO.md)")
    for audio, ref in pairs:
        segs = asr.transcribe(str(audio))
        heard = " ".join(s["text"] for s in segs)
        recall = asr.word_recall(ref.read_text(encoding="utf-8"), heard)
        assert recall >= 0.7, f"{audio.name}: only {recall:.0%} of the spoken words were recognised"
        assert segs and segs.info["language"] == "en" and all(s["end"] > s["start"] for s in segs)
        r = process(segments=segs, source="audio")
        assert r["notes"] and len(r["concepts"]) >= 3 and r["graph"]["nodes"] and all(t["time"] for t in r["transcript"])
        # and once more through the real web route with the real file
        c = create_app().test_client()
        resp = c.post("/api/process", data={"file": (io.BytesIO(audio.read_bytes()), audio.name)}, content_type="multipart/form-data")
        assert resp.status_code == 200 and resp.get_json()["meta"]["source"] == "audio"
