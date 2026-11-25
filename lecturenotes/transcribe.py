"""Audio -> timed segments with faster-whisper (loaded lazily, CPU int8 by default).

Settings (environment variables):
  WHISPER_MODEL         default "base" ("tiny.en"/"base.en" are faster for English, "small" is more accurate; a local folder also works)
  WHISPER_DEVICE        default "cpu"            WHISPER_COMPUTE   default "int8"
  WHISPER_LANGUAGE      default: auto-detect     WHISPER_BEAM_SIZE     default 5
  WHISPER_INITIAL_PROMPT optional subject vocabulary / technical terms to guide recognition
  LN_MAX_AUDIO_MINUTES  default 180
The first use downloads the model (internet needed once). Pre-download it with:  python -m lecturenotes.transcribe --prepare
Check a recording without the web app:  python -m lecturenotes.transcribe lecture.wav [--reference lecture.txt]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import Counter

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".mp4", ".aac"}
SAMPLE_RATE = 16000
_MODEL = None  # loaded on first use. Not safe for simultaneous calls: the web app (app.py) serialises them with one lock.


class TranscriptionUnavailable(RuntimeError):
    """faster-whisper (or its model) cannot be used in this environment."""


class InvalidAudio(ValueError):
    """The uploaded file is not usable audio (corrupt, wrong format, too short or too long)."""


class Segments(list):
    """[{'text','start','end'}, ...] plus .info (language, duration_seconds, model)."""

    def __init__(self, items=(), info=None):
        super().__init__(items)
        self.info = dict(info or {})


def settings():
    return dict(model=os.environ.get("WHISPER_MODEL", "base"), device=os.environ.get("WHISPER_DEVICE", "cpu"),
                compute_type=os.environ.get("WHISPER_COMPUTE", "int8"), language=os.environ.get("WHISPER_LANGUAGE") or None,
                beam_size=max(1, min(10, int(os.environ.get("WHISPER_BEAM_SIZE", "5")))),
                initial_prompt=os.environ.get("WHISPER_INITIAL_PROMPT", "").strip() or None,
                max_minutes=float(os.environ.get("LN_MAX_AUDIO_MINUTES", "180")))


def _faster_whisper():
    try:
        import faster_whisper
    except ImportError as e:
        raise TranscriptionUnavailable("faster-whisper is not installed. Run: pip install faster-whisper") from e
    except Exception as e:  # e.g. a missing system DLL on Windows
        raise TranscriptionUnavailable(f"faster-whisper is installed but could not be loaded: {type(e).__name__}: {e}") from e
    return faster_whisper


def _cached(name):
    if os.path.isdir(name):
        return True
    try:
        from faster_whisper.utils import download_model
        download_model(name, local_files_only=True)
        return True
    except Exception:
        return False


def status():
    """Readiness report for the UI and for demos. Never raises and never downloads anything."""
    s = settings()
    base = {k: s[k] for k in ("model", "device", "compute_type", "language", "beam_size")}
    try:
        _faster_whisper()
    except TranscriptionUnavailable as e:
        return dict(base, available=False, loaded=False, cached=False, note=str(e))
    cached = _cached(s["model"])
    note = None if cached or _MODEL is not None else f"Model '{s['model']}' is not downloaded yet: the first transcription downloads it (internet needed)."
    return dict(base, available=True, loaded=_MODEL is not None, cached=cached, note=note)


def _load_model():
    global _MODEL
    if _MODEL is None:
        fw, s = _faster_whisper(), settings()
        try:
            _MODEL = fw.WhisperModel(s["model"], device=s["device"], compute_type=s["compute_type"])
        except Exception as e:
            raise TranscriptionUnavailable(
                f"Could not load the Whisper model '{s['model']}': {e}. The first run needs internet to download it "
                f"(run: python -m lecturenotes.transcribe --prepare).") from e
    return _MODEL


def transcribe(path: str, initial_prompt: str | None = None):
    """Return timed Segments. An optional glossary prompt can improve technical-term recognition."""
    fw = _faster_whisper()
    model, s = _load_model(), settings()
    try:
        decode = getattr(fw, "decode_audio", None) or __import__("faster_whisper.audio", fromlist=["decode_audio"]).decode_audio
        audio = decode(path, sampling_rate=SAMPLE_RATE)
    except Exception as e:
        raise InvalidAudio(f"Could not decode this audio file ({type(e).__name__}). Is it a real audio recording?") from e
    duration = len(audio) / SAMPLE_RATE
    if duration < 1.0:
        raise InvalidAudio("The recording is shorter than one second.")
    if duration > s["max_minutes"] * 60:
        raise InvalidAudio(f"The recording is {duration / 60:.0f} minutes long; the limit is {s['max_minutes']:.0f} minutes "
                           f"(set LN_MAX_AUDIO_MINUTES to change it).")
    prompt = (initial_prompt or s["initial_prompt"] or "").strip()[:1000] or None
    kwargs = dict(language=s["language"], beam_size=s["beam_size"], vad_filter=True)
    if prompt:
        kwargs["initial_prompt"] = prompt
    raw, info = model.transcribe(audio, **kwargs)
    out = [dict(text=x.text.strip(), start=float(x.start), end=float(x.end)) for x in raw if x.text.strip()]
    return Segments(out, dict(language=getattr(info, "language", None), language_probability=getattr(info, "language_probability", None),
                              duration_seconds=round(duration, 1), model=s["model"]))


def word_recall(reference: str, heard: str) -> float:
    """Share of the reference's words (as a multiset) found in the recognised text: a simple, transparent speech-to-text check."""
    tokens = lambda t: Counter(re.findall(r"[a-z0-9]+", t.lower()))
    ref, hyp = tokens(reference), tokens(heard)
    return sum(min(c, hyp[w]) for w, c in ref.items()) / max(1, sum(ref.values()))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check speech-to-text outside the web app.")
    ap.add_argument("audio", nargs="?", help="audio file to transcribe and run through the notes pipeline")
    ap.add_argument("--prepare", action="store_true", help="download/load the Whisper model now")
    ap.add_argument("--reference", help="text file with what is spoken in the recording (reports word recall)")
    a = ap.parse_args(argv)
    try:
        print("settings:", settings())
        if a.prepare or a.audio:
            t = time.time()
            _load_model()
            print(f"model ready ({time.time() - t:.1f}s)")
        if a.audio:
            t = time.time()
            segs = transcribe(a.audio)
            print(f"transcribed in {time.time() - t:.1f}s: {len(segs)} segments, info={segs.info}")
            for s in segs[:5]:
                print(f"  [{s['start']:7.1f}s] {s['text']}")
            if a.reference:
                ref = open(a.reference, encoding="utf-8").read()
                print(f"word recall vs reference: {word_recall(ref, ' '.join(s['text'] for s in segs)):.2f}")
            from .pipeline import process
            r = process(segments=segs, source="audio")
            print("pipeline:", r["meta"], "| first topics:", [t["title"] for t in r["topics"][:4]])
        elif not a.prepare:
            ap.print_help()
    except (TranscriptionUnavailable, InvalidAudio) as e:
        print("ERROR:", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
