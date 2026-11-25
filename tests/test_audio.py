"""Audio-input tests, in three clearly separated kinds:

  test_mock_*       the speech-to-text call is REPLACED by a function that returns Whisper-style segments
                    (upload route, validation, timestamps, hand-over to the notes pipeline, exports).
  test_stubengine_* a FAKE faster_whisper module exercises OUR wrapper (settings, segments, errors, locking, status).
  test_real_*       in tests/test_audio_real.py: REAL speech-to-text; skipped unless faster-whisper and demo audio exist.

Nothing here proves that Whisper itself transcribes well; that is what test_audio_real.py and the Windows steps in AUDIO_DEMO.md are for."""
import io
import math
import os
import struct
import sys
import tempfile
import threading
import types
import wave
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from app import create_app
from lecturenotes import transcribe as asr
from lecturenotes.pipeline import process

ROOT = Path(__file__).parent.parent
LECTURES = {"biology": (ROOT / "evaluation" / "data" / "dev" / "biology_plain.txt").read_text(encoding="utf-8"),
            "machine learning": (ROOT / "samples" / "sample_lecture.txt").read_text(encoding="utf-8")}


# ---------------------------------------------------------------- helpers
def wav_bytes(seconds=2.0, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate))))
    return buf.getvalue()


def pseudo_whisper(text, words=14, seconds=3.0, language="en"):
    """Segments shaped like Whisper output: leading space, cut in the middle of sentences, increasing times."""
    w = text.split()
    segs = [dict(text=" " + " ".join(w[i:i + words]), start=k * seconds, end=(k + 1) * seconds) for k, i in enumerate(range(0, len(w), words))]
    return asr.Segments(segs, dict(language=language, language_probability=0.99, duration_seconds=len(segs) * seconds, model="base"))


def upload(client, name, data):
    return client.post("/api/process", data={"file": (io.BytesIO(data), name)}, content_type="multipart/form-data")


# ================================================================ MOCK tests (speech-to-text replaced)
def test_mock_upload_validation_returns_clear_json_errors():
    c = create_app().test_client()
    with mock.patch.object(asr, "transcribe") as t:
        r = upload(c, "empty.wav", b"")
        assert r.status_code == 400 and "empty" in r.get_json()["error"].lower()
        r = upload(c, "slides.pdf", b"%PDF")
        assert r.status_code == 400 and "Unsupported" in r.get_json()["error"]
        r = c.post("/api/process", data={})
        assert r.status_code == 400 and "Paste a transcript" in r.get_json()["error"]
        assert not t.called  # invalid uploads never reach the transcriber
    app = create_app()
    app.config["MAX_CONTENT_LENGTH"] = 2000
    r = upload(app.test_client(), "big.wav", b"\0" * 10_000)
    assert r.status_code == 413 and r.is_json and "too large" in r.get_json()["error"]  # JSON, not an HTML page


def test_mock_every_supported_extension_is_accepted_in_any_letter_case():
    c, seen = create_app().test_client(), []
    segs = pseudo_whisper(LECTURES["biology"])

    def fake(path):
        seen.append(Path(path).suffix.lower())
        return segs

    with mock.patch.object(asr, "transcribe", side_effect=fake):
        for ext in sorted(asr.AUDIO_EXTS):
            for name in ("lecture" + ext, "LECTURE" + ext.upper()):
                r = upload(c, name, b"audio-bytes")
                assert r.status_code == 200 and r.get_json()["meta"]["source"] == "audio", name
    assert sorted(set(seen)) == sorted(asr.AUDIO_EXTS)


def test_mock_uploaded_bytes_reach_the_transcriber_intact_and_the_temp_file_is_deleted():
    data, seen = wav_bytes(2.0), {}

    def fake(path):
        seen.update(path=path, exists=os.path.exists(path), same=Path(path).read_bytes() == data)
        with wave.open(path) as w:
            seen["frames"] = w.getnframes()
        return pseudo_whisper(LECTURES["biology"])

    with mock.patch.object(asr, "transcribe", side_effect=fake):
        assert upload(create_app().test_client(), "tone.wav", data).status_code == 200
    assert seen["exists"] and seen["same"] and seen["frames"] == 32000
    assert not os.path.exists(seen["path"])


def test_mock_transcriber_failures_become_json_errors_and_leave_no_temp_files():
    paths = []
    for exc, code in [(RuntimeError("decoder exploded"), 500), (asr.InvalidAudio("not audio"), 422),
                      (asr.TranscriptionUnavailable("faster-whisper is not installed"), 501)]:
        def fake(path, exc=exc):
            paths.append(path)
            raise exc
        with mock.patch.object(asr, "transcribe", side_effect=fake):
            r = upload(create_app().test_client(), "x.mp3", b"bytes")
        assert r.status_code == code and r.is_json and r.get_json()["error"], (code, r.data[:80])
        assert not os.path.exists(paths[-1])
    with mock.patch.object(asr, "transcribe", return_value=asr.Segments([])):
        r = upload(create_app().test_client(), "silence.wav", b"bytes")
    assert r.status_code == 400 and "No speech" in r.get_json()["error"]


def test_mock_whisper_style_segments_flow_through_the_whole_pipeline_for_different_subjects():
    for subject, text in LECTURES.items():
        segs = pseudo_whisper(text)
        c = create_app().test_client()
        with mock.patch.object(asr, "transcribe", return_value=segs):
            r = upload(c, "lecture.wav", wav_bytes(1.0))
        assert r.status_code == 200, subject
        j = r.get_json()
        assert j["meta"]["source"] == "audio" and j["meta"]["audio"]["language"] == "en" and j["meta"]["audio"]["segments"] == len(segs)
        times = [t["time"] for t in j["transcript"]]
        assert all(times) and times == sorted(times) and times[0] == "00:00", subject  # every sentence is timestamped, in order
        sent_time = {t["sentence"]: t["time"] for t in j["transcript"]}
        bullets = [b for n in j["notes"] for b in n["bullets"]]
        assert bullets and all(b["time"] == sent_time[b["sentence"]] for b in bullets), subject  # notes point to when it was said
        assert len(j["concepts"]) >= 3 and any(x["definition"] for x in j["concepts"]), subject
        assert j["quick_revision"]["terms"] and j["graph"]["nodes"] and j["graph"]["edges"], subject
        base = f"/files/{j['id']}/"
        assert c.get(base + "concept-map.png").data[:4] == b"\x89PNG" and b"<svg" in c.get(base + "concept-map.svg").data[:600]
        assert c.get(base + "notes.json").status_code == 200
        txt = c.get(base + "transcript.txt").data.decode("utf-8").splitlines()
        assert txt[0].startswith("[00:00] ") and len(txt) == len(j["transcript"]), subject
        # the audio path must give the same sentences as pasting the same words as text
        assert [s["text"] for s in process(text=text)["transcript"]] == [s["text"] for s in j["transcript"]], subject


def test_mock_a_sentence_cut_across_segments_is_timed_where_it_starts():
    segs = [dict(text=" Gradient descent is an optimization", start=0.0, end=4.0),
            dict(text=" algorithm that minimizes the loss function. The learning", start=4.0, end=9.0),
            dict(text=" rate controls the size of each step. A large learning rate makes the loss function diverge.", start=9.0, end=15.0),
            dict(text=" Gradient descent updates every weight using the loss function gradient.", start=15.0, end=21.0)]
    r = process(segments=segs, source="audio")
    at = lambda prefix: next(s["time"] for s in r["transcript"] if s["text"].startswith(prefix))
    assert at("Gradient descent is an") == "00:00"  # starts in segment 1
    assert at("The learning rate controls") == "00:04"  # starts at the end of segment 2, finishes in segment 3
    assert at("A large learning rate") == "00:09" and at("Gradient descent updates") == "00:15"


def test_mock_non_english_audio_gets_a_visible_warning():
    with mock.patch.object(asr, "transcribe", return_value=pseudo_whisper(LECTURES["biology"], language="hi")):
        j = upload(create_app().test_client(), "lecture.wav", b"bytes").get_json()
    assert "English" in " ".join(j["meta"]["warnings"]) and j["meta"]["audio"]["language"] == "hi"


def test_mock_audio_status_route_is_json_even_without_faster_whisper():
    c = create_app().test_client()
    with mock.patch.dict(sys.modules, {"faster_whisper": None}):
        j = c.get("/api/audio-status").get_json()
    assert j["available"] is False and "not installed" in j["note"] and ".wav" in j["formats"] and j["max_upload_mb"] > 0


# ================================================================ STUB-ENGINE tests (fake faster_whisper module)
class _Seg:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


class _Info:
    def __init__(self, language="en", probability=0.97):
        self.language, self.language_probability = language, probability


class _Model:
    inits, calls, script, delay = [], [], None, 0.0

    def __init__(self, name, device=None, compute_type=None):
        _Model.inits.append((name, device, compute_type))

    def transcribe(self, audio, **kw):
        _Model.calls.append((len(audio), kw))
        if _Model.delay:
            import time
            time.sleep(_Model.delay)
        segs = _Model.script or [_Seg(" Hello there. ", 0.0, 2.5), _Seg("   ", 2.5, 3.0), _Seg(" Second part.", 3.0, 6.0)]
        return iter(segs), _Info()


def _decode(path, sampling_rate=16000):
    import numpy as np
    with wave.open(path) as w:  # raises wave.Error for anything that is not a WAV: stands in for a decoder failure
        return np.zeros(int(w.getnframes() * sampling_rate / w.getframerate()), dtype="float32")


@contextmanager
def stub_engine(cached=(), **env):
    fw = types.ModuleType("faster_whisper")
    fw.WhisperModel, fw.decode_audio = _Model, _decode
    utils = types.ModuleType("faster_whisper.utils")

    def download_model(name, local_files_only=False):
        if local_files_only and name not in cached:
            raise RuntimeError("not cached")
    utils.download_model = download_model
    saved_env = {k: os.environ.get(k) for k in ("WHISPER_MODEL", "WHISPER_DEVICE", "WHISPER_COMPUTE", "WHISPER_LANGUAGE", "LN_MAX_AUDIO_MINUTES")}
    for k in saved_env:
        os.environ.pop(k, None)
    os.environ.update(env)
    _Model.inits, _Model.calls, _Model.script, _Model.delay, old = [], [], None, 0.0, asr._MODEL
    asr._MODEL = None
    try:
        with mock.patch.dict(sys.modules, {"faster_whisper": fw, "faster_whisper.utils": utils}):
            yield
    finally:
        asr._MODEL = old
        for k, v in saved_env.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def _wav_file(seconds):
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(wav_bytes(seconds))
    f.close()
    return f.name


def test_stubengine_wrapper_cleans_segments_and_passes_settings_to_the_engine():
    path = _wav_file(3.0)
    try:
        with stub_engine(WHISPER_MODEL="tiny.en", WHISPER_LANGUAGE="en", WHISPER_DEVICE="cpu", WHISPER_COMPUTE="int8"):
            segs = asr.transcribe(path)
            assert isinstance(segs, asr.Segments) and [s["text"] for s in segs] == ["Hello there.", "Second part."]  # stripped, empty dropped
            assert segs[1]["start"] == 3.0 and segs[1]["end"] == 6.0
            assert segs.info == dict(language="en", language_probability=0.97, duration_seconds=3.0, model="tiny.en")
            assert _Model.inits == [("tiny.en", "cpu", "int8")]
            n, kw = _Model.calls[0]
            assert n == 48000 and kw["language"] == "en" and kw["beam_size"] == 5 and kw["vad_filter"] is True
    finally:
        os.unlink(path)



def test_stubengine_uses_optional_technical_vocabulary_prompt():
    path = _wav_file(3.0)
    try:
        with stub_engine(WHISPER_INITIAL_PROMPT="machine learning, gradient descent"):
            asr.transcribe(path)
            _n, kw = _Model.calls[0]
            assert kw["beam_size"] == 5
            assert kw["initial_prompt"] == "machine learning, gradient descent"
    finally:
        os.unlink(path)

def test_stubengine_simultaneous_uploads_are_serialised_and_load_the_model_once():
    text = LECTURES["biology"].split()
    app, codes = create_app(), []
    with stub_engine():
        _Model.script = [_Seg(" " + " ".join(text[i:i + 14]), 3.0 * k, 3.0 * (k + 1)) for k, i in enumerate(range(0, len(text), 14))]
        _Model.delay = 0.05
        post = lambda: codes.append(upload(app.test_client(), "lecture.wav", wav_bytes(1.0)).status_code)
        ts = [threading.Thread(target=post) for _ in range(4)]
        [t.start() for t in ts], [t.join() for t in ts]
        assert codes == [200] * 4 and len(_Model.inits) == 1 and len(_Model.calls) == 4


def test_stubengine_rejects_undecodable_too_short_and_too_long_audio():
    good, short = _wav_file(3.0), _wav_file(0.5)
    junk = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    junk.write(b"this is not audio")
    junk.close()
    try:
        with stub_engine():
            for path, word in [(junk.name, "decode"), (short, "shorter")]:
                try:
                    asr.transcribe(path)
                except asr.InvalidAudio as e:
                    assert word in str(e)
                else:
                    raise AssertionError(word)
        with stub_engine(LN_MAX_AUDIO_MINUTES="0.01"):
            try:
                asr.transcribe(good)
            except asr.InvalidAudio as e:
                assert "limit" in str(e) and "LN_MAX_AUDIO_MINUTES" in str(e)
            else:
                raise AssertionError("too long")
    finally:
        for p in (good, short, junk.name):
            os.unlink(p)


def test_stubengine_status_distinguishes_ready_and_model_not_downloaded():
    with stub_engine(cached=()):
        s = asr.status()
        assert s["available"] and not s["cached"] and s["beam_size"] == 5 and "downloads it" in s["note"]
    with stub_engine(cached=("base",)):
        s = asr.status()
        assert s["available"] and s["cached"] and s["note"] is None and s["model"] == "base"


def test_stubengine_real_wav_upload_through_the_unmocked_route_end_to_end():
    text = LECTURES["biology"].split()
    with stub_engine(WHISPER_MODEL="base.en"):
        _Model.script = [_Seg(" " + " ".join(text[i:i + 14]), 3.0 * k, 3.0 * (k + 1)) for k, i in enumerate(range(0, len(text), 14))]
        j = upload(create_app().test_client(), "lecture.wav", wav_bytes(2.0)).get_json()  # decoder + wrapper + route + pipeline, only the model is fake
    assert j["meta"]["source"] == "audio" and j["meta"]["audio"]["duration_seconds"] == 2.0 and j["meta"]["audio"]["model"] == "base.en"
    assert j["concepts"] and j["notes"] and j["quick_revision"]["terms"]


def test_stubengine_corrupt_file_with_audio_extension_is_a_clean_422():
    with stub_engine():
        r = upload(create_app().test_client(), "broken.wav", b"this is not a wav file at all")
    assert r.status_code == 422 and "decode" in r.get_json()["error"]


def test_missing_faster_whisper_gives_a_clear_501_not_a_crash():
    with mock.patch.dict(sys.modules, {"faster_whisper": None}):
        r = upload(create_app().test_client(), "lecture.wav", wav_bytes(1.0))
    assert r.status_code == 501 and "pip install faster-whisper" in r.get_json()["error"]


def test_word_recall_is_case_and_punctuation_insensitive_and_counts_repeats():
    assert asr.word_recall("The cell, the CELL!", "the cell the cell") == 1.0
    assert asr.word_recall("a b c d", "a b") == 0.5
    assert asr.word_recall("go go go", "go") == 1 / 3
