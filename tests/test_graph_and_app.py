import json
from pathlib import Path
from unittest import mock

from app import create_app
from lecturenotes import transcribe as asr
from lecturenotes.graph import render
from lecturenotes.pipeline import process

SAMPLE = (Path(__file__).parent.parent / "samples" / "sample_lecture.txt").read_text(encoding="utf-8")


def test_graph_selection_is_consistent():
    g = process(SAMPLE)["graph"]
    ids = {n["id"] for n in g["nodes"]}
    assert 3 <= len(ids) <= 14
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])


def test_render_png_and_svg():
    r = process(SAMPLE)
    png, svg = render(r["graph"], r["topics"], "png"), render(r["graph"], r["topics"], "svg")
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 10_000
    assert b"<svg" in svg[:600] and b"</svg>" in svg[-20:]


def test_render_empty_graph_does_not_crash():
    assert render({"nodes": [], "edges": []}, [], "png")[:4] == b"\x89PNG"


def test_index_and_sample():
    c = create_app().test_client()
    assert c.get("/").status_code == 200 and b"Generate notes" in c.get("/").data
    assert "neural network" in c.get("/api/sample").get_json()["text"].lower()


def test_process_and_downloads():
    c = create_app().test_client()
    r = c.post("/api/process", data={"text": SAMPLE})
    assert r.status_code == 200
    j = r.get_json()
    assert {"notes", "quick_revision", "concepts", "relations", "graph", "id"} <= set(j)
    base = f"/files/{j['id']}/"
    assert json.loads(c.get(base + "notes.json").data)["meta"]["topics"] == j["meta"]["topics"]
    assert c.get(base + "concept-map.png").data[:4] == b"\x89PNG"
    assert b"<svg" in c.get(base + "concept-map.svg").data[:600]
    dl = c.get(base + "concept-map.png?download=1")
    assert "attachment" in dl.headers["Content-Disposition"]
    assert c.get(base + "evil.txt").status_code == 404
    assert c.get("/files/nope/notes.json").status_code == 404


def test_input_errors():
    c = create_app().test_client()
    assert c.post("/api/process", data={"text": "  "}).status_code == 400
    assert c.post("/api/process", data={"text": "Too short."}).status_code == 400
    import io
    bad = c.post("/api/process", data={"file": (io.BytesIO(b"x"), "notes.pdf")}, content_type="multipart/form-data")
    assert bad.status_code == 400 and "Unsupported" in bad.get_json()["error"]


def test_text_file_upload():
    import io
    c = create_app().test_client()
    r = c.post("/api/process", data={"file": (io.BytesIO(SAMPLE.encode()), "lecture.txt")}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["meta"]["source"] == "text"


def test_audio_route_with_mocked_transcriber():
    import io
    segs = [dict(text=s.strip().rstrip(".") + ".", start=i * 4.0, end=i * 4.0 + 4) for i, s in enumerate(SAMPLE.replace("\n", " ").split(". "))]
    c = create_app().test_client()
    with mock.patch.object(asr, "transcribe", return_value=segs) as m:
        r = c.post("/api/process", data={"file": (io.BytesIO(b"RIFF"), "lecture.wav")}, content_type="multipart/form-data")
    assert m.called and r.status_code == 200 and r.get_json()["meta"]["source"] == "audio"


def test_audio_without_whisper_gives_clear_error():
    import io
    c = create_app().test_client()
    with mock.patch.object(asr, "transcribe", side_effect=asr.TranscriptionUnavailable("faster-whisper is not installed")):
        r = c.post("/api/process", data={"file": (io.BytesIO(b"RIFF"), "a.mp3")}, content_type="multipart/form-data")
    assert r.status_code == 501 and "faster-whisper" in r.get_json()["error"]
