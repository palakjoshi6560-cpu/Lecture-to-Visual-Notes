import sys
import threading
from types import SimpleNamespace

import pytest

from app import validate_lecture_url, parse_vtt_segments, fetch_link_segments


def test_accepts_public_http_video_urls():
    assert validate_lecture_url("https://vimeo.com/123456") == "https://vimeo.com/123456"
    assert validate_lecture_url("https://www.youtube.com/watch?v=abcdefghijk").startswith("https://")


@pytest.mark.parametrize("url", [
    "", "file:///etc/passwd", "ftp://example.com/video", "http://localhost/a",
    "http://127.0.0.1/video", "http://192.168.1.2/video", "http://[::1]/video",
    "https://user:pass@youtube.com/watch?v=abcdefghijk",
])
def test_rejects_invalid_or_local_urls(url):
    with pytest.raises(ValueError):
        validate_lecture_url(url)


def test_parses_webvtt_timestamps_and_text(tmp_path):
    p = tmp_path / "captions.en.vtt"
    p.write_text(
        "WEBVTT\r\n\r\n"
        "00:00:01.000 --> 00:00:03.500 align:start position:0%\r\n"
        "Hello <b>world</b>.\r\n\r\n"
        "00:00:04.000 --> 00:00:05.000\r\nSecond line.\r\n",
        encoding="utf-8",
    )
    got = parse_vtt_segments(str(p))
    assert got == [
        {"text": "Hello world.", "start": 1.0, "end": 3.5},
        {"text": "Second line.", "start": 4.0, "end": 5.0},
    ]


def test_parses_minute_only_timestamp_and_multiline_cue(tmp_path):
    p = tmp_path / "captions.vtt"
    p.write_text(
        "WEBVTT\n\n"
        "00:01.250 --> 00:03.500\n"
        "First <i>caption</i> line\nsecond caption line &amp; more\n\n"
        "01:00:00.000 --> 01:00:02.000\nOne hour mark.\n",
        encoding="utf-8",
    )
    got = parse_vtt_segments(str(p))
    assert got == [
        {"text": "First caption line second caption line & more", "start": 1.25, "end": 3.5},
        {"text": "One hour mark.", "start": 3600.0, "end": 3602.0},
    ]


def test_link_fetch_prefers_captions_without_loading_whisper(tmp_path, monkeypatch):
    class FakeYoutubeDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            template = self.options["outtmpl"]
            path = template.replace("%(id)s", "lecture").replace("%(ext)s", "en.vtt")
            if self.options.get("writesubtitles") or self.options.get("writeautomaticsub"):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nDatabase is organized data.\n")
            return {"title": "Demo lecture", "extractor_key": "Youtube"}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    import app
    monkeypatch.setattr(app.asr, "transcribe", lambda *_: pytest.fail("Whisper should not run when captions exist"))
    segments, info = fetch_link_segments("https://www.youtube.com/watch?v=abcdefghijk", threading.Lock())
    assert segments == [{"text": "Database is organized data.", "start": 1.0, "end": 3.0}]
    assert info["method"] == "captions"
    assert info["title"] == "Demo lecture"


def test_link_fetch_uses_whisper_when_no_captions_exist(tmp_path, monkeypatch):
    class FakeYoutubeDL:
        calls = 0

        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def extract_info(self, url, download=False):
            type(self).calls += 1
            if self.options.get("format"):
                path = self.options["outtmpl"].replace("%(id)s", "lecture").replace("%(ext)s", "webm")
                with open(path, "wb") as f:
                    f.write(b"mock audio bytes")
            return {"title": "Lecture without captions", "extractor_key": "Youtube"}

    monkeypatch.setitem(sys.modules, "yt_dlp", SimpleNamespace(YoutubeDL=FakeYoutubeDL))
    import app
    fake_segments = app.asr.Segments(
        [{"text": "A database stores related data.", "start": 0.0, "end": 3.0}],
        info={"language": "en", "duration_seconds": 3.0, "model": "tiny.en"},
    )
    monkeypatch.setattr(app.asr, "transcribe", lambda path: fake_segments)
    segments, info = fetch_link_segments("https://www.youtube.com/watch?v=abcdefghijk", threading.Lock())
    assert segments == fake_segments
    assert info["method"] == "whisper_audio"
    assert info["language"] == "en"
    assert FakeYoutubeDL.calls == 2


def test_caption_segments_enter_common_pipeline_and_keep_timestamps():
    from lecturenotes.pipeline import process

    segments = [
        {"text": "A database is an organized collection of related data.", "start": 1.0, "end": 4.0},
        {"text": "A database management system manages stored data.", "start": 4.0, "end": 7.0},
        {"text": "A table contains rows and columns.", "start": 7.0, "end": 10.0},
        {"text": "A primary key uniquely identifies each row.", "start": 10.0, "end": 13.0},
        {"text": "Normalization reduces unnecessary duplication.", "start": 13.0, "end": 16.0},
        {"text": "SQL is used to query relational data.", "start": 16.0, "end": 19.0},
    ]
    result = process(segments=segments, source="link")
    assert result["meta"]["source"] == "link"
    assert result["transcript"][0]["time"] == "00:01"
    assert result["transcript"][1]["time"] == "00:04"
    assert result["notes"]
    assert result["graph"] is not None
