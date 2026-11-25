"""Flask app: upload transcript/audio -> notes + concept map; download PNG/SVG/JSON."""
from __future__ import annotations

import io
import json
import ipaddress
import re
import socket
import html
import os
import tempfile
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

from flask import Flask, abort, jsonify, render_template, request, send_file

from lecturenotes import transcribe as asr
from lecturenotes.graph import render
from lecturenotes.pipeline import process

ROOT = Path(__file__).parent
TEXT_EXTS = {".txt", ".md", ""}
MIMES = {"png": "image/png", "svg": "image/svg+xml"}



def validate_lecture_url(url: str) -> str:
    """Accept public HTTP(S) lecture URLs; reject local/private network targets."""
    raw = (url or "").strip()
    if not raw or len(raw) > 2048:
        raise ValueError("Paste a valid public lecture URL.")
    parsed = urlparse(raw)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Use a public http:// or https:// lecture link.")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Links containing embedded usernames or passwords are not supported.")
    try:
        _ = parsed.port  # reject malformed ports before passing the URL to yt-dlp
    except ValueError as e:
        raise ValueError("The lecture link contains an invalid port.") from e
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        raise ValueError("Local or private network links are not supported.")
    try:
        ip = ipaddress.ip_address(host)
        if not ip.is_global:
            raise ValueError("Local or private network links are not supported.")
    except ValueError as e:
        if "not supported" in str(e):
            raise
        # A normal DNS hostname is allowed; yt-dlp resolves the provider.
    return raw


def parse_vtt_segments(path: str):
    """Read WebVTT captions into clean, timestamped Whisper-compatible segments.

    Handles cue settings, multi-line cues, HTML entities/tags and duplicate
    adjacent captions while preserving the original cue times.
    """
    content = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    time_re = re.compile(
        r"(?:(\d{1,}):)?(\d{2}):(\d{2})[.,](\d{3})\s+-->\s+"
        r"(?:(\d{1,}):)?(\d{2}):(\d{2})[.,](\d{3})(?:\s+[^\n]*)?"
    )

    def seconds(groups, offset):
        h = int(groups[offset] or 0)
        m, sec, ms = int(groups[offset + 1]), int(groups[offset + 2]), int(groups[offset + 3])
        return h * 3600 + m * 60 + sec + ms / 1000

    lines = content.splitlines()
    segments, i = [], 0
    while i < len(lines):
        match = time_re.search(lines[i])
        if not match:
            i += 1
            continue
        start, end = seconds(match.groups(), 0), seconds(match.groups(), 4)
        i += 1
        cue = []
        while i < len(lines) and lines[i].strip():
            raw = lines[i].strip()
            if not raw.startswith(("NOTE", "STYLE", "REGION")):
                raw = re.sub(r"<[^>]*>", "", raw)
                raw = html.unescape(raw)
                raw = raw.replace("\u200b", "").replace("\ufeff", "").strip()
                if raw:
                    cue.append(raw)
            i += 1
        text = re.sub(r"\s+", " ", " ".join(cue)).strip()
        # Do not collapse repeated spoken phrases: repetition may be meaningful.
        if text and end >= start:
            segments.append({"text": text, "start": start, "end": end})
        i += 1
    return segments


def fetch_link_segments(url: str, audio_lock):
    """Use yt-dlp supported extractors: captions first, then permitted public audio + Whisper."""
    url = validate_lecture_url(url)
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError("Link input support is not installed. Run: python -m pip install -r requirements.txt") from e

    with tempfile.TemporaryDirectory(prefix="lecture_link_") as tmpdir:
        common = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 30,
            "retries": 2,
            "extractor_retries": 2,
            "js_runtimes": {
                "node": {}
            },
            "remote_components": ["ejs:github"],
            "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        }
        info = {}
        try:
            caption_opts = dict(common, skip_download=True, writesubtitles=True,
                                writeautomaticsub=True, subtitleslangs=["en.*", "hi.*"],
                                subtitlesformat="vtt")
            with yt_dlp.YoutubeDL(caption_opts) as ydl:
                info = ydl.extract_info(url, download=True) or {}
            caption_files = list(Path(tmpdir).glob("*.vtt"))
            # Prefer English, then Hindi, and prefer non-"orig" tracks where
            # providers publish both a translated and original caption file.
            def caption_rank(path):
                stem = path.stem.lower()
                lang = stem.split(".")[-1]
                return (0 if lang.startswith("en") else 1 if lang.startswith("hi") else 2,
                        1 if "orig" in lang else 0, len(stem))
            for caption in sorted(caption_files, key=caption_rank):
                segments = parse_vtt_segments(str(caption))
                if segments:
                    return segments, {"title": info.get("title", ""), "extractor": info.get("extractor_key", ""),
                                      "method": "captions", "language": caption.stem.split(".")[-1]}
        except Exception as caption_error:
            # Continue to audio fallback only when the extractor can still provide media.
            info = info or {}

        try:
            audio_opts = dict(common, format="bestaudio/best", max_filesize=500 * 1024 * 1024)
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                info = ydl.extract_info(url, download=True) or info
            media_files = [p for p in Path(tmpdir).iterdir()
                           if p.is_file() and p.suffix.lower() in asr.AUDIO_EXTS]
            if not media_files:
                raise RuntimeError("The link did not provide accessible captions or a supported audio stream.")
            media = max(media_files, key=lambda p: p.stat().st_size)
            with audio_lock:
                technical_prompt = (
                    "Technical computer networking lecture. "
                    "OSI model, Open Systems Interconnection, physical layer, "
                    "data link layer, network layer, transport layer, session layer, "
                    "presentation layer, application layer, TCP, UDP, IP address, "
                    "MAC address, packet, protocol, source, destination. "
                    "Preserve English technical terms when spoken in Hindi-English lectures."
                )

                segments = asr.transcribe(
                    str(media),
                    initial_prompt=technical_prompt
                )
            if not segments:
                raise RuntimeError("No speech was detected in the linked lecture.")
            return segments, {"title": info.get("title", ""), "extractor": info.get("extractor_key", ""),
                              "method": "whisper_audio", "language": getattr(segments, "info", {}).get("language")}
        except (asr.TranscriptionUnavailable, asr.InvalidAudio):
            raise
        except Exception as e:
            raise RuntimeError(
                "Could not read this link. It may be unsupported, private, login-protected, geo-restricted, "
                "or have no accessible captions/audio. Try a public link or upload the audio/transcript. "
                f"Details: {type(e).__name__}"
            ) from e


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("LN_MAX_UPLOAD_MB", "200")) * 1024 * 1024
    store, lock = OrderedDict(), threading.Lock()  # in-memory results, newest 25 kept
    audio_lock = threading.Lock()  # speech-to-text is CPU bound and loads one shared model: one recording at a time

    def fail(msg, code=400):
        return jsonify(error=msg), code

    @app.errorhandler(413)
    def too_large(_e):  # JSON, so the page can show it instead of failing to parse an HTML error
        mb = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
        return fail(f"The upload is too large (limit {mb:g} MB). Use a shorter or compressed recording, or set LN_MAX_UPLOAD_MB.", 413)

    @app.get("/api/audio-status")
    def audio_status():
        return jsonify(dict(asr.status(), formats=sorted(asr.AUDIO_EXTS), max_upload_mb=app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)))

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/sample")
    def sample():
        return jsonify(text=(ROOT / "samples" / "sample_lecture.txt").read_text(encoding="utf-8"))

    @app.post("/api/process")
    def api_process():
        text, f = (request.form.get("text") or "").strip(), request.files.get("file")
        link_url = (request.form.get("link_url") or "").strip()
        try:
            if link_url:
                segments, link_info = fetch_link_segments(link_url, audio_lock)
                result = process(segments=segments, source="link")
                result["meta"]["link"] = dict(url=link_url, segments=len(segments), **link_info)
                result["meta"]["audio"] = {"segments": len(segments), "language": link_info.get("language", "unknown"),
                                           "duration_seconds": round(max((s["end"] for s in segments), default=0), 1)}
            elif f and f.filename:
                ext = Path(f.filename).suffix.lower()
                if ext in asr.AUDIO_EXTS:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                        f.save(tmp)
                    try:
                        if os.path.getsize(tmp.name) == 0:
                            return fail("The uploaded audio file is empty.")
                        with audio_lock:
                            segments = asr.transcribe(tmp.name)
                    except (asr.TranscriptionUnavailable, asr.InvalidAudio):
                        raise
                    except Exception as e:  # unexpected decoder/model failure: report it as JSON, not an HTML error page
                        app.logger.exception("transcription failed")
                        return fail(f"Transcription failed ({type(e).__name__}: {e}).", 500)
                    finally:
                        os.unlink(tmp.name)
                    if not segments:
                        return fail("No speech was detected in the audio file.")
                    result = process(segments=segments, source="audio")
                    info = dict(getattr(segments, "info", None) or {})
                    result["meta"]["audio"] = dict(filename=f.filename, segments=len(segments), **info)
                    if info.get("language") and info["language"] != "en":
                        result["meta"]["warnings"] = [f"Detected language '{info['language']}'. The notes pipeline is built for English, so results may be poor."]
                elif ext in TEXT_EXTS:
                    result = process(text=f.read().decode("utf-8", errors="replace"))
                else:
                    return fail(f"Unsupported file type '{ext}'. Use .txt, .md, or audio ({', '.join(sorted(asr.AUDIO_EXTS))}).")
            elif text:
                result = process(text=text)
            else:
                return fail("Paste a transcript or choose a file first.")
        except asr.TranscriptionUnavailable as e:
            return fail(str(e), 501)
        except asr.InvalidAudio as e:
            return fail(str(e), 422)
        except RuntimeError as e:
            return fail(str(e), 422)
        except ValueError as e:
            return fail(str(e))
        except Exception as e:
            app.logger.exception("lecture processing failed")
            return fail(f"Processing failed ({type(e).__name__}). Check the server log for details.", 500)
        rid = uuid.uuid4().hex[:12]
        result["id"] = rid
        with lock:
            store[rid] = dict(result=result, cache={})
            while len(store) > 25:
                store.popitem(last=False)
        return jsonify(result)

    @app.get("/files/<rid>/<name>")
    def files(rid, name):
        with lock:
            entry = store.get(rid)
        if entry is None or name not in ("notes.json", "concept-map.png", "concept-map.svg", "transcript.txt"):
            abort(404)
        r = entry["result"]
        if name == "notes.json":
            data, mime = json.dumps(r, indent=2, ensure_ascii=False).encode("utf-8"), "application/json"
        elif name == "transcript.txt":  # timestamped when the source was audio
            data = "\n".join((f"[{s['time']}] " if s["time"] else "") + s["text"] for s in r["transcript"]).encode("utf-8")
            mime = "text/plain"
        else:
            fmt = name[-3:]
            data = entry["cache"].get(fmt) or entry["cache"].setdefault(fmt, render(r["graph"], r["topics"], fmt))
            mime = MIMES[fmt]
        download = request.args.get("download", "").strip().lower() in {"1", "true", "yes"}
        return send_file(io.BytesIO(data), mimetype=mime, as_attachment=download, download_name=name)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=False, port=5000)
