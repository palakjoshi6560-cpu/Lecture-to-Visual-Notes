Universal Lecture Link Input
============================

This update adds a third input mode: Lecture link. Existing Transcript and File or audio modes remain.

Supported approach:
- Uses yt-dlp extractors for many public video/lecture platforms and direct media URLs.
- Attempts English/Hindi subtitles/captions first.
- If captions are unavailable but an accessible audio stream exists, downloads audio to a temporary folder and sends it through the existing faster-whisper pipeline.
- Temporary media is deleted after processing.
- Transcript download is enabled for link-based results too.

Limitations:
- This is not literally every website. The site must be supported by yt-dlp or expose a direct media stream.
- Private, login-protected, paid, age/geo-restricted, DRM-protected or blocked content may fail.
- Audio fallback requires faster-whisper and a working local Whisper model; some formats may require FFmpeg/PyAV support.
- Only use links you are authorized to access and process. The app does not bypass logins, DRM or access restrictions.

Install/update:
    python -m pip install -r requirements.txt

Run:
    python app.py
