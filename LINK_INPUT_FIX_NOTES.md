# Link input reliability patch

Changes in this patch:
- WebVTT parser now normalizes CRLF/CR newlines, accepts cue settings, strips inline markup, decodes HTML entities, supports multiline cue text, and handles both minute-second and hour-minute-second timestamps.
- Caption track selection is deterministic: English is preferred, then Hindi, before other available languages; original tracks are lower priority when a translated track is available.
- URL validation now rejects embedded credentials and malformed ports in addition to non-HTTP(S), localhost, and private IP URLs.
- Added focused tests for caption parsing, URL validation, caption-first retrieval, Whisper fallback, and preservation of timestamps through the shared notes pipeline.

Scope and limitations:
- Link retrieval still depends on the website being supported by the installed yt-dlp version or exposing a direct media stream.
- Private, login-protected, DRM-protected, age/geo-restricted or blocked videos are not bypassed.
- Live end-to-end downloading was not verified in this build environment because outbound internet, yt-dlp, Flask and faster-whisper were unavailable here. The caption and Whisper branches were tested with mocked downloader/ASR components; run the supplied local smoke test with a public video before presenting the feature.
