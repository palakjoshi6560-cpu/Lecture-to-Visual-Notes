# Quality patch — 2026-11-23 (B)

This patch targets the observed OS lecture failures while keeping the processing engine subject-neutral.

## Changes
- Recover missing sentence boundaries before `Next/Now, let us discuss ...` transitions.
- Join dependent `Such as ...` / `Including ...` fragments to the preceding sentence for readable notes while preserving raw transcript content.
- Recognize colon-style definitions such as `CPU scheduling: Process of selecting ...`.
- Extract a source-backed `CPU scheduling — selects — ready process` relation from that colon-style definition.
- Add context-gated OS terminology corrections for observed ASR variants: `inference structure`, `re-stores`, FCFS, SJF, page-table mapping, and page-fault wording.
- Ensure every non-empty topic has at least one substantive source-backed note bullet if ranking/deduplication would otherwise leave it empty.

## Verification
- `tests/test_signposting_filter.py`, `tests/test_terminology.py`, and `tests/test_content_quality_regressions.py`: 29 passed in the local patch environment.
- Python compilation: passed.
- Flask-dependent graph/app tests and the full evaluation suite were not run in this patch environment. Re-run them in the project's Windows `.venv` before submission.
