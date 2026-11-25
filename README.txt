Pytest collection delay fix

What changed:
- Moved the expensive full Operating Systems lecture process() call out of module import time.
- Added a module-scoped pytest fixture so the lecture is still processed only once, but after test collection starts executing.
- Updated OS regression tests to use the fixture result.

Apply:
Extract this ZIP into the project root, preserving folder paths.

Verify collection quickly:
python -m pytest --collect-only -q tests/test_os_lecture.py

Expected: 8 tests collected. This command must not process the full lecture.

Then run the OS tests when ready:
python -m pytest -vv tests/test_os_lecture.py

Use `python -m pytest`, not `pytest.exe`, because the copied virtual environment's pytest launcher may contain a stale OneDrive path.
