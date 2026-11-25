# Final-submission repair — 23 Nov 2026

This update addresses the four regressions reported by the full test run:

1. **Graph rendering endpoint mismatch:** canonicalized graph endpoints are now stored on the selected edge as well as the node set. This prevents render-time `KeyError` when a quantifier-prefixed concept is canonicalized.
2. **High-signal one-off concepts:** a single-mention technical endpoint is retained when it has an explicit evidence-backed relation to an established concept. Narrative proper names remain excluded.
3. **Concept-map size and connectivity:** removed topic-count-based expansion beyond the configured node cap. Removed isolated map nodes; standalone definitions remain in the concepts/notes rather than appearing as disconnected map nodes.
4. **Instructional sentence detection:** student-directed prompts such as “You should check/review/remember…” are marked instructional while remaining in the transcript.

## Verification performed in this environment
- `tests/test_context_map.py::test_graph_keeps_a_high_signal_one_off_technical_concept` — passed.
- `tests/test_stage2.py::test_metadata_header_and_direct_address_are_excluded_but_preserved` — passed.
- Direct graph-render check — PNG signature verified.
- Synthetic multi-topic graph cap check — 14-node cap respected; every map node is an edge endpoint.
- Python compilation checks for edited modules — passed.

The audio route test and long OS lecture fixture were not rerun here: this runtime does not have Flask installed, and the long fixture exceeded the available execution window. The full test suite must therefore be rerun in the project's existing Windows `.venv` before submission.
