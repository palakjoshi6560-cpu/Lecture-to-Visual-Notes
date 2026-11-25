# Quality patch – 23 November 2026

This patch addresses the three failures reported in the latest Windows test run.

## Changes

1. **Canonical graph endpoints:** graph selection now resolves leading list determiners to an existing canonical concept when available. This keeps a relation such as `candidate key → selected as → primary key` attached to the concept node instead of a phrase fragment.
2. **Evidence-backed context edges:** context/list relations are enriched only from other source sentences that explicitly mention both endpoints (including ordinary plural forms). Context edges with only one supporting sentence are omitted rather than padded with duplicate or invented evidence.
3. **Quick revision coverage:** a source-backed contextual relation is no longer discarded merely because one of its evidence sentences is also a concept-definition sentence. Direct pattern relations retain the existing redundancy guard.
4. **Domain-neutral implementation:** removed the flagged subject-specific comments, phrases and special-case wording; the generic list-follow-up rule now handles noun phrases without naming a particular subject.
5. **Concept-map noise filtering:** generic structural phrases are filtered without adding subject-specific vocabulary to the implementation.

## Verification performed in the patch environment

- `tests/test_content_quality_regressions.py`: **5 passed**
- `tests/test_context_map.py`: **5 passed**
- `test_no_new_lecture_specific_terms_in_source` and `test_known_debt_list_is_current`: **2 passed**
- DBMS headed evaluation evidence audit: all emitted context edges had at least two source sentences mentioning both endpoints.
- Python compilation and ZIP integrity checked before packaging.

The complete test suite was not completed in this environment because the long-running evaluation/OS test batch exceeded the execution time limit. Please run `python -m pytest -q` on the Windows environment as the final full-suite check.
