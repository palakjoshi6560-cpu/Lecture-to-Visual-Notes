# Final quality fixes (23 Nov 2026)

- Expanded subject-neutral signposting exclusion to catch transitions such as “Next, let us discuss …” while preserving the raw transcript and explanatory sentences.
- Improved concept-map selection so source-backed relation clusters (ACID/property lists, prerequisite chains, key-selection links, composite-key groups) are prioritized and multi-topic lectures can retain a representative concept for each section.
- Canonicalized leading quantifiers in displayed graph endpoints when the canonical concept is present (for example, “one candidate key” to “candidate key”).
- Tightened contextual relation evidence: a supporting sentence is added only when it mentions both endpoints; a single explicit enumeration sentence remains valid evidence on its own.
- Added a regression test for “Next, let us discuss …”.

The full test suite has not been run to completion in this build environment. The Flask-dependent app tests also require Flask, which is not installed in this isolated environment. Run the full suite in the project's existing Windows virtual environment before submission.
