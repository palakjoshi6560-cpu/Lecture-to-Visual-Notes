# Context-aware terminology correction

The audio pipeline applies conservative corrections to copied ASR segments before sentence analysis.

- The correction engine in `lecturenotes/terminology.py` is subject-independent.
- Vocabulary and context rules live in JSON files under `lecturenotes/terminology_packs/`.
- A pack is enabled only when its required anchor and minimum number of independent supporting anchors are present in the recording.
- Applied changes are returned in `meta.terminology_corrections`; the web UI summarizes them. The original ASR wording remains available in `raw_transcript` and in the correction audit trail.
- Typed transcripts are not silently rewritten, and fuzzy matching is intentionally disabled.

## Adding another subject pack

Create a JSON file in `lecturenotes/terminology_packs/` with this structure:

```json
{
  "name": "short_pack_identifier",
  "required_anchor": "\\brequired\\b",
  "supporting_anchors": ["\\banchor one\\b", "\\banchor two\\b"],
  "min_supporting": 2,
  "word_rules": [
    {"id": "pack.term", "pattern": "\\bmisheard form\\b", "replacement": "Correct Term"}
  ],
  "phrase_rules": []
}
```

`phrase_rules` use the same `id`, `pattern`, and `replacement` fields. Use exact, conservative patterns and multiple independent context anchors. Avoid rules that could alter a valid word in an unrelated context. Do not expand incomplete cut-off words based only on a guess.
