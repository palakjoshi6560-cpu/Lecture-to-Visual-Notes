# Final regression correction

- Quick Revision now performs high-threshold deduplication (TF-IDF cosine >= 0.80) across terms, relationships, and remember items. The full notes, concept list, and full relation list retain source material; Quick Revision removes genuinely repeated entries while preserving distinct coordinated relationships.
- Concept importance scoring remains phrase-sensitive, but the capped concept shortlist now reserves space for recurring, explicitly defined single-word concepts before less specific long phrases. This is domain-neutral and protects terms such as semaphore, mutex, or deadlock when they are defined and recur in the lecture.

Run the full suite in the project's Windows `.venv-audio` with the installed `en_core_web_sm` model before treating this correction as fully verified.
