# Combined content and concept-map fix

- Separates `requires` and `refers to` from `uses` so relation labels preserve meaning and direction.
- Prevents definition clauses such as `Consistency means the transaction...` from becoming a false edge to the first noun in the definition body.
- Recovers introduced enumerations across adjacent sentences as contextual links with both source sentences retained as evidence. The graph renders these as dashed arrows, distinct from direct in-sentence statements.
- Preserves both ends of a one-off, multi-word prerequisite/dependency/reference relation so named concepts are not discarded solely because they occur once.
- Selects context-linked groups together and fills remaining map capacity with strong direct relations; displayed nodes remain connected to an evidence-backed edge.
- Adds regression coverage for relationship direction, ACID list grouping, evidence grounding, and cross-domain map invariants.

Context links are explicitly marked `kind=context`; they are not treated as direct `pattern` relations in Quick Revision or the relation-quality metric.
