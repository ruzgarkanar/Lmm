# The gate, drawn twice

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/architecture-dark.svg">
  <img alt="LMM architecture" src="../assets/architecture-light.svg">
</picture>

The gate appears twice on purpose. On the way in it decides what may become
a record; on the way out it re-reads the sentence the engine produced and
drops any claim the memory does not support. A diagram that draws it once
has drawn a retrieval pipeline instead.

## The pieces

**Identity.** A concept is a numeric key; a word is only a label. One
spelling can be two entities, and one entity carries labels in several
languages — which is how the same memory answers in Turkish, English and
Spanish without a translation layer.

**The graph.** Records — subject, predicate, value — each stamped with
source, trust and time, linked by *cause*, *then*, *contradicts*. Closed
triangles witnessed twice teach a predicate transitivity, and derived facts
are written with an `#inference` stamp **below the speaking threshold**.

**The evidence index.** The document's own sentences, kept verbatim, indexed
at several window scales. Retrieval is deterministic word/graph lookup —
explainable, and the same answer every run.

**The trust ladder.**

| level | source | trust | |
|---|---|---|---|
| 5 | operator | 0.75 | the person running the system |
| 4 | document | 0.60 | an ingested file, with its citation |
| 3 | model extraction | 0.50 | a candidate — if the gate admits it |
| 2 | inference | 0.35 | derived — below the speaking threshold |
| 1 | stranger | 0.30 | someone nobody has vouched for |

**The exit gate.** Candidates are generated per evidence subset, then judged:
a digit the evidence does not carry is vetoed outright, a mixture sentence —
evidence words wrapped around content the evidence never supplied — is
dropped, and the read-back re-extracts the claims and verifies each one.
What fails is **dropped, not repaired**: a repaired sentence has no source.

## In a multi-document store

Every evidence line opens with its document's name — a dateline — so the
answer can say *which* programme, and the verifier can confirm it. A question
that names a document gets that document's sentences (weighed by how many
source names share the word: a word every source carries weighs exactly
zero). One document cannot monopolise the block when others hold evidence.
And the census — which documents speak of this, counted — rides into the
answer as one line of store-attested fact.
