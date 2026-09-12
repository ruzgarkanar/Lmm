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
at several window scales. Retrieval by words is deterministic word/graph
lookup — explainable, and the same answer every run.

**The meaning channel.** A word search cannot cross a paraphrase: a lease
writes RESIDES and a reader asks where somebody LIVES. So a 30 MB static
multilingual matrix ships in the wheel and is attached by default — one
lookup per word-piece and a mean, no transformer, no warm-up, no network.

It sits at the **document** layer, and which layer was measured rather than
assumed. Over lines it moved nothing; over document profiles it took the
store's own retrieval from 92% to 99% on 103 known-item queries, because a
catalogue line reads "Aidiyet ve Motivasyon." — four words carry no subject,
so vectors over lines compare fragments while vectors over a profile compare
subjects. The opposite was measured too: choosing the lines INSIDE a proposed
document by meaning rather than by words took the answering line from 78% to
31%. **Meaning says which document; inside it, the question's own words are
sharper.**

The two rankings are fused by reciprocal rank (RRF, K=60) — which reads
order, not score, so there is no weight to tune and a line the words already
found cannot be pushed out by a vector's opinion.

**Late interaction.** A single vector averages a line into one point, and the
averaging is where a vague question loses. So every piece of the question is
scored against the piece of the line it fits best and the bests are summed —
ColBERT's idea without ColBERT's model, since the piece vectors are already
the table. It only ever REORDERS: no line enters by it and none leaves, so
nothing it decides can reach the gates. Measured, it is worth more the vaguer
the question gets (six terms removed from the query: MRR 0.707 → 0.807), and
end to end it takes the answering line's chance of reaching the engine from
60% to 79%.

Both are parametric: `Memory(encoder=..., reranker=..., dense=False)`.

**The entity graph.** The document's own, built with no model call. An
entity is a phrase whose words occur together beyond chance (pointwise
mutual information over the phrase's weakest split), which keeps varied
company, and whose parts do not choose freely. An edge says one thing —
*these two are mentioned together, here* — and carries the witnessing
sentence, weighted by Dunning's log-likelihood ratio so an entity that
appears everywhere does not become everyone's neighbour. An edge cannot be
fabricated: it is an observation. It is also the partition — a question
naming two entities reads the intersection of two posting lists rather
than scoring the store. On a 36,472-line novel: 417 ms to 0.01 ms.

**The composer.** A memory that answers only the shapes somebody wired by
hand answers a finite set of questions, and shapes are endless. So the
engine proposes a PLAN over verified primitives — anchor a phrase to its
dated line, take the latest, gather lines, span two dates, order them,
filter by a cutoff, count, tally by month — and the interpreter executes
only operations it knows, on phrases the store can anchor. The sentence
is built from the final step's typed value by the library's own template:
the engine contributes operation names and phrases, never an output word.
An unknown operation, an unanchorable phrase, a dangling reference: the
plan dies and no claim is born. *Plan is a proposal; the primitives are
the law.*

**The event organs.** A count is the length of a VERIFIED list (the
engine lists, the store checks each item, the number is what survives); a
total is the sum of verified amounts; "which came first" and "how many
days between" are date arithmetic. The shape reader routes at the door
and its verdict binds both ways — organs silent and plan dead, the turn
refuses rather than gambling prose.

**Dialogue's two columns.** WHEN an event happened is read from the
sentence before the envelope's stamp (the engine proposes the date, the
arithmetic admits it or keeps the stamp); WHO SPOKE is kept per line, so
the asker's own lines seat first. `Memory.distil()` writes the events a
chatty passage reports into the graph — listed by the engine, admitted by
the passage's own words, gated like any other fact.

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
