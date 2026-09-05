# Multi-document stores

A memory holding one manual behaves differently from one holding sixty-two
sibling documents that share a template — a corpus class most retrieval
systems meet in production and few are tested on. LMM's multi-document
behaviour was built against exactly that trial, and every mechanism below is
general: no keyword, no language rule, nothing tuned to any corpus.

## What changes when the store is plural

**Datelines.** Every evidence line the answer path reads opens with its
document's name: `Alpha Programme — the trust module opens with…`. The
answer can then say *which* document — and because the verifier reads the
same line, naming your source is the one claim that is grounded by
construction.

**The source-name channel.** A question that names a document gets that
document's sentences. A query word is weighed by its power to separate
sources — log(S/s) over how many of the S source names carry it — so a word
every sibling shares weighs exactly zero, and a single-document store is
provably unaffected.

**The named document speaks first.** When the question named sources, their
seats order to the front of the block; nothing enters or leaves it.

**No monopoly.** One document may hold at most the corroboration share of
the block when other documents hold matching evidence — the same rule that
stops one *region* of a document monopolising an answer, one level up.

**The census.** "Which documents cover X?" is not a retrieval problem — it
is a count:

```python
m.where("psychological safety")
# [("Trust Basics", 62), ("Natural Leadership", 51), ...]
```

Milliseconds, no engine, and the counts are the receipts. The same tally
rides into conversational answers as one line of store-attested fact.

**The informed refusal.** A turn that must decline while the census is full
offers the tally instead — with datelined reasons — gated so it can only
name and count what the store attests. The trigger is a *state* (refused,
tally in hand), never a reading of the question's wording, which is what
lets "what would you recommend?" get a useful, sourced reply without anyone
classifying intent.

**Follow-ups.** "And who is it for?" names its subject by pointing. When a
turn's subject resolves to nothing the memory knows and the previous turn's
did, the previous subject rides along — the signal is resolution failing,
which reads the same in any language.

## Composition

```python
text, sources = m.compose(
    "draft a one-day sales programme covering objection handling")
```

The engine is handed *material* — datelined evidence lines plus the tally —
and asked to organise, not to know. The draft is re-read line by line on the
way out: a number the material does not carry is vetoed, a mixture line is
dropped, structure stands. What survives returns with the documents it
rests on.
