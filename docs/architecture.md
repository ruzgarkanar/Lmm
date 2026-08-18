# Architecture

This note describes the system as it stands. The README is the shop window; this
is the floor plan.

## The shape of it

```
                 ┌──────────── the GATE decides everything ────────────┐
 user/message →  extract (engine)  →  gate.admit  →  GRAPH (facts, sources,
                                                      trust, contradictions,
 document     →  learn_text ──────→  evidence index  transitive derivation)
 spreadsheet  →  learn_rows ──────→  graph directly (no model calls)
                                                      │
 question     →  graph lookup + evidence retrieval →  answer (engine)
                                                      │
                 coverage gate → digit discipline → support check → verify
                 (an answer whose claims are not in the given evidence DROPS)
```

Two layers, and the split is the design:

**`lmm.core`** — the discrete substrate. Identities, records, links, trust
ordering, contradiction arbitration, transitive derivation. Pure python; no
model, no network, no GPU. This is why the whole thing can be tested in seconds
on a laptop, and why `pip install lmm` pulls in nothing.

**`lmm`** — the layer that gives it language. Extraction, retrieval, answer
generation, the verification chain, the document adapters. It talks to an
engine; the engine is pluggable and is the only part that needs a model.

## The one rule everything rests on

**The engine cannot write records, and cannot speak unsupported facts.**

It produces *candidates* — extracted triples, phrasings, answers. The gate
admits or rejects; the gate releases or withholds. Nothing the model says
reaches memory without passing `gate.admit`, and nothing reaches the user
without surviving the verification chain. Every other property in this document
is downstream of that sentence.

## Ingestion

A document is not one material but two, and they want opposite treatment.

*Tables* state their own structure — row is entity, column is predicate, cell is
value — so they go **straight into the graph** through `learn_rows`, with no
model call at all. This is why a spreadsheet loads in milliseconds and why a
spec sheet's values stay attached to the fields that name them. Flattening a
table into prose is precisely how that attachment is lost.

*Prose* goes into the **evidence index**: sentences, stored verbatim with their
source, indexed at several context scales. Not everything fits a triple —
ranges, qualifications, conditions — and the evidence layer is where those
survive. Neighbouring sentences are indexed together as well, because a warning
line often carries the section's subject without repeating it.

With `deep=True` the engine additionally mines each sentence for triples. It is
off by default for files: a large document's structure is already taken by the
adapters, and an extractor call per sentence costs minutes to add little.

## Answering

1. Resolve the question's subject against the graph.
2. Gather candidate records and retrieve evidence sentences.
3. Hand *only that material* to the engine, in a structural frame — field names,
   not sentences in any user language.
4. Check what came back: does it cover the question, are its digits present in
   the evidence, are its claims re-extractable and supported?
5. Release, or abstain.

Step 4 is the fabrication gate. An answer whose claims are not in the evidence
it was given is dropped rather than repaired.

## Abstention is a stamp, not a phrase

Whether a turn abstained is decided **structurally**: the spoken sentence is
re-extracted, and a turn that asserted no claim is an abstention — in whatever
language it came out in. `Session.last_abstained` carries this, and
`Memory.ask(explain=True)` surfaces it.

The alternative, matching refusal phrases against a list, was in this codebase
and was a real defect: it made the *measurement* language-dependent where the
engine was not, and would have scored every honest English refusal as a
fabrication. Tests K1–K6 defend the replacement.

## Derivation

Transitive facts are derived symbolically on the graph, in microseconds, with no
model call — and the derived fact goes through the same gate as any other, so it
cannot write around a contradiction. This is the operation chunk-retrieval
cannot perform at all: two facts that never co-retrieve still compose here.

## Growth without retraining

Knowledge goes to the graph, never to weights. A periodic LoRA pass ("sleep
consolidation") refreshes *behaviour* — fluency, phrasing — from gate-approved
conversation logs, never from the model's raw output. The distinction matters:
training on your own unfiltered generations is how a system drifts.

## Persistence

The graph is written to the memory path; the evidence index to
`<path>.evidence` beside it. **Ingested sentences are stored verbatim**, because
provenance is the point — an answer must be able to show the line it rests on.
Treat a memory file as being as sensitive as the documents that made it.

## Further reading

- `CONTRIBUTING.md` — the two rules (no document-specific constants, no
  hand-written language rules) and what breaking them cost when it was found.
- `tests/test_core.py` — every test is a pathology that was measured on this
  code, written back as an assertion. It is the most honest description of the
  system's failure modes in the repository.
