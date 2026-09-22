# Measurements

Every figure below is the median of three repeats, produced by the same
side-blind scorer, on the same engine for both sides. The corpora, the
questions, every answer and every scoring decision are in the repository —
re-run them, and if you find a scoring bug, open an issue: two have been
found so far, and both were penalising the competitor.

## Retrieval, 0.6

Measured on this repository's own field corpus, engine-free. A known-item
query is a line from the store with its most distinctive terms removed, so
the words alone cannot find it — the protocol needs no hand-written gold and
cannot be tuned to.

**Which encoder** — 20,000 lines, 180 queries, two terms removed:

| encoder | first hit | MRR | index build |
|---|---|---|---|
| a distillation of our own | 27% | 0.306 | 1.2 s |
| `paraphrase-multilingual-MiniLM-L12-v2` (torch) | 66% | 0.710 | 36.1 s |
| **the bundled static matrix** | **84%** | **0.883** | **0.3 s** |

**Which layer** — 103 documents, the store's own `find`:

| | first-3 hit |
|---|---|
| words alone | 92% |
| + meaning at the DOCUMENT layer | **99%** |
| + meaning at the LINE layer | no change |
| lines chosen by meaning INSIDE a document | 78% → **31%** |

**End to end** — 115,913 lines, 304 queries, by whether the ANSWERING line
reaches the five the engine is shown:

| | first | in five |
|---|---|---|
| words alone | 31% | 32% |
| + meaning channel | 31% | 60% |
| + late-interaction reordering | 34% | **79%** |

**How much reordering is worth, by how vague the question is** — the same
corpus, by terms removed from the query:

| terms removed | single vector | late interaction |
|---|---|---|
| 2 | 0.864 | 0.884 |
| 4 | 0.779 | 0.825 |
| 6 | 0.707 | **0.807** |

## Cost retractions, 0.6

Three hedges against weak retrieval were re-measured after it improved, and
each had stopped earning. Eleven field questions with gold answers:

| hedge | measurement | now |
|---|---|---|
| 3 candidate subsets per question | every admitted answer came from the WIDE block; the narrow ones won nothing | `CANDIDATES = 1` — 139 → 104 calls, 4/11 → 5/11 |
| 2 views per judge | the second view confirmed nothing, on either judge | `VIEWS = 1` — 8 fewer calls in 104 |
| the offline expansion index | kept zero of a hand-written perfect query set | deleted, −607 lines |

| | |
|---|---|
| first question on a 115k-line store | 10.6 s → **0.13 s** (the derived index is saved) |
| a deterministic question asked twice in one turn | asked once — one refusal turn, −2 calls, −11k tokens |

## Against GraphRAG

| same engine (gpt-4o-mini), 3 samples each | LMM | GraphRAG |
|---|---|---|
| Fictional corpus, 17 q | **17/17** · spread 0 | 16/17 · spread 0 |
| NIST SP 800-63B (45k tokens), 13 q | **11/13** · spread 0 | 7/13 · spread **2** |

![Both build a graph; only one keeps the number](../assets/fig-64-survives.svg)

*GraphRAG loses specific values.* Its indexer summarises entity
descriptions, and a summary abstracts the number away — asked for a maximum
length the standard states as **64**, it answered "verifiers should
establish minimum length requirements", a sentence in which the value never
appears. *LMM's misses are refusals* — reworded questions whose answer was
in the document and not retrieved. One failure mode approximates; the other
declines. They are not the same failure.

## Cost

| NIST SP 800-63B | LMM | GraphRAG |
|---|---|---|
| Indexing | **0 model calls** | 220 calls · 498,230 prompt tokens |
| Per question | ~5 calls | 2 calls · ~10,100 prompt tokens |

Ingestion is the structural difference: LMM reads the document verbatim and
decides nothing at write time, so reading is nearly free and nothing is
lost before anyone asks.

## Language independence

The same fictional world, line for line, in three languages — same
questions, same gold keys, same invented names:

| | median of 3 | samples |
|---|---|---|
| Turkish | **17/17** | 17 · 17 · 17 |
| English | **16/17** | 15 · 16 · 16 |
| Spanish | **15/17** | 15 · 15 · 15 |

## What the gate is actually for — a correction from the field

An integration measuring 44 supplier/requirement cells against two real
tender documents published a correction to its own earlier report, and it
corrects ours too. Its first comparison put the whole document in the
prompt — **no retrieval at all** — and the gate looked like it was buying
two fewer fabrications. That was the wrong control.

The right one is the same retrieval LMM uses (`evidence.find`, zero model
calls) with a single engine call after it instead of the gate. Same engine,
same questions, 44 cells:

| | calls | prompt tokens | seconds | false-covered |
|---|---|---|---|---|
| document in the prompt | 44 | 292,749 | 125 | 5 |
| **the same retrieval, one call, no gate** | 44 | **28,083** | **88** | **3** |
| LMM | 378 | 313,173 | 491 | **2** |

Read that column in two steps. Giving the model six retrieved lines instead
of the whole document removes two invented coverages; **adding the gate on
top of those same lines removes one more** — inside a run-to-run variance
the reporter measured at about two cells. So on that corpus the gate's
contribution to non-fabrication sits at the noise floor, for a reason worth
stating plainly: **once retrieval is good, the fabrication the gate exists
to stop was mostly not happening.** Naive RAG over the same retriever was
11× cheaper in tokens and 5.6× faster.

What the same run shows the gate DOES buy is something neither side had
measured. Counting what each approach was willing to say across the 44
cells:

| | "partly" | "fully" | "not covered" |
|---|---|---|---|
| same retrieval, no gate | **31** | 7 | 6 |
| document in the prompt | 14 | 29 | **0** |
| LMM | 0 | 34 | 10 |

Unverified answering **hedges**: 31 of 44 cells came back "partly covered",
and the whole-document variant did not once say "not covered" in 44 tries.
LMM commits — 34 to 10, and 8 of those 10 refusals are correct. A matrix
where 31 rows of 44 say "partly" tells its reader nothing.

That is the property verification sells, and it is not accuracy: **the
answers do not drift to the middle.** A system that never commits is never
wrong and never useful. The shape of the number, for anyone repeating this:
over a fixed question set, how often does the system return the
non-committal option. It costs nothing to compute and separates verified
from unverified answering far more sharply than accuracy does.

The same run also reports **zero fabrications in 44 cells** — the promise
held exactly as written. Both of LMM's false-covered cells were checked
against the document by hand: one is the open abstention-stamp defect, and
the other is a true, sourced sentence answering a *different* question than
the one asked. Which points at the limit that actually bites, below.

## The scorer's own bugs

A benchmark whose author only ever finds errors that flatter him is not
evidence. Two defects were found in this project's scorer, both moving
points **away** from the competitor; both fixes were published with the
corrected numbers:

1. Abstention detection was a Turkish-only phrase list; GraphRAG's four
   correct English refusals scored as fabrications. Fix: 13/17 → 16/17.
2. The fabricated-value check read the question's own echoed subject as an
   invented designation, scoring an honest refusal as a fabrication.
