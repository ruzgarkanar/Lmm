# LongMemEval runs

Raw output of `benchmarks/longmemeval.py`, one file per run, kept so a
number in a release note can be checked rather than believed. This is the
thing that was missing: the project measured itself on this benchmark
twice before and could not do it a third time, because the harness and
the question list lived outside the repository.

**The configuration is part of the measurement.** Every file's name says
which data variant and which caller settings produced it, and the
harness prints them in its own header. A figure quoted without them
means nothing — the same thirty questions score 10% or 23% on the same
code depending on how the caller asks.

| file | variant | caller | judged | calls |
|---|---|---|---|---|
| `0.12.1_s_30.json` | `longmemeval_s` (~50 sessions) | default | 3/30 · 10% | 289 |
| `0.12.1_s_30_ask_quoted.json` | `longmemeval_s` | `shape="ask", quoted=True` | 7/30 · 23% | 248 |
| `0.12.1_oracle_30.json` | `longmemeval_oracle` (evidence only) | default | 13/30 · 43% | 237 |
| `0.12.1_oracle_30_ask_quoted.json` | `longmemeval_oracle` | `shape="ask", quoted=True` | 14/30 · 47% | 214 |

Two effects, of very different size. **Configuration** is worth 13 points
where there are distractors (10% → 23%) and almost nothing without them
(43% → 47%), because the default path sends count- and span-shaped turns
to their organs or refuses them (W93 — measured right on a document
corpus, and LongMemEval is full of "how many days"). **Distraction** is
worth about 23 points under either setting.

Three types do not move under any cell: `multi-session` is **0/5 in all
four** — composition across sessions, neither a retrieval nor a
configuration problem; `single-session-preference` scores 1/5 because its
gold is a behaviour rather than a fact (this memory answers questions, it
does not compose preference-aware recommendations); and
`temporal-reasoning` 1/5 wants a computed interval.

### Where the loss actually is — measured engine-free

Retrieval is not the main suspect. Using the benchmark's own
`answer_session_ids`, the session carrying the answer is among the first
six seats in **25 of 30** questions on `longmemeval_s` (100% on the
oracle) and in the first twenty in 28. At the SENTENCE level, over the
eighteen questions whose gold is a literal string: the gold is written in
the store in 15, reaches the six seats in **9**, and of those nine is
answered correctly in **4**. The loss is split roughly evenly between a
stored sentence that never reaches a seat and a seated sentence that does
not become an answer.

Two readings, both in the files: `strict` is whether the gold string
appears in the answer, `judge` is an engine asked whether the answer says
what the gold says — the second is what published LongMemEval figures
use. Neither is called accuracy on its own.

The slice is deterministic: questions sorted by id, taken round-robin
across the six types, so `--most 30` is the same thirty questions on
every machine and in every release.

### What speaking costs here, and the plan organ's part in it

Accuracy is not the sharp number on this corpus — **precision when the
memory speaks is**. With `shape="ask", quoted=True` and the question's
own date supplied, the memory spoke 11 times and was wrong 5 of them.
The document side of this project measures zero fabrications in 44
cells; this is a different corpus and a different picture, and it is the
one that matters to the promise.

Three of those five came from the **plan organ**, all on the route
`chain · refuse · plan` — the chain refused, and the rescue seat then
answered with a confident number: `212` where the gold is 25, `0 days`
where it is 7, `14 days` where it is 18. Without the seat those three
turns abstain.

| | correct | spoke | wrong | abstained | calls |
|---|---|---|---|---|---|
| plan organ on | 8/30 | 11 | **5** | 19 | 247 |
| plan organ off | 7/30 | 7 | **2** | 23 | 222 |

It trades one right answer for three wrong ones and costs 25 more calls.
The arithmetic is not the problem — that is ours and verified. The
ANCHORING is: in a fifty-session store a phrase like "finished reading
X" matches lines in several sessions, and the organ picks one. Measured
on NIST two days earlier the same seat rescued **nothing**; here it
rescues wrongly. Two corpora, two different harms, one organ.

A separate pair, also wrong and not the plan organ's: two
`knowledge-update` questions answered from the quoted path, one giving
the CURRENT value where the past was asked and one the OLD value where
the current was asked.

`benchmarks/longmemeval.py --no-plan-rescue` reproduces the second row.

### A harness bug, recorded because it moved a number

Every instance carries a `question_date` in 2023 and the first runs did
not pass it, so "how many months since I last visited a museum" was
measured against the real calendar — the organ answered **1442 days
(about 47 months)** where the gold is 5 months. `asked_at` exists for
exactly this. Fixed, the slice went 7/30 to 8/30 and temporal-reasoning
0/5 to 1/5: one cell, not the three the error looked like it was worth.
