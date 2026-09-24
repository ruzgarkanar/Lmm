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
