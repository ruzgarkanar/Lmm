# Honest limits

Kept current, and deliberately specific. A limits page that only flatters is
marketing; this one is part of the measurement.

- **Relevance is the dominant failure mode, and the gate does not touch
  it.** The gate guarantees non-fabrication, not perfect relevance — that
  sentence has always been here, and a field integration measuring 44
  supplier/requirement cells showed it is not one limit among several but
  *the* one. Zero fabrications in 44 cells; both of its wrong answers were
  relevance failures, and one of them is instructive: every content word
  verbatim in the document, a real source stamp, a true sentence — the
  requirement asked for a simulation feature in the product and the
  document described unit testing during the project. The gate had nothing
  to object to, because there was nothing false in it.

  *"The memory has something to say about X"* is not *"the supplier offers
  X"*, and an integration doing coverage rather than question-answering
  carries that distance itself. Nothing here judges it.
- **An assertion with no stamp is worth suspecting** — and the defect that
  made it necessary is closed as of 0.8.0. A refusal spoken with
  `abstained=False` and `sources == ()` was reported twice from the field,
  and it had two doors, both of the same shape. The answer path's fallback
  excluded only what the relation gate had refused, so it revived what the
  grounding score had already eliminated; the widened rescue pass took its
  own refusal SENTENCE and asked a word-overlap reading whether it was a
  claim. Both now defer to the structural stamp: **a gate that asks whether
  what a sentence says is in the evidence cannot fail a sentence that says
  nothing**, so a sentence carrying none of its evidence is not an answer,
  and a pass that left by the refusal door did not answer whatever its
  words look like. Measured on a fifteen-question slice: unsourced
  assertions 2 → 0, honest abstentions 1 of 3 → 3 of 3.

  The diagnostic is still worth keeping. Requiring a source was measured as
  a poor *rule* — it removed two wrong answers and cost two right ones —
  but as a *signal* it caught every occurrence of this class, and it will
  catch the next one before a report does.
- **The synonym ceiling is lower than it was, and it is not gone.** A word
  search alone cannot reach a paraphrase, and two instruments were built
  against that. The offline expansion channel was **measured not to close it**
  — its filter can only keep queries that copy a line, and a genuine
  rephrasing shares no distinguishing word with its own line — so it was
  deleted rather than left as a switch nobody should turn on.

  The meaning channel that replaced it is measured to close much of the class:
  the line that answers reaches the engine 32% → 79% of the time on a
  115,913-line corpus. It does **not** close the example this page has always
  carried. The line reads *"Memorized secrets SHALL be at least 8 characters
  in length"*; asked *"what is the shortest password"*, the channel still
  misses it, while *"minimum length for a memorised secret"* now lands first.
  Vectors move the boundary; they do not abolish it.
- **A corpus's subjects are readable, and still not answerable.**
  `m.themes()` groups the documents that belong together — community
  detection over the entity graph, deterministic, zero model calls. It
  does not say what a group is ABOUT: scoping the composer to a
  community's documents was measured and returned one document's
  outline, because nothing in the material states a theme.
- **Global, thematic questions are the competitor's home ground.** "What are
  the main themes of this corpus" is answered well by community summaries;
  LMM has no equivalent, and the honest cost of building one the summary way
  is the fabrication channel it opens. The census (`where`) covers the
  countable half — *which documents speak of X* — exactly, in milliseconds.
- **Per-question latency pays for verification.** The exit gate's read-back
  is model calls; on a hosted engine a question runs seconds, not
  milliseconds. The floor is different: questions the graph can settle are
  answered in microseconds with zero calls.
- **The judge can wobble.** A model asked "does the evidence say this" does
  not always answer the same way twice at temperature zero. Mutual-coverage
  claims skip the jury entirely; the partial band still rides on it, and one
  conversational turn was measured flipping between runs.
- **LMM does not OCR.** A PDF without a text layer teaches it nothing, and
  `learn()` says so instead of reporting success.
- **Two hedges were retired on measurements of eleven questions.** The
  candidate ladder (3 → 1) and the judges' second view (2 → 1) were each
  removed because they won nothing in that run — the narrow candidates never
  won, the second view never confirmed. Eleven questions is a small sample to
  retire machinery on. Both are numbers rather than deletions
  (`session.CANDIDATES`, `session.VIEWS`), so a corpus that proves us wrong
  can put them back in one line.

- **Chat memory is this project's weaker ground.** On a fixed 30-question
  LongMemEval slice with the harness in this repository, 0.13.0 measures
  **13/30** on `longmemeval_s` (0.12.1: 7/30 in the nearest comparable
  configuration). An earlier figure of 57-60% for 0.7 cannot be reproduced
  and should not be relied on: its harness lived outside the repository.
  Thirty of five hundred questions is a small slice, and the judge is the
  configured engine, `gpt-4o-mini`, the same model that answered, so the
  figure compares this system's own versions and not published
  leaderboards (Zep reports ~71% under a GPT-4o judge). Of the questions
  it stays silent on, the answer's session was in the evidence block for
  most: the remaining loss is between having the evidence and stating a
  conclusion from it.
- **A count verifies that its items exist, not that they meet the
  question.** Asked how many appointments the user went to in March, the
  counting organ named three doctors where the answer is two: every name
  was written in the evidence, but one appointment was scheduled for
  April. Membership in the set the question describes is not checked.
- **Extraction is not byte-stable across processes.** At temperature
  zero, the same passage can distil into slightly different record sets
  run to run, and a 30-question score moves ±2 with it. The 0.7 guards
  make the ANSWERS stable against this mood (a cancelled thing is never
  counted, a stated tally outranks an enumeration); the variance itself
  is the engine's, not the store's.

- **The sample is small.** The published comparisons rest on a handful of
  documents and a few dozen questions. Every question, answer and scoring
  decision is in the repository; widen it and tell us what breaks.

## The vocabulary gap, measured

Sentence retrieval scores a line by how many of the question's content
words it carries. When a reader asks in their own words and the document
answers in its own, there is nothing for that scoring to find: NIST
SP 800-63B answers "what is the shortest password a user may choose" under
*memorized secret*, and the question reaches the wrong lines because that
is where its words happen to occur.

The library has a defence built for exactly this — `WIDEN`, a second
retrieval over wordings the engine proposes, keeping only words the store
actually holds. On 5 October 2026 it was measured where it should pay, at
the level of retrieval alone so that no answering noise is in the way
(`benchmarks/widen_recall.py`, 11 engine calls): **it rescued nothing and
broke nothing**, recall 8 of 11 either way.

The reason is structural rather than a matter of tuning.
`generate.phrasings` is shown the QUESTION and never the store, so it can
only guess generic rewordings — and it proposed, for that question, the
word "password", which the question had already used. The document's own
vocabulary is in the index and would have passed the filter: "memorized"
appears on 1,173 lines, "director" on 24. It is simply never proposed.

Three cheaper repairs were measured and refused. Letting the meaning
channel choose candidate LINES, rather than reorder the ones the words
admitted, puts the answering line at rank 410 and 930 of 12,253. A
word-level bridge built from the bundled meaning table ranks "hash" and
"captcha" above "director" for the word "heads". And grounding the
proposal in the block retrieval did return helps only when retrieval was
already close: the block for the password question does contain
*memorized secret*, and the block for "who currently heads NIST" contains
neither "director" nor "acting".

So this is an open limit with a known mechanism, which is a better place
to be than an open limit without one, and it is not closed.

## Counting membership, and a repair that was refused

A count verifies that its items exist, not that they meet the question.
Asked how many doctor's appointments the user went to in March, the
organ named three doctors where the answer is two: every name was
written in the evidence, so every name passed, and one appointment was
in April. Every organ that verifies items against lines inherits this.

The obvious repair was built and measured on 5 October 2026, and it does
not work. `find` already refuses to let a word that narrows nothing vote;
the converse seemed sound — after the survivors are verified, ask each
demand of the question how many of their home lines carry it, and treat
a demand that MOST carry but not all as the condition the question is
selecting on. On the doctors it is right: "march" sits on two home lines
of three, and the April appointment leaves the count.

It is wrong by exactly the same arithmetic on the invariant that has
guarded this organ since 0.7. Asked how many projects the user leads,
three lines answer, and one of them reads *"my second project, Quiet
Lantern, started well"* — a project the user leads, written without the
word "lead". The demand "lead" sits on two home lines of three, so the
rule drops a true item with the same confidence that it drops the false
one. The two cases have the same shape, and no lexical signal separates
them: a qualifier the question adds ("in March") and a relation the
question names ("lead") are both just words the lines may or may not
carry.

What distinguishes them is that the April line states an ALTERNATIVE —
another date, in the slot the question constrains — while the Quiet
Lantern line states nothing instead of "lead". Reading that difference
means knowing which words are values of the same kind, and the only
instrument here that knows it is the store's own date machinery. So the
repair, if there is one, is narrower than the rule tried: dates checked
as dates, not demands checked as words. The limit stands until that is
built and measured.

## No line reads its own date

Measured on LongMemEval, 5 October 2026. Asked how many doctor's
appointments the user went to in March, the memory answers three where
the answer is two, and the three lines say:

| named | the sentence says | the envelope says |
|---|---|---|
| Dr Johnson | "on **April 1st**" | chat 2023/03/27 |
| Dr Smith | "on March 3rd" | chat 2023/03/27 |
| Dr Thompson | "on March 20th" | chat 2023/03/27 |

All three were *mentioned* on the same day, and one of the appointments
is in April. Counting the mentions gives three; counting the
appointments gives two.

The library already knows that a date in a sentence outranks the date on
its envelope — that is what `anchor` does, reading the event's date from
the text with an engine proposal that arithmetic confirms. But `anchor`
resolves ONE event. Every primitive that works over a SET of lines —
`lines`, `before_lines`, `after_lines`, `month_tally` — takes each
line's date from its SOURCE STAMP, by reading the digits out of it. When
a conversation reports several events in one message, those primitives
see one date for all of them.

So the gap is narrow and nameable: **no primitive reads each line's own
date from its own text.** It is not a retrieval failure (the answering
session is seated), not a gate failure (nothing was rejected), and not a
reading failure (the engine is never asked). The organ is working
correctly over dates that are not the ones the question is about.

`month_tally` does not close it either: it answers which month is
busiest, not how many fell in a named one.

Closing it means a reader that is given many lines and returns a date
for each — one call for the set, not one per line — and primitives that
prefer that date to the stamp when it exists, under the same arithmetic
confirmation `anchor` already applies. Until that exists, every count
and every window over a conversation is a count of MENTIONS, and this
page says so rather than the number pretending otherwise.

