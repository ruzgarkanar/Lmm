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
- **An assertion with no stamp is worth suspecting.** While the
  abstention-stamp defect stays open (`abstained=False` on a sentence that
  is itself a refusal), the same field run found that every instance
  carried the same signature: `abstained=False` **and** `sources == ()`.
  Requiring a source is measured as a poor *rule* — it removed two wrong
  answers and cost two right ones — but as a *diagnostic* it caught every
  occurrence. A careful caller treats an answer that asserts without
  provenance as suspect.
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

- **Chat memory is measured mid-climb, not conquered.** On a fixed
  30-question LongMemEval slice (gpt-4o-mini as engine) 0.7 measures
  57-60%, from 43% at the cycle's start; published chat-memory products
  claim ~71% under stronger engines and different protocols. The
  remainder is named, not mysterious: summing durations across records,
  ordering THREE events, reflecting a stored preference in a
  recommendation, a ±1-day anchor — and roughly half of what is still
  missed, the bare engine handed the same lines also misses, confidently,
  where this memory abstains.
- **Extraction is not byte-stable across processes.** At temperature
  zero, the same passage can distil into slightly different record sets
  run to run, and a 30-question score moves ±2 with it. The 0.7 guards
  make the ANSWERS stable against this mood (a cancelled thing is never
  counted, a stated tally outranks an enumeration); the variance itself
  is the engine's, not the store's.

- **The sample is small.** The published comparisons rest on a handful of
  documents and a few dozen questions. Every question, answer and scoring
  decision is in the repository; widen it and tell us what breaks.
