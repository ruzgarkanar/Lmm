# Honest limits

Kept current, and deliberately specific. A limits page that only flatters is
marketing; this one is part of the measurement.

- **Answer selection can still pick a true-but-off-target sentence.** The
  gate guarantees non-fabrication, not perfect relevance.
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

- **The sample is small.** The published comparisons rest on a handful of
  documents and a few dozen questions. Every question, answer and scoring
  decision is in the repository; widen it and tell us what breaks.
