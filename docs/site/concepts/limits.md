# Honest limits

Kept current, and deliberately specific. A limits page that only flatters is
marketing; this one is part of the measurement.

- **Answer selection can still pick a true-but-off-target sentence.** The
  gate guarantees non-fabrication, not perfect relevance.
- **Purely lexical retrieval has a synonym ceiling.** "How many inches is
  the screen", where the line reads `Screen 15.6" LCD` and never says
  *inches*, is the open class. The offline expansion channel was built for
  exactly this and **measured not to close it**: its filter can only keep
  queries that copy a line — a genuine rephrasing shares no distinguishing
  word with its own line, so there is nothing lexical to keep it by. The
  code stays in the repository, off unless asked for, and closing the class
  needs a relevance judgement rather than a lexical one.
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
- **The sample is small.** The published comparisons rest on a handful of
  documents and a few dozen questions. Every question, answer and scoring
  decision is in the repository; widen it and tell us what breaks.
