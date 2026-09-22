# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.10] — 2026-09-22

### Added

- **`ask(..., quoted="only")`** — the store's refusal may stand (W161),
  and **`Session.quoted_refused`** says which failure happened (W160).

  Diagnosing the quoted reading over fifteen questions turned its own
  scoreboard around. It carried nine; the engine DECLINED four, three
  of them correctly (the store holds nothing on those subjects); and
  the store REFUSED two. Those two were not misses — they were saves.
  Asked a regulation's number, the engine had answered with the PAGE
  number from a table of contents ("96", "140"), and the check caught
  both, because the digit appeared nowhere in the line the engine said
  it had used. **The ordinary path then spoke one of them anyway.**

  So a decline and a refusal are different facts and the turn now
  keeps them apart. With `quoted="only"` a refusal stands; a decline
  still falls back, having offered nothing to refuse.

  **Measured, gold read off the document itself:** falling back
  rescued one correct answer and spoke one false one; letting the
  refusal stand asserted nothing false and missed three — same score,
  a different KIND of error, and 54 → 32 calls, 72 → 48 seconds.
  Neither dominates, so neither is the default: the caller says which
  error is the expensive one where they work.

### Recorded

- A third measurement trap caught the author of the other two: the
  first version of the W161 invariant asked one question twice in one
  `Memory` and compared a cached answer with a live one. `cache=False`
  in a test that measures two paths against the same question.

## [0.7.9] — 2026-09-22

### Added

- **`ask(..., quoted=True)`** — an answer that quotes the store verifies
  itself for nothing (W159). Verification is 46% of a turn's engine
  calls because it judges free prose AFTER it exists: a read-back
  re-extracts the claims, a relation check reads the edges. Every other
  answering path here already avoids that — a record, a count, a span
  are spoken from the store's own value through our template, and need
  no checking because nothing new was said. This is that discipline
  applied to prose.

  One call returns the answer AND the number of the evidence line it
  used. The store then checks, with no model and no network: is that a
  line it offered, does the answer's every content word come from that
  line or from the question (W152), and does every digit appear in it.
  What survives is assembled out of a line the store holds, so the
  fabrication the gates exist to catch is structurally impossible
  rather than judged unlikely.

  **Measured, two runs of fifteen questions over a 355-line document:**
  calls 90/84 → 54/69, prompt 368k/349k → 224k/273k, seconds 105/105 →
  76/100. About **30% fewer calls and tokens**, 16% less wall clock; the
  quoted reading carried 8–9 of 15 turns and the rest fell back. Both
  arms abstained on the same absence questions.

  The cost is named rather than hidden: an answer spread over several
  lines gets the one line it quoted. Asked which exams a study guide
  serves, the ordinary path lists four and the quoted path names one.
  So this is a caller's choice, not a default — the promise is
  unchanged, only the mechanism keeping it and its price.

### Refuted on the way

- The first cut asked the engine to reproduce the evidence line
  character for character and matched it against the store. Measured on
  the same fifteen questions, it matched ONCE: copying a long line
  exactly is a task engines are bad at and nothing needs them to do,
  since the store already holds the line. Naming it by NUMBER cannot be
  misspelled and a number nobody offered is refused by construction.

- The first A/B ran both arms in one process and reported a 99 → 34
  second "improvement". That was the deterministic call pool this
  release's predecessor documented — the arms must each have their own
  process. Written down here because the author of that warning fell
  into it one release later.

## [0.7.8] — 2026-09-22

Documentation only, and it revises this project's central claim.

### Corrected

- **What the gate is for.** A team integrating this library published a
  correction to their own report, and it corrects the argument this
  README had been making. Their first comparison put the whole document
  in the prompt — no retrieval at all — which made the gate look like it
  was buying two fewer invented answers.

  Against the right control (the same retrieval LMM uses, one engine
  call after it instead of the gate), across 44 cells on two real tender
  documents: stuffing the document produced 5 false coverages, the same
  retrieval with one ungated call produced 3, and LMM produced 2 —
  inside a run-to-run variance of about two cells. **Retrieval buys two;
  the gate buys the third, at the noise floor**, because once retrieval
  is good the fabrication the gate exists to stop was mostly not
  happening. Naive RAG over the same retriever was 11× cheaper in tokens
  and 5.6× faster, and anyone choosing between them deserves to read
  that here.

### Added to the record

- **What the gate does buy, which neither side had measured.** Across
  the same 44 cells: unverified answering returned "partly covered" 31
  times out of 44, and the whole-document variant never once said "not
  covered". LMM committed 34 to 10, and 8 of those 10 refusals are
  correct. A matrix where 31 rows of 44 say "partly" tells its reader
  nothing.

  So the claim is not "the gate stops the model inventing things". It is
  that **the gate stops a system from refusing to answer while appearing
  to** — the answers do not drift to the middle. A system that never
  commits is never wrong and never useful. The shape of the number, for
  anyone repeating it: over a fixed question set, how often the system
  returns the non-committal option. It costs nothing and separates
  verified from unverified answering far more sharply than accuracy.

- **Relevance is the dominant failure mode**, promoted on the limits
  page from one item among several to the first. The same run reports
  zero fabrications in 44 cells; both wrong answers were relevance
  failures, one of them with every content word verbatim in the document
  and a real source stamp — a true sentence answering a different
  question than the one asked. The gate had nothing to object to.

- **An assertion with no stamp is worth suspecting.** While the
  abstention-stamp defect stays open, every instance in that run carried
  one signature: `abstained=False` and `sources == ()`. Measured as a
  rule it is poor (it removes two wrong answers and costs two right
  ones); as a diagnostic it caught every occurrence, and it is now
  documented as one.

## [0.7.7] — 2026-09-22

### Added

- **`Memory.reset()` and `ask(..., standalone=True)`** — a turn may
  decline the conversation it did not have (W158). Measured from the
  field, both directions, same question and document: a cell answers
  when asked first and abstains when asked after nine unrelated cells
  in the same `Memory`. Nothing but the order differed, and the
  reporter first attributed the change to a release — the sequence is
  part of the input, and anyone comparing runs needs to know it.

  That is the follow-up inheritance doing its job: a turn naming no
  document of its own reads the one the conversation was about, which
  is right in a conversation and contamination in a matrix of
  independent cells. It is NOT turned off — the same reporter measured
  it earning its place even there (77.3% with it, 72.7% with a fresh
  memory per cell; isolation traded three invented coverages for three
  missed ones). What was missing is the caller's say: `reset()` ends
  the conversation, `standalone=True` does it for one turn, before and
  after.

  What neither does is forget. The graph, the evidence index and the
  aids are the memory; a conversation is not, and the invariant checks
  that `facts` and every stored line survive a reset.

### Documentation

- **`cache=False` is not "nothing is reused"** — and a benchmarker must
  know it. Below `Memory` the engine pools its own deterministic calls
  (temperature 0, byte-exact prompt, per PROCESS and so across `Memory`
  instances) and the extractor memoises sentences already read. Asking
  one question twice in a single process can cost ZERO calls with the
  cache off and a freshly built memory — measured by somebody timing
  this library, who briefly recorded a spectacular and false "0 calls"
  for the change they were evaluating. The pooling is right in
  production and wrong to measure through; the fix is one process per
  variant, and it is now written down in both the code and the API
  page.

### Recorded

- `Answer.covered` (0.7.5) does NOT separate its proposer's three
  grades on their corpus, and they published the distribution showing
  it: partial cells at 0.31 and 0.82, a false-covered cell at 0.89.
  Partialness there comes from a QUALIFICATION — a footnote assigning
  the work elsewhere, a scope note contradicting the price table — and
  no word count reads that, in either direction. `covered` measures
  what it says and stays; it is simply not a grader, which is the
  second reason grades do not belong in this library.

## [0.7.6] — 2026-09-22

### Added

- **`ask(..., shape=...)`** — a caller who knows the turn's kind is not
  asked to prove it (W157). Traced from outside, one cell of a batch:
  six engine calls and 5 015 prompt tokens, of which routing is 26% —
  one call reading the message's shape, another asking whether the
  message is about the responder ITSELF. In a conversation both are
  necessary. In a batch they are a constant: 44 questions of one known
  kind, and the second question asked 44 times whether the caller is
  asking the assistant about itself.

  A declared kind seats both readings: **6 calls → 4**, on every cell
  of a batch. Everything after the door is untouched — the organs, the
  gates, the read-back, the 46% of that trace which buys the ability
  to say no.

  It is a declaration, not a hint: a wrong shape costs the organ that
  shape would have reached, exactly as a wrong shape read by the
  engine would. It is read in ONE place (`Session._turn_shape`), it
  lives for the turn and is cleared after it, so a batch's declaration
  cannot leak into a later ordinary turn. Left unset, every turn reads
  as it always did.

### Held deliberately

- The third proposal from the same report — `ask()` on a declarative
  sentence should ask rather than chat — is not in this release, and
  not because it is wrong: the reporter measured the catalogue's own
  wording retrieving better than a paraphrase (15/20 against 11/20,
  17/20 with both). It changes retrieval behaviour, and the run that
  would judge it is in flight. Shipping it now would make those
  numbers incomparable.

## [0.7.5] — 2026-09-22

### Added

- **`Answer.covered` and `Answer.missing`** (W156) — how much of the
  question the answer carried, and which of its demands it did not.
  `abstained` is binary, and an integration measuring 44
  supplier/requirement cells hit a structural ceiling on it: three of
  its gold cells are PARTLY covered and no configuration could emit
  that.

  The obvious repair was measured by the reporter and thrown away:
  asking the engine to grade completeness over the proof lines took a
  79.5% strict score to 38.6%, because a model shown six retrieved
  windows and asked "how complete is this?" always finds something
  missing. What ships instead is a COUNT, not a judgement — the same
  instrument as `evidence.coverage` pointed the other way, engine-free
  and unable to fabricate, in the same class as `where()` and
  `themes()`.

  The demands are the question's content words, and deliberately not a
  cleverer subset: two cleverer subsets were built and measured here —
  dropping what the store carries often kept the modal ("shall") and
  dropped a real demand ("material classification") for the crime of
  being written down, and a corpus-relative median cuts where nothing
  was asked. There is no stopword list in this codebase and there will
  not be one, so a question's quieter words ride along in the
  denominator the same way in every question, which is what leaves the
  measure comparable — one question against another, one supplier
  against the next.

  Three grades ("fully / partly / not covered") are NOT shipped: that
  is a procurement vocabulary, not a memory concept, and a library
  that learns one customer's matrix stops being a library. The
  measurement is the library's; the grades are the caller's.

### Open, with the reasoning

- **A refusal sentence can be stamped as an assertion** — reported
  from the same integration, two cells of 44, stable. A short refusal
  restates the question, W152 correctly subtracts the question's own
  words, and what remains is the refusal's register ("specific",
  "information", "have"), which an English business document carries.
  Reproduced here.

  Two repairs were built and refused. **Togetherness** — a claim
  quotes one region, a refusal gathers words from several (W135's
  doctrine) — measured backwards: True for both refusals, False for
  W97's genuine claim. **Store frequency**, the reporter's own
  suggestion, is the right instrument and has no boundary that is not
  a dial: their own table puts "and" at 48% of sentences, under the
  majority this codebase uses everywhere, and a corpus-relative median
  was measured cutting the wrong words.

  It also runs with the grain of a deliberate choice: this reading
  "errs towards ASSERTED", because claiming an abstention that did not
  happen is the damaging direction. The reporter measured the obvious
  guard (require a source stamp) at net zero — two false-covered
  removed, two true positives lost. So it waits on the chat slice
  being measurable again rather than on another idea.

### Measurement

- The ingestion-shape test (E5) was measuring the machine's mood. Its
  timed loop runs in single-digit milliseconds at the small size, and
  it was caught FAILING on a tree whose ingestion had got faster — the
  small run sped up more than the large one and the quotient rose
  past the bound. It now collects the garbage of two hundred earlier
  tests first and takes the best of three runs at each size; the bound
  did not move, and the shape it reports is 5.8x for 5x the data
  instead of a noisy 11x.

## [0.7.4] — 2026-09-22

### Fixed

- **A strict minority needs three regions to exist** (W154). The
  reported "a second document destroys the correct answer" was real,
  and the mechanism was in neither place we looked. Both of us hunted
  a lost SEAT — three synthetic reproductions of that failed. What
  changed between the reporter's two runs was the number of REGIONS
  the proof folds to: one, then two.

  At exactly two regions the load-bearing test is unsatisfiable.
  `half` is 1.0, and a word must hold `holders >= 1` to be in the
  proof at all and `holders < 1.0` to be in a minority of it. No word
  can do both, so nothing without a digit in it could ever be
  load-bearing and every such answer was stamped an abstention — a
  dead band exactly one width wide, entered only by going from one
  region to two, which is why a second document looked like the
  cause.

  The boundary stands: register is what the proof BLANKETS. But a
  strict minority needs three regions to exist — with two, every word
  present is in one or both and neither is a minority, so there is no
  register to subtract and the reading defers to coverage, exactly as
  it already did with one region.

  The report proposed widening the comparison to `holders <= half`
  instead. Measured on W97's own fixture, that breaks it: at four
  regions "have" and "information" (two lines each) become
  load-bearing and an honest refusal is stamped an assertion — the
  thing W97 exists to prevent. The population bound leaves every
  three- and four-region proof byte for byte as it was.

### Still open

- Seat displacement in a multi-source store: at `most=6`, a second
  document's line can take the seat of a lower-ranked line from the
  document that holds the answer (at `most=10` the two orders agree).
  Reported with data. It may no longer be reachable end to end now
  that the dead band above is gone — the same report shows the rescue
  pass finding the target line and the turn refusing anyway, which is
  what that dead band did — so it is measured again before the
  seating rules, whose monopoly behaviour was measured on 62 sibling
  documents, are touched.

- A refusal came back in mixed language ("Ich do not know.", route
  `('chain', 'widened')`). The answer path, not `_refuse`, so the
  language match that refusals get never ran.

## [0.7.3] — 2026-09-22

### Fixed

- **Repeating the question is not a claim of its own** (W152). The
  gate before the one 0.7.2 fixed, and the same reporter's corpus
  found it: a document writes `4-Augenprinzips`, the engine answers
  with the same term spelled `Vier-Augen-Prinzips`. A numeral against
  its written-out twin, a compound against its split halves — three
  tokens read as unsupported, coverage lands at 0.44, and a correct,
  sourced answer is stamped `abstained=True` over a spelling variant
  of the question's own subject. 3/3 reproducible, and reproduced here
  before it was touched.

  The reading already strips what is not a claim: the engine's
  parentheses, its `#stamps`. Restating what was ASKED belongs in that
  class — an answer that says the subject back has asserted nothing by
  doing so, and what it asserts is what it says BEYOND the question.
  `evidence.coverage` has taken a `question` argument all along for
  exactly this; the abstention reading simply never passed it, and now
  both of its callers do. Nothing is loosened: the gates that decide
  what may be SPOKEN are elsewhere and untouched. This reading only
  decides the stamp.

### Reported, not reproduced

- **"A second document destroys the correct answer."** Reported as
  3/3 stable: one vendor's answer is correct alone and refused once a
  second vendor's document shares the store. It matters more than
  anything else on this list — a tender store holds eight vendors —
  and it is not shipped as a guess. Three synthetic reproductions were
  built (word channel alone, meaning channel on, target ranked low
  inside its own document) and the target line survived every one, so
  the mechanism is not yet in hand. Changing the seating rules on a
  hypothesis would risk the monopoly behaviour measured on 62 sibling
  documents. A reduced case, or the two documents, closes this.

- A refusal came back in mixed language ("Ich do not know."). No
  reproduction yet; the turn's `.route` would say which path spoke.

## [0.7.2] — 2026-09-22

The same reporter, measuring 0.7.1 against the same two tender
documents: the abstention bug was not closed. 0.7.1's fold asked for
strict containment, and the evidence index seats a passage as windows
SLID by a sentence — each carries a few words of a neighbour, so none
is a subset of another, nothing folded, and a correct sourced answer
was still stamped `abstained=True`.

### Fixed

- **Sliding windows of one passage are one attestation** (W148, second
  cut). The fold now asks `evidence._same_region` — the question this
  codebase already had an organ for, majority by Jaccard or by
  containment ratio, whose own docstring warns that answering it with
  a second rule is how two organs end up disagreeing about what "the
  same" means. It is transitive, because three windows of one passage
  are one attestation and not two.

  And it **stops at the source**. Measured while building it: "The
  Alpha course runs for two full days in Berlin" and its Beta sibling
  read as one region on words alone — folding them would have erased
  exactly the independence W39 exists to protect (the same sentence in
  two documents is two attestations). Lines from different sources
  never merge.

  The folding is now a named organ, `Session._regions_of`, rather than
  a block inside the abstention reading: one place to test, one place
  to read. A proof that folds to a single region has no register to
  subtract and the reading defers to coverage, exactly as before W97 —
  which is right for a document seen at several scales, and leaves
  chat evidence, whose lines subsume nothing, measuring unchanged
  (W97's own fixture folds nothing at all).

### Known, and not silently patched

- A refusal can still be read as an assertion when one of its words
  (an "I do not have **that**" sort of word) happens to sit in a
  minority of proof lines. Present in 0.7.1 and before, not introduced
  here, and closing it properly needs the chat slice re-measured
  rather than a guess.

## [0.7.1] — 2026-09-22

Five findings from somebody integrating 0.7.0 against two real tender
documents — the kind of report that only comes from outside. Two of
them inverted this project's central promise, and both are closed.

### Fixed

- **A correct, sourced answer could be stamped `abstained=True`**
  (W148). Reproduced 3/3 by its finder: every word of the answer was
  written in the document, and the turn reported it as a refusal with
  no sources. The abstention reading (W97) subtracts "register" — words
  the proof blankets — from the content that carries a claim, and it
  stands on proof lines being INDEPENDENT attestations. The
  multi-scale evidence index breaks that premise: it seats the same
  passage at several widths, so the answer's most distinctive words sit
  in a MAJORITY of proof lines and are struck as blanket. Now a line
  whose words are a subset of another proof line's is the same region
  read at two scales and casts one vote. Distinct lines — the chat
  evidence W97 was written on — subsume nothing and measure exactly as
  before.

- **An unreachable engine is no longer indistinguishable from honest
  ignorance** (W149). With no engine — client package absent, key
  refused, network gone — every question returned "I don't know",
  `abstained=True`, empty route: infrastructure failure wearing the
  exact signature of the thing this project asks to be trusted for. In
  the reporter's use it marked a vendor's real capability as "not
  covered". The dispatch's failures now carry one name
  (`runtime.EngineDown`), the turn records an `engine-error` step in
  its route, and **`Answer.engine_error`** states it. The turn still
  abstains — nothing was asserted — but a caller can retry or alert
  instead of writing down an absence.

  The configuration case keeps both readings: `runtime.EngineMissing`
  is an `EngineDown` *and* an `ImportError`, so a fresh machine still
  gets the message naming the four engines and the `pip install` line
  (W62), while the turn still reports an engine error.

- **The package stated two versions** (W150). `importlib.metadata`
  said 0.7.0 while `lmm.__version__` said 0.6.5 — the release bumped
  `pyproject.toml` and the hand-written string was missed. The string
  is now derived from the installed distribution's metadata, and the
  invariant checks the build and the report agree, so they cannot
  drift again.

### Added

- **`Learned.calls`** — what a reading cost in engine calls (W151),
  counted at the engine's one door and only on real dispatches. `deep`
  defaults to False for a file and True for text handed in directly,
  which is sound and was invisible: a reader who cleans a document into
  a string first, the ordinary thing to do, paid one call per sentence
  without being told. The trade is now a number beside what was gained.

### Documentation

- The API page carries `Memory`'s full signature with the defaults the
  code actually holds (`warmth` and `reply_tokens` are `None`, not 0.6
  and 1200), and documents the entry points it had been missing:
  `learn(source=, deep=)` — the parameter the whole multi-document
  story rests on — `ask(fluent=)`, `compose(seats=)`,
  `themes(least=, most=)`, and `Answer.engine_error` with the
  three-branch pattern a caller should write.

## [0.7.0] — 2026-09-13

Conversation memory learns what "living" means: a told fact can END, a
question is asked ON a day, a thing HAS a kind, and a small telling can
be read whole. Six mechanisms, each generic (no word of any language in
code), each behind a failing test first. Measured on the same
30-question chat-memory slice throughout: **43% → 57-60%**, wrong
claims 7 → 5-6, no category below its start — while the 31-question
document exam stayed at **30/31, byte for byte** the same behaviour on
undated stores. Invariants 211 → **218**.

### Added

- **The graph count asks which facts still stand** (W141). "I canceled
  my Forbes subscription" distils into a record whose own value says it
  ended; the deterministic walk cannot read an ending in any language,
  counted a cancelled thing, and the number flipped run to run with the
  extraction's mood. Now the walk GATHERS, the candidates are laid out
  by day in their records' own words, and the engine — the one
  instrument that reads language — is asked which still stand. An offer
  no candidate wrote is discarded; an empty or failed reading falls
  back to the full gather; the count never exceeds what the graph
  gathered.

- **A count's block reads in time order when the stamps are dated**
  (W140), laid down before the asker's stable priority seat so W98
  holds — and `items_of` is told it may trust that order only when it
  is real. An undated corpus builds its block byte for byte as before.

- **The plan knows the day the question is asked** (W142). `now` is a
  primitive beside `anchor`; the operator may set `session.asked_at` (a
  benchmark replays last year's questions, a letter is answered a week
  late), and with none set it is the calendar's today. Verified by
  construction — the clock is not a claim. A span that covers whole
  months says so in months too. "How many days ago did I harvest" went
  from an abstention to the exact answer.

- **A distilled thing carries its KIND as an index key, never a claim**
  (W143). "I'm also getting Architectural Digest" holds no word of
  'magazine' or 'subscription'; the kind the distiller files
  ("magazine subscription") lets the question reach the record. It
  lives in the evidence aids beside the bridges — a side-file, never
  the graph — widens what the gather can FIND, and no gate, label or
  spoken word reads it. Deliberately unverified against the passage,
  for the same reason a bridge can be: it licenses retrieval, never
  speech. The distiller also lists ONGOING facts the speaker reports
  about themselves — a subscription told in the present tense never
  reached the graph before.

- **A small dated telling is read whole when every organ dies** (W147).
  Measured on twelve missed chat questions: handed the same lines whole
  and in time order, the bare engine answered six the selective gather
  threw away — dispersed clues connect only when they sit in one view.
  A new organ in the rescue seat, NOT the chain: it runs only on an ASK
  that still asserted nothing, only when every line is dated, only
  under a measured reading budget, and its output passes the chain's
  own two-tier gate — digits non-negotiable (an invented "14 years"
  dies structurally), word residue by read-back. "Which mode of
  transport did I use most recently" and "what are the two hobbies" now
  answer correctly, stamped.

### Fixed

- **A list is not the name of one thing** (W144). The distiller offered
  "A, B, and C" as ONE thing and the count answered "1:" while listing
  three. The guard is punctuation, not language: comma-split parts, two
  or more of them each carrying two or more content words, are a list —
  skipped, never rewritten; the evidence line stays.

- **Membership by name or kind, never by story.** "Using Mendeley to
  organize my project sources" carries 'project', and the tool was
  measured entering a count of projects; a kind is matched as a UNIT
  ("project management tool" cannot vote on 'project' alone). The
  projects count went 8 → 2.

- **An enumeration neither outvotes a stated tally nor counts one
  breath** (W145). The widened distiller filled the graph and the graph
  count began preempting every "how many" — four stated-tally answers
  ("20 playlists", "32 species", "15 videos", "10-12 hours") broke at
  once and came back with three structural guards: candidates wearing
  one stamp are one telling; a gathered line carrying a number beside
  met question-words outranks the enumeration; and that tally must live
  in the ASKER's line — an assistant's chatty digit silences nothing.

- **A leading ordinal is layout, not a name** — the engine's "1. " no
  longer leaks into an item's name.

### Recorded

- A seventh mechanism was built, measured, and REVERTED before release:
  handing the organ-dead turn to the factual chain contradicted W93's
  own measured basis (the chain's 23 confident wrong claims were
  measured on chat, not documents). The telling organ above is the
  corrected design — a new organ with its own gates, not a loosened
  chain. The conflict and the reasoning are in the tests.

- Scoring fidelity, harness-side: LongMemEval's insufficiency questions
  (gold text says the information is not enough) are abstention
  questions; an explicit abstention now scores as the correct
  behaviour it is.

## [0.6.5] — 2026-09-13

Three defects reported by somebody testing a bot built on this library,
which is the only place this kind of thing is ever found. Two reproduce
and one was worse than described; the third does not reproduce. A
fourth was found by re-running the exam afterwards and is a cost of
0.6.3's own change.

### Fixed

- **An underscore is a space that survived a filename** (W138). `\w`
  counts it as a letter, so a document named `Alpha_Sales_101` was ONE
  token and a reader asking about "Alpha Sales 101" — two tokens — could
  never name it. Six of a 103-document catalogue were unreachable that
  way. The report said such a question went unanswered; it is worse,
  because naming BINDS the turn (W130): the name failed, a neighbouring
  document's name won, and the answer came back confident and stamped
  with the wrong document.

- **The graph speaks the spelling it was given, not its index key**
  (W137). Folding is how two spellings become one concept and it is
  load-bearing, but it was applied at EXTRACTION, before identity was
  ever consulted, so the surface a document used was gone before
  anything could keep it — and the graph path speaks what it keeps. A
  document written with its reader's own accented letters came back with
  them flattened: correct, and misspelt in that reader's language.
  Extraction now only cleans invisible combining marks, which is its
  actual job, and the identity layer keeps the surface beside the folded
  key with `label_of` preferring it. Every lookup still folds first.

- **A document that does not write a field can still answer** (W139).
  The record path owns a turn that names a document, and when its field
  reading came back empty the turn REFUSED — right before naming bound
  the turn, wrong after, because the scope IS that document now and the
  ordinary path reads it and nobody else's. Measured: a corpus writing
  its duration in days, asked for HOURS, bridged to a different head,
  came back empty, and refused a question the document answers in its
  own first line.

### Did not reproduce

A refusal arriving in English against a Turkish setting. Refusals come
back in the question's language.

### Note for existing stores

Both index-key changes here — and the case-folding change in 0.6.2 —
mean a store written by an earlier version searches under keys the
current one no longer produces. Re-learn the documents and re-run
`bridge()` to get the recall back.

## [0.6.4] — 2026-09-13

Measured against a 101-question slice of LongMemEval and a 31-question
document exam whose gold is extracted FROM the documents. The document
exam holds at 30/31 (field questions 20/20); the chat-memory slice
scores 34.7%, and the reason it did not move is written below because
it is the most useful thing this release found.

### Added

- **`Memory.themes()`** (W136) — the subjects a corpus falls into,
  with NO engine. A corpus has subjects no single document names: a
  catalogue's programme families, a manual's subsystems. GraphRAG finds
  them by running community detection over an LLM-extracted graph and
  then paying an LLM per community summary; RAPTOR clusters embeddings
  and summarises each cluster. Detection costs nothing here because the
  graph already existed — entities are phrases whose words co-occur
  beyond chance, edges are witnessed co-mentions weighted by
  log-likelihood. Measured on 103 documents: 501 entities, 5,671 edges,
  groups in a hundredth of a second, zero calls, and the largest fell on
  the catalogue's own families.

  It is deterministic (the textbook shuffles its nodes; this cannot),
  it writes no summary, and an entity carried by most of the corpus is
  left out — without that filter the top group was the field headings
  every document repeats.

  **It does not answer "what is this family about", and was measured
  trying.** Scoping the composer to a community's documents returned one
  course's outline, not the shared subject. `themes()` is a structural
  reading and claims nothing more.

### Fixed

- **Two events told the same day are not ordered by the clock** (W134).
  The ordering organ already declined a tie, and the guard never fired
  because the comparison took every digit in the stamp — including the
  time the message was SENT. Two events reported on one day were ordered
  by which message was typed first. Three wrong claims became
  abstentions, each of which had been printing both events with the same
  date beside it.

- **A counted name is attested as a unit, not word by word** (W135).
  Verification asked whether every word of an item appears SOMEWHERE in
  the block, and the block is many lines — so a whole clause passed when
  its words were scattered across them. The engine offered "which are
  your Data Mining project and your Database Systems project" as ONE
  item and the turn answered "1:" while listing two things. One line
  must now carry the whole name, which is the rule the sum organ already
  kept.

### Why the chat-memory score did not move

The retrieval work in 0.6 took the answering line's chance of reaching
the engine from 32% to 79% — on a corpus of 103 documents. This
benchmark gives each question a median of TWO sessions and 24 messages.
There is no "which document" problem in it to solve. Its failures are
aggregation and arithmetic: of 32 wrong claims, 22 came from the
counting, planning and ordering organs and none from retrieval. Two of
those classes are fixed above; the largest remaining one is the class
the README already names — a long conversation reports the same thing
in different words each time, and the list comes back short.

## [0.6.3] — 2026-09-13

Found by tracing one failing turn end to end instead of guessing at it.
Every step before the last was right, and the last one threw the answer
away.

### Fixed

- **A field is its words, not its capitalisation** (W132). One document
  writes "COURSE LENGTH:" and its neighbour "Course Length:". Keyed by
  the literal spelling, each looked like a field ONE document uses — so
  the reading that keeps what the corpus REPEATS threw both away. What
  followed all pointed somewhere else: the bridge nominated a field
  nobody held, the one-head rule saw a tie and declined, the record path
  stood aside, the chain ran, and the field veto fell back to matching
  heads by kinship, reached a DIFFERENT head that merely shared a stem
  with the question's word, and vetoed a correct answer for not carrying
  that field's digits. Measured on the live catalogue: the turn now
  answers from the record in 2.7 s where it refused in 25.3 s.

- **A greeting is not a failed question** (W133). The rescue seats read
  `last_abstained`, which means "this turn asserted nothing" — true of
  every greeting by construction. So "hello" opened the delivery bridge
  and the composer's general door and paid for both. Seven engine calls
  for a greeting, of which one wrote the reply; six now, 13.6 s to 8.7 s.

- The second record door stamps the route like its twin (W110's third
  case): a path that speaks says so.

### Measured, and worth knowing before tuning anything

Latency is round trips, not model size and not prompt size. On a hosted
deployment a bare call costs about 1.13 s with a 2,000-character prompt
and 1.55 s with a tiny one; 200 output tokens costs 2.87 s. A turn's
wall time is very nearly its call count times that. Our own code is not
in the picture: metered over a greeting, the library's share was
NEGATIVE against the sum of its calls, because two of them run in
parallel.

An optimisation was tried and refuted: the door's three readings are
independent and could be fired side by side, but a speculative prefetch
puts a model call into the graph path, whose whole claim is that it has
none. Four invariants said so. A second is not worth the property.

## [0.6.2] — 2026-09-13

Found by asking a hundred real documents every kind of question a
reader asks, with the gold answers extracted FROM the documents rather
than written by hand. Field questions — the most basic thing this
library does — were answered 8 times in 20. They are now answered 20
in 20, and two of the three causes were ours for years.

### Fixed

- **A word folded from capitals now reaches the same key as from lower
  case** (W131). `casefold` is not idempotent across case for every
  script: 'I' folds to 'i' and 'ı' folds to 'ı', so a word written in
  capitals and the same word in lower case became TWO index keys — and
  documents write their field headings in capitals. Measured:
  a capitalised field heading was indexed under one key while the same
  words typed in lower case searched another, so the line stating the
  answer could not be reached by the question that asked for it. Every gate then behaved correctly, so it read as an honest
  "I don't have that" about something the document plainly states.
  Raising the letter before folding it makes the fold agree with
  itself — same Unicode table, no language rule, length still
  preserved; it also merges Greek final sigma.

  **This changes index keys.** A store written by an earlier version
  keeps answering, but to get the recall back, re-learn the documents.

- **A question that names a document is answered from it** (W130).
  Naming used only to clear the previous subject; the turn then read
  the whole store. On a catalogue where every course states the same
  fields, the named course ranked first and its own answering line was
  not in the top forty. Naming now binds — and inside that binding the
  name stops scoring, because every line in play is from that document
  and the words that spelled it were pulling the masthead to the top.

- **Lines inside a proposal are ranked by what a word is worth**, not by
  how many matched. This was the one place in retrieval that ignored
  rarity: a document about one subject writes that subject's word on
  every line, so the question's topic words outvoted the two that named
  the field.

### Known, named rather than hidden

- A compound value can lose a part: asked one course's duration, the
  memory answered "2 full days" where the document says "4 modules x 2
  full days". Everything said is written; the gate cannot see what was left
  out.
- A count-shaped question the corpus cannot answer can still be
  answered by counting the wrong things: "how many people completed X"
  came back with a list of other courses. Each item is written
  somewhere, so the gates pass it.

## [0.6.1] — 2026-09-13

Two defects found while testing 0.6.0 the way a user would — installed
from PyPI, against a hundred real documents.

### Fixed

- **The reordering is given something to reorder** (W127). The proposal
  was cut to the seats being filled and THEN reordered, so the reranker
  could only shuffle what the words had already chosen. It hid behind
  the multi-document case (five documents giving up three lines each is
  fifteen candidates); on a ONE-document store, asked for three seats,
  the answering line was missed entirely and with a pool it comes first.
  The pool is now a floor on the whole proposal, shared across the
  documents proposed — widening the per-document share instead was
  measured and was worse (78% → 75-77% in five).

- **A line is credited to the document that offered it** (W128). The
  fused reading recovered each line's source by looking the TEXT back up
  in the store, first match wins — and a catalogue repeats lines across
  documents, so a line was credited to whichever document sat first.
  Read off a live 103-document corpus, four different questions came
  back stamped with the same alphabetically-first training. Provenance
  is now recorded as the line is proposed, and the lexical channel's own
  sources are carried through the fusion.

### Measured, and named as a limit

The bundled static matrix is weak at mapping a PROBLEM STATEMENT onto a
training title — a stated problem onto the course that treats it.
On eight such questions over a 103-document catalogue it placed the
right course in the top three once; a torch multilingual encoder
(`paraphrase-multilingual-MiniLM-L12-v2`) placed it four times. Known-item
retrieval on the same corpus is 100% either way, so this is a property of
the class, not of the wiring: content matching is solved, need-statement
matching is not. Where that class matters, pass `encoder=`.

## [0.6.0] — 2026-09-13

The release where retrieval stopped being only words, and where three
hedges against weak retrieval were retired because measurement said they
had stopped earning. Every number below was produced by a protocol in the
repository; none of it is an estimate.

### Added

- **The meaning channel ships with the library and is on by default.** A
  30 MB static multilingual matrix travels in the wheel, so a reader who
  paraphrases — a lease that says RESIDES against a question that says
  LIVES — is understood with no download, no account and no network. It is
  a matrix, not a model: one lookup per word-piece and a mean, so there is
  no torch and no warm-up. Measured on 20,000 lines and 180 known-item
  queries it beats the transformer it would otherwise take (MRR **0.883**
  against 0.710) and builds its index a hundred times faster (0.3 s against
  36.1 s). Source: `minishlab/potion-multilingual-128M` (MIT, the model2vec
  method), reduced to 108k pieces and 128 dimensions; the segmentation is
  thirty lines of our own, so no tokenizer library is required. See NOTICE.

- **Rank fusion, and late interaction over the same matrix.** The word
  ranking and the meaning ranking are fused by reciprocal rank (RRF, K=60)
  — order, not score, so there is no weight to tune. The lines inside a
  proposal are then reordered by late interaction (ColBERT's idea without
  ColBERT's model). End to end, by whether the ANSWERING line reaches the
  five the engine is shown, on a 115,913-line corpus: **32% → 60% → 79%**.
  Reordering is worth more the vaguer the question gets (six terms removed
  from the query: MRR 0.707 → 0.807), which is the way round that matters.

- **A parametric surface for all of it.** `Memory(encoder=...)` takes any
  callable from strings to vectors; `reranker=` any callable from
  `(query, texts)` to an order, or `None`; `dense=False` removes the
  channel entirely. With no numpy, no bundled data, or an encoder that
  raises, the channel is **absent, never degraded** — the words carry the
  turn.

- **W123**: the derived field-name index is written beside the store. It
  was built on first use, once per PROCESS: 10.6 s before the first answer
  on a 115,913-line store, paid again by every command-line run. Now
  0.7 MB on disk and **0.13 s**, with the sentence count and the bound it
  was built under, so a file that does not match is rebuilt rather than
  trusted.

- **W117**: a deterministic question is asked once. At temperature 0 the
  engine is a function, and one refusal turn was asking the amount reader
  twice on the same 45,755-character block. Byte-exact pooling, and warmth
  is never pooled.

- **W125**: every shipped entry point imports what it runs — including the
  lazy imports inside functions, which is the class of breakage no
  import-time check can see.

### Changed

- **The meaning channel sits at the DOCUMENT layer, and the layer was
  measured.** Over lines it moved nothing; over document profiles the
  store's own retrieval went **92% → 99%** on 103 known-item queries. The
  opposite arrangement was measured too: choosing the lines INSIDE a
  proposed document by meaning rather than by words took the answering line
  from 78% to **31%**. Meaning says which document; inside it, the
  question's own words are sharper.

- **`CANDIDATES` 3 → 1** (W124). Three evidence subsets were answered per
  question and the gate chose among them — insurance bought before the
  meaning channel existed. Measured after it: across eleven field questions
  every admitted answer came from the wide block and the narrow candidates
  won **nothing**. Collapsed, the run went from 139 engine calls to 104 and
  correctness from 4/11 to 5/11. It stays a number, not a deletion.

- **`VIEWS` 2 → 1** (W126). Both judges could buy a second, wider view when
  the first declined. In the same run the second view confirmed **nothing**,
  on either judge — eight calls of a hundred and four.

- **One dependency: numpy**, named in `pyproject.toml` with its reason.
  "Zero dependencies" was true and is no longer; hiding that would be worse
  than losing it.

- `learn()` lost its `expand=` argument, and `Learned` its `expanded` count.

### Fixed

- **The memory answers the identity question it was asked** (W120). Asked
  WHO WROTE IT, it answered with its own NAME — true, attested, past every
  gate, and an answer to a different question. The name is one field among
  the rows the operator declared, and the question is now mapped onto those
  fields — including the fields we could hold but do not, because a mapping
  shown only what it has can never discover that something is missing.

- **The meaning channel is attached where the store is built** (Session,
  not Memory). A caller using the older class got a store with no vectors
  and no way to tell; the failure was silent.

- **The graph-direct path records its route** (W110's fifth path). It spends
  no model call, so it was never instrumented, and a record question came
  back with an empty route — indistinguishable from a turn that never
  happened.

- **`lmm/mind.py` restored.** It was retired as "an idle derivation loop on
  no product path" and is on the product path: the `lmm` console command
  builds it at startup. The import sits inside `main()`, so the package
  imported cleanly, every test passed, and the command crashed on launch.
  W125 exists so that cannot recur.

### Removed

- **The offline expansion channel, entirely** (−607 lines). It generated
  the questions each line answers and indexed them; measured on NIST
  SP 800-63B its keep filter is inverted — a real rephrasing never reaches
  its own line, so it kept only the queries that COPIED the line. Handed
  perfect queries by hand, it kept zero. The README recorded that for
  months while the channel stayed, off by default, costing a rung on the
  channel ladder, a side-file key, a prompt, a session pass, an api
  argument and five invariants. A switch nobody should turn on is a
  liability, not an option.

- `generate.wants_material` — it was `turn_shape`'s "material" asked a
  second time with a second prompt on the same sentence.

## [0.5.0] — 2026-09-11

The release where question shapes stopped being hand-written. Everything
here was found by measurement — a public benchmark, a live consultation,
a drowned block read line by line — and every fix is a class, never a
case: the two rules in CONTRIBUTING (no document-specific constants, no
hand-written language rules) held through all of it.

### Added

- **The composer** (`Session._plan_answer`). A memory that answers only
  the shapes somebody wired by hand answers a finite set of questions.
  The engine now proposes a PLAN over the verified primitives — anchor a
  phrase to its dated line, take the latest, gather lines, span two
  dates, order them, filter by a cutoff, count, tally by month — and the
  interpreter executes only operations it knows, on phrases the store
  can anchor. The sentence is built from the final step's TYPED value by
  our own template: the engine contributes operation names and phrases,
  never an output word. An unknown operation, an unanchorable phrase, a
  dangling reference — the plan dies and no claim is born.

  Measured on a dated store: five shapes nobody wrote an organ for —
  *did A happen before B*, *in which month most*, *how many days
  between*, *when first*, *when last* — answered correctly with no new
  code. And the seat binds the plan's output type: a count-shaped turn
  will not voice a month tally, however correct its arithmetic.

- **Four event organs, each the arithmetic its question asks for.** A
  count is the length of a VERIFIED LIST — the engine lists the items,
  the store checks each against the block, and the number is the length
  of what survives, so a hallucinated item cannot be counted (and where
  the things counted are documents, the census counts them with no model
  call at all). A total is the SUM of verified amounts. "Which came
  first" is date arithmetic over anchors. "How many days between" is
  subtraction, not counting.

  The shape reader that routes them is one small call at the door
  (material / count / order / sum / when / none), replacing three
  separate classifiers asked after the fact — and its verdict binds both
  ways: organs silent and plan dead, the turn REFUSES rather than
  gambling the prose chain. Measured on a benchmark slice, that chain
  had earned two right answers and twenty-three wrong ones.

- **Dialogue's two missing columns.** WHEN an event happened is the
  sentence's word before the envelope's: people tell events days later,
  so the engine reads the candidate lines and proposes the date, and the
  arithmetic decides whether the reading may stand — the date must match
  a candidate's stamp, or its day must be written as a number in that
  line with the year in the stamp's neighbourhood. An invented date
  fails both and the envelope speaks. No month table, no date grammar.
  WHO SPOKE is metadata as real as the date: `learn(..., speaker=)`
  keeps it per line, `session.asker` says who is asking, and the event
  organs seat that speaker's lines first.

- **`Memory.distil(text)`** — the write-time reading for passages where
  events melt into talk. The engine lists each event as
  thing/what-happened, the passage's own words admit it, and each
  survivor becomes one gated dated record, while the passage is kept as
  evidence so nothing is lost. Documents that state their own structure
  need none of it; their rows reach the graph for free.

- **`Memory.bridge()`** — one call per field head, once, teaching the
  store what words readers ask each field with. Measured on a live
  62-document corpus: a duration question fell from 12 s and six model
  calls to **2.7 ms and none**.

- **`Answer.route`** — which organs a turn passed through, in order
  (`("record",)`, `("chain", "refuse", "count")`). The orchestration was
  always there; now it is a fact on the answer instead of a story in the
  stack, and an organ a router can be asked about is an organ a planner
  can schedule.

- **`LMM_SMALL_BACKEND`** — the turn's yes/no classifiers may take a
  cheaper road (a small local model agreed with the hosted engine 19/20
  on fresh multilingual examples). OFF by default: one measured
  classifier is not a licence to reroute sixteen.

### Fixed

- **No apparatus of ours can be spoken.** Asked which documents state a
  field, the memory answered "K1, K2, K3, K4" — reciting the labels this
  code puts in front of evidence lines. The label is ours to choose, so
  it is now the name of the source that wrote the line: an answer that
  echoes a label cites a document.
- **An untrusted language name is never a command.** The language
  classifier was handed the question bare, the engine began answering it
  ("I'm sorry, but"), the token cap clipped that to "Im" — and refusals
  came out in an invented language. The classifier's text is quoted
  material now; the name only compares; the retry may name a language
  only when a second, differently-phrased reading agrees. Six languages,
  six correct refusals.
- **A refusal in chat register is not an assertion.** Chat evidence
  speaks a refusal's own words, so fifty-three honest shrugs wore
  provenance marks and were scored as wrong claims. An assertion now
  needs something load-bearing besides word overlap: a shared digit, or
  a shared word sitting in a minority of the proof's lines.
- **`LMM_TIMEOUT=0` removes the limit, not the patience.** It was
  removing the retry, so under load every rate-limit wait was raised raw
  — forty-nine of fifty answers in one benchmark run came back 429.
- **A derived row is spoken as a sentence**, with the arithmetic
  preserved: "3: Dr. Lee, Dr. Smith, Dr. Patel." became "You visited 3
  different doctors." A phrasing that changes or drops a number is
  refused and the row stands.
- The answer cache now fingerprints the retrieval aids, so a question
  asked before `bridge()` is not replayed after it. The record path no
  longer matches field names by short-stem kinship — it speaks without a
  gate, and kinship called two sibling fields relatives. Newer Azure
  deployments (which refuse `max_tokens` and any non-default
  temperature) are handled by reading the endpoint's own refusal, not a
  model-name list.

### Measured, and stated plainly

The field set is unchanged: 40/40 factual · 15/15 comparisons · 10/10
traps. On LongMemEval — a long-personal-conversation benchmark, not a
document one — this release scores 37.6% with gpt-4o-mini, against
published 63.8% (Zep) and 49.0% (Mem0) measured with far stronger
answering models. Its temporal "how many" slice moved from 2 correct to
11 across this work; its cross-session counting slice did not move
beyond noise, and README's Honest limits says so. 182 invariants.

## [0.4.0] — 2026-09-10

The release where retrieval stopped being one channel. Two new readings,
both built from the document itself with no model call and no vendor, and
both able to show why a line arrived — which is the property this project
would have had to give up to buy the same abilities from an embedding.

### Added

- **The document's own entity graph** (`lmm.mentions`). GraphRAG hands
  each chunk to a model and asks for the entities and relations; what
  comes back is a claim, invented at index time and unverifiable
  afterwards. This graph is built by the document: an entity is a phrase
  whose words occur together beyond chance (pointwise mutual
  information over the phrase's weakest split), which also keeps varied
  company (no neighbour taking more than half its appearances) and whose
  parts do not choose freely (seeing one, the other is nearly
  determined). On a public novel the three readings take 2,318
  candidates to 487, headed by the book's actual names.

  An edge says one thing — THESE TWO ARE MENTIONED TOGETHER, HERE — and
  carries the sentence that witnesses it, weighted by Dunning's
  log-likelihood ratio so an entity that appears everywhere does not
  become everyone's neighbour. An edge cannot be fabricated: it is an
  observation. What a relation MEANS is still a claim, and claims go to
  the gates with the sentence in hand.

  What it buys is scale, and the honest version of that claim is the
  point: on a 36,472-line novel the lexical search already found a line
  carrying both entities, six pairs out of six, and took 417 ms a
  question doing it. The same lines come out of two posting lists in
  0.01 ms.

- **The corpus as its own thesaurus** (`lmm.affinity`). A text says a
  tenant RESIDES at an address and a reader asks where she LIVES; lexical
  retrieval cannot bridge that. Two words used for the same thing keep
  the same company, which is arithmetic over counts the store already
  holds: positive pointwise mutual information for what company is
  surprising, cosine between profiles for how much two words share, and
  an inverted context index so a query compares against the words
  sharing two of its contexts rather than against the vocabulary
  squared. A neighbour comes back WITH the shared company that earned
  it.

  A substitute keeps the same company without being in it: words
  appearing together in more than half the lines of either are
  companions, not alternatives — measured, the first cut answered
  "lives" with "nearby" and "visitor".

- **One harness for any document** (`benchmarks/document_probe.py`).
  Point it at a file or a folder; it builds the memory the way a user
  would, generates its own questions from the document's structure, and
  reports what each organ did — records, census, extremes, comparison,
  absent fields, nonsense, a rephrased question, the language of a
  refusal, whether claims carry stamps, and the calls, tokens and
  seconds each question cost. It asks the kind of question the document
  can answer: a head that opens a hundredth of the units is a field, one
  that opens three lines of a novel is a colon.

### Fixed

- **A row is reachable by the name a question uses.** A spreadsheet
  writes people as "Taylor, Mr. Elmer Zebley" and a question arrives as
  words; the graph kept only the written spelling, so on an 891-row
  public table the graph-first path — the one that answers with no model
  call at all — matched none of the questions a reader would ask. Row
  questions 0/8 to 7/8, and the cost per question 9.3 calls and 7,523
  prompt tokens to 2.4 and 1,761.
- **The language judge did not work.** Asked whether a reply was in the
  same language as the question, the engine answered NO for an English
  question answered in English — a verdict carrying no information, so
  every refusal was rewritten once and then kept whatever came back.
  Naming one sentence's language is a smaller question the engine
  answers reliably; two names compare by arithmetic. Twelve refusals
  across six languages, twelve right.
- **No person is named in a shipped prompt.** The chat prompt introduced
  the library's author by name, so every product built on this told its
  users who wrote the framework. Identity is the operator's declaration
  (`Memory(identity=...)`); a memory that was told nothing says what it
  can attest, which is that it is a memory.

Regression across the release: the 73-question field set 40/40 factual ·
15/15 comparisons · 7-8/8 frontier · 10/10 traps; twelve hardware
specifications the system has never seen 8/8 · 4/4 · 3/3 · 4/4; a
thousand generated documents 25/25 · 25/25; NIST 11/13; EN 17/17; 158
invariants.

## [0.3.6] — 2026-09-09

### Changed

- **A document that never uses the words is not asked about them.** The
  mirror of the record answer, on the refusal path: a question naming
  one document and one of the corpus's own field heads, where that
  document never uses the head's words anywhere — record or prose — has
  no answer in the only source it may be answered from, so the three
  candidates and the three judgements that would refuse them are not
  paid for. Measured before it was written: the condition holds for zero
  of 123 factual questions across three corpora, and for 24 of 25 traps
  at a thousand documents. The fifty-question run over that store takes
  109 s, from 205 s and from 620 s at the start of the day; quality is
  unchanged everywhere (field set 40/40 · 15/15 · 7/8 · 10/10, scale
  25/25 · 25/25, NIST 11/13, EN 17/17), and the mechanically-derived set
  rises to 56/58 factual.

## [0.3.5] — 2026-09-09

The release where the commonest question stopped costing anything, and
the field set went to a full card.

Measured, three runs, medians, zero flips in every category. 62-document
field set (73 q): factual **40/40**, comparisons **15/15**, frontier 7/8,
traps 10/10 — and a factual question that names a document now takes
**0.01 s** instead of 10.4 s. Twelve hardware specifications the system
has never seen, with no adaptation: **8/8 · 4/4 · 3/3 · 4/4**. A thousand
generated specifications (50 q): **25/25 · 25/25**. NIST 11/13, EN 17/17;
151 invariants.

### Added

- **A field of a named document is read, not generated.** The commonest
  question a document store is asked needs no engine: the row exists,
  the source is named, the head is named. On the question door it is
  read before the router as well, because `ask()` is asked a question by
  contract. Tried once before and reverted for firing on one question in
  twelve — the cause was the naming underneath it, and a bullet being
  read as part of a head, both fixed here.

### Fixed

- **A value belongs to the field it was written under.** A
  specification carrying MEMORY: 128 GB and no STORAGE line was asked
  for its storage and answered "128 GB", stamped with that document —
  and every gate agreed, because the source was right, the number was in
  the evidence, and the sentence answered the shape of the question.
  Only the FIELD was wrong. A local veto now requires a claim's digits
  to appear on a line that carries the asked head. On the thousand-
  document set, traps 23/25 → 25/25.
- **A bullet is not part of a head.** Documents open list items with
  "·", so the record reader saw "· DURATION" as a second field beside
  DURATION — the same rows twice in the corpus's own vocabulary, and no
  head ever unique.
- **A source stamp survives the space in a document's name.**
  `Answer.sources` was read by splitting the answer on whitespace and
  keeping words that start with '#', so "#docx:Course 02" arrived as
  "#docx:Course" — a stamp two siblings share, which is not provenance.
- **A head is easier to recognise beside one of its values.** Shown a
  bare list of field names, the engine could not see that a question
  about a SCREEN means DISPLAY; shown "DISPLAY (e.g. 13 inches)" it
  could. The values are the documents' own, and a question no field fits
  still returns NONE.

## [0.3.4] — 2026-09-09

The release where the readings were measured above sixty documents for
the first time, and where the comparison category — the one this project
kept calling its weakest — went to 15/15 with zero flips.

Measured, three runs, medians. 62-document field set (73 q): factual
37→39/40, comparisons 10→15/15, frontier 7/8, traps 10/10. Twelve
hardware specifications the system had never seen (19 q): 8/8, 4/4, 3/3,
4/4 — a complete card on a corpus with no adaptation. 1,000 generated
specifications (50 q): factual 25/25, traps 23/25. NIST 11/13 and EN
17/17 unchanged; 148 invariants.

### Fixed

- **The extremes of a field are the corpus's, not the block's.** The
  first measurement above sixty documents found the worst kind of bug
  this system can have: a confident wrong answer with a source stamp on
  it. The census lays one row per source and stops at sixty, and the
  extremes row was computed from the rows that fitted — so on a corpus
  of a thousand the memory reported the highest price among the first
  sixty documents in alphabetical order. Every word was attested; only
  the superlative was false, and the superlative is the part nobody can
  check by reading one line. What is shown may be capped; what is
  claimed may not.
- **One question over a thousand documents took 315 seconds.** The
  record rider rebuilt the head index — a read of every sentence — once
  per source. Built when the store changes and kept otherwise: same
  reading, 315 s → 11 s, and a 50-question run 620 s → 348 s.
- **The verdict speaks before the lines it is drawn from.** The
  comparison's verdict row was appended after every line the two
  documents contributed, so the engine answered off two raw lines
  ("1 full day" versus "1 day"), concluded they differ, and the jury
  refused it — the turn abstaining with the answer two lines below where
  the reader stopped. The census learned this months ago; the comparison
  layout now has it too.
- **A sibling is not the document you named.** Sibling programmes share
  their opening phrase and differ only in a tail, so a question about ONE
  of them was read as naming six, and the comparison layout fired for a
  question that compares nothing. Among the sources a question calls, the
  one whose name it accounts for most completely is the one it names; a
  tie keeps everyone, because two names matched in full is what a
  comparison is.
- **A missing engine says which engine is missing.** `pip install
  living-memory-model` pulls in nothing, so a first `learn()` on a fresh
  machine reached the default local engine and reported
  `ModuleNotFoundError: No module named 'torch'` — a sentence that names
  neither the choice to be made nor the four ways to make it. The name in
  the message is the distribution name, kept in one place, because
  telling someone to install `lmm[local]` sends them to a different
  project.
- **CI had been red since the 0.3.2 rename, and nothing in the repository
  showed it.** The dependency-free check struck out the package's old
  name from pip's list and so reported the package itself as a
  dependency; three consultation tests reached the engine through the
  small helpers a turn asks on its way, and failed on a machine with
  nothing installed. The suite's own claim — no model, no network — is
  now verified the only way it can be, in a virtualenv with neither.

### Changed

- **The question door does not gamble on a chat.** Every question was
  paying for a full chat completion that was thrown away: the
  speculation buys latency in a consultation, where the router's verdict
  and the chat reply are wanted at once, and `ask()` inherited it only
  because it passes `teach=False`. Same for the delivery bridge, which
  offers a composed catalogue when a consultation cannot answer — the
  question door has `compose()` for that. Factual questions 12.0 → 9.0
  calls and 18,800 → 16,300 prompt tokens; traps 12.0 → 8.0 calls.
  `Session.respond` takes `conversational=` for callers that want the
  consultation behaviour explicitly.
- **The documentation site follows `main`.** Publishing was a command
  someone had to remember, and the site had been serving 0.3.2 while the
  sources for 0.3.3 sat in the repository. It builds with `--strict`, so
  a broken link fails in CI rather than in a reader's tab.

## [0.3.3] — 2026-09-08

The release where the memory leaves the process. A question asked in
other words now reaches the documents' own vocabulary, a document that
argues with itself says so, and the same three verbs are available over
HTTP, in LangChain and in TypeScript — each with the audit intact.

Measured, three runs, medians. On a corpus of English hardware
specifications this system had never seen, with no adaptation: frontier
0/3 → 2/3, factual 8/8, comparisons 4/4, traps 4/4. On the 73-question
Turkish field set: frontier 5/8 → 7/8, factual 38/40, traps 10/10,
comparisons 12/15. NIST 11/13 and EN 17/17 unchanged; 142 invariants.

### Added

- **The same three verbs over HTTP, one memory per user** (`lmm.serve`).
  A memory that lives in one process is a library; it becomes a product
  the moment a second person asks it something. Every user's store is
  chosen by their id and nothing else, no route reads across users, an
  id that could climb a path is refused before it becomes a filename,
  and each memory holds its own lock so different users answer in
  parallel. It is a transport, not a second brain: the abstention and
  the source stamps cross the wire intact. Localhost by default;
  `LMM_TOKEN` adds a bearer check. Zero dependencies.
- **LangChain adapters** (`lmm.adapters.langchain`). A retriever whose
  passages carry their document stamps, and an agent tool that returns
  the answer that was AUDITED — with its sources, and with the refusal
  as itself, because an agent handed an empty string writes its own
  answer over the silence. LangChain is not a dependency: without
  `langchain_core` the same calls return duck-typed equivalents.
- **A TypeScript client** (`sdk/typescript`). `ask` resolves to the same
  four fields a Python caller receives; a client returning a bare string
  would discard the only property that separates this from a chat
  completion. No dependencies. Its wire test starts the real Python
  server and checks across the two languages that what one user teaches,
  another cannot read.

### Fixed

- **A refusal is spoken in the language it was asked in.** Hidden behind
  a Turkish corpus: six English questions with no answer in the store
  were refused in Dutch, French and Spanish — never once in English. The
  question is now handed back as a language sample inside the
  instruction, the rule is repeated as the last thing read before
  writing, and the sentence is read back by the same kind of small judge
  the gates use; on a no it is written once more. No canned phrase in
  any language: the memory speaks in the engine's voice or it does not
  speak. Measured 3/9 → 9/9 across four languages.
- **The store's own vocabulary answers the question's.** Asked which
  machine is the "priciest", every seat went to prose about value —
  the documents write LIST PRICE. The engine is shown the field names
  the corpus repeats and asked which one the question reaches for, so a
  bridge can only land on vocabulary that provably exists in the store,
  and it is asked only when the question touches no field at all. The
  finding then steers the writer's WORDS while the facts block stays
  byte for byte what it was — putting a synonym into the evidence would
  assert a synonymy the store cannot attest, inside the one channel that
  must stay attested.
- **Naming is a property of what was asked.** Retrieval's query carries
  the previous turn's subject so a pointer-shaped follow-up stays on
  topic, and names were read off that query: in a corpus of siblings a
  question naming nobody came back "naming" eight of twelve documents,
  the comparison layout fired for eight sources, and it overwrote the
  corpus-wide reading that had just been laid out correctly. The ride
  still steers retrieval; it no longer confers namehood.
- **A line may hold several records.** Ingested line by line, a
  specification shows the record channel three fields; ingested by the
  bulk path — how every benchmark and every customer load runs — the
  three arrive glued into one window and a partition at the first colon
  saw one. How a document was chunked is not supposed to change what the
  memory can read.
- **A source that contradicts itself is read as a contradiction.** One
  value spoken, one hidden, and the stamp attesting both is what the
  gate exists to prevent. Scoped by what the field corpus taught: a
  repeated head is usually a list and sometimes a zoom, so three values
  under one head is an enumeration, one value carried inside another is
  the same length seen closer up, and what remains is a head stated
  twice with two values that exclude each other.

## [0.3.2] — 2026-09-07

Two identities, one relation: the release where the comparison reading
grew up. Measured on a 73-question field set (40 factual, 15
comparisons, 10 traps, 8 frontier), three runs, medians: comparisons
8/15 → 13/15, factual 37/40 → 38/40, traps 10/10 throughout. NIST
11/13 and EN 17/17 unchanged; 130 invariants.

### Fixed

- **In a comparison, two sources saying the same thing say it twice.**
  Sibling documents share a template, so two items of equal value write
  the same record line word for word; the assembly deduped by TEXT and
  dropped the second, leaving one value where the question asked about
  two — the verdict row could never be written, and the case is
  precisely the one comparisons are asked about most. Identity is per
  (line, source), the assembly-layer twin of the store's own rule.
- **Two inflections of one short root are kin.** A comparison asking
  about a "süre" could not reach a record headed "SÜRESİ": four letters
  of shared root, a different ending on each, and neither relation
  admitted it — so the question never reached the line that answers it,
  and the record rider's "did the question ask about this field"
  ranking was blind in every short-rooted language. Kinship now means a
  shared opening of at least three letters with each remainder within
  TAIL, and it is spent on the retrieval side only: the gates keep
  same_stem, so kinship widens what can be FOUND and nothing about what
  may be SAID.

### Added

- The comparison verdict is written into the evidence: two named
  sources answering the same field head yield one system-written row —
  head, each source's value, joined by `=` when their digit sets agree
  and `≠` when they differ. The engine reads a verdict instead of
  building one, and every reader is taught the notation.
- `benchmarks/conversation_eval.py`: the multi-run referee — a fresh
  store per run, per-category medians and the FLIP set, AND-golds for
  multi-part questions, and a frontier category. Sorting-layer changes
  are not judged without it.

## [0.3.1] — 2026-09-06

The audit release: a day of field dialogue, a corpus-wide format audit,
and a head-to-head benchmark against LangChain, LlamaIndex and Mem0 —
with every finding, ours included, fixed architecturally or filed
openly. Published numbers re-measured unchanged after every batch
(NIST 11/13, EN 17/17; 124 invariants, W27-W42).

### Fixed

- **One-row Word tables are fact boxes, and fact boxes are prose.** The
  docx reader dropped single-row grids whole (no header row to read),
  and every duration/seats/audience box of a sibling-template corpus
  lived in one; cells now enter the evidence as "LABEL: value." lines,
  joining the record channel, and an unpunctuated Word paragraph closes
  with its full stop so headings stop gluing into mega-lines. Body
  coverage audited to 100% across the corpus.
- **The same sentence in two documents is two attestations.** The
  duplicate guard keyed on text alone, silencing every sibling that
  shares a template line — cross-document comparisons refused forever.
- **Provenance notation belongs to the system.** Generated candidates
  shed anything stamp-shaped; the mark names the line the answer
  actually rests on; the history carries unstamped text; a turn that
  says nothing carries no stamp.
- **An answer's substance cannot be borrowed from the question** — one
  rule at both doors (read-back and informed refusal), and a claim that
  names its source is judged by that source's lines alone.
- **An echo cannot turn a question into its own assertion.** On an ASK
  turn the chat echo draws on prior turns only; the trap-set
  fabrication ("yes, there is a certificate") is gone.
- The prior subject rides retrieval as a challenger, not a preemption —
  restoring a benchmark answer to its honest, attested path.

### Added

- **The comparison reading**: a question naming two or more sources
  lays them side by side — per-source seats, record rows riding along,
  the named-source jury holding each claim to the source it names. A
  source is named by a pointer or by its bigram; family letters call
  nobody.
- `inflect.kin`: retrieval-only kinship for short-rooted languages
  (four-letter roots and their suffixed forms), spent where surfacing a
  real line is the worst it can do; the gates keep same_stem.
- An expansion is judged by the words it ADDS — the keep filter no
  longer discards a record line's honest bridge on the corpus-common
  words it shares with every line.
- In a multi-document store, a same-region twin steps aside so a
  document's second seat reaches its other regions.

## [0.3.0] — 2026-09-05

The consultation release. One evening of field dialogue against a live
customer bot — a RAG competitor on the other screen — surfaced every gap in
turn, and every fix below is architectural: no keyword, no language rule, no
corpus-specific line anywhere. The published numbers were re-measured
unchanged after every change batch (NIST 11/13, EN 17/17; 110 unit
invariants, W14–W26).

### Added

- **The consultation surface.** `respond(msg, teach=False)`: a statement is
  CONTEXT (routes to the chat voice; its extracted terms accumulate in the
  consultation's brief), a question searches with the whole consultation
  riding along, and a delivery request routes straight to the composer,
  which returns a source-stamped catalogue built from the brief. The bare
  refusal is reserved for surfaces that have no conversation — when the
  answering chain comes back empty, the already-gated conversational reply
  speaks instead.
- **The echo rule.** In the chat reading, repeating what the user themselves
  said this conversation is not an assertion — a consultant may sound like
  one. A word the user never spoke still answers to the graph.
- **Gated streaming.** `compose(..., on_line=)` (and `respond(...,
  on_line=)`): each draft line is judged by the composer's gate the moment
  its newline arrives and handed to the callback while the engine is still
  writing. A refused line is never seen; a backend that cannot stream
  degrades to batch by itself.
- **The operator's knobs.** `Memory(..., persona=, style=, warmth=,
  reply_tokens=)`. The persona is the voice and stops at the document's
  edge; the style is the document's shape and travels only to the
  composer; warmth and reply_tokens tune the voice surfaces alone. None of
  them can reach a classifier or a verifier.
- `compose(..., topics=)` gathers per topic — dedicated seats, dedicated
  queries.

### Changed

- **Latency, halved and then some** (measured, same dialogue): a greeting
  turn 15.9 s → ~6 s, a catalogue's first line on screen 13.4 s → ~5 s.
  The verifier's per-sentence re-reads now go to the engine side by side
  where the backend allows; the re-extractor memoises (the gate and the
  abstention stamp used to read the same sentences twice); on the
  consultation surface the chat voice and the delivery question ride out
  beside the router and the loser is discarded.
- Per-topic composer queries use the topic alone — appending the brief
  made every query converge on the same generic winners (measured on a
  61-document store).
- An ordinal list marker ("1.", "2)") at the head of a draft line is
  format, not a quantity — the digit gate no longer kills the engine's own
  section numbering.

### Fixed

- A context statement in a no-teach conversation no longer falls through to
  the question treatment (it used to end in a refusal wearing the
  persona's greeting).
- A stale census line can no longer leak from a previous turn into the
  informed refusal.

## [0.2.0] — 2026-09-05

The multi-document release. Everything below came out of one field trial: a
corpus of 62 sibling documents sharing a template — a class no benchmark had
covered — worked through question by question, with the published
single-document numbers re-measured unchanged after every change (NIST 11/13,
EN corpus 17/17).

### Added
- `Memory.compose(brief)` — long-form drafts from the evidence: the engine
  organises, the material speaks, and every line is re-read on the way out
  (invented numbers and unsupported mixtures are dropped, structure stands).
  Returns the draft with the documents it rests on.
- `Memory.where(term)` — the census: which documents mention a term, with
  sentence counts as receipts. A count, not a retrieval; no engine, ms.
- The source-name channel: a question that names a document gets that
  document's sentences, weighed by log(S/s) — a name-word every source
  shares weighs exactly zero, which keeps single-document stores byte-stable.
- Datelines: in a multi-document store every evidence line opens with its
  document's name, so answers can say *which programme* — and the verifier
  can confirm it.
- Source seat cap: one document cannot monopolise the block when others hold
  evidence (the region rule, one level up).
- The census rides into conversation: listing questions name the documents
  that actually speak, counted at answer time.
- The informed refusal: a turn that declines while the census is full offers
  the tally — with datelined reasons — gated so it can name and count only
  what the store attests.
- Conversational follow-ups: a subject that resolves to nothing the memory
  knows points backwards, and the previous turn's subject rides along. No
  pronoun lists — the signal is resolution failing, in any language.
- `ask()` cannot write memory (`teach=False` throughout): a question API is
  not a lesson, however imperative its grammar. Teaching stays on
  `Session.respond`, deliberately.

### Fixed
- The abstention stamp is read off the evidence, not asked of a model — a
  full correct answer was being recorded as a refusal when the re-extractor
  found no tidy triple in a long list-sentence. One model call fewer per
  answering turn.
- A restated evidence line needs no jury: mutual word-coverage between claim
  and a single evidence line settles support without the judge — whose
  verdict on that class also wobbled across identical runs. One-way coverage
  still goes to the jury; a subset of a line can invert it.
- The named document speaks first: admission was right, order was not, and
  the engine reads the block top-down.
- learn_rows aliasing could fuse a repeated cell value with a row's identity
  and silently drop the very fact it was drawn from; the alias now requires
  a phrase unique in the whole table, on a row that has no name of its own.
- The offline expansion channel is no longer listed as a capability: measured
  on a real standard, its filter keeps only the queries that copy a line —
  the ones that buy nothing. The code stays, off unless asked for, with the
  measurement in Honest limits.

## [0.1.0] — 2026-08-18

First public release. The layer had been working for some time; this release is
the work of making it installable, readable and lawful to publish.

### Added

- **`Memory` — one front door.** `Memory("mind.lmm")`, `.learn(...)`,
  `.ask(...)`, `.save()`. `learn()` takes a file path or the text itself and
  routes on the extension: PDF and Word split into tables (straight to the
  graph) and prose (to the evidence layer); spreadsheets and CSV are all table;
  anything else readable as text is text. It is a wrapper — `Memory.session` is
  the real `Session`, and the older API is unchanged.
- **`ask(explain=True)`** returns an `Answer`, which *is* the answer string and
  additionally carries `.abstained`, `.sources`, `.subject` and `.wrote`. The
  abstention flag is the session's own structural stamp, not a phrase list.
- **Word support** (`lmm.tables.read_docx` / `learn_docx`), on the same contract
  as the PDF adapter.
- **CSV support**, through the standard library, so the most common "here is a
  table" format costs the core install nothing.
- `LICENSE` (Apache 2.0), `NOTICE`, `CONTRIBUTING.md`, `SECURITY.md`,
  `examples/`, and CI that runs the tests on Python 3.9–3.13 and verifies that
  the core installs with no dependencies.
- Six tests for the new surface (59 total), none of which needs a model.

### Changed

- **The core install has no dependencies.** `pandas` was imported at
  `tables.py`'s top level, which meant the pure-python layer could not be
  installed without it. Every optional reader is imported at call time now and
  reports the exact install line — naming the PyPI distribution, since
  `pip install docx` fetches a different, abandoned package.
- **src-layout.** Everything importable moved to `src/`, so `import lmm` no
  longer depends on the working directory and an installed copy cannot be
  confused with the working tree.
- **`v3` is now `lmm.core`.** The substrate was named after an internal version
  number. It is the discrete layer — memory, gate, geometry, dynamics,
  transitivity — and it is now named for what it is. `bench/` is `benchmarks/`.

### Removed

- Third-party and private benchmark material: a device manufacturer's manual, an
  institution's internal strategy document and an inspection report naming real
  individuals, together with every derived result file that quoted them.
  The invented corpora (`benchmarks/corpus*.txt`) remain — they are the
  publishable half of the benchmark, written so no answer can leak from a
  model's training data. Only aggregate scores are published for the rest.
- Turkish internal design notes and conversation logs from the published tree.

### Known limits

Documented in full in the README under "Honest limits". In short: the gate
guarantees non-fabrication, not perfect relevance; there is a lexical-retrieval
ceiling on spec lines reachable only through very common words; and the engine
sometimes appends an explanation to a refusal that the memory contradicts —
which is detected and scored against the system, but not prevented.

[0.1.0]: https://github.com/ruzgarkanar/lmm/releases/tag/v0.1.0
