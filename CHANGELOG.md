# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
training title — "ekipte dedikodu var" onto "Etkin Çatışma Yönetimi".
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
