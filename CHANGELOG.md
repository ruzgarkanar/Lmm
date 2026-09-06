# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
