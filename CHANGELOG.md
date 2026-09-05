# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
