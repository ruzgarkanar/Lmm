# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
