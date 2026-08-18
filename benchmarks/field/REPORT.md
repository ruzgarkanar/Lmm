# Field trial: LMM against real documents off the public internet

**This is a diagnosis, not a score.** Nothing in the library was changed to make
these numbers better, and no rule specific to any of these documents was added.
The question was never "how many does it get right" but **which format, and
which part of a document, breaks it — and at which layer**.

Ten defects are named below. Four of them produce a confidently wrong answer;
three of them are silent, which is worse.

## How it was run — as a stranger would

The library was installed the way a user installs it, not the way a maintainer
runs it:

```bash
python3.11 -m build --wheel
python3.11 -m venv venv
venv/bin/pip install 'dist/lmm-0.1.0-py3-none-any.whl[pdf,xlsx,docx,azure]'
```

Every trial script imports `from lmm import Memory` and nothing else — no
`sys.path` insert, no editable install, no repository on the path — and every
script was run from a directory outside the repository, so a stray local `lmm/`
could not shadow the installed package. The first run printed `lmm.__file__`,
which is the proof that what was measured is the built wheel:

```
lmm from: .../field/venv/lib/python3.11/site-packages/lmm/__init__.py
```

Engine: `LMM_BACKEND=azure`. Documents: fetched by
[`fetch_documents.py`](fetch_documents.py), which carries every URL, its licence
and the reason that document is in the set. **No document is committed** — the
download directory is in `.gitignore`. Only public-domain or openly licensed
material was used: NIST, NASA/NTRS, NARA, US Census (public domain, US
Government); Geoscience Australia, the Kubernetes Authors, the European Court of
Auditors (CC BY 4.0); UK ONS (OGL v3); RFC 9110 (IETF Trust).

`python3.11 tests/test_core.py` → **59/59**, after this work as before it.

---

## 1. What went in

Ingesting a *file* makes **zero model calls** — `learn()` passes `deep=False` for
files — so these times are pure CPU and reproducible.

| Document | Format | Lang | Size | Ingest | Facts→graph | "Tables" | Evidence entries |
|---|---|---|---|---|---|---|---|
| `nist_sp800-63b.pdf` — NIST SP 800-63B, 80 pp | PDF, text layer | EN | 1.5 MB | 5.9 s | 160 | 37 | 12 395 |
| `naca_report_scan.pdf` — NACA Report 1270, 36 pp | PDF, **poor OCR** | EN | 4.1 MB | 3.5 s | 0 | 0 | 11 665 |
| `us_constitution_scan.pdf` — NARA page scans, 4 pp | PDF, **image only** | EN | 2.3 MB | **0.00 s** | **0** | **0** | **0** |
| `mh370_debris_analyses.docx` — Geoscience Australia Rec 2017/11 | DOCX | EN | 14.1 MB | **0.59 s** | 373 | 12 | 2 252 |
| `us_state_population.xlsx` — Census NST-EST2023 | XLSX | EN | 15 KB | 0.32 s | 173 | 1 | 70 |
| `uk_cpi_timeseries.csv` — ONS CPI series D7G7 | CSV | EN | 11 KB | **0.04 s** | 641 | 1 | 644 |
| `rfc9110.txt` — HTTP Semantics | TXT | EN | 503 KB | 2.1 s | 0 | 0 | 21 841 |
| `k8s_nodes.md` — Kubernetes docs | Markdown | EN | 15 KB | 0.06 s | 0 | 0 | 585 |
| `rfc9110.html` — **the same RFC** | HTML — *undocumented* | EN | 1.2 MB | 7.8 s | 0 | 0 | **44 773** |
| `eca_sr14_2023_en.pdf` — ECA SR 14/2023, 53 pp | PDF | EN | 4.5 MB | 2.6 s | 32 | 18 | 5 771 |
| `eca_sr14_2023_el.pdf` — **the same report**, 64 pp | PDF | **Greek** | 3.7 MB | 3.4 s | 44 | 22 | 7 292 |
| `eca_sr14_2023_bg.pdf` — **the same report**, 56 pp | PDF | **Bulgarian** | 4.5 MB | 3.4 s | 70 | 22 | 7 299 |

Ingestion speed is not the problem anywhere. A 14 MB Word document with twelve
tables is read in 0.59 s and a 350-page-class PDF in under 6 s, with no model
call and no network.

Two rows are already findings, before a single question was asked:

* the image-only PDF took **0.00 s, produced zero of everything, and raised
  nothing** — `learn()` returned a cheerful `<Learned 0 facts, 0 tables via pdf>`;
* HTML, which the library does not document as supported, was **accepted
  silently** and filled the evidence index with 44 773 entries, 80 % of which
  contain markup such as `<meta content="Common,Latin" name="scripts">`.

---

## 2. What came back

Every document was read by hand and given 5–9 questions: most of them facts
stated plainly in the text, at least two of them things the document does not
contain. Each answer was graded against the document.

| Document | Correct | Correct abstention | **Missed** (in the document, refused) | **WRONG** |
|---|---|---|---|---|
| `nist_sp800-63b.pdf` | 7/7 | 2/2 | 0 | 0 |
| `naca_report_scan.pdf` (bad OCR) | 4/6 | 2/2 | 2 | 0 |
| `us_constitution_scan.pdf` (image only) | — | 5/5 | — | 0 |
| `mh370_debris_analyses.docx` | 7/7 | 2/2 | 0 | **1** |
| `us_state_population.xlsx` | **0/5** | 2/2 | **5** | 0 |
| `uk_cpi_timeseries.csv` | 4/5 | 2/2 | 1 | 0 |
| `rfc9110.txt` | 5/5 | 0/2 | 0 | **2** |
| `rfc9110.html` (same document) | 4/5 | 0/1 | 1 | **1** |
| `k8s_nodes.md` | 5/5 | 2/2 | 0 | 0 |
| `eca_sr14_2023_en.pdf` | 4/5 | 2/2 | 0 | **1** |
| `eca_sr14_2023_el.pdf` (Greek) | 4/5 | 2/2 | 0 | **1** |
| `eca_sr14_2023_bg.pdf` (Bulgarian) | 3/5 | 2/2 | 2 | 0 |
| **Total** | **47/60** | **23/28** | **11** | **6** |

Three separate probes were run outside the main set and are reported with the
findings they belong to: a multi-column table probe (Finding 3), an empty-memory
control (Finding 6), and an edge-case battery (Finding 8).

---

## 3. The architectural holes

### 1. An ungated second model call is appended to every gated answer

**This is the single most serious finding, and it is the source of four of the
six wrong answers.**

*Symptom.* A correct or abstaining answer is followed by a fabricated clause:

* `rfc9110.txt` — *"What does the 419 status code mean?"* →
  **"I do not know. El código de estado 419 generalmente se refiere a 'Session
  Expired'."** Reproduced three times independently. RFC 9110 assigns no 419.
* `rfc9110.txt` — *"Which RFC obsoletes RFC 9110?"* →
  **"I do not know. RFC 9110 is obsoleted by RFC 9111."** False, and not in the
  document.
* `us_state_population.xlsx` — →
  **"I do not know. La estimación de la población de California a partir del 1
  de julio de 2023 es de aproximadamente 39,5 millones."** The document says
  38 965 193, and the memory contains no such claim in any form.
* `mh370_debris_analyses.docx` — *"Sediment below what size was not analysed?"* →
  **"Any sediment below 100 µm was not analysed. Sediment below 63
  micrometers…"** The string `63` in that sense appears nowhere in the document.

*Root cause, traced.* `Session._answer` (`session.py:1004-1006`) and
`Session._hedge` (`session.py:1273-1274`) both do this:

```python
note = generate.hedge_note(src, question)
if note:
    safe = f"{safe} {note}"
```

`generate.hedge_note` (`generate.py:158`) is a **separate model call at
`temperature=0.3`** whose only protection is the sentence *"Contain NO facts —
only the caveat and the source"* in its own prompt. Its docstring states *"since
it carries no fact it also passes verify"* — but **verify is never re-run on the
concatenation**. The single property the whole architecture rests on — *"an
answer whose claims are not in the given evidence DROPS"* — is enforced on
`safe`, and then `safe` is mutated afterwards. Everything downstream of the gate
is outside the gate.

The same call is also where the wrong-language text enters (Finding 6), and it
corrupts the abstention stamp: `last_abstained` is computed at `session.py:344`
by re-extracting claims from the *spoken* string, which now includes the note.

*Structural fix.* The hedge is presentation, not content. Either

  (a) build it structurally from what the system already holds — printing
  `(uncertain · #pdf:x.pdf)` requires no model call at all, is deterministic,
  cannot fabricate, and saves one API round-trip per answer; or

  (b) if it must be generated, pass the **concatenated** sentence back through
  the same `verify` / `digits_ok` / `supported` chain and drop the note when it
  fails.

Option (a) does not violate rule 2: a parenthesis, a `#` stamp and an ellipsis
are *format*, exactly like the colon separator CONTRIBUTING already permits.

### 2. A PDF with no text layer is indistinguishable from a successful ingest

*Symptom.* `us_constitution_scan.pdf`, an image-only PDF of four NARA page
scans — precisely what an office scanner produces — ingested in 0.00 s, returned
`<Learned 0 facts, 0 tables via pdf from '#pdf:us_constitution_scan.pdf'>`, wrote
**zero** evidence entries, and raised nothing. Every subsequent question was
correctly refused (the gate held perfectly), so from the user's side a document
that was never read is indistinguishable from a document about which the system
happens to know nothing.

The degraded-OCR case is the same failure with a mask on: `naca_report_scan.pdf`
produced 11 665 evidence entries of OCR noise (`JEROME C. HUNSAKEB`,
`Washington 2ti, D. C.`), which cost two of six answerable questions — the two
whose answers were proper names.

*Root cause.* `Learned` carries `facts` and `tables`, and `facts=0` is the
*normal* result for any prose document (`.txt`, `.md`, and every PDF ingested
with `deep=False`). There is therefore no value in the report that distinguishes
"nothing extractable" from "extracted fine, no triples". The evidence count —
which `learn()` already knows — is not reported.

*Structural fix.* Add the evidence count to `Learned`, and when a PDF yields
zero extractable characters say the one thing the user needs: this file has no
text layer, LMM does not OCR, run OCR first. Zero is not a tuned constant.

### 3. Multi-column table cells reach the evidence index in the wrong company — and no gate can catch it

*Symptom (targeted probe).* NIST Table 4-1 is a four-column requirement matrix
(AAL1 / AAL2 / AAL3).

> *"According to Table 4-1, what is the reauthentication requirement at AAL1?"*
> → **"The reauthentication requirement at AAL1 is 12 hours of inactivity."**
> The document says **30 days**.

*Root cause.* `pdfplumber` flattens that table so the cells arrive out of
reading order:

```
12 hours or 30      12 hours or 15 minutes
minutes inactivity;  inactivity; SHALL use
Reauthentication  30 days
MAY use one        both authentication
```

The AAL1 value `30 days` is emitted *after* the AAL2 and AAL3 cells, so the
evidence window containing the word `Reauthentication` contains `12 hours` and
does not contain `30 days`. **No gate can reject this**: the claim genuinely is
in the evidence, verbatim. The coverage gate, the digit discipline and the
support check all pass, correctly.

The code already names this exact pathology in `learn_text` — *"a window that
spanned two rows put cells side by side that the DOCUMENT NEVER PUT TOGETHER"* —
and solves it for rows via `evidence.table_windows`. Columns are unsolved.

*Structural fix.* `read_pdf` receives cell geometry from pdfplumber and throws it
away. A cell should enter the graph as (row label, **column label**, value), not
as a token in a flattened line; and where geometry is unavailable, a table cell
should not enter the evidence index as free prose at all. This is the one
finding where the gate cannot be the answer — the representation has to be.

### 4. A spreadsheet with a two-row header loses half its columns, silently

*Symptom.* **0 of 5** questions answered from a memory holding 173 records built
from a 15 KB spreadsheet. Not one wrong answer — five honest refusals about
numbers that are in the file.

*Root cause, traced.* `read_xlsx` (`tables.py:108-110`) picks exactly **one**
header row. The Census sheet's headers occupy rows 3 **and** 4, and the sheet
says so itself in its own first line: *"table with row headers in column A and
column headers in rows 3 through 4. (leading dots indicate sub-parts)"*. Three
consequences, all silent:

1. Row 4 (`2020 | 2021 | 2022 | 2023`) becomes a **data row**.
2. The 2021, 2022 and 2023 columns all have `NaN` headers, so `row[h or ""]`
   maps all three onto the **single dict key `""`** — later keys overwrite
   earlier ones, so **2021 and 2022 are discarded outright** and 2023 is stored
   under an empty predicate, reachable by no question that names a year.
3. The leading-dot convention is kept verbatim, so the subject label is
   `.California`; `m.about("California")` returns nothing.

Verified directly: `m.about(".Alabama")` yields three records — the 2020 base,
the 2020 estimate, and one with `predicate=None` — for a row that has five
numbers in it. `"What is .California?"` answers *".California is a geographic
area"*, so identity resolution works; it is the predicate that was destroyed.

The footer row `Release Date: December 2023` also becomes a data row.

*Structural fix.* Header detection must be allowed to span a *block* of rows
(the block ends where the value types stop being text — read off the sheet, no
constant), and a column with no name must never be merged into another: it
inherits its neighbour's name or is dropped loudly.

### 5. HTML: not supported, not refused, accepted silently

*Symptom.* `api._UNSUPPORTED` names `.doc`, `.ppt`, `.pptx`, `.pages`,
`.numbers`, and their messages are genuinely good (`"PowerPoint is not
supported"`, `"legacy Word (.doc) is not readable; save it as .docx"`). `.html`
is in neither `_UNSUPPORTED` nor `_TEXT`, so it falls through to the
try-it-as-text branch and "succeeds".

RFC 9110 as HTML: 44 773 evidence entries, **80 % containing markup**, ingested
in 7.8 s versus 2.1 s for the identical document as `.txt` — and it still scored
4/5, because this particular page is mostly prose between tags. That is the
finding, not the score: on a navigation-heavy page the evidence index would be a
menu, and nothing in the API would tell the user. The user is not warned, and
`Learned` reports `adapter="text"`, which is technically true and practically a
lie.

*Structural fix.* Either support it — stripping tags is a *format* operation,
not a language rule — or name it in `_UNSUPPORTED` with the same clarity as
PowerPoint. The current middle position is the worst of the two.

### 6. There is no language gate — only a language instruction

*Control experiment (deliberate).* Seven English questions were put to a
brand-new, completely empty `Memory()`:

| | |
|---|---|
| abstained | **7 / 7** — the gate is perfect |
| answered in English | **1 / 7** |

> *"How often does the node controller check the state of each node by default?"*
> → **"Je ne sais pas encore. Voulez-vous que je le recherche ?"**
> *"What is the default value of `--unhealthy-zone-threshold`?"*
> → **"Ik weet het nog niet. Moet ik het opzoeken?"**
> *"How many passengers were on board MH370?"*
> → **"Maaf, saya tidak tahu tentang itu. Apakah saya perlu mencarinya?"**

The same thing happened in the real runs: three of five refusals on the
image-only PDF came out in Dutch and Spanish, three of five on the spreadsheet
in Spanish and Italian.

*Root cause.* Every other property in this system is structural. Language
fidelity is a **prompt instruction** (`generate._MATCH_LANGUAGE`) plus
`temperature=0.3`, and nothing ever re-reads the produced sentence. It fails
exactly where it matters most: when retrieval finds nothing there is no evidence
in the prompt to anchor the language, so the engine falls back on the
distribution of its multilingual few-shot examples. Where evidence *does* exist
the problem disappears — the Greek and Bulgarian runs answered in Greek and
Bulgarian throughout, including their refusals.

This is a sharper statement of the limit the README already lists ("the engine
sometimes appends an explanation to a refusal"): it is not a stylistic tic, it is
a missing gate, and it is 6/7 on an empty memory.

*Structural fix.* The system already owns the organ. `_asserted_a_fact` re-reads
the spoken sentence to decide whether it asserted anything; the same shape of
check — *"is this sentence in the same language as the question, yes or no"* —
would turn the language contract into a gate. It needs no word list, so rule 2
is untouched.

### 7. A path typo is learned as a fact

*Symptom.* `m.learn("manual.pdf")` when that file is not there:

* with an engine configured → `<Learned 0 facts, 0 tables via text from
  '#document'>` — a **silent success** for a document that was never opened;
* on the core install → **`ModuleNotFoundError: No module named 'torch'`**, 22
  traceback lines, for a typo.

A directory path gives both the same way.

*Root cause.* `_looks_like_path` returns False when `os.path.isfile` fails, so
the *string* `"manual.pdf"` falls to the "this is the text itself" branch and is
learned with `deep=True` (because text typed at the memory is "a fact being
taught"), which reaches for the default local engine.

*Structural fix.* An argument that has the *shape* of a path — short, no
newline, a known document extension — and is not a file is a mistake, not a
sentence. That is a format test, not a heuristic about content, and it makes the
core install's error message about the user's typo rather than about torch.

### 8. Broken and protected files leak third-party exceptions

| input | what the user gets |
|---|---|
| truncated PDF | `PdfminerException: Unexpected EOF` · 51 traceback lines |
| empty PDF | `PdfminerException: No /Root object! - Is this really a PDF?` · 29 lines |
| **password-protected PDF** | **`PdfminerException:` — an empty message** · 38 lines |
| `.xlsx` renamed `.pdf` | `PdfminerException: No /Root object!` |
| `.pdf` renamed `.txt` | `ValueError: … is not a text file and '.txt' is not a format LMM reads (pdf, docx, xlsx, csv, txt, md)` ✔ |
| 10 MB text file | ingested in 7.0 s, no error ✔ |

The last two are exactly right. The protected-PDF case is the worst possible
outcome: an exception class the user has never heard of, carrying **nothing at
all**. The README promises "a missing reader reports the exact install line
rather than a traceback" and delivers it (see §4) — a *broken file* has no such
contract, and it is the more common event.

### 9. "Tables" in a prose PDF are layout noise, and the count is not stable across languages

The same ECA report, three official language versions, identical content:

| | tables detected | facts written to the graph |
|---|---|---|
| English | 18 | 32 |
| Greek | 22 | 44 |
| Bulgarian | 22 | **70** |

Inspecting them, the "tables" are the cover page, a chart's data labels and text
panels — pdfplumber's ruled-box detection firing on typography. The graph's
contents are therefore a function of the *typesetting*, not of the document.
Nothing wrong came out of it here, but "facts in the graph" is not a meaningful
number for a prose PDF, and a user comparing two language editions of one report
would find three different memories.

### 10. Answer selection picks the true-but-off-target half of a sentence — identically in every script

The document reads: *"…defined the priority areas … for seven years (2021-2027)
and set the financial envelope for the first four years (2021-2024)."*

> *"For how many years was the financial envelope … set?"*
> EN → **"…was set for seven years."**
> EL → **"…καθορίστηκε για επτά έτη."**
> BG → honest refusal.

The same wrong half, in Latin and in Greek script. That is the useful part: this
is a **selection** defect, not a language defect. The README names the class
("answer selection can still pick a true-but-off-target sentence"); this is a
measured instance in which the off-target pick flatly contradicts the attribute
that was asked about, and it survives every gate because the sentence is real.

### 11. CSV: a headerless file is parsed with its first data line as the header

The ONS series has a seven-line key/value preamble and **no header row**.
`_read_csv` hands it to `csv.DictReader`, which takes line 1 —
`"Title","CPI ANNUAL RATE 00: ALL ITEMS 2015=100"` — as the field names. It
works, and scored 4/5 — but by luck: the two columns really are (label, value),
so `1989 → 5.2` lands correctly. The cost is that `"Release date" →
"22-07-2026"` becomes a fact of the same standing as an inflation rate, and the
one miss is `"2026 JUN"`, the row whose label is not a bare year.

### Non-Latin alphabets: the measured cost

Same report, same five factual questions, same two absent-fact questions:

| | correct | missed | wrong |
|---|---|---|---|
| English | 4/5 | 0 | 1 |
| **Greek** | 4/5 | 0 | 1 |
| **Bulgarian (Cyrillic)** | 3/5 | 2 | 0 |

Text extraction is clean in all three (accented Greek and Cyrillic come out of
pdfplumber intact), answers stay in the question's language throughout, and the
graph ingests all three. Bulgarian loses two answers to abstention and gains
none to fabrication — the failure mode under a non-Latin script is *silence*,
not error, which is the right way round. The English and Greek runs make the
same mistake (Finding 10), which is the evidence that the defect is
script-independent.

---

## 4. Installation experience

1. **`.env` is looked for next to the installed package, not next to the user.**
   `runtime_azure._root()` returns `dirname(dirname(__file__))` — the repository
   root when running from a checkout, and **`site-packages/`** after
   `pip install`. Reproduced verbatim on a clean venv with a correct `.env` in
   the working directory:

   ```
   .env looked for at: .../venv/lib/python3.11/site-packages/.env
   cwd: .../field   | .env here? True
   → KeyError: 'AZURE_OPENAI_ENDPOINT'
   ```

   A raw `KeyError` with no message, for a file the user placed exactly where the
   documentation implies. Fix: search the working directory and walk upward, or
   honour an `LMM_ENV` variable.

2. **Missing extras are handled beautifully.** Core-only install, verbatim:
   `reading PDF files needs pdfplumber, which is not installed.  pip install
   'lmm[pdf]'  (or: pip install pdfplumber)` — and the same for `docx` and
   `xlsx`, while `.csv` and `.md` keep working with zero dependencies. This is
   the standard the rest of the error surface should be held to, and it is the
   reason Findings 7 and 8 stand out as much as they do.

3. **Build and install were clean.** `python3.11 -m build --wheel`, a fresh venv,
   `pip install 'lmm-0.1.0-py3-none-any.whl[pdf,xlsx,docx,azure]'`, and
   `from lmm import Memory` from an unrelated directory — first try, no import
   problems, no missing package data, no path tricks.

4. **The provenance stamp is a filename snapshot.** Files renamed after ingest
   still answer with the old `#pdf:` tag. Cosmetic, but provenance is what the
   README sells.

---

## 5. The wrong answers, in full

| # | Document | Question | Answer given | Document says | Layer |
|---|---|---|---|---|---|
| 1 | `rfc9110.txt` / `.html` | What does the 419 status code mean? | "I do not know. **El código de estado 419 … 'Session Expired'**" | 419 is unassigned | **hedge note** (Finding 1) — 3 reproductions |
| 2 | `rfc9110.txt` | Which RFC obsoletes RFC 9110? | "I do not know. **RFC 9110 is obsoleted by RFC 9111.**" | nothing does | **hedge note** (Finding 1) |
| 3 | `us_state_population.xlsx` | California, 1 July 2023 | "I do not know. **… aproximadamente 39,5 millones**" | 38 965 193 | **hedge note** (Finding 1) |
| 4 | `mh370_debris_analyses.docx` | Sediment below what size was not analysed? | "…below 100 µm… **Sediment below 63 micrometers**" | 100 µm only | **hedge note** (Finding 1) |
| 5 | `eca_sr14_2023_en.pdf` + `_el.pdf` | For how many years was the financial envelope set? | "**seven years**" / "**επτά έτη**" | the first four (2021-2024) | **retrieval / answer selection** (Finding 10) |
| 6 | `nist_sp800-63b.pdf` (probe) | Table 4-1, reauthentication at AAL1? | "**12 hours of inactivity**" | 30 days | **ingestion — column collapse** (Finding 3) |

**Four of six are the hedge note.** One line of concatenation after the gate
accounts for two thirds of every false statement this system made in a
twelve-document field trial — and it is the cheapest of all these defects to
close, because the note carries no information the system does not already hold
in structured form.

The other two are the two things a gate can never fix: an answer assembled from
evidence that says the wrong thing (Finding 3, an ingestion bug) and an answer
that quotes the wrong true sentence (Finding 10, a retrieval bug). Those two are
the honest residue. The rest is a hole in the wall.

---

## 6. What held

Worth recording as carefully as what broke, because it is most of the system:

* **The gate never let an ungrounded claim through the answer path.** Every
  fabrication in this report entered through the hedge note appended *after* it.
* **Empty memory abstains, 7/7.** A memory that knows nothing says so.
* **All 12 absent-fact questions were refused** — not one invented answer to a
  question whose subject the document does not discuss.
* **The clean PDF is excellent**: 9/9 on an 80-page NIST specification,
  including four separate numeric requirements and two correct refusals.
* **DOCX is excellent**: 7/7 on a 14 MB technical record, ingested in 0.59 s.
* **Markdown is excellent**: 5/5 including four flag defaults.
* **Non-Latin scripts work.** Greek matched English exactly; Bulgarian lost two
  answers to silence and none to error.
* **Ingestion is fast and needs nothing** — no model, no network, no GPU, on
  every format tested.
