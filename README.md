# LMM — Living Memory Model

**A verifiable memory layer for language models.** Facts live in a graph, every
claim carries its source, and an answer the memory does not support cannot leave
the system.

[![CI](https://github.com/ruzgarkanar/lmm/actions/workflows/ci.yml/badge.svg)](https://github.com/ruzgarkanar/lmm/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%20%E2%80%93%203.13-blue.svg)](https://pypi.org/project/lmm/)

> In an LLM, knowledge is frozen into weights at training time.
> In LMM, knowledge lives in a memory you can write to, inspect and audit —
> which is what makes "I don't know" trustworthy rather than polite.

```bash
pip install lmm
```

```python
from lmm import Memory

m = Memory("mind.lmm")            # persistent; Memory() is transient
m.learn("manual.pdf")             # pdf · docx · xlsx · csv · md · txt · or text
print(m.ask("what is the screen's diagonal?"))
m.save()
```

That is the whole API. One method reads every format, and the file extension
picks the adapter.

---

## Why not embedding RAG

|  | Embedding RAG | LMM |
|---|---|---|
| Ingestion | chunks + embeddings | facts into a **graph** + sentences into an **evidence index** |
| Retrieval | similarity gamble | deterministic word/graph lookup, explainable |
| Multi-hop | fails when chunks don't co-retrieve | **derives** new facts symbolically (µs, no model call) |
| Fabrication | a plea in the prompt | **structural gate**: unsupported claims cannot leave |
| Provenance | none | every fact carries its source; uncertain answers are flagged |
| Tables | flattened into text | **rows go straight into the graph** — zero model calls |
| A 350-page manual | seconds of embedding, lossy retrieval | queryable in **~2 s** |

The core install has **no dependencies at all**. The graph, the gate, the trust
ordering and the derivation are pure python — no model, no network, no GPU.

## Measured

Same questions to both sides. RAG baseline: LangChain + multilingual embeddings
+ Chroma + gpt-4o-mini, best-practice grounding prompt. **LMM's engine here is
the same gpt-4o-mini**, so the column compares architectures, not model sizes.
LMM figures are medians of three samples — these benchmarks are noisy and a
single run is not a measurement.

| Benchmark | RAG | LMM |
|---|---|---|
| Fictional corpus, 17 q (no parametric leakage possible) | 14/17 | **17/17** |
| A strategy document, 16 q | 13/16 | **14/16** |
| A 350-page device manual, 30 q | 14/30 | **27/30** — ingested in **2.3 s** vs 9.2 s |
| An inspection spreadsheet, 15 q | 15/15 | 14/15 — ingested in **86 ms** vs 4.7 s |
| **Wrong facts asserted, all runs** | 1 | **0** |

The bigger the document, the wider the gap: retrieval collapse is RAG's
structural ceiling, composition is LMM's structural strength.

Only the invented corpus is in this repository. The other three documents belong
to their owners — a manufacturer's product manual, an institution's internal
strategy document, an inspection report naming real people — and are not ours to
redistribute. `benchmarks/corpus*.txt` is a fictional world written so that no
answer can come from what a model already knows.

**These numbers are lower than this file used to show, and that is the point.**
An audit removed everything that had been fitted to the documents being scored:
few-shot examples quoting the manual's own voltages, a prompt clause teaching the
exact distinction one question turned on, and thresholds (a 700-character
window, a 40-character cell, a 0.75 similarity) chosen while reading one PDF.
The honest cost was **62/63 → 59/63**, of which the prompt content alone was 4
points — that was the share of the score that came from having seen the test.
Removing the last Turkish from the runtime cost one point more.

## Language independence, measured

No hand-written sentences, no word lists, no per-language rules anywhere in the
pipeline. That used to be an argument about the code; every benchmark was
Turkish, so nothing tested it.

`corpus_en.txt` and `corpus_es.txt` are the fictional corpus line for line — the
same facts in the same order, the same 17 questions, the same gold keys
translated. The invented names (*zerbalit, vorlin, norgul, Nortlann*) are
identical in all three, so no answer in any language can come from a model's own
knowledge. Only the language moves.

| The same world, the same 17 questions | median of 3 | samples |
|---|---|---|
| Turkish | **17/17** | 17 · 17 · 17 |
| English | **16/17** | 15 · 16 · 16 |
| Spanish | **15/17** | 15 · 15 · 15 |

**The measuring tool was the language-bound part.** Scoring an abstention meant
recognising "I don't know" — a phrase list, in one language. Against an English
document every honest refusal would have scored as a fabrication. So the
judgement moved inside: a turn now records whether it *asserted* anything,
decided by re-extracting the claims from what was spoken. `explain=True`
surfaces that stamp.

**The two-point gap is one defect.** Every remaining loss in English and Spanish
is a refusal that keeps talking — *"Vorlin is not a recognized substance"*, which
the corpus flatly contradicts. Retrieval, derivation and the gates behave
identically in three languages (the graph ingests 61–62 facts and derives 8 in
each); the engine's *stylistic* habit of explaining its refusals varies, and this
system's scoring is strict enough to charge it.

## Architecture

```
                 ┌──────────── the GATE decides everything ────────────┐
 user/message →  extract (engine)  →  gate.admit  →  GRAPH (facts, sources,
                                                      trust, contradictions,
 document     →  learn_text ──────→  evidence index  transitive derivation)
 spreadsheet  →  learn_rows ──────→  graph directly (no model calls)
                                                      │
 question     →  graph lookup + evidence retrieval →  answer (engine)
                                                      │
                 coverage gate → digit discipline → support check → verify
                 (an answer whose claims are not in the given evidence DROPS)
```

- **The engine cannot write records and cannot speak unsupported facts.** It
  produces candidates; the gate admits, the gate releases. Every other property
  here is downstream of that one sentence.
- **Two speeds.** Symbolic reasoning — derivation, causality, contradiction —
  runs in microseconds on the graph. The engine is only the language I/O.
- **A document is two materials.** Tables state their own structure and go
  straight to the graph with no model call; prose goes to the evidence index,
  where ranges and qualifications that don't fit a triple survive verbatim.
- **Growth without retraining.** Knowledge goes to the graph, never to weights.
  A periodic LoRA pass refreshes *behaviour* from gate-approved logs — never
  from the model's own raw output.

Full description: [`docs/architecture.md`](docs/architecture.md).

## Install

```bash
pip install lmm                      # core: pure python, zero dependencies
pip install 'lmm[pdf]'               # pdfplumber + pypdf
pip install 'lmm[xlsx]'              # pandas + openpyxl
pip install 'lmm[docx]'              # python-docx
pip install 'lmm[local]'             # torch + transformers + peft (local engine)
pip install 'lmm[gguf]'              # llama.cpp, CPU
pip install 'lmm[azure]'             # hosted API
```

Extras are opt-in and lazy: nothing is imported until you hand LMM a file of
that kind, and a missing reader reports the exact install line rather than a
traceback.

**Engines.** The default is a local Qwen2.5-3B-Instruct; `LMM_BACKEND=gguf` runs
on CPU through llama.cpp, `LMM_BACKEND=azure` against a hosted API. The
graph/gate layer is engine-agnostic — and most of it needs no engine at all.

## Using it

```python
from lmm import Memory

m = Memory("mind.lmm")

m.learn("spec.pdf")                       # tables → graph, prose → evidence
m.learn("inspection.xlsx")                # rows → graph, milliseconds
m.learn("report.docx")
m.learn("The probe was built on Nortlann in 2031.")   # plain text works too

answer = m.ask("where was the probe built?", explain=True)
print(answer)                             # it IS the answer string
print(answer.abstained)                   # did this turn assert anything?
print(answer.sources)                     # the stamps it rests on

m.save()
```

`learn()` returns a small report (`.facts`, `.tables`, `.adapter`, `.source`).
`m.about("Nortlann")` reads the graph directly, and `m.session` is the full
`Session` API if you need it — `Session`, `learn_text` and `tables.*` are
unchanged and still supported.

See [`examples/`](examples/) — including one that runs with no engine at all.

## Honest limits

Kept current, and deliberately specific.

- Answer *selection* can still pick a true-but-off-target sentence. The gate
  guarantees non-fabrication, not perfect relevance.
- Spec lines reachable only through very common words ("how many inches is the
  screen", where the line reads `Screen 15.6" LCD` and never says *inches*) are
  a lexical-retrieval ceiling. One such case is rescued by an append-only
  mechanism; the class is not closed.
- Two benchmark questions are stable failures and are named rather than hidden:
  one whose answer sits in a table row sharing a single stem with the question,
  and one needing a heading plus a line eight sentences below it in one window.
- The engine sometimes appends an explanation to a refusal that the memory
  contradicts. This is **detected** — the turn is scored as having asserted
  something, and loses the point — but not prevented. It is the whole of the
  remaining English/Spanish gap. (The *other* source of appended text, a second
  model call after the gate, is gone: the hedge is now the stored source stamp
  in brackets, `(~ #pdf:manual.pdf)`, and nothing is appended to a refusal at
  all.)
- **LMM does not OCR.** A PDF without a text layer — what an office scanner
  produces — teaches it nothing, and `learn()` now says so instead of reporting
  a cheerful zero. Run OCR first and learn the result.
- **There is no HTML reader.** An `.html` file is read as plain text and its
  markup lands in the evidence index; `learn()` warns, with the share of
  characters that were tags. Convert to text first.
- Table *columns* are still flattened by the PDF reader: a cell can reach the
  index next to a neighbouring column's value, and no gate can reject that,
  because the claim really is in the evidence. Rows are repaired; columns are
  not.
- A spreadsheet whose header spans more than one row loses the unnamed columns.
- An engine call is bounded by `LMM_TIMEOUT` (90 s by default; 0 removes the
  limit) on the API backend, which raises `EngineTimeout` rather than waiting
  out a quota. The local engine's generation loop is **not** interruptible, so
  the budget does not cover it.
- The coverage gate cannot distinguish negation affixes at word level (language
  lists are forbidden by design); the engine-level support check covers most of
  that class.
- Pre-1.0. The API surface may still move.

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) — it leads with the two rules
that make this project what it is: **no document-specific constants** and **no
hand-written language rules**. Both are stated with what enforcing them cost.

```bash
python3.11 tests/test_core.py     # 63 tests · no model · no GPU · no network
```

Every test is a pathology that was measured on this code, written back as an
assertion so it cannot return.

## Licence and attribution

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

**LMM is not a language model trained from scratch.** It is a memory and
reasoning layer on top of one. The default engine is
[Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (Apache
2.0, Alibaba Cloud), downloaded by the user from its own source; any LoRA
adapter this project trains is a derivative of it. What is original here is the
layer — the graph, the gate, the evidence index and the adapters. The base model
is credited, never hidden.
