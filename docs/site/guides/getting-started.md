# Getting started

## Install

```bash
pip install living-memory-model      # core: pure python, zero dependencies
```

The core install has **no dependencies at all** — the graph, the gate, the
trust ordering and the derivation are plain Python. Readers and engines are
opt-in extras:

```bash
pip install 'living-memory-model[pdf]'      # pdfplumber + pypdf
pip install 'living-memory-model[xlsx]'     # pandas + openpyxl
pip install 'living-memory-model[docx]'     # python-docx
pip install 'living-memory-model[openai]'   # OpenAI, OpenRouter, Groq, vLLM…
pip install 'living-memory-model[local]'    # torch + transformers, local engine
```

Nothing is imported until you hand LMM a file of that kind, and a missing
reader reports the exact install line rather than a traceback.

## Teach it, ask it

```python
from lmm import Memory

m = Memory("mind.lmm")

m.learn("spec.pdf")                       # tables → graph, prose → evidence
m.learn("inspection.xlsx")                # rows → graph, milliseconds
m.learn("The probe was built on Nortlann in 2031.")   # plain text works too

answer = m.ask("where was the probe built?", explain=True)
print(answer)                             # it IS the answer string
print(answer.abstained)                   # did this turn assert anything?
print(answer.sources)                     # the stamps it rests on
```

A spreadsheet reaches the graph with **no model call at all** — a table
states its own structure. That is why a sheet loads in milliseconds and a
350-page manual is queryable in about two seconds.

## Three calls you will use next

```python
m.where("psychological safety")
# → [("Trust Basics", 62), ("Team Field Guide", 51), ...]
# which documents speak of a term — counted, with receipts, no engine, ms

text, sources = m.compose("draft a one-day onboarding programme")
# a structured draft: the engine organises, the material speaks, and every
# line is re-read on the way out — invented numbers and unsupported claims drop

m.about("Nortlann")
# the raw records the graph holds on a concept
```

`ask()` **cannot write memory** — a question is not a lesson, however
imperative its grammar. Teaching is the conversational surface's job:
`m.session.respond(text, teach=True)`.
