# Getting started

## Install

```bash
pip install living-memory-model      # core: one dependency, numpy
```

The core install has **one dependency, numpy**, and it is named rather than
hidden. The graph, the gate, the trust ordering and the derivation are plain
Python and call nothing; what numpy buys is the **meaning channel** — a 30 MB
static multilingual matrix that travels in the wheel and is attached by
default, so a question asked in other words than the document used is
understood with no download, no account and no network.

If you do not want it, `Memory(..., dense=False)` runs without it and without
numpy; the channel is then absent rather than broken.

Readers and engines are opt-in extras:

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


## Asking in other words than the document used

This is the case a word search cannot reach, and the reason the meaning
channel ships on:

```python
m = Memory("mind.lmm")
m.learn("lease.pdf")            # the document says "the tenant RESIDES at ..."

m.ask("where does the tenant live?")   # LIVES appears nowhere in the document
```

Nothing was configured for that. The channel picks the document by meaning,
the words pick the lines inside it, the two rankings are fused by rank, and a
late-interaction pass orders what the engine finally sees.

If you have a stronger encoder, plug it in and ours is never loaded:

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
m = Memory("mind.lmm",
           encoder=lambda texts: model.encode(texts, normalize_embeddings=True).tolist())
```

## When the first question feels slow

The field-name index is derived from every sentence in the store, so it is
built once and then **written beside the memory** — `mind.lmm.expansion`,
about 0.7 MB on a 115,000-line store. Call `m.save()` after ingesting and the
next process starts warm: measured, the first question went from 10.6 s to
0.13 s. Delete the file and it is rebuilt; a file that does not match the
sentences is not trusted at all.
