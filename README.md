# LMM — Living Memory Model

> In an LLM, knowledge is frozen into weights at training time.
> In LMM, knowledge lives in a **verifiable memory**: writable at any moment,
> persistent, source-stamped — and the foundation of the system's ability to
> distinguish "I know" from "I don't know".

LMM is a **memory-and-reasoning layer** that wraps any language model
(a local 3B, a cloud API — the engine is pluggable). It replaces
embedding-based RAG for factual work:

| | Embedding RAG | LMM |
|---|---|---|
| Ingestion | chunks + embeddings | facts into a **graph** + sentences into an **evidence index** |
| Retrieval | similarity gamble | deterministic word/graph lookup, explainable |
| Multi-hop | fails when chunks don't co-retrieve | **derives** new facts symbolically (ms, no model call) |
| Fabrication | prompt-level plea | **structural gate**: unsupported claims cannot leave the system |
| Provenance | none | every fact carries its source; uncertain answers carry a caveat |
| Tables / spreadsheets | flattened into text | **rows go straight into the graph** — zero model calls |
| A 350-page manual | seconds of embedding, lossy retrieval | **queryable in ~0.5 s** (instant-ready mode) |

## Measured (four benchmarks, same questions to both sides)

RAG baseline: LangChain + multilingual embeddings + Chroma + **gpt-4o-mini**
(a stronger engine than LMM's local default), best-practice grounding prompt.

| Benchmark | RAG | LMM |
|---|---|---|
| Fictional corpus (17 q — no parametric leakage possible) | 14/17 | **15/17** |
| Real strategy PDF (16 q) | 13/16 | **15/16** |
| Site-inspection spreadsheet (15 q) | 15/15 | 14/15 — ingested in **86 ms** vs 4.7 s |
| 350-page device manual (30 q) | 14/30 | **22/30** — ingested in **0.44 s** vs 9.2 s |
| Wrong facts asserted across all runs | 1 | **0** |

The pattern: the bigger the document, the wider the gap — retrieval collapse
is RAG's structural ceiling, composition is LMM's structural strength.

## Architecture (one screen)

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

- **The engine cannot write records and cannot speak unsupported facts.**
  It only produces candidates; the gate admits, the gate releases.
- **Two speeds:** symbolic reasoning (derivation, causality, contradiction)
  runs in microseconds on the graph; the engine is only the language I/O.
- **Growth without retraining:** knowledge goes to the graph, never to
  weights. A periodic LoRA ("sleep consolidation") refreshes *behavior* —
  language quality — from gate-approved conversation logs, never from the
  model's own raw output.
- **Language-independent by construction:** no hand-written sentences, no
  word lists, no per-language rules anywhere in the pipeline. Users converse
  in their own language; the system's confirmations and refusals are
  engine-generated in that language.

## Quick start

```bash
python3.11 -m lmm.chat                 # interactive session, local engine
```

```python
from lmm.session import Session
from lmm import tables

s = Session("memory.lmm")
s.learn_text(open("manual.txt").read(), deep=False)   # 350 pages ≈ 0.5 s
tables.learn_xlsx(s, "inspection.xlsx")               # spreadsheet → graph, ms
print(s.respond("What is the probe's water protection rating?"))
s.save()                                              # memory + evidence persist
```

Engines: local Qwen (default, `models/qwen-3b` + optional LoRA in
`models/lmm/lora`), `LMM_BACKEND=gguf` (CPU, llama.cpp), `LMM_BACKEND=azure`
(experiments). The graph/gate layer is engine-agnostic.

## Honest limits (kept current)

- Answer *selection* can still pick a true-but-off-target sentence; the gate
  guarantees non-fabrication, not perfect relevance.
- Spec lines only reachable through very common words ("how many inches is
  the screen") are a lexical-retrieval ceiling; the PDF-table adapter
  (tables → graph, like xlsx) is the planned fix.
- The coverage gate cannot distinguish negation affixes at the word level
  (language lists are forbidden by design); the engine-level support check
  covers most of this class.
- Base language model: Qwen2.5-3B-Instruct (Apache 2.0). LMM's originality
  is the memory/gate/evidence layer on top of an open engine — the base
  model is credited, never hidden.
