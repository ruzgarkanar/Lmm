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

## Measured (same questions to both sides)

RAG baseline: LangChain + multilingual embeddings + Chroma + **gpt-4o-mini**,
best-practice grounding prompt. LMM's engine here is the same gpt-4o-mini, so
the column compares ARCHITECTURES, not model sizes. LMM numbers are medians of
three samples of the same configuration (these benchmarks are noisy; a single
run is not a measurement).

| Benchmark | RAG | LMM |
|---|---|---|
| Fictional corpus (17 q — no parametric leakage possible) | 14/17 | **17/17** |
| Real strategy PDF (16 q) | 13/16 | **14/16** |
| 350-page device manual (30 q) | 14/30 | **27/30** — ingested in **2.3 s** vs 9.2 s |
| Site-inspection spreadsheet (15 q) | 15/15 | 14/15 — ingested in **86 ms** vs 4.7 s |
| Wrong facts asserted across all runs | 1 | **0** |

The pattern: the bigger the document, the wider the gap — retrieval collapse
is RAG's structural ceiling, composition is LMM's structural strength.

**These numbers are lower than the ones this file used to show for the manual
and the strategy PDF, and that is the point.** An audit went through the code
looking for anything that had been fitted to the three documents being scored:
few-shot examples that quoted the manual's own voltages and dimensions and told
the engine what a particular ingress-protection code means, a prompt clause that
taught the very warranty-vs-service-life distinction one absence question turns
on, and numeric thresholds (a 700-character record window, a 40-character table
cell, a 0.75 similarity, an inflection rule with a special case for short
spec-table subjects) that had been chosen while reading one PDF. All of it is
gone; thresholds that survive are either read off the material at hand or stated
as what they mean. The honest cost, measured step by step: **62/63 → 59/63**,
of which the prompt content alone was 4 points — that was the share of the score
that came from having seen the test. Removing the last Turkish from the runtime
(next section) cost **one point more: 59/63 → 58/63**, on a manual question
whose answer the engine now declines to guess at.

## Language independence, measured

The line above — no hand-written sentences, no word lists, no per-language rules
— used to be an argument about the code. Every benchmark here was Turkish, so
nothing tested it. `bench/corpus_en.txt` and `bench/corpus_es.txt` are the
fictional corpus line for line: the same facts in the same order, the same 17
questions of the same four types, the same gold keys translated, nothing added.
The invented names (zerbalit, vorlin, norgul, Nortlann) are identical in all
three, so no answer in any language can come from what the model already knows.
Only the language moves.

| The same world, the same 17 questions | median of 3 | samples |
|---|---|---|
| Turkish (the original corpus) | **17/17** | 17 · 17 · 17 |
| English | **16/17** | 15 · 16 · 16 |
| Spanish | **15/17** | 15 · 15 · 15 |

**What had to come out first.** Three things had quietly made Turkish the
system's home language. Every answer in its history was framed by hand-written
Turkish labels (`OLGULAR:` / `SORU:`) wrapped around a question that could be in
any language. Every few-shot example in every classifier — causality, identity,
extraction, both read-back judges — was Turkish, so each learned its pattern in
one grammar. And the LoRA data builder pasted subjects into a hand-written
Turkish question, teaching the weights that a question looks Turkish. The labels
are structural field names now, the examples are spread across four languages,
and the question form is supplied with the dataset whose language it belongs to.

**The measuring tool was the language-bound part.** Scoring an absence question
means deciding whether the system declined to answer, and from outside that
leaves only the wording — a phrase list, in one language. Against an English
document every honest refusal would have scored as a fabrication, and the
benchmark would have reported a language dependency the engine does not have. So
the judgement moved inside: a turn carries whether it ASSERTED anything, decided
by re-extracting the claims from what was spoken (the same re-extractor the
fabrication gate already trusts). The phrase list survives only as a fallback for
result files written before the stamp existed. One structural guard outranks the
stamp in every language: an "abstention" that names a figure is a fabrication.

**What the three-language run found is the reason to have it.** Two defects were
invisible while every benchmark spoke one language, and both were real, not
scoring artifacts. The answer prompt had illustrated "reply in the question's
language" with a German example, and the illustration beat the rule exactly where
the model had no evidence to anchor on: asked in English with nothing to say, it
answered *"Ich weiß es nicht."* four times out of four. And a refusal that keeps
talking is a fabrication — asked at how many degrees vorlin boils, the engine
volunteered *"Vorlin is not a recognized substance"*, which the corpus flatly
contradicts. Spanish did the same; Turkish, tersely, did not.

**The two-point gap is not closed, and it is one defect.** Every remaining loss
in English and Spanish is that same trailing clause: *"...la población de la isla
puede variar"*, *"Vorlin is not a recognized substance"*. Tightening the prompt
recovered part of it and the rest is stable across all three samples. The claim
the gap qualifies is worth stating exactly: retrieval, derivation and the gates
behave the same in three languages — the graph ingests 61-62 facts and derives 8
in each — while the engine's *stylistic* habit of explaining its refusals varies
by language, and this system's abstention scoring is strict enough to charge it.

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
- **Language-independent, and now measured:** no hand-written sentences, no
  word lists, no per-language rules anywhere in the pipeline. Users converse
  in their own language; the system's confirmations and refusals are
  engine-generated in that language. See the table below — the same fictional
  world, the same 17 questions, in three languages.

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
  the screen") are a lexical-retrieval ceiling. One such line is rescued by an
  append-only mechanism (a question that asks for a quantity gets one extra
  evidence seat); the class is not closed.
- Two questions are stable failures and are named rather than hidden: a manual
  question whose answer sits in a table row sharing only one stem with the
  question, and a strategy-document question whose answer needs a heading and a
  line eight sentences below it in one window — the widest window scale now
  refuses to index prose that long, and this document's records fall on the
  wrong side of the bound it derives.
- The engine sometimes appends an explanation to a refusal ("X is not a
  recognized substance") that the memory contradicts. The gate on the evidence
  path scores candidates rather than re-verifying the chosen sentence, so such
  a clause is not stripped; it is DETECTED — the turn is scored as having
  asserted something, and loses the point — but not prevented. It is the whole
  of the remaining English/Spanish gap.
- The coverage gate cannot distinguish negation affixes at the word level
  (language lists are forbidden by design); the engine-level support check
  covers most of this class.
- Base language model: Qwen2.5-3B-Instruct (Apache 2.0). LMM's originality
  is the memory/gate/evidence layer on top of an open engine — the base
  model is credited, never hidden.
