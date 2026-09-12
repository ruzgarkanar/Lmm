# LMM — Living Memory Model

**A verifiable memory layer for language models.** Facts live in a graph,
every claim carries its source, and an answer the memory does not support
cannot leave the system.

```bash
pip install living-memory-model      # the import is `lmm`
```

One dependency (numpy), and a 30 MB multilingual embedding matrix travels in
the wheel — so a question asked in other words than the document used is
understood **out of the box**, offline, with nothing to configure.

```python
from lmm import Memory

m = Memory("mind.lmm")            # persistent; Memory() is transient
m.learn("manual.pdf")             # pdf · docx · xlsx · csv · md · txt · or text
print(m.ask("what is the screen's diagonal?"))
m.save()
```

That is the whole quickstart. One method reads every format, and the file
extension picks the adapter.

## The one rule

> In an LLM, knowledge is frozen into weights at training time.
> In LMM, knowledge lives in a memory you can write to, inspect and audit —
> which is what makes "I don't know" trustworthy rather than polite.

The engine cannot write records, and cannot speak unsupported facts. Every
other property of this system is downstream of that sentence.

## Measured, not promised

| same engine, median of 3 | LMM | GraphRAG |
|---|---|---|
| Fictional corpus, 17 questions | **17/17** · spread 0 | 16/17 |
| NIST SP 800-63B, 13 questions | **11/13** · spread 0 | 7/13 · spread 2 |
| Indexing that standard | **0 model calls** | 220 calls · 498k tokens |
| Wrong facts asserted | **0** | 1 |

Retrieval in 0.6, measured on this repository's own corpora — by whether the
line that answers actually reaches the engine:

| | first | in five |
|---|---|---|
| words alone | 31% | 32% |
| + the bundled meaning channel | 31% | 60% |
| + late-interaction reordering | 34% | **79%** |

No model call in any of it.

Every number on this site carries the commit it was measured at, and the
[two bugs found in our own scorer — both penalising the competitor —
were published with the corrected numbers](reference/measurements.md).

## Where to go

- **[Getting started](guides/getting-started.md)** — install, learn, ask.
- **[Why a memory](concepts/why.md)** — the argument, in five minutes.
- **[The API surface](reference/api.md)** — everything callable, and what
  needs an engine (most of it does not).
- **[The architecture](concepts/architecture.md)** — the three channels, the
  two gates, and which layer each instrument sits at.
- **[Honest limits](concepts/limits.md)** — what this does not do, measured.
