# Cost — measured, not estimated

Every token below is the `usage` field the server itself returned, collected by [`meter.py`](meter.py), which wraps the OpenAI chat SDK and the local embedding call and touches no library code. Every figure is the **median of 3 repeats** of the same configuration.

Both sides run the **same engine**, Azure `gpt-4o-mini`, so the comparison is between architectures rather than model sizes — the same discipline the accuracy table in the README uses.

Corpora are the two reproducible fictional ones in this repository (`corpus.txt` + `questions.json`, `corpus_en.txt` + `questions_en.json`) — no third-party document is involved, so anyone can re-run this. **Measured 2026-08-19, at commit `7ec9851`.** Every table in sections 1–6 is that one run: the answer path has moved several times in this repository's history and a cost table with no commit under it cannot be checked against the tree that produced it.

Reproduce:

```bash
sh benchmarks/cost_all.sh          # 3 sides x 2 languages x 3 repeats
python3.11 benchmarks/cost_report.py > benchmarks/COST.md
```

## 1. Ingestion cost (paid once per document)

**EN corpus** — 1,642 characters

| | model calls | prompt tok | completion tok | embedding calls | wall |
|---|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 0 | 0 | 0 | 1 **local, 0 API tokens** (0.1 s CPU) | 4.7 s |
| LMM `deep=False` (evidence-only) | 0 | 0 | 0 | 0 — none | 0.0 s |
| LMM `deep=True` (graph extraction) | 121 | 43,529 | 1,056 | 0 — none | 22.9 s |

**TR corpus** — 1,529 characters

| | model calls | prompt tok | completion tok | embedding calls | wall |
|---|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 0 | 0 | 0 | 1 **local, 0 API tokens** (0.1 s CPU) | 4.7 s |
| LMM `deep=False` (evidence-only) | 0 | 0 | 0 | 0 — none | 0.0 s |
| LMM `deep=True` (graph extraction) | 119 | 42,995 | 1,182 | 0 — none | 20.5 s |

**The embedding zero is real and it matters.** RAG's ingestion here spends no API tokens at all, because the embedding model is a local `sentence-transformers` checkpoint. What it spends instead is CPU seconds and roughly half a gigabyte of dependencies (section 4). Had a hosted embedding API been used instead, that column would carry a real token bill; this setup deliberately gives RAG the cheaper option.

**LMM `deep=False` ingests for free** — literally zero model calls, zero tokens, 6 ms. Building the evidence index is pure python: no model, no network, no embedding. This is the cheapest ingestion in the table by a wide margin, and it is the mode meant for a large document.

**`deep=True` is where LMM's ingestion cost lives** — 121 calls and 44,585 tokens to read this small corpus into a graph, because every candidate fact is extracted and then re-read by the gate before it is admitted. That is the provenance and the refusal guarantee being paid for up front. It buys 62 admitted facts and the 8 further ones the graph *derives* symbolically — no model call, microseconds — which is what makes multi-hop answers possible. It is paid once per document rather than once per question.

For scale: that corpus is ~1.5 KB. Ingestion cost on this path grows with the document, so a 350-page manual is not a `deep=True` job — which is exactly why `deep=False` exists.

## 2. Cost per question

**EN**

| | calls/q | prompt tok/q | completion tok/q | wall/q |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 1.0 | 512 | 11 | 1.5 s |
| LMM `deep=False` (evidence-only) | 7.6 | 5,141 | 59 | 11.2 s |
| LMM `deep=True` (graph extraction) | 5.3 | 3,840 | 47 | 7.8 s |

**TR**

| | calls/q | prompt tok/q | completion tok/q | wall/q |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 1.0 | 593 | 9 | 1.5 s |
| LMM `deep=False` (evidence-only) | 7.8 | 5,374 | 65 | 11.7 s |
| LMM `deep=True` (graph extraction) | 5.2 | 3,735 | 46 | 7.9 s |

### Where LMM's calls go

Calls per question, by kind. This is the breakdown that says where to attack the cost, and it is not flattering: the verification read-back — the gate re-extracting the claims out of a sentence before it is allowed to leave — is a large share of the bill. That is fabrication-0 being paid for in tokens.

| | extract calls/q | answer calls/q | read-back calls/q | relation-check calls/q | prompt tok/q (largest bucket) |
|---|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) (en) | 0.0 | 1.0 | 0.0 | 0.0 | answer: 512 |
| LMM `deep=False` (evidence-only) (en) | 1.0 | 2.9 | 2.8 | 1.0 | answer: 2,249 |
| LMM `deep=True` (graph extraction) (en) | 0.8 | 2.2 | 2.1 | 0.1 | answer: 1,747 |
| RAG (embed + Chroma + 4o-mini) (tr) | 0.0 | 1.0 | 0.0 | 0.0 | answer: 593 |
| LMM `deep=False` (evidence-only) (tr) | 1.0 | 2.9 | 2.9 | 1.0 | answer: 2,314 |
| LMM `deep=True` (graph extraction) (tr) | 0.8 | 2.1 | 2.0 | 0.3 | answer: 1,684 |

## 3. Zero-call answers

How many questions LMM answered with **no model call at all** — pure graph lookup, no engine involved. The architecture permits it: a derived fact or a table row is already an answer. Whether it *happens* on this corpus is a measurement, and the honest answer is below.

This row read **0 of 17 at every commit** until `lmm/lookup.py` was written. The architecture permitted the zero-call answer and the code never took it: `respond` spent a call turning a held triple into a sentence and more calls reading that sentence back. `lookup` takes it, for the questions it can settle without guessing.

| | questions | answered with 0 model calls | share |
|---|---|---|---|
| LMM `deep=False` (evidence-only) (en) | 17 | 0 | 0% |
| LMM `deep=True` (graph extraction) (en) | 17 | 3 | 18% |
| LMM `deep=False` (evidence-only) (tr) | 17 | 0 | 0% |
| LMM `deep=True` (graph extraction) (tr) | 17 | 3 | 18% |

**What those questions have in common is what the path requires: the question NAMES what it asks about.** "is vorlin a liquid" names `liquid`, and the graph holds `vorlin -[type]-> liquid` as a record it DERIVED — so the answer is the record, returned in microseconds with no engine to consult and therefore nothing for an engine to invent. These are the multi-hop questions, which is the pleasing part: the step that was supposed to prove "the symbolic layer needs no model" is the step now proving it in the bill.

**The rest still go to the engine, and the reason is a limit rather than an oversight.** A question that names no field at all — "norgul nedir" — is indistinguishable, structurally, from one naming a field the graph has never heard of — "Nortlann'ın başkenti neresidir". Both are a known subject followed by words no node and no document carries. Answering the first from the subject's single taxonomy value means answering the second with it too, which turns a correct abstention into a confident wrong answer. The eight "what is X" questions per language stay on the paid path, and that is the price of the guarantee.

## 4. Infrastructure

What each architecture requires on disk before it can answer anything. Measured with two clean virtualenvs ([`cost_infra.sh`](cost_infra.sh)) — package counts and megabytes, not adjectives.

| | pip packages | site-packages | plus model weights |
|---|---|---|---|
| **LMM** — `pip install living-memory-model` | **1** (`dependencies = []` in pyproject.toml) | **1.1 MB** | none — the graph, gate, trust ordering and derivation are pure python |
| **RAG** — `pip install langchain-chroma langchain-openai langchain-text-splitters sentence-transformers` | **114** | **1.35 GB** | **458 MB** embedding checkpoint, fetched at first use |

Both venvs also carry the same ~21 MB of `pip`/`setuptools`, excluded from both rows. The RAG figure is dominated by `torch` (534 MB), `transformers` (110 MB), `scipy` (100 MB), `onnxruntime` (81 MB) and — via Chroma — `kubernetes` (84 MB). That is roughly **1,200x the disk and 114x the packages**, before a single question is asked. To be fair to RAG: this weight is what buys it the zero-API-token ingestion in section 1, and an LMM deployment that runs its own local engine pulls in a comparable stack (`pip install living-memory-model[local]`). The 1.1 MB figure is the *core* — the part that stores, derives and gates, which is the part that has no dependencies.

## 5. Price, as a separate layer

The tables above are token counts, which are true regardless of what anyone charges. Below is one multiplication applied to them: published gpt-4o-mini rate used for this table: $0.15 / 1M input tokens, $0.60 / 1M output tokens. **Prices change; re-multiply rather than trust this table.** Run the same architecture on a local GGUF engine (`LMM_BACKEND=gguf`) and every dollar figure here becomes **exactly $0** — replaced by seconds of your own CPU/GPU, which is the trade this project is built around.

**EN — total spend for N questions over one ingested document**

| | ingestion (once) | per question | 1,000 questions | 100,000 questions |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | $0.0000 | $0.0001 | $0.0834 | $8.34 |
| LMM `deep=False` (evidence-only) | $0.0000 | $0.0008 | $0.8065 | $80.65 |
| LMM `deep=True` (graph extraction) | $0.0072 | $0.0006 | $0.6110 | $60.39 |

**TR — total spend for N questions over one ingested document**

| | ingestion (once) | per question | 1,000 questions | 100,000 questions |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | $0.0000 | $0.0001 | $0.0943 | $9.43 |
| LMM `deep=False` (evidence-only) | $0.0000 | $0.0008 | $0.8451 | $84.51 |
| LMM `deep=True` (graph extraction) | $0.0072 | $0.0006 | $0.5952 | $58.81 |

### Break-even

- **LMM `deep=False` (evidence-only) vs RAG (en)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.
- **LMM `deep=True` (graph extraction) vs RAG (en)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.
- **LMM `deep=False` (evidence-only) vs RAG (tr)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.
- **LMM `deep=True` (graph extraction) vs RAG (tr)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.

The one crossing that exists is *inside* LMM:

- **`deep=False` vs `deep=True` (en)** — `deep=False` ingests free but asks dearer; `deep=True` overtakes it at **~35 questions** on one document. Below that, shallow is the cheaper LMM; above it, the graph has repaid its own extraction.
- **`deep=False` vs `deep=True` (tr)** — `deep=False` ingests free but asks dearer; `deep=True` overtakes it at **~28 questions** on one document. Below that, shallow is the cheaper LMM; above it, the graph has repaid its own extraction.

**Treat those two numbers as an order of magnitude, not a threshold.** They divide a fixed ingestion cost by a small per-question difference, so the noise in the per-question figure is amplified — which is exactly why the two languages disagree by several-fold on a corpus that is otherwise line-for-line identical. What is solid is the shape: shallow wins for a handful of questions, deep wins once you are asking hundreds.

## 6. Honest summary

**Where we are more expensive: everywhere that is measured in tokens.** On the EN corpus LMM `deep=True` spends **5.3 model calls per question** against RAG's 1.0 — about **5x the calls and 7x the prompt tokens**. Ingestion is worse in relative terms: 44,585 tokens against RAG's 0, because RAG's embedding step is a local model and spends none at all. There is no reading of these numbers in which LMM is the cheap option on a hosted per-token engine, and no break-even where that reverses — the gap grows with every question asked.

**We are also slower per question**: 7.8 s against RAG's 1.5 s, because six or seven sequential calls cannot beat one. On the same rate-limited deployment, measured back to back.

**Which LMM mode is cheaper is a question of how many questions you will ask.** `deep=False` ingests for nothing but asks dearer (7.6 calls/q vs 5.3); `deep=True` pays 44,585 tokens up front and then asks cheaper, because the graph answers more directly. They cross somewhere in the **low hundreds of questions** on one document (~35 here, but see the caveat above — the two languages disagree several-fold). `deep=False` is also the only workable mode for a large document, since extraction cost scales with the text while the evidence index does not.

**Some questions now cost nothing at all — on the engine that ingested them well.** Section 3 counts 6 answers across these runs that never reached the engine — the graph settled them itself, in microseconds, for zero tokens and zero seconds of anyone's GPU. It is a minority of the questions and it does not move the per-question average much; what it moves is the FLOOR. On that share of the traffic the architecture is not merely cheaper than RAG, it is free. **That floor is not free of the engine that built it** — see §7, where the same corpus on a local 3B model never reaches it at all, because the derivation the floor rests on never closes there.

**Where we are cheaper: the axes this table cannot bill.** The tokens above buy three things RAG does not have at any price — every answer carrying its source, a structural gate that stops an unsupported claim from leaving, and multi-hop facts *derived* symbolically in microseconds with no model call. Section 4's disk figures are the other axis: a zero-dependency core against an embedding stack and a vector database. And the whole dollar column collapses to zero on a local engine, where the cost becomes your own seconds — which is the deployment this project is actually built for.

**So the fair sentence is this:** if you are paying per token for a hosted model and you only need one-hop lookup, embedding RAG is cheaper than LMM and will stay cheaper. LMM's case is accuracy, provenance and refusal (see the README's benchmark table), bought with tokens at ingestion and at verification — or bought with CPU seconds instead, on hardware you already own.
## 7. Does the zero-call floor survive a weak local engine?

Section 3's zero-call share is real, and it is measured on Azure `gpt-4o-mini`
— a strong engine. The question this project actually turns on is whether the
same corpus, ingested by a **weak local model**, produces the same floor. It
does not, and the reason is worth stating precisely because it is not a defect
in the lookup path — it is a defect **upstream of it**, in extraction.

Measured (`benchmarks/gguf_run.py`, TR corpus, `LMM_BACKEND=gguf
LMM_N_GPU_LAYERS=99`, the LoRA-merged 3B int4 build, one session ingested once
and asked both ways):

| | ingest facts | ingest derived (`#inference`) | zero-call answers |
|---|---|---|---|
| Azure gpt-4o-mini (§1, §3) | 59–62 | **8** | **3/17 (18%)** |
| Local 3B int4 (this section) | 60 | **0** | **0/17 (0%)** |

`lookup.find` never fired once, and the ingested graph shows why: the local
extractor's SUBJECTS are frequently whole garbled clauses instead of single
concepts — `"kelvit gri renklidir"` as a subject, `"bataklık büyürse"` as
another — where the strong engine reads `kelvit -[colour]-> grey`. Two
sentences that should name the SAME node (`vorlin`, `içecek`, `sıvı`) instead
each mint their own, so `core/transitive.py`'s two-independently-witnessed-
triangle threshold is never met, `tür` is never marked transitive, and nothing
is ever derived — hence `derived: 0`. `lookup.find` only ever answers a
TRANSITIVE relation (see `lmm/lookup.py`'s docstring for why that restriction
exists), so with nothing transitive it has nothing to say, correctly: it
declined every time rather than guessing from the noisy graph it was given.

### 7.1 That diagnosis was incomplete — an attempted fix, and what it measured

The paragraph above named the garbled subject as THE reason, and a fix was
built for exactly it: a document-wide second pass that reduces a long subject
to a shorter one when the document itself uses the shorter form as a subject
elsewhere (whole words only — never a character prefix, so `kart`/`kartal`,
`organ`/`organizma`, `şeker`/`şekersiz` cannot merge). **It changed nothing,
and the reason it changed nothing is the finding.**

To measure it without paying for a fresh 61-sentence ingestion per variant, the
local engine's readings were cached once (`extract.reextract` + `is_causal`,
sentence by sentence) and replayed through the real `learn_text` write path.
The replay reproduces the baseline exactly — 60 facts, 1 skipped, 0 derived —
so the columns below differ only in the variable named.

`normalization=oracle` is not a candidate implementation. It is the CEILING:
every multi-word subject reduced to its first word, which on this corpus is the
right concept nearly every time. It answers "what is the most this class of fix
could possibly buy".

| subject normalization | causal routing | facts | derived | transitive | zero-call (`lookup`) |
|---|---|---|---|---|---|
| off (baseline) | as the engine reads it | 60 | 0 | — | 0/17 |
| on (the fix) | as the engine reads it | 60 | 0 | — | 0/17 |
| **oracle (perfect)** | as the engine reads it | 60 | **0** | — | **0/17** |
| off | disabled | 57 | 0 | — | 0/17 |
| on | disabled | 57 | 0 | — | 0/17 |
| **oracle (perfect)** | **disabled** | 56 | **7** | `tür` | **2/17** |

Read the third row first: **even a perfect subject normalization derives
nothing.** The fix was aimed at a real defect that is not the binding one, so
it was reverted rather than kept for the sake of the diff.

The binding one is in the row below it. `generate.is_causal` — the engine
deciding whether a sentence asserts causality — answers YES for **41 of the 61
sentences** of this corpus on the local 3B, including textbook is-a sentences
that the prompt's own few-shot examples cover (`an eagle is a bird -> NONE`):

    Zerbalit bir metaldir.   -> CAUSE: zerbalit -> EFFECT: metaldir
    Kuş bir canlıdır.        -> CAUSE: kuş      -> EFFECT: canlıdır
    Gölün suyu tatlıdır.     -> CAUSE: ...      -> EFFECT: tatlıdır

Four of those 41 are genuinely causal. The same eight sentences on Azure
`gpt-4o-mini`, same prompt, same code: **every one classified correctly** — the
two causal ones causal, the six declaratives NONE. So this is engine quality,
not a prompt defect.

The consequence is structural. `learn_text` checks the causal reading FIRST,
and a sentence read as causal never reaches the triple path at all — its fact
is written under `#causes` instead of `tür`. The taxonomy is therefore SPLIT
ACROSS TWO PREDICATES, and `core/transitive.py` counts triangles within ONE
predicate. `zerbalit -[#causes]-> metaldir` beside `metal -[tür]-> maddedir`
closes nothing, however clean the subjects are. That is why the two defects
only pay off together, and why neither alone moves the number.

**Why the safe fix cannot reach the oracle, measured rather than assumed.** For
normalization to recover `kelvit` from `"kelvit bir metaldir"`, something has
to attest `kelvit` as a concept, and on this corpus the local extractor never
once emits it alone — it appears only inside four clauses. The document
statistic that would capture it (a word recurring across distinct subjects)
does not separate it from the glue: `bir` occurs inside 17 distinct subjects,
`kelvit` in 4, `canlıdır` in 6, `renklidir` in 2. Ranking by that frequency in
either direction picks `renklidir`, `maddedir`, `sıvıdır` — the predicate
words — as often as it picks the entity. There is no threshold that separates
those bands on this corpus, and a threshold chosen on this corpus is exactly
the per-document constant this repository refuses. So the honest statement is
that the safe version of this fix is worth nothing here, and the unsafe version
was not built.

**One more thing the oracle row says, and it is not encouraging.** The 7 derived
facts it produces include `nortlann -[tür]-> yazar` and `norgul -[tür]-> bir` —
derivation over a noisy graph derives noise, and `lookup` then answers from it
(`norgul çoğalırsa ne olur` → `norgul — tür → çoğalır`). Reaching the zero-call
floor on this engine would mean reaching it with fabrication-shaped junk inside
it. Whatever restores the floor locally has to fix EXTRACTION QUALITY, not the
route between the extractor and the graph.

**The obvious next move was tried, and it is closed.** The paragraph above names
the causal reading as the thing that steals the sentences, so the cheapest
imaginable fix is to stop asking it first: read the FACTS first and ask
`is_causal` only when no triple comes back. It is an ordering, not a language
rule, and it costs one line. Both arms were replayed through the real
`learn_text` write path on cached local readings — two independent 61-sentence
readings of the same corpus, taken a day apart, agreeing exactly — and the
fact-first arm was then run end to end on the local engine as well:

| reading order | facts | derived | `#causes` edges | zero-call (`lookup`) | ingest calls |
|---|---|---|---|---|---|
| causal first (shipped) | 60 | **0** | **38** | 0/17 | 88 |
| fact first (the fix) | 57 | **0** | **0** | 0/17 | 63 |

**It derives nothing and it deletes the causal graph.** Not one derived fact
appears, for the reason §7.1 already gives — the subjects the local extractor
writes are whole clauses (`kelvit bir metaldir`), so the taxonomy does not
close a triangle no matter which predicate it is filed under. The routing was
never the binding constraint; the oracle row above had already said so, and
this is the confirmation from the other direction. What the change does buy is
25 fewer engine calls per ingestion, because the second read is skipped
whenever the first one succeeds — real, and not worth what it costs.

**What it costs is the whole causal graph, and that is not a local-engine
artifact.** The reason is a fact about sentences, not about weak models: a
sentence that asserts a cause almost always also states an extractable
relation, so whichever read is asked first is simply the one that wins. On
Azure `gpt-4o-mini`, where `is_causal` classifies all eight probe sentences
correctly, **all five causal sentences of this corpus also return a triple**:

    Yağmur yağarsa bataklık büyür.  is_causal=(yağmur, bataklık)
                                    reextract=[yağmur, şart, yağarsa]
                                              [bataklık, büyüme, artar]
    Norgul çoğalırsa morlan çoğalır. is_causal=(norgul, morlan)
                                     reextract=[norgul, çoğalma, morlan]

So fact-first would empty `#causes` on the STRONG engine too, where the causal
reading is the correct one — a certain loss on the two causal questions'
graph path, in exchange for a gain that measures zero. The change was reverted;
the finding is kept here and the ordering is now pinned by a model-free test
(`O3`), so the next reader does not have to spend the ingestion to learn it.

### 7.1.1 Making the wrong subject UNSAYABLE — a grammar, and what it bought

§7.1 ends by saying the fix has to be in EXTRACTION QUALITY. A prompt asks for
a short subject and the local 3B ignores it. llama.cpp can do better than ask:
`create_chat_completion` takes a GBNF grammar and constrains the DECODER, so
what the grammar cannot spell the engine cannot say. The attempt below built,
per sentence, a grammar whose `subject` (and optionally `value`) field can only
be **a contiguous run of that sentence's own words, the whole sentence
excluded** — a prefix-sharing trie, so an n-word sentence costs n rules rather
than n²/2 literals, and nothing in it is a list, a constant or a keyword: every
literal comes from the sentence being read. Beside it, the second half of
§7.1's diagnosis was addressed by writing BOTH readings of a sentence — the
causal edge AND the triple — so the taxonomy stops being split across two
predicates without `#causes` being emptied, which is what the reversal in §7.1
did.

Measured end to end on the local engine (`benchmarks/gguf_run.py`, TR corpus,
61 sentences, `LMM_BACKEND=gguf`, Metal, one ingestion per arm):

| arm | facts | derived | transitive | zero-call | ingest calls | ingest wall |
|---|---|---|---|---|---|---|
| base (shipped) | 60 | **0** | — | 0/17 | 88 | 40 s |
| span grammar only | 59 | **0** | — | 0/17 | 88 | 43 s |
| both readings only | 95 | **0** | — | 0/17 | 126 | 76 s |
| **grammar + both readings** | 93 | **0** | **`tür`** | **0/17** | 128 | 83 s |
| grammar + both, values constrained too | 92 | **0** | `tür` | 0/17 | 127 | 79 s |

**The grammar does exactly what it was built to do, and the engine walks around
it.** "Subject = the whole sentence" became unsayable, so the model said the
longest thing that was still sayable — the sentence minus its last word:

    Kelvit bir metaldir.   base: subject `kelvit bir metaldir`
                           grammar: subject `kelvit bir`
    Selvin bir içecektir.  base: `selvin bir içecektir` -> grammar: `selvin bir`

Which is not nothing — those subjects are now CONSISTENT across sentences, and
`core/transitive.py` marks `tür` transitive for the first time on this engine
(no predicate ever qualified in any earlier local run). But no derivation
follows, and the reason is worth more than the attempt: the triangles that
qualified `tür` are triangles the corpus STATES, so the closing edge is already
an observed fact and there is nothing left to derive. Adding `Mind.run()`'s
closure pass afterwards changes none of the numbers above.

**The floor is blocked one layer lower down, and that layer is a gate.** Even
with a clean subject, `lookup` answers "is vorlin a liquid" only if the question
word `sıvı` reaches the node the document wrote, which is `sıvıdır` — and
`inflect.ROOT` is five letters, so a four-letter Turkish root cannot be joined
to its inflected form. Loosening that is loosening the identity rule that keeps
`kart`/`kartal` and `organ`/`organizma` apart, which is not a trade this
repository makes for a benchmark number.

The oracle rows confirm the shape of it, replayed offline from the cached
readings (`benchmarks/cost/gguf_readings_gbnf.json`, the same method as §7.1 —
no engine, so the columns differ only in the variable named):

| readings | both readings | subject normalization | facts | derived | lookup |
|---|---|---|---|---|---|
| base | no | as read | 60 | 0 | 0/17 |
| base | yes | as read | 95 | 0 | 0/17 |
| base | yes | **oracle (first word)** | 94 | **7** | **1/17** |
| grammar | yes | as read | 93 | 0 | 0/17 |
| grammar | yes | document-attested shortening | 93 | **0** | 0/17 |
| grammar | yes | **oracle (first word)** | 90 | **7** | **1/17** |

Only the oracle derives, only on this corpus, and §7.1's warning about what it
derives still holds — `norgul -[tür]-> bir`, `kar -[tür]-> çoğalır`. The
"document-attested shortening" row is §7.1's reverted fix re-tested on top of
the grammar, on the theory that the grammar now attests the short forms it
needs: it does attest `zerbalit`, `morlan`, `telvas` alone, and it never once
attests `metal`, which is the node the only interesting chain runs through.

**Both changes were reverted.** They are not in the tree: the grammar cost
nothing but bought nothing, and writing both readings costs 40 more engine
calls per ingestion for a graph that answers identically. What is kept is this
section and the 61 cached readings beside it, so the next attempt starts from
the measurement instead of paying 128 calls to reach it.

### 7.2 The local engine now uses the hardware it is running on

Unrelated to the graph, and measured on the same machine: `runtime_gguf.py`
defaulted `n_gpu_layers` to 0, so anyone who did not know to export
`LMM_N_GPU_LAYERS` ran the local engine on the CPU. Same prompt, same 160-token
cap, same binary, back to back:

| | first call | second call |
|---|---|---|
| forced CPU (`LMM_N_GPU_LAYERS=0`) | 2.30 s | 1.80 s |
| accelerator detected (default now) | **0.68 s** | **0.62 s** |

The detection asks llama.cpp's own `llama_supports_gpu_offload` whether THIS
build was compiled with a GPU backend, so a CPU-only wheel on a GPU machine
correctly stays on the CPU; no platform string is parsed. `LMM_N_GPU_LAYERS`
still overrides in both directions. The 61-sentence `deep=True` ingestion above
now takes 63 s.

**This is a finding about extraction quality, not about the gate.** The two
guarantees this task actually targeted held up on the local engine exactly as
they did on Azure: `python3.11 tests/test_core.py` is model-free by
construction, and comparing the local engine's answers word-for-word against
the pre-change baseline (same corpus, same 17 questions) found **zero new
fabrications** — 14 of 17 answers are byte-identical, the 3 that differ are
ordinary local-sampling variance (a different garbled non-answer to the same
undecidable question), and the one true residual — *"vorlin sıcak içilir"*
answering *"at how many degrees does vorlin boil"* — is a real sentence from
the document answering the WRONG field (drinking temperature, not boiling
point), present **before this task's changes too**. It is the
answers-a-question-nobody-asked class `session._relation_held` exists for, not
something `lookup` or the sentence filter introduced or could fix — `lookup`
never engaged on this run, and the sentence filter only removes a MIXED
sentence, and this one is not mixed (it is a single, fully-grounded claim, just
aimed at the wrong slot).

**The honest conclusion:** the graph-first path is engine-independent *once the
graph exists* — nothing about `lookup.find` reads a token or asks a question of
any model. Whether the graph closes over a transitive relation at all is a
property of ingestion, and ingestion quality is exactly as engine-dependent as
every other measurement in this document. A deployment that wants the
zero-call floor on a local model needs either a better local extractor or
`deep=True` ingestion on a stronger engine even when serving answers locally
afterwards — the graph, once built, does not care which engine built it.


## 8. Two optimizations, measured on five documents

Sections 1–3 measure one corpus in two languages. This section measures **five
documents, ten questions each, three samples, median** — the short protocol,
run by [`benchmarks/field/measure.py`](field/measure.py). Three of the
documents are the invented corpora (a fixed 10 of their 17 questions, the same
10 every time); two are real documents off the public internet, fetched by
[`fetch_documents.py`](field/fetch_documents.py): a NIST PDF and a US Census
spreadsheet. **Two of the ten questions in each field set are not answerable
from the document**, so an abstention has something to be right about.

The question sets are committed (`benchmarks/field/questions_*.json`); the
documents, the graphs and the per-sample results are not.

**The ingestion is paid once and reused.** Every configuration and every sample
is loaded from the same saved graph, because ingestion variance is the largest
noise source in this system and has nothing to do with the answer path being
compared. What is re-run per sample is the answering.

Three arms, selected by the two controls the library carries for exactly this:

| arm | what it is |
|---|---|
| `base` | `LMM_DIRECT_LOOKUP=0` — no graph-first path at all |
| `words` | graph-first, naming a node ONE WORD AT A TIME (what shipped before) |
| `graph` | graph-first, naming a node by the question's contiguous word RUNS |

### 8.1 The graph-first path, and what naming a multi-word node buys

| document | arm | correct | abstain | **WRONG** | zero-call | calls/q | s/q |
|---|---|---|---|---|---|---|---|
| corpus TR (10) | base | 10 | 0 | **0** | 0 | 6.9 | 10.9 |
| corpus TR (10) | words | 10 | 0 | **0** | 3 | 4.4 | 7.0 |
| corpus TR (10) | **graph** | **10** | 0 | **0** | **3** | **4.4** | **6.8** |
| corpus EN (10) | base | 10 | 0 | **0** | 0 | 7.0 | 11.2 |
| corpus EN (10) | words | 10 | 0 | **0** | 3 | 4.5 | 6.7 |
| corpus EN (10) | **graph** | **10** | 0 | **0** | **3** | **4.5** | **6.6** |
| corpus ES (10) | base | 9 | 1 | **0** | 0 | 6.6 | 11.0 |
| corpus ES (10) | words | 9 | 1 | **0** | 3 | 4.3 | 6.4 |
| corpus ES (10) | **graph** | **9** | 1 | **0** | **3** | **4.3** | **6.8** |
| NIST SP 800-63B, pdf (10) | base | 10 | 0 | **0** | 0 | 8.0 | 13.6 |
| NIST SP 800-63B, pdf (10) | words | 10 | 0 | **0** | 0 | 8.0 | 12.3 |
| NIST SP 800-63B, pdf (10) | **graph** | **10** | 0 | **0** | **0** | **8.0** | **13.0** |
| US Census, xlsx (10) | base | 5 | 5 | **0** | 0 | 6.1 | 9.3 |
| US Census, xlsx (10) | words | 8 | 2 | **0** | 4 | 4.1 | 5.9 |
| US Census, xlsx (10) | **graph** | **9** | 1 | **0** | **6** | **2.7** | **4.2** |

**The WRONG column is zero in all fifteen cells.** That is the column the
decision rested on: a cheaper path that trades a fabrication for a saving is
not a cheaper path, and this one trades nothing.

**Read the spreadsheet row first, because it is where the whole gain is.** The
graph holds `united states -[population estimate (as of july 1)]-> 331526933`
as a row of a table it read with no model call at all, and the question
"united states population estimate as of july 1" says exactly that. The
one-word scan could not name the node: neither `united` nor `states` resolves
to anything, so the question went to the engine, which answered five of ten
correctly and abstained on the other five. Naming the node by the question's
own word RUNS settles six of the ten in microseconds, and the engine — still
carrying the other four — brings the document to **9 correct, 1 abstention, 0
wrong, 2.7 calls per question against the baseline's 6.1**. A spreadsheet is
the case this rule was built for, and a spreadsheet is where it pays.

**The three invented corpora do not move at all, and that is the honest
half.** Their subjects are single invented words — `zerbalit`, `vorlin` — so
there is no multi-word node to name and the runs have nothing to find. Both
graph arms sit at the same 3 of 10, the same figure §3 reports as 3 of 17. The
rule cost them nothing either: same answers, same abstentions, same zero wrong.

**The NIST PDF does not move either, and the reason is a limit rather than a
defect.** Its graph is the document's own structure — section names, front
matter fields, table rows — and its ten questions are prose questions
("how many characters shall a memorized secret be"), answered out of the
evidence index rather than the graph. All three arms answer 10 of 10 at 8.0
calls per question. The graph-first path declined every one of them, correctly:
nothing in that graph is the record those questions ask for.

**Across the fifty questions:** zero-call answers go 0 → 13 → **15**, calls per
question 6.9 → 5.1 → **4.8**, and correct answers 44 → 47 → **48**, with the
wrong count fixed at 0 throughout.

### 8.2 The answer cache: the same question, over a memory that has not moved

The second pass asks the identical ten questions against the identical memory.
Its cost is the measurement.

| document | pass | calls/q | s/q | answers identical |
|---|---|---|---|---|
| corpus TR (10) | 1 | 4.4 | 7.0 | 9/10 |
| corpus TR (10) | **2** | **0.9** | **1.3** | 9/10 |
| corpus EN (10) | 1 | 4.5 | 7.2 | 9/10 |
| corpus EN (10) | **2** | **0.9** | **1.3** | 9/10 |
| corpus ES (10) | 1 | 4.3 | 6.7 | 9/10 |
| corpus ES (10) | **2** | **0.9** | **1.3** | 9/10 |
| NIST SP 800-63B, pdf (10) | 1 | 8.0 | 12.9 | 10/10 |
| NIST SP 800-63B, pdf (10) | **2** | **0.0** | **0.0** | 10/10 |
| US Census, xlsx (10) | 1 | 2.7 | 3.7 | 10/10 |
| US Census, xlsx (10) | **2** | **0.0** | **0.0** | 10/10 |

**On the two field documents the second pass is free** — 0.0 calls, 0.0
seconds, and every answer byte-identical to the first.

**On the three corpora it is 0.9 calls per question rather than 0.0, and the
0.9 is a correctness decision rather than a miss.** Two of those ten questions
sit inside the research-offer flow: "melvarit nedir" names a subject the
document does not define, which makes the session OFFER to look it up, and the
turn after that offer means yes or no to it. A replayed answer would not
consume the offer, so it would still be standing when the next turn arrived and
that turn would be read as the approval. Both turns therefore bypass the cache
entirely — and the one answer of ten that differs between the passes is exactly
the regenerated refusal.

**What invalidates a kept answer is not a record count.** A count does not move
when a second source reinforces a fact, when `sleep()` fades one, or when a
contradiction lowers a rival's trust — and each of those changes what is spoken
or whether it is hedged. The stamp sums the trust and the source counts as well
as the counts, so every mutation the graph has moves it. The residue is stated
in `Memory._state` rather than hidden: two trust changes that cancel to the
same total would not move it, and nothing in the graph moves trust in pairs.

### 8.4 What a table loses on its way to an answer — three repairs, one kept

`benchmarks/field/REPORT.md` names three table defects. Each was built, each
was measured on its own, and ONE of the three is in the shipped path — which is
the only reason that one can be trusted.

**KEPT — the row's name (§4).** A spreadsheet cell has no margin, so a sheet
that nests rows draws the indentation with a character: the Census file writes
`.Alabama`, `.Puerto Rico`. Every nested row was unreachable by the name a
question uses — `about("puerto rico")` came back empty while
`about(".puerto rico")` held the whole row. The plain name is now registered as
an ALIAS of the row and the cell's own text stays the label, so nothing the
document wrote is edited (`-5` would lose its sign to the same rule). What
counts as layout is Unicode's category, asked once in `core/dataset.bare`.

**REVERTED — the cell's company (§3).** A flattened PDF table emits its cells
out of reading order, so the evidence window holding `Reauthentication` held
AAL2's `12 hours` and not AAL1's `30 days` — a wrong answer no gate can refuse,
because the claim really is in the evidence. `learn_rows` was given a second
evidence unit per CELL (`row · column: value`) so that a correctly bound unit
could compete with the mis-ordered line on both words instead of on the row's
alone. It does not work, and the report's own probe says so directly: *"According
to Table 4-1, what is the reauthentication requirement at AAL1?"* answers
`12 hours of inactivity` with the units and without them. 1260 units were
written into the NIST graph and not one carries the binding, because pdfplumber
never hands `learn_rows` a clean row for that table at all — the loss is
upstream of the index, in the rendering. The census evidence store grew from
12 KB to 22.5 KB for it and no column of the measurement moved.

**BUILT, OFF BY DEFAULT — the header block (§4).** `read_xlsx` picked one
header row where the Census sheet has two, so three columns had no name and
`row[h or ""]` collapsed them onto one dict key: 2021 and 2022 were overwritten
out of existence. The sheet says where its header ends in its own merges
(`A3:A4` — this name occupies both rows; `C3:F3` — this name covers four
columns), and the reader now reads them, inheriting a spanning name into the
columns under it. No row number is written down: `_header_row` still decides
where the header begins.

Measured, 5 documents × 10 questions × 3 samples, median, the shipped answer
path, one ingestion per arm:

| document | arm | correct | abstain | WRONG | zero-call | calls/q |
|---|---|---|---|---|---|---|
| census | before | 9 | 1 | 0 | 6 | 2.7 |
| census | **row name (shipped)** | **10** | 0 | **0** | **7** | **2.0** |
| census | row name + cell units | 10 | 0 | 0 | 7 | 2.0 |
| census | row name + cell units + header block | 6 | 2 | **2** | 3 | 4.4 |
| nist | before | 10 | 0 | 0 | 0 | 8.0 |
| nist | row name (shipped) | 10 | 0 | 0 | 0 | 8.0 |
| nist | row name + cell units | 10 | 0 | 0 | 0 | 8.0 |
| tr · en · es | before | 10 · 10 · 9 | 0 · 0 · 1 | 0 | 3 · 3 · 3 | 4.4 · 4.5 · 4.3 |
| tr · en · es | row name (shipped) | 10 · 10 · 9 | 0 · 0 · 1 | 0 | 3 · 3 · 3 | 4.4 · 4.5 · 4.3 |

**48/50 → 49/50 correct, zero-call 15/50 → 16/50, and zero wrong answers in
every arm.** The one question that moved is `puerto rico population estimate as
of july 1`, refused at every commit in this history, now answered with NO MODEL
CALL AT ALL — the row was always in the graph, wearing a dot. The three corpora
do not touch `learn_rows` and do not move.

**The header block's own row said why it was off.** It recovered two columns the
reader was destroying AND it cost two correct answers and bought two wrong ones,
and those were the same fact: the four recovered columns are all named
`Population Estimate (as of July 1) <year>`, so "population estimate as of july
1" — a question naming no year — went from one answer to four equally good
ones, and the path picked 2023 where the question set's gold is 2020. The gold
is not obviously right and the answer is not obviously wrong; what is certain
is that the choice became arbitrary, and an arbitrary choice among four is not
something to ship because a benchmark happens to prefer one of them.

**§8.4.1 below is the next turn of that measurement: the arbitrary choice was
the defect, it was fixed where it lived, and the block now ships ON.**

### 8.4.1 The silent choice, and the four arms that decided it

Two silences, one shape. This project had already called the first one a defect
and fixed it — a document the reader could not parse is now REPORTED rather
than dropped (`M2`) — and then shipped the second: four columns the sheet
distinguishes, one question that distinguishes none of them, and a path that
answered from one of the four without saying that the other three were equally
named. Swallowing a column silently and choosing among columns silently are the
same fault seen from two sides.

**The mechanism (`lmm/lookup.py`, shape 3).** The graph path already had the
right instinct — it declines whenever the question does not settle to ONE
record — and the fix generalises that condition instead of opening a route
past it. What must be unique is not the record, it is the QUESTION'S READING:
the records are grouped by which words of the question their field name
carries, the widest reading wins (the same judgement `_subjects` makes between
a long naming and a short one), and a reading that names several records is
answered with ALL of them, each under its own full field name. The
distinguishing word — `2020`, `2021` — is in that name because the SHEET put
it there, inherited from the merge `C3:F3` says is one cell. Nothing in the
code knows what a year is. Ordering matters and is measured: the widest
reading is chosen BEFORE the several-records test, so "the 2021 population
estimate" is a single half-named field, which this path does not answer at all
— it falls through to the semantic path rather than being answered with the
three columns the question ruled out.

**Four questions were added to the census set first**, because the old ten
never asked for a column the single-row reader destroys — a set that cannot
see the loss cannot judge the repair. Three name the column the way the sheet
names it (`… as of july 1 2022`, `… 2023`, `… 2021`); one names the year
loosely ("what is the 2021 population estimate for the united states"), which
is the phrasing the graph path deliberately declines. The census set is
**14 questions** now, so the field total is 54 and every number below is
recomputed rather than compared against the old 50.

Same protocol — 5 documents, 3 samples, median, one ingestion per arm, the
shipped answer path:

| arm | reading | answer path | census correct | census WRONG | census zero-call | all five correct | all five WRONG |
|---|---|---|---|---|---|---|---|
| a | single row | choose silently | 10/14 | **3** | 10 | 49/54 | **3** |
| b | header block | choose silently | 10/14 | **2** | 6 | — | — |
| c | header block | **say the candidates** | **14/14** | **0** | **10** | **53/54** | **0** |
| d | single row | say the candidates | 10/14 | **3** | 10 | — | — |

Arms **b** and **d** are census-only: the header block changes ONE of the five
documents and leaves the other four graphs byte-identical, so re-asking them
would measure the same graph twice. Arms **a** and **c** are full, and their
tr · en · es · nist rows are identical to the row — 10 · 10 · 9 · 10 correct,
0 wrong, 3 · 3 · 3 · 0 zero-call — which is the other half of the result: the
mechanism does not touch a document whose columns are not shared-named.

**What each arm says.**

- **(a) is the previous HEAD, and the new questions show what it was hiding.**
  Three of its four year-named questions are answered WRONG **with no model
  call and full confidence** — "population estimate as of july 1 2022 for the
  west region" is answered `78661381`, which is the 2020 number, because 2022
  never survived ingestion and the field's name does not carry the year that
  would have ruled it out. This is the cost of the single-row reading, and it
  had never appeared in a score before because no question asked for it.
- **(d) is (a) exactly.** The mechanism alone changes nothing: with the columns
  already collapsed there is no group of several to speak, and the three wrong
  answers stay wrong. The repair is not in the answer path alone.
- **(b) is the block alone, and it is the arbitrary choice caught in the act.**
  The engine answers "united states population estimate as of july 1" with the
  2023 figure and "south population estimate as of july 1" with the 2023
  figure, each in a fluent sentence, each costing 6–7 calls; two more of the
  same shape abstain. Four right answers, one picked, nothing said about it.
- **(c) is both, and it is the arm that ships.** Every year-named question is
  answered exactly and free (`west — population estimate (as of july 1) 2022 →
  78759506`, 0 calls); every year-less question is answered with all four
  columns and their names, also free; the loosely-phrased one still goes to the
  engine and comes back right at 7 calls. Census zero-call rises 6 → 10 against
  arm (b) and the calls per question fall 3.5 → 1.9, because a question this
  path can read is a question nobody pays for.

**Read the 14/14 honestly.** The scorer marks an answer correct when it carries
the gold string, so a four-line answer that includes the gold value scores the
same as a one-line one. What the measurement establishes is therefore NOT that
four lines are better prose — it is that **no arm produces a wrong answer** and
that the three questions arm (a) got wrong are now right, while the year-less
questions stop depending on which of four columns a model happened to pick. The
alternative shape — asking the reader WHICH year they meant — was rejected
before it was built: it cannot answer a question the document answers four
times over, and this system's own rule is that a refusal is for what is not in
the document.

**Decision.** Both default ON: `LMM_XLSX_HEADER_BLOCK=0` restores the
single-row reading and `LMM_LOOKUP_CANDIDATES=0` restores the silent decline,
and each is one variable, so any of the four rows above can be reproduced.
Field baseline, recomputed: **53/54 correct · 0 wrong · 19/54 zero-call**.

**The third question the report asked answered itself, and was measured anyway.**
Whether a document triple may stand in the answer block beside the evidence: the
rule that drops them tests the `#doc:` stamp, and only `Memory.learn(text)`
writes that stamp — a table adapter stamps `#xlsx:`/`#pdf:`, so a cell written
by `learn_rows` was never excluded by that rule. It only ever reached prose
extraction, which is what it was built for. Lifting it entirely was then run
over the three corpora, which are the only sets carrying the stamp:

| corpus | rule as shipped | rule lifted |
|---|---|---|
| tr | 10 correct · 0 wrong · 3 zero-call · 4.4 calls/q | 10 · 0 · 3 · 4.4 |
| en | 10 correct · 0 wrong · 3 zero-call · 4.5 calls/q | 10 · 0 · 3 · 4.5 |
| es | 9 correct · 0 wrong · 3 zero-call · 4.3 calls/q | 9 · 0 · 3 · 4.3 |

Not one column moves, because with evidence present the evidence is what the
answer is built from either way. The switch that produced this table is not in
the tree — a knob with no measured effect is not a finding, it is a knob — and
what is kept is the comment in `_answer` saying what the rule reaches.

### 8.3 What this does not say

It does not say the architecture became cheaper than embedding RAG. §6's
sentence stands: on a hosted per-token engine, RAG is cheaper for one-hop
lookup and there is no break-even. What moved is the FLOOR — the share of
questions that cost nothing at all — and on a document whose facts are a table,
that share is now most of them.
