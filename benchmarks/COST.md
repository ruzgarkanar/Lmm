# Cost — measured, not estimated

Every token below is the `usage` field the server itself returned, collected by [`meter.py`](meter.py), which wraps the OpenAI chat SDK and the local embedding call and touches no library code. Every figure is the **median of 3 repeats** of the same configuration.

Both sides run the **same engine**, Azure `gpt-4o-mini`, so the comparison is between architectures rather than model sizes — the same discipline the accuracy table in the README uses.

Corpora are the two reproducible fictional ones in this repository (`corpus.txt` + `questions.json`, `corpus_en.txt` + `questions_en.json`) — no third-party document is involved, so anyone can re-run this. Measured 2026-08-18.

Reproduce:

```bash
sh benchmarks/cost_all.sh          # 3 sides x 2 languages x 3 repeats
python3.11 benchmarks/cost_report.py > benchmarks/COST.md
```

## 1. Ingestion cost (paid once per document)

**EN corpus** — 1,642 characters

| | model calls | prompt tok | completion tok | embedding calls | wall |
|---|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 0 | 0 | 0 | 1 **local, 0 API tokens** (0.2 s CPU) | 7.7 s |
| LMM `deep=False` (evidence-only) | 0 | 0 | 0 | 0 — none | 0.0 s |
| LMM `deep=True` (graph extraction) | 121 | 43,529 | 1,057 | 0 — none | 24.4 s |

**TR corpus** — 1,529 characters

| | model calls | prompt tok | completion tok | embedding calls | wall |
|---|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 0 | 0 | 0 | 1 **local, 0 API tokens** (0.2 s CPU) | 7.4 s |
| LMM `deep=False` (evidence-only) | 0 | 0 | 0 | 0 — none | 0.0 s |
| LMM `deep=True` (graph extraction) | 118 | 42,449 | 1,174 | 0 — none | 22.7 s |

**The embedding zero is real and it matters.** RAG's ingestion here spends no API tokens at all, because the embedding model is a local `sentence-transformers` checkpoint. What it spends instead is CPU seconds and roughly half a gigabyte of dependencies (section 4). Had a hosted embedding API been used instead, that column would carry a real token bill; this setup deliberately gives RAG the cheaper option.

**LMM `deep=False` ingests for free** — literally zero model calls, zero tokens, 6 ms. Building the evidence index is pure python: no model, no network, no embedding. This is the cheapest ingestion in the table by a wide margin, and it is the mode meant for a large document.

**`deep=True` is where LMM's ingestion cost lives** — 121 calls and 44,586 tokens to read this small corpus into a graph, because every candidate fact is extracted and then re-read by the gate before it is admitted. That is the provenance and the refusal guarantee being paid for up front. It buys 62 admitted facts and the 8 further ones the graph *derives* symbolically — no model call, microseconds — which is what makes multi-hop answers possible. It is paid once per document rather than once per question.

For scale: that corpus is ~1.5 KB. Ingestion cost on this path grows with the document, so a 350-page manual is not a `deep=True` job — which is exactly why `deep=False` exists.

## 2. Cost per question

**EN**

| | calls/q | prompt tok/q | completion tok/q | wall/q |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 1.0 | 512 | 11 | 1.6 s |
| LMM `deep=False` (evidence-only) | 7.6 | 5,141 | 59 | 10.6 s |
| LMM `deep=True` (graph extraction) | 5.3 | 3,839 | 46 | 7.0 s |

**TR**

| | calls/q | prompt tok/q | completion tok/q | wall/q |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 1.0 | 593 | 9 | 1.6 s |
| LMM `deep=False` (evidence-only) | 7.8 | 5,400 | 65 | 11.1 s |
| LMM `deep=True` (graph extraction) | 5.5 | 3,944 | 48 | 7.5 s |

### Where LMM's calls go

Calls per question, by kind. This is the breakdown that says where to attack the cost, and it is not flattering: the verification read-back — the gate re-extracting the claims out of a sentence before it is allowed to leave — is a large share of the bill. That is fabrication-0 being paid for in tokens.

| | extract calls/q | answer calls/q | read-back calls/q | relation-check calls/q | prompt tok/q (largest bucket) |
|---|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) (en) | 0.0 | 1.0 | 0.0 | 0.0 | answer: 512 |
| LMM `deep=False` (evidence-only) (en) | 1.0 | 2.9 | 2.8 | 1.0 | answer: 2,249 |
| LMM `deep=True` (graph extraction) (en) | 0.8 | 2.2 | 2.1 | 0.1 | answer: 1,748 |
| RAG (embed + Chroma + 4o-mini) (tr) | 0.0 | 1.0 | 0.0 | 0.0 | answer: 593 |
| LMM `deep=False` (evidence-only) (tr) | 1.0 | 2.9 | 2.9 | 1.0 | read-back: 2,318 |
| LMM `deep=True` (graph extraction) (tr) | 0.8 | 2.2 | 2.1 | 0.3 | answer: 1,784 |

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
| **LMM** — `pip install lmm` | **1** (`dependencies = []` in pyproject.toml) | **1.1 MB** | none — the graph, gate, trust ordering and derivation are pure python |
| **RAG** — `pip install langchain-chroma langchain-openai langchain-text-splitters sentence-transformers` | **114** | **1.35 GB** | **458 MB** embedding checkpoint, fetched at first use |

Both venvs also carry the same ~21 MB of `pip`/`setuptools`, excluded from both rows. The RAG figure is dominated by `torch` (534 MB), `transformers` (110 MB), `scipy` (100 MB), `onnxruntime` (81 MB) and — via Chroma — `kubernetes` (84 MB). That is roughly **1,200x the disk and 114x the packages**, before a single question is asked. To be fair to RAG: this weight is what buys it the zero-API-token ingestion in section 1, and an LMM deployment that runs its own local engine pulls in a comparable stack (`pip install lmm[local]`). The 1.1 MB figure is the *core* — the part that stores, derives and gates, which is the part that has no dependencies.

## 5. Price, as a separate layer

The tables above are token counts, which are true regardless of what anyone charges. Below is one multiplication applied to them: published gpt-4o-mini rate used for this table: $0.15 / 1M input tokens, $0.60 / 1M output tokens. **Prices change; re-multiply rather than trust this table.** Run the same architecture on a local GGUF engine (`LMM_BACKEND=gguf`) and every dollar figure here becomes **exactly $0** — replaced by seconds of your own CPU/GPU, which is the trade this project is built around.

**EN — total spend for N questions over one ingested document**

| | ingestion (once) | per question | 1,000 questions | 100,000 questions |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | $0.0000 | $0.0001 | $0.0834 | $8.34 |
| LMM `deep=False` (evidence-only) | $0.0000 | $0.0008 | $0.8065 | $80.66 |
| LMM `deep=True` (graph extraction) | $0.0072 | $0.0006 | $0.6107 | $60.36 |

**TR — total spend for N questions over one ingested document**

| | ingestion (once) | per question | 1,000 questions | 100,000 questions |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | $0.0000 | $0.0001 | $0.0945 | $9.45 |
| LMM `deep=False` (evidence-only) | $0.0000 | $0.0008 | $0.8487 | $84.87 |
| LMM `deep=True` (graph extraction) | $0.0071 | $0.0006 | $0.6277 | $62.07 |

### Break-even

- **LMM `deep=False` (evidence-only) vs RAG (en)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.
- **LMM `deep=True` (graph extraction) vs RAG (en)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.
- **LMM `deep=False` (evidence-only) vs RAG (tr)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.
- **LMM `deep=True` (graph extraction) vs RAG (tr)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.

The one crossing that exists is *inside* LMM:

- **`deep=False` vs `deep=True` (en)** — `deep=False` ingests free but asks dearer; `deep=True` overtakes it at **~35 questions** on one document. Below that, shallow is the cheaper LMM; above it, the graph has repaid its own extraction.
- **`deep=False` vs `deep=True` (tr)** — `deep=False` ingests free but asks dearer; `deep=True` overtakes it at **~31 questions** on one document. Below that, shallow is the cheaper LMM; above it, the graph has repaid its own extraction.

**Treat those two numbers as an order of magnitude, not a threshold.** They divide a fixed ingestion cost by a small per-question difference, so the noise in the per-question figure is amplified — which is exactly why the two languages disagree by several-fold on a corpus that is otherwise line-for-line identical. What is solid is the shape: shallow wins for a handful of questions, deep wins once you are asking hundreds.

## 6. Honest summary

**Where we are more expensive: everywhere that is measured in tokens.** On the EN corpus LMM `deep=True` spends **5.3 model calls per question** against RAG's 1.0 — about **5x the calls and 7x the prompt tokens**. Ingestion is worse in relative terms: 44,586 tokens against RAG's 0, because RAG's embedding step is a local model and spends none at all. There is no reading of these numbers in which LMM is the cheap option on a hosted per-token engine, and no break-even where that reverses — the gap grows with every question asked.

**We are also slower per question**: 7.0 s against RAG's 1.6 s, because six or seven sequential calls cannot beat one. On the same rate-limited deployment, measured back to back.

**Which LMM mode is cheaper is a question of how many questions you will ask.** `deep=False` ingests for nothing but asks dearer (7.6 calls/q vs 5.3); `deep=True` pays 44,586 tokens up front and then asks cheaper, because the graph answers more directly. They cross somewhere in the **low hundreds of questions** on one document (~35 here, but see the caveat above — the two languages disagree several-fold). `deep=False` is also the only workable mode for a large document, since extraction cost scales with the text while the evidence index does not.

**Some questions now cost nothing at all — on the engine that ingested them well.** Section 3 counts 6 answers across these Azure runs that never reached the engine — the graph settled them itself, in microseconds, for zero tokens and zero seconds of anyone's GPU. It is a minority of the questions and it does not move the per-question average much; what it moves is the FLOOR. **That floor is not free of the engine that built it** — see §7, where the same corpus on a local 3B model never reaches it at all, because the derivation the floor rests on never closes there.

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

