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
| LMM `deep=True` (graph extraction) | 121 | 43,529 | 1,058 | 0 — none | 23.3 s |

**TR corpus** — 1,529 characters

| | model calls | prompt tok | completion tok | embedding calls | wall |
|---|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 0 | 0 | 0 | 1 **local, 0 API tokens** (0.2 s CPU) | 7.4 s |
| LMM `deep=False` (evidence-only) | 0 | 0 | 0 | 0 — none | 0.0 s |
| LMM `deep=True` (graph extraction) | 118 | 42,449 | 1,176 | 0 — none | 21.8 s |

**The embedding zero is real and it matters.** RAG's ingestion here spends no API tokens at all, because the embedding model is a local `sentence-transformers` checkpoint. What it spends instead is CPU seconds and roughly half a gigabyte of dependencies (section 4). Had a hosted embedding API been used instead, that column would carry a real token bill; this setup deliberately gives RAG the cheaper option.

**LMM `deep=False` ingests for free** — literally zero model calls, zero tokens, 13 ms. Building the evidence index is pure python: no model, no network, no embedding. This is the cheapest ingestion in the table by a wide margin, and it is the mode meant for a large document.

**`deep=True` is where LMM's ingestion cost lives** — 121 calls and 44,587 tokens to read this small corpus into a graph, because every candidate fact is extracted and then re-read by the gate before it is admitted. That is the provenance and the refusal guarantee being paid for up front. It buys 62 admitted facts and the 8 further ones the graph *derives* symbolically — no model call, microseconds — which is what makes multi-hop answers possible. It is paid once per document rather than once per question.

For scale: that corpus is ~1.5 KB. Ingestion cost on this path grows with the document, so a 350-page manual is not a `deep=True` job — which is exactly why `deep=False` exists.

## 2. Cost per question

**EN**

| | calls/q | prompt tok/q | completion tok/q | wall/q |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 1.0 | 512 | 11 | 1.6 s |
| LMM `deep=False` (evidence-only) | 7.6 | 5,059 | 57 | 10.7 s |
| LMM `deep=True` (graph extraction) | 6.7 | 4,940 | 57 | 9.1 s |

**TR**

| | calls/q | prompt tok/q | completion tok/q | wall/q |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | 1.0 | 593 | 9 | 1.6 s |
| LMM `deep=False` (evidence-only) | 7.8 | 5,400 | 65 | 11.6 s |
| LMM `deep=True` (graph extraction) | 6.6 | 4,790 | 59 | 9.2 s |

### Where LMM's calls go

Calls per question, by kind. This is the breakdown that says where to attack the cost, and it is not flattering: the verification read-back — the gate re-extracting the claims out of a sentence before it is allowed to leave — is a large share of the bill. That is fabrication-0 being paid for in tokens.

| | extract calls/q | answer calls/q | read-back calls/q | relation-check calls/q | prompt tok/q (largest bucket) |
|---|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) (en) | 0.0 | 1.0 | 0.0 | 0.0 | answer: 512 |
| LMM `deep=False` (evidence-only) (en) | 1.0 | 2.9 | 2.7 | 1.0 | answer: 2,249 |
| LMM `deep=True` (graph extraction) (en) | 1.0 | 2.8 | 2.8 | 0.1 | read-back: 2,162 |
| RAG (embed + Chroma + 4o-mini) (tr) | 0.0 | 1.0 | 0.0 | 0.0 | answer: 593 |
| LMM `deep=False` (evidence-only) (tr) | 1.0 | 2.9 | 2.9 | 1.0 | read-back: 2,319 |
| LMM `deep=True` (graph extraction) (tr) | 1.0 | 2.6 | 2.6 | 0.4 | answer: 2,118 |

## 3. Zero-call answers

How many questions LMM answered with **no model call at all** — pure graph lookup, no engine involved. The architecture permits it: a derived fact or a table row is already an answer. Whether it *happens* on this corpus is a measurement, and the honest answer is below.

| | questions | answered with 0 model calls | share |
|---|---|---|---|
| LMM `deep=False` (evidence-only) (en) | 17 | 0 | 0% |
| LMM `deep=True` (graph extraction) (en) | 17 | 0 | 0% |
| LMM `deep=False` (evidence-only) (tr) | 17 | 0 | 0% |
| LMM `deep=True` (graph extraction) (tr) | 17 | 0 | 0% |

**Zero. Not one.** Every question on this corpus went through the engine. The graph decides *what* may be said and the derivation runs in microseconds without a model — but `respond` still spends a call turning the retrieved facts into a sentence, and more calls reading that sentence back. The zero-call path exists in the architecture and is not reached here; claiming otherwise would be the easiest number in this document to fake.

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
| LMM `deep=False` (evidence-only) | $0.0000 | $0.0008 | $0.7929 | $79.29 |
| LMM `deep=True` (graph extraction) | $0.0072 | $0.0008 | $0.7825 | $77.54 |

**TR — total spend for N questions over one ingested document**

| | ingestion (once) | per question | 1,000 questions | 100,000 questions |
|---|---|---|---|---|
| RAG (embed + Chroma + 4o-mini) | $0.0000 | $0.0001 | $0.0945 | $9.45 |
| LMM `deep=False` (evidence-only) | $0.0000 | $0.0008 | $0.8492 | $84.92 |
| LMM `deep=True` (graph extraction) | $0.0071 | $0.0008 | $0.7610 | $75.40 |

### Break-even

- **LMM `deep=False` (evidence-only) vs RAG (en)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.
- **LMM `deep=True` (graph extraction) vs RAG (en)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.
- **LMM `deep=False` (evidence-only) vs RAG (tr)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.
- **LMM `deep=True` (graph extraction) vs RAG (tr)** — dearer on both axes; **there is no break-even.** RAG is cheaper at every N, and the gap widens.

The one crossing that exists is *inside* LMM:

- **`deep=False` vs `deep=True` (en)** — `deep=False` ingests free but asks dearer; `deep=True` overtakes it at **~407 questions** on one document. Below that, shallow is the cheaper LMM; above it, the graph has repaid its own extraction.
- **`deep=False` vs `deep=True` (tr)** — `deep=False` ingests free but asks dearer; `deep=True` overtakes it at **~74 questions** on one document. Below that, shallow is the cheaper LMM; above it, the graph has repaid its own extraction.

**Treat those two numbers as an order of magnitude, not a threshold.** They divide a fixed ingestion cost by a small per-question difference, so the noise in the per-question figure is amplified — which is exactly why the two languages disagree by several-fold on a corpus that is otherwise line-for-line identical. What is solid is the shape: shallow wins for a handful of questions, deep wins once you are asking hundreds.

## 6. Honest summary

**Where we are more expensive: everywhere that is measured in tokens.** On the EN corpus LMM `deep=True` spends **6.7 model calls per question** against RAG's 1.0 — about **7x the calls and 10x the prompt tokens**. Ingestion is worse in relative terms: 44,587 tokens against RAG's 0, because RAG's embedding step is a local model and spends none at all. There is no reading of these numbers in which LMM is the cheap option on a hosted per-token engine, and no break-even where that reverses — the gap grows with every question asked.

**We are also slower per question**: 9.1 s against RAG's 1.6 s, because six or seven sequential calls cannot beat one. On the same rate-limited deployment, measured back to back.

**Which LMM mode is cheaper is a question of how many questions you will ask.** `deep=False` ingests for nothing but asks dearer (7.6 calls/q vs 6.7); `deep=True` pays 44,587 tokens up front and then asks cheaper, because the graph answers more directly. They cross somewhere in the **low hundreds of questions** on one document (~407 here, but see the caveat above — the two languages disagree several-fold). `deep=False` is also the only workable mode for a large document, since extraction cost scales with the text while the evidence index does not.

**Where we are cheaper: the axes this table cannot bill.** The tokens above buy three things RAG does not have at any price — every answer carrying its source, a structural gate that stops an unsupported claim from leaving, and multi-hop facts *derived* symbolically in microseconds with no model call. Section 4's disk figures are the other axis: a zero-dependency core against an embedding stack and a vector database. And the whole dollar column collapses to zero on a local engine, where the cost becomes your own seconds — which is the deployment this project is actually built for.

**So the fair sentence is this:** if you are paying per token for a hosted model and you only need one-hop lookup, embedding RAG is cheaper than LMM and will stay cheaper. LMM's case is accuracy, provenance and refusal (see the README's benchmark table), bought with tokens at ingestion and at verification — or bought with CPU seconds instead, on hardware you already own.

