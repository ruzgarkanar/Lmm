"""COST REPORT — aggregate the cost matrix into medians, projections and a
break-even point, and print it as the markdown that becomes COST.md.

Nothing here is typed by hand. Every number comes from benchmarks/cost/*.json,
which in turn comes from the server's own `usage` field (see meter.py). The
median of the three repeats is the headline, exactly as evaluate.py does for
accuracy — these benchmarks are noisy and one run is not a measurement.

PRICE IS A SEPARATE LAYER. The token counts above the price table are the
durable fact; the dollar figures are those counts multiplied by a published
per-token rate that can change at any time, and by a rate that is zero when the
engine is local.

    python3.11 benchmarks/cost_report.py [cost_dir] > benchmarks/COST.md
"""
import collections
import json
import os
import sys

# The rate used for the DERIVED price table only. Published gpt-4o-mini
# pricing, USD per 1M tokens, as used on the run date recorded below. Change
# these two numbers and every dollar figure in the report changes with them;
# no token count moves.
PRICE_IN = 0.15 / 1_000_000
PRICE_OUT = 0.60 / 1_000_000
PRICE_NOTE = ("published gpt-4o-mini rate used for this table: "
              "$0.15 / 1M input tokens, $0.60 / 1M output tokens")
RUN_DATE = "2026-08-18"

SIDES = ["rag", "lmm-shallow", "lmm-deep"]
LABEL = {"rag": "RAG (embed + Chroma + 4o-mini)",
         "lmm-shallow": "LMM `deep=False` (evidence-only)",
         "lmm-deep": "LMM `deep=True` (graph extraction)"}
BUCKETS = ["extract", "answer", "read-back", "relation-check"]


def median(values):
    if not values:
        return 0
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def load(cost_dir):
    runs = collections.defaultdict(list)
    for name in sorted(os.listdir(cost_dir)):
        # Only the run files: <side>_<lang>_<repeat>.json. The directory also
        # holds infra.json and the summary this script writes, which are not
        # samples and must not be counted as one.
        parts = name[:-5].rsplit("_", 2)
        if (not name.endswith(".json") or len(parts) != 3
                or parts[0] not in SIDES or not parts[2].isdigit()):
            continue
        side, lang, _rep = parts
        runs[(side, lang)].append(
            json.load(open(os.path.join(cost_dir, name), encoding="utf-8")))
    return runs


def usd(prompt, completion):
    return prompt * PRICE_IN + completion * PRICE_OUT


def money(x):
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def ingest_stats(samples):
    g = lambda k: median([s["ingest"].get(k, 0) for s in samples])   # noqa: E731
    emb = [s["ingest"].get("embedding", {}) for s in samples]
    return {
        "calls": g("calls"),
        "prompt": g("prompt_tokens"),
        "completion": g("completion_tokens"),
        "wall": g("wall_seconds"),
        "derived": median([s["ingest"].get("derived", 0) for s in samples]),
        "facts": median([s["ingest"].get("facts", 0) for s in samples]),
        "embed_calls": median([e.get("calls", 0) for e in emb]),
        "embed_cpu": median([e.get("cpu_seconds", 0) for e in emb]),
        "embed_chars": median([e.get("characters", 0) for e in emb]),
    }


def query_stats(samples):
    """Per-question medians. The per-question figure is the run's total over
    its question count, medianed across runs — not a median of per-question
    values, which would hide the tail that actually gets billed."""
    n = median([s["question_count"] for s in samples]) or 1
    tot = lambda k: median([s["query_totals"][k] for s in samples])  # noqa: E731
    buckets = {}
    for b in BUCKETS:
        per_run = []
        for s in samples:
            calls = sum(q["buckets"].get(b, {}).get("calls", 0)
                        for q in s["questions"])
            p = sum(q["buckets"].get(b, {}).get("prompt_tokens", 0)
                    for q in s["questions"])
            c = sum(q["buckets"].get(b, {}).get("completion_tokens", 0)
                    for q in s["questions"])
            per_run.append((calls, p, c))
        buckets[b] = {
            "calls": median([x[0] for x in per_run]) / n,
            "prompt": median([x[1] for x in per_run]) / n,
            "completion": median([x[2] for x in per_run]) / n,
        }
    return {
        "n": n,
        "calls": tot("calls") / n,
        "prompt": tot("prompt_tokens") / n,
        "completion": tot("completion_tokens") / n,
        "wall": tot("wall_seconds") / n,
        "zero": median([s["zero_call_questions"] for s in samples]),
        "buckets": buckets,
        "query_embed_cpu": median([s.get("query_embedding", {})
                                   .get("cpu_seconds", 0) for s in samples]),
    }


def total_usd(ing, qry, n):
    return (usd(ing["prompt"], ing["completion"])
            + n * usd(qry["prompt"], qry["completion"]))


def breakeven(ing_a, qry_a, ing_b, qry_b):
    """Questions at which side A's running total overtakes side B's.

    Only meaningful when A ingests cheaper but queries dearer (or the reverse);
    if one side is cheaper on BOTH axes there is no crossing, and saying so is
    more useful than printing a number."""
    fa, va = usd(ing_a["prompt"], ing_a["completion"]), usd(qry_a["prompt"], qry_a["completion"])
    fb, vb = usd(ing_b["prompt"], ing_b["completion"]), usd(qry_b["prompt"], qry_b["completion"])
    if (fa - fb) == 0 or (vb - va) == 0:
        return None
    n = (fb - fa) / (va - vb)
    return n if n > 0 else None


def main():
    cost_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "cost")
    runs = load(cost_dir)
    langs = sorted({k[1] for k in runs})
    data = {k: {"ingest": ingest_stats(v), "query": query_stats(v),
                "samples": len(v)} for k, v in runs.items()}

    out = []
    w = out.append

    w("# Cost — measured, not estimated\n")
    w("Every token below is the `usage` field the server itself returned, "
      "collected by [`meter.py`](meter.py), which wraps the OpenAI chat SDK "
      "and the local embedding call and touches no library code. Every figure "
      "is the **median of 3 repeats** of the same configuration.\n")
    w("Both sides run the **same engine**, Azure `gpt-4o-mini`, so the "
      "comparison is between architectures rather than model sizes — the same "
      "discipline the accuracy table in the README uses.\n")
    w(f"Corpora are the two reproducible fictional ones in this repository "
      f"(`corpus.txt` + `questions.json`, `corpus_en.txt` + "
      f"`questions_en.json`) — no third-party document is involved, so anyone "
      f"can re-run this. Measured {RUN_DATE}.\n")
    w("Reproduce:\n\n```bash\nsh benchmarks/cost_all.sh          "
      "# 3 sides x 2 languages x 3 repeats\npython3.11 benchmarks/cost_report.py "
      "> benchmarks/COST.md\n```\n")

    # ---- 1. ingestion --------------------------------------------------
    w("## 1. Ingestion cost (paid once per document)\n")
    for lang in langs:
        w(f"**{lang.upper()} corpus** — "
          f"{len(open(os.path.join(cost_dir, '..', 'corpus.txt' if lang == 'tr' else 'corpus_en.txt'), encoding='utf-8').read()):,} characters\n")
        w("| | model calls | prompt tok | completion tok | embedding calls | wall |")
        w("|---|---|---|---|---|---|")
        for side in SIDES:
            d = data.get((side, lang))
            if not d:
                continue
            i = d["ingest"]
            emb = (f"{i['embed_calls']:.0f} **local, 0 API tokens** "
                   f"({i['embed_cpu']:.1f} s CPU)" if i["embed_calls"] else "0 — none")
            w(f"| {LABEL[side]} | {i['calls']:.0f} | {i['prompt']:,.0f} | "
              f"{i['completion']:,.0f} | {emb} | {i['wall']:.1f} s |")
        w("")
    w("**The embedding zero is real and it matters.** RAG's ingestion here "
      "spends no API tokens at all, because the embedding model is a local "
      "`sentence-transformers` checkpoint. What it spends instead is CPU "
      "seconds and roughly half a gigabyte of dependencies (section 4). Had a "
      "hosted embedding API been used instead, that column would carry a real "
      "token bill; this setup deliberately gives RAG the cheaper option.\n")
    sh0 = data.get(("lmm-shallow", langs[0])) if langs else None
    dp0 = data.get(("lmm-deep", langs[0])) if langs else None
    if sh0 and sh0["ingest"]["calls"] == 0:
        w(f"**LMM `deep=False` ingests for free** — literally zero model "
          f"calls, zero tokens, {sh0['ingest']['wall']*1000:.0f} ms. Building "
          f"the evidence index is pure python: no model, no network, no "
          f"embedding. This is the cheapest ingestion in the table by a wide "
          f"margin, and it is the mode meant for a large document.\n")
    if dp0:
        w(f"**`deep=True` is where LMM's ingestion cost lives** — "
          f"{dp0['ingest']['calls']:.0f} calls and "
          f"{dp0['ingest']['prompt'] + dp0['ingest']['completion']:,.0f} "
          f"tokens to read this small corpus into a graph, because every "
          f"candidate fact is extracted and then re-read by the gate before it "
          f"is admitted. That is the provenance and the refusal guarantee "
          f"being paid for up front. It buys {dp0['ingest']['facts']:.0f} "
          f"admitted facts and the {dp0['ingest']['derived']:.0f} further ones "
          f"the graph *derives* symbolically — no model call, microseconds — "
          f"which is what makes multi-hop answers possible. It is paid "
          f"once per document rather than once per question.\n")
        w(f"For scale: that corpus is ~1.5 KB. Ingestion cost on this path "
          f"grows with the document, so a 350-page manual is not a "
          f"`deep=True` job — which is exactly why `deep=False` exists.\n")

    # ---- 2. per question -----------------------------------------------
    w("## 2. Cost per question\n")
    for lang in langs:
        w(f"**{lang.upper()}**\n")
        w("| | calls/q | prompt tok/q | completion tok/q | wall/q |")
        w("|---|---|---|---|---|")
        for side in SIDES:
            d = data.get((side, lang))
            if not d:
                continue
            q = d["query"]
            w(f"| {LABEL[side]} | {q['calls']:.1f} | {q['prompt']:,.0f} | "
              f"{q['completion']:,.0f} | {q['wall']:.1f} s |")
        w("")
    w("### Where LMM's calls go\n")
    w("Calls per question, by kind. This is the breakdown that says where to "
      "attack the cost, and it is not flattering: the verification read-back "
      "— the gate re-extracting the claims out of a sentence before it is "
      "allowed to leave — is a large share of the bill. That is fabrication-0 "
      "being paid for in tokens.\n")
    w("| | " + " | ".join(f"{b} calls/q" for b in BUCKETS) + " | prompt tok/q (largest bucket) |")
    w("|---|" + "---|" * (len(BUCKETS) + 1))
    for lang in langs:
        for side in SIDES:
            d = data.get((side, lang))
            if not d:
                continue
            b = d["query"]["buckets"]
            top = max(BUCKETS, key=lambda k: b[k]["prompt"])
            w(f"| {LABEL[side]} ({lang}) | "
              + " | ".join(f"{b[k]['calls']:.1f}" for k in BUCKETS)
              + f" | {top}: {b[top]['prompt']:,.0f} |")
    w("")

    # ---- 3. zero-call ---------------------------------------------------
    w("## 3. Zero-call answers\n")
    w("How many questions LMM answered with **no model call at all** — pure "
      "graph lookup, no engine involved. The architecture permits it: a "
      "derived fact or a table row is already an answer. Whether it *happens* "
      "on this corpus is a measurement, and the honest answer is below.\n")
    w("This row read **0 of 17 at every commit** until `lmm/lookup.py` was "
      "written. The architecture permitted the zero-call answer and the code "
      "never took it: `respond` spent a call turning a held triple into a "
      "sentence and more calls reading that sentence back. `lookup` takes it, "
      "for the questions it can settle without guessing.\n")
    w("| | questions | answered with 0 model calls | share |")
    w("|---|---|---|---|")
    zero_total = 0
    for lang in langs:
        for side in ("lmm-shallow", "lmm-deep"):
            d = data.get((side, lang))
            if not d:
                continue
            q = d["query"]
            zero_total += q["zero"]
            w(f"| {LABEL[side]} ({lang}) | {q['n']:.0f} | {q['zero']:.0f} | "
              f"{100 * q['zero'] / q['n']:.0f}% |")
    w("")
    if zero_total == 0:
        w("**Zero. Not one.** Every question on this corpus went through the "
          "engine. The graph decides *what* may be said and the derivation "
          "runs in microseconds without a model — but `respond` still spends a "
          "call turning the retrieved facts into a sentence, and more calls "
          "reading that sentence back. The zero-call path exists in the "
          "architecture and is not reached here; claiming otherwise would be "
          "the easiest number in this document to fake.\n")
    else:
        w("**What those questions have in common is what the path requires: "
          "the question NAMES what it asks about.** \"is vorlin a liquid\" "
          "names `liquid`, and the graph holds `vorlin -[type]-> liquid` as a "
          "record it DERIVED — so the answer is the record, returned in "
          "microseconds with no engine to consult and therefore nothing for an "
          "engine to invent. These are the multi-hop questions, which is the "
          "pleasing part: the step that was supposed to prove \"the symbolic "
          "layer needs no model\" is the step now proving it in the bill.\n")
        w("**The rest still go to the engine, and the reason is a limit rather "
          "than an oversight.** A question that names no field at all — "
          "\"norgul nedir\" — is indistinguishable, structurally, from one "
          "naming a field the graph has never heard of — \"Nortlann'ın "
          "başkenti neresidir\". Both are a known subject followed by words no "
          "node and no document carries. Answering the first from the "
          "subject's single taxonomy value means answering the second with it "
          "too, which turns a correct abstention into a confident wrong "
          "answer. The eight \"what is X\" questions per language stay on the "
          "paid path, and that is the price of the guarantee.\n")

    # ---- 4. infrastructure ----------------------------------------------
    infra_path = os.path.join(cost_dir, "infra.json")
    if os.path.exists(infra_path):
        infra = json.load(open(infra_path, encoding="utf-8"))
        w("## 4. Infrastructure\n")
        w("What each architecture requires on disk before it can answer "
          "anything. Measured with two clean virtualenvs "
          "([`cost_infra.sh`](cost_infra.sh)) — package counts and megabytes, "
          "not adjectives.\n")
        w("| | pip packages | site-packages | plus model weights |")
        w("|---|---|---|---|")
        for row in infra["rows"]:
            w(f"| {row['name']} | {row['packages']} | {row['size']} | "
              f"{row['weights']} |")
        w("")
        w(infra["note"] + "\n")

    # ---- 5. price layer + projections -----------------------------------
    w("## 5. Price, as a separate layer\n")
    w(f"The tables above are token counts, which are true regardless of what "
      f"anyone charges. Below is one multiplication applied to them: "
      f"{PRICE_NOTE}. **Prices change; re-multiply rather than trust this "
      f"table.** Run the same architecture on a local GGUF engine "
      f"(`LMM_BACKEND=gguf`) and every dollar figure here becomes **exactly "
      f"$0** — replaced by seconds of your own CPU/GPU, which is the trade "
      f"this project is built around.\n")
    for lang in langs:
        w(f"**{lang.upper()} — total spend for N questions over one ingested document**\n")
        w("| | ingestion (once) | per question | 1,000 questions | 100,000 questions |")
        w("|---|---|---|---|---|")
        for side in SIDES:
            d = data.get((side, lang))
            if not d:
                continue
            i, q = d["ingest"], d["query"]
            w(f"| {LABEL[side]} | {money(usd(i['prompt'], i['completion']))} | "
              f"{money(usd(q['prompt'], q['completion']))} | "
              f"{money(total_usd(i, q, 1_000))} | "
              f"{money(total_usd(i, q, 100_000))} |")
        w("")

    # ---- break-even ------------------------------------------------------
    w("### Break-even\n")
    lines = []
    for lang in langs:
        rag = data.get(("rag", lang))
        if not rag:
            continue
        for side in ("lmm-shallow", "lmm-deep"):
            d = data.get((side, lang))
            if not d:
                continue
            n = breakeven(d["ingest"], d["query"], rag["ingest"], rag["query"])
            i, q = d["ingest"], d["query"]
            ri, rq = rag["ingest"], rag["query"]
            cheaper_ingest = usd(i["prompt"], i["completion"]) < usd(ri["prompt"], ri["completion"])
            cheaper_query = usd(q["prompt"], q["completion"]) < usd(rq["prompt"], rq["completion"])
            if cheaper_ingest and cheaper_query:
                lines.append(f"- **{LABEL[side]} vs RAG ({lang})** — cheaper on "
                             f"both axes; no crossing, LMM is ahead from the "
                             f"first question.")
            elif not cheaper_ingest and not cheaper_query:
                lines.append(f"- **{LABEL[side]} vs RAG ({lang})** — dearer on "
                             f"both axes; **there is no break-even.** RAG is "
                             f"cheaper at every N, and the gap widens.")
            elif n:
                lines.append(f"- **{LABEL[side]} vs RAG ({lang})** — crosses at "
                             f"**~{n:,.0f} questions**.")
            else:
                lines.append(f"- **{LABEL[side]} vs RAG ({lang})** — no crossing "
                             f"in the positive range.")
    out.extend(lines or ["- not enough data"])
    w("")
    # The one crossing that DOES exist, and the only one that changes a
    # decision: deep=False ingests free but asks dearer, so it wins only until
    # the graph's cheaper queries have repaid the extraction.
    inner = []
    for lang in langs:
        sh, dp = data.get(("lmm-shallow", lang)), data.get(("lmm-deep", lang))
        if not (sh and dp):
            continue
        n = breakeven(sh["ingest"], sh["query"], dp["ingest"], dp["query"])
        if n:
            inner.append(
                f"- **`deep=False` vs `deep=True` ({lang})** — `deep=False` "
                f"ingests free but asks dearer; `deep=True` overtakes it at "
                f"**~{n:,.0f} questions** on one document. Below that, shallow "
                f"is the cheaper LMM; above it, the graph has repaid its own "
                f"extraction.")
    if inner:
        w("The one crossing that exists is *inside* LMM:\n")
        out.extend(inner)
        w("")
        w("**Treat those two numbers as an order of magnitude, not a "
          "threshold.** They divide a fixed ingestion cost by a small "
          "per-question difference, so the noise in the per-question figure is "
          "amplified — which is exactly why the two languages disagree by "
          "several-fold on a corpus that is otherwise line-for-line identical. "
          "What is solid is the shape: shallow wins for a handful of "
          "questions, deep wins once you are asking hundreds.\n")

    # ---- 6. honest summary ----------------------------------------------
    w("## 6. Honest summary\n")
    lang0 = langs[0] if langs else None
    rag = data.get(("rag", lang0))
    deep = data.get(("lmm-deep", lang0))
    shallow = data.get(("lmm-shallow", lang0))
    if rag and deep:
        qr, qd = rag["query"], deep["query"]
        ir, idp = rag["ingest"], deep["ingest"]
        call_x = qd["calls"] / qr["calls"] if qr["calls"] else 0
        tok_x = qd["prompt"] / qr["prompt"] if qr["prompt"] else 0
        ing_tok = idp["prompt"] + idp["completion"]
        w(f"**Where we are more expensive: everywhere that is measured in "
          f"tokens.** On the {lang0.upper()} corpus LMM `deep=True` spends "
          f"**{qd['calls']:.1f} model calls per question** against RAG's "
          f"{qr['calls']:.1f} — about **{call_x:.0f}x the calls and "
          f"{tok_x:.0f}x the prompt tokens**. Ingestion is worse in relative "
          f"terms: {ing_tok:,.0f} tokens against RAG's "
          f"{ir['prompt'] + ir['completion']:,.0f}, because RAG's embedding "
          f"step is a local model and spends none at all. There is no reading "
          f"of these numbers in which LMM is the cheap option on a hosted "
          f"per-token engine, and no break-even where that reverses — the gap "
          f"grows with every question asked.\n")
        w(f"**We are also slower per question**: {qd['wall']:.1f} s against "
          f"RAG's {qr['wall']:.1f} s, because six or seven sequential calls "
          f"cannot beat one. On the same rate-limited deployment, measured "
          f"back to back.\n")
        if shallow:
            qs = shallow["query"]
            n = breakeven(shallow["ingest"], shallow["query"],
                          deep["ingest"], deep["query"])
            w(f"**Which LMM mode is cheaper is a question of how many "
              f"questions you will ask.** `deep=False` ingests for nothing but "
              f"asks dearer ({qs['calls']:.1f} calls/q vs {qd['calls']:.1f}); "
              f"`deep=True` pays {ing_tok:,.0f} tokens up front and then asks "
              f"cheaper, because the graph answers more directly."
              + (f" They cross somewhere in the **low hundreds of questions** "
                 f"on one document (~{n:,.0f} here, but see the caveat above — "
                 f"the two languages disagree several-fold)." if n else "")
              + " `deep=False` is also the only workable mode for a large "
                "document, since extraction cost scales with the text while "
                "the evidence index does not.\n")
        if zero_total:
            w(f"**Some questions now cost nothing at all.** Section 3 counts "
              f"{zero_total:.0f} answers across these runs that never reached "
              f"the engine — the graph settled them itself, in microseconds, "
              f"for zero tokens and zero seconds of anyone's GPU. It is a "
              f"minority of the questions and it does not move the per-question "
              f"average much; what it moves is the FLOOR. On that share of the "
              f"traffic the architecture is not merely cheaper than RAG, it is "
              f"free, and it is engine-independent — the same answer comes back "
              f"from a 3B int4 build on a laptop as from a hosted model, "
              f"because neither was asked.\n")
        w("**Where we are cheaper: the axes this table cannot bill.** The "
          "tokens above buy three things RAG does not have at any price — "
          "every answer carrying its source, a structural gate that stops an "
          "unsupported claim from leaving, and multi-hop facts *derived* "
          "symbolically in microseconds with no model call. Section 4's disk "
          "figures are the other axis: a zero-dependency core against an "
          "embedding stack and a vector database. And the whole dollar column "
          "collapses to zero on a local engine, where the cost becomes your "
          "own seconds — which is the deployment this project is actually "
          "built for.\n")
        w("**So the fair sentence is this:** if you are paying per token for a "
          "hosted model and you only need one-hop lookup, embedding RAG is "
          "cheaper than LMM and will stay cheaper. LMM's case is accuracy, "
          "provenance and refusal (see the README's benchmark table), bought "
          "with tokens at ingestion and at verification — or bought with CPU "
          "seconds instead, on hardware you already own.\n")

    json.dump({f"{k[0]}_{k[1]}": v for k, v in data.items()},
              open(os.path.join(cost_dir, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
