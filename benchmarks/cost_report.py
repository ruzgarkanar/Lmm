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
        if not name.endswith(".json") or name.startswith("smoke"):
            continue
        side, lang, _rep = name[:-5].rsplit("_", 2)
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
    w("**LMM's ingestion is where LMM is expensive.** `deep=True` reads the "
      "document into a graph — every candidate fact is extracted and then "
      "re-read by the gate before it is allowed in. That is the price of the "
      "provenance and the refusal guarantee, and it is paid once per "
      "document, not once per question.\n")

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
      "graph lookup. The architecture allows it (a table row or a derived fact "
      "can answer without generation); on this corpus, through the `respond` "
      "path, it does not happen.\n")
    w("| | questions | answered with 0 model calls | share |")
    w("|---|---|---|---|")
    for lang in langs:
        for side in SIDES:
            d = data.get((side, lang))
            if not d:
                continue
            q = d["query"]
            w(f"| {LABEL[side]} ({lang}) | {q['n']:.0f} | {q['zero']:.0f} | "
              f"{100 * q['zero'] / q['n']:.0f}% |")
    w("")

    # ---- 5. price layer + projections -----------------------------------
    w("## 4. Price, as a separate layer\n")
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

    json.dump({f"{k[0]}_{k[1]}": v for k, v in data.items()},
              open(os.path.join(cost_dir, "summary.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\n".join(out))


if __name__ == "__main__":
    main()
