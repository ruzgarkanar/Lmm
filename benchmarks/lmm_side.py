"""LMM side — same corpus, same questions. learn_text (read once → graph) +
respond (graph-grounded answer). Engine: LOCAL Qwen-3B+LoRA (no cloud, no API).

Output JSON field names ('soru', 'cevap', 'ms', 'ingest_ms', 'cevaplar') match
the Turkish benchmark data format read by score.py — do not rename them.

Output: bench/sonuc_lmm.json  [{soru, cevap, ms}] + ingestion statistics.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))   # benchmarks/
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)


def main():
    from lmm.session import Session
    from lmm.mind import Mind

    corpus = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "corpus.txt"),
                  encoding="utf-8").read()
    s = Session(None)               # fresh memory — no disk writes
    m = Mind(s)
    # INSTANT-READY vs DEEP ingestion: a 350-page manual is read shallow (the
    # evidence index answers it), a small corpus deep (facts into the graph, so
    # derivation can compose). LMM_DEEP=0 selects the shallow mode.
    deep = os.environ.get("LMM_DEEP", "1") != "0"
    t0 = time.time()
    wrote, skipped = s.learn_text(corpus, source="#doc:corpus", deep=deep)
    m.run()                         # derivations (ms) — multi-hop forms here
    ingest_ms = round((time.time() - t0) * 1000)
    derived = sum(1 for r in s.memory.records.values()
                  if r.source == "#inference")
    print(f"ingest: {wrote} facts, {skipped} skipped, {derived} derived, "
          f"{ingest_ms}ms", flush=True)
    for sent in getattr(s, "unread", []):
        print(f"  ⚠ not learned: {sent}", flush=True)

    questions = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "questions.json"),
                               encoding="utf-8"))
    # THE A/B CONTROL for the graph-first path. `fluent=True` sends every
    # question to the engine, which is exactly the behaviour before
    # `lmm/lookup.py` existed — so a regression can be attributed by
    # MEASUREMENT rather than by argument: same binary, same corpus, same
    # questions, one flag. It lives here rather than in the library because it
    # is a benchmark control, not a setting anyone deploys.
    fluent = os.environ.get("LMM_FLUENT", "0") == "1"
    results = []
    for q in questions:
        t0 = time.time()
        said = s.respond(q["soru"], fluent=fluent)
        ms = round((time.time() - t0) * 1000)
        # THE SYSTEM'S OWN ABSTENTION SIGNAL. Whether a turn declined to answer
        # is something the session knows for certain (it took the refusal
        # path); reading it back out of the refusal SENTENCE means recognising
        # "I don't know" in whichever language the engine wrote it, which is
        # how the scorer came to hold a list of Turkish phrases. Carrying the
        # flag here makes the measurement work on a document in any language.
        results.append({"soru": q["soru"], "cevap": said, "ms": ms,
                        "abstained": bool(s.last_abstained),
                        # which path spoke — so the result file itself says
                        # whether a question cost anything
                        "from_graph": bool(s.last_from_graph)})
        print(f"> {q['soru']}\n  {said}   ({ms}ms)", flush=True)

    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "result_lmm.json")
    json.dump({"ingest_ms": ingest_ms, "yazilan": wrote, "atlanan": skipped,
               "turetilen": derived, "cevaplar": results},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
