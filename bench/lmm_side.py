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
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def main():
    from lmm.session import Session
    from lmm.mind import Mind

    corpus = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bench", "corpus.txt"),
                  encoding="utf-8").read()
    s = Session(None)               # fresh memory — no disk writes
    m = Mind(s)
    t0 = time.time()
    wrote, skipped = s.learn_text(corpus, source="#doc:korpus")
    m.run()                         # derivations (ms) — multi-hop forms here
    ingest_ms = round((time.time() - t0) * 1000)
    derived = sum(1 for r in s.memory.records.values()
                  if r.source == "#inference")
    print(f"ingest: {wrote} facts, {skipped} skipped, {derived} derived, "
          f"{ingest_ms}ms", flush=True)
    for sent in getattr(s, "unread", []):
        print(f"  ⚠ not learned: {sent}", flush=True)

    questions = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "bench", "questions.json"),
                               encoding="utf-8"))
    results = []
    for q in questions:
        t0 = time.time()
        said = s.respond(q["soru"])
        ms = round((time.time() - t0) * 1000)
        results.append({"soru": q["soru"], "cevap": said, "ms": ms})
        print(f"> {q['soru']}\n  {said}   ({ms}ms)", flush=True)

    out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "bench", "result_lmm.json")
    json.dump({"ingest_ms": ingest_ms, "yazilan": wrote, "atlanan": skipped,
               "turetilen": derived, "cevaplar": results},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
