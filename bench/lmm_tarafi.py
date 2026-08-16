"""LMM tarafı — aynı korpus, aynı sorular. learn_text (bir kez oku → grafa) +
respond (graf-temelli cevap). Motor: YEREL Qwen-3B+LoRA (bulut yok, API yok).

Çıktı: bench/sonuc_lmm.json  [{soru, cevap, ms}] + yutma istatistiği.
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

    corpus = open(os.path.join(ROOT, "bench", "korpus.txt"),
                  encoding="utf-8").read()
    s = Session(None)               # taze bellek — diske yazmaz
    m = Mind(s)
    t0 = time.time()
    wrote, skipped = s.learn_text(corpus, source="#doc:korpus")
    m.run()                         # türetimler (ms) — çok-adım burada oluşur
    ingest_ms = round((time.time() - t0) * 1000)
    derived = sum(1 for r in s.memory.records.values()
                  if r.source == "#inference")
    print(f"ingest: {wrote} olgu, {skipped} atlandı, {derived} türetildi, "
          f"{ingest_ms}ms", flush=True)

    questions = json.load(open(os.path.join(ROOT, "bench", "sorular.json"),
                               encoding="utf-8"))
    results = []
    for q in questions:
        t0 = time.time()
        said = s.respond(q["soru"])
        ms = round((time.time() - t0) * 1000)
        results.append({"soru": q["soru"], "cevap": said, "ms": ms})
        print(f"> {q['soru']}\n  {said}   ({ms}ms)", flush=True)

    out = os.path.join(ROOT, "bench", "sonuc_lmm.json")
    json.dump({"ingest_ms": ingest_ms, "yazilan": wrote, "atlanan": skipped,
               "turetilen": derived, "cevaplar": results},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"→ {out}", flush=True)


if __name__ == "__main__":
    main()
