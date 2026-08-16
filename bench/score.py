"""Scoring — objective, keyword-based (NO judge model: the judge would be
RAG's own engine, hence biased).

direct/multi-hop/causal: does one of the gold words appear in the answer (fold).
absence: does the answer ABSTAIN — something like "I don't know" AND no
        fabricated value. Criterion: we don't scan the answer for "confident
        claim" patterns absent from the corpus/gold; the practical proxy: pass
        if an abstention word is present, fail if absent and the answer is
        short-and-assertive.

Note: JSON field names ('soru', 'altin', 'tip', 'cevap', 'ms', 'ingest_ms',
'cevaplar') are Turkish because the benchmark data files are Turkish — they
are part of the data format and must not be renamed here.
"""
import json
import os
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Turkish abstention words — FUNCTIONAL: the benchmark answers are in Turkish,
# so abstention ("I don't know") is detected via these Turkish phrases.
ABSTAIN = ("bilmiyorum", "bilgi yok", "bilgim yok", "bilgiye sahip değil", "bilemem",
           "bulunmuyor", "yer almıyor", "belirtilmemiş", "bilinmiyor",
           "bilmem", "ulaşamadım", "emin değilim", "yoktur")


def fold(t):
    t = unicodedata.normalize("NFC", t)
    return "".join(ch.casefold()[0] for ch in t if not unicodedata.combining(ch))


def score(path, questions):
    data = json.load(open(path, encoding="utf-8"))
    by_q = {r["soru"]: r for r in data["cevaplar"]}
    rows, per_type = [], {}
    for q in questions:
        r = by_q[q["soru"]]
        ans = fold(r["cevap"] or "")
        if q["altin"] is None:
            ok = any(fold(s) in ans for s in ABSTAIN)
        else:
            ok = any(fold(g) in ans for g in q["altin"])
        rows.append((q["tip"], q["soru"], ok, r["ms"]))
        t = per_type.setdefault(q["tip"], [0, 0, 0])
        t[0] += ok; t[1] += 1; t[2] += r["ms"]
    return data, rows, per_type


def main():
    qpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bench", "questions.json")
    questions = json.load(open(qpath, encoding="utf-8"))
    global RAG_PATH, LMM_PATH

    for name, path in (("RAG (langchain+embedding+gpt-4o-mini)",
                        sys.argv[2] if len(sys.argv) > 2 else "bench/result_rag.json"),
                       ("LMM (graph+gate+local-3B)", sys.argv[3] if len(sys.argv) > 3 else "bench/result_lmm.json")):
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            print(f"{name}: no results ({path})"); continue
        data, rows, per_type = score(full, questions)
        print(f"\n=== {name} ===")
        for qtype, ((ok, n, ms)) in ((k, v) for k, v in sorted(per_type.items())):
            print(f"  {qtype:10}  {ok}/{n}   avg {round(ms/n)}ms")
        top = sum(v[0] for v in per_type.values())
        n = sum(v[1] for v in per_type.values())
        print(f"  TOTAL       {top}/{n}   ingest {data.get('ingest_ms','?')}ms")
        for qtype, question, ok, ms in rows:
            if not ok:
                print(f"    ✗ [{qtype}] {question}")


if __name__ == "__main__":
    main()
