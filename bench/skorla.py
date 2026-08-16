"""Skorlama — nesnel, anahtar-sözcük temelli (yargıç-model YOK: yargıç RAG'in
kendi motoru olurdu, taraflı).

dogrudan/cok-adim/nedensel: altın sözcüklerden biri cevapta geçiyor mu (fold).
yokluk: cevap SAKINIYOR mu — bilmiyorum-benzeri VE uydurulmuş değer yok.
        Ölçüt: cevapta korpusta/altında OLMAYAN "kesin iddia" kalıbı aranmaz;
        pratik vekil: sakınma sözcüğü varsa geç, yoksa ve cevap kısa-kesinse kal.
"""
import json
import os
import sys
import unicodedata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SAKINMA = ("bilmiyorum", "bilgi yok", "bilgim yok", "bilgiye sahip değil", "bilemem",
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
            ok = any(fold(s) in ans for s in SAKINMA)
        else:
            ok = any(fold(g) in ans for g in q["altin"])
        rows.append((q["tip"], q["soru"], ok, r["ms"]))
        t = per_type.setdefault(q["tip"], [0, 0, 0])
        t[0] += ok; t[1] += 1; t[2] += r["ms"]
    return data, rows, per_type


def main():
    qpath = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bench", "sorular.json")
    questions = json.load(open(qpath, encoding="utf-8"))
    global RAG_YOL, LMM_YOL

    for name, path in (("RAG (langchain+embedding+gpt-4o-mini)",
                        sys.argv[2] if len(sys.argv) > 2 else "bench/sonuc_rag.json"),
                       ("LMM (graf+kapı+yerel-3B)", sys.argv[3] if len(sys.argv) > 3 else "bench/sonuc_lmm.json")):
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            print(f"{name}: sonuç yok ({path})"); continue
        data, rows, per_type = score(full, questions)
        print(f"\n=== {name} ===")
        for tip, ((ok, n, ms)) in ((k, v) for k, v in sorted(per_type.items())):
            print(f"  {tip:10}  {ok}/{n}   ort {round(ms/n)}ms")
        top = sum(v[0] for v in per_type.values())
        n = sum(v[1] for v in per_type.values())
        print(f"  TOPLAM      {top}/{n}   ingest {data.get('ingest_ms','?')}ms")
        for tip, soru, ok, ms in rows:
            if not ok:
                print(f"    ✗ [{tip}] {soru}")


if __name__ == "__main__":
    main()
