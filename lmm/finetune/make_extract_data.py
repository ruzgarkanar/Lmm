"""EXTRACTION eğitim verisi — tanim-temiz.jsonl'nin YAPISAL alanlarından.

En büyük darboğaz extract (cümle → üçlü). tanim-temiz.jsonl her satırda ZATEN
hizalı yapı taşıyor: {kavram, ilişki, hedef, cümle}. Yani (cümle → {kind:WRITE,
triples:[[kavram, ilişki, hedef]]}) hazır bir denetimli örnek — Qwen'e sormaya
GEREK YOK, ~75k temiz örnek. condition-5 korunur: hedef üçlü GERÇEK VERİNİN
yapısından gelir, elle yazılmaz.

Ek olarak SORU (ASK) örnekleri de üretir: "{kavram} nedir" → {kind:ASK,
triples:[[kavram,"",""]]} — böylece LoRA WRITE/ASK ayrımını da öğrenir (sohbette
"X nedir"i yanlışlıkla WRITE sanma sorunu buydu).

A100'de asıl eğitim bunu + grounding self-distillation'ı birlikte tüketir.

Kullanım:
    python3.11 -m lmm.finetune.make_extract_data --out data/train/extract.jsonl
"""
import argparse
import json
import os

from lmm import prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/train/tanim-temiz.jsonl")
    ap.add_argument("--out", default="data/train/extract.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0=tümü")
    ap.add_argument("--ask-ratio", type=float, default=0.4,
                    help="her N WRITE için ASK örneği oranı")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    os.chdir(root)

    rows = []
    with open(args.src, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    if args.limit:
        lines = lines[:args.limit]

    for i, rec in enumerate(lines):
        kavram = rec.get("kavram", "").strip()
        hedef = rec.get("hedef", "").strip()
        ileti = rec.get("ilişki", "").strip()
        cumle = rec.get("cümle", "").strip()
        if not (kavram and hedef and cumle):
            continue
        # WRITE: gerçek cümle → yapısal üçlü (hizalı, gerçek)
        triple = [kavram.lower(), ileti, hedef.lower()]
        rows.append({"messages": [
            {"role": "system", "content": prompts.EXTRACT_SYSTEM},
            {"role": "user", "content": cumle},
            {"role": "assistant", "content": json.dumps(
                {"kind": "WRITE", "triples": [triple]}, ensure_ascii=False)},
        ]})
        # ASK: "{kavram} nedir" → soru (WRITE/ASK ayrımını öğret)
        if args.ask_ratio and (i % max(1, int(1 / args.ask_ratio))) == 0:
            rows.append({"messages": [
                {"role": "system", "content": prompts.EXTRACT_SYSTEM},
                {"role": "user", "content": f"{kavram} nedir"},
                {"role": "assistant", "content": json.dumps(
                    {"kind": "ASK", "triples": [[kavram.lower(), "", ""]]},
                    ensure_ascii=False)},
            ]})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} extraction örneği → {args.out} "
          f"({len(lines)} kaynak satırdan)")


if __name__ == "__main__":
    main()
