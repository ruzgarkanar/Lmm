"""LoRA eğitim verisi — GRAFTAN/GERÇEK VERİDEN üretilir, ELLE cümle YAZILMAZ.

condition-5 (elle dil yok) burada da korunur: hedef cümleyi biz uydurmayız.
İki kaynak, iki strateji:

  1) GROUNDING (topraklama disiplini): modele "yalnız ENJEKTE edilen olgudan
     konuş" davranışını öğretir. Girdi = olgu bloğu + soru; hedef = o olguyu
     söyleyen KISA cümle. Hedefi ELLE kalıpla yazmak condition-5'i ihlal ederdi;
     bu yüzden hedef, MEVCUT çalışan istem davranışının ÖZ-DAMITIMIDIR (Qwen
     kendi üretir, biz yalnız 'ne yap' deriz) — yani sistemin şu anki doğru
     davranışını ağırlığa taşırız. Kalıp yok.

  2) REFUSAL (uydurmama refleksi): olgu bloğu BOŞ iken hedef, generate.refusal
     çıktısı — "bilmiyorum"u da model kendi dilinde üretir.

  3) IDENTITY: kimlik olgusu (lmm→üretici→rüzgar) graftan; hedef yine öz-damıtım.

Çıktı: JSONL, her satır {"messages":[{system},{user},{assistant}]} — chat SFT
formatı. train.py bunu tüketir.

Kullanım:
    python3.11 -m lmm.finetune.make_data --limit 800 --out data/train/lora.jsonl

Not: öz-damıtım Qwen'i çağırır (yavaş). --limit ile küçük tut; kalite için
gerçek `cümle`'yi de kullanmıyoruz çünkü o, blokta OLMAYAN olgular içerir
(modele fazladan olgu = uydurma öğretirdi). Sadece bloktaki olgu hedefe girer.
"""
import argparse
import json
import os
import random

from lmm import prompts, retrieve, link, generate
from v3.memory import Memory, OPERATOR
from v3.gate import Gate


def _grounding_rows(limit):
    """tanim-temiz.jsonl'den: her kavram için (olgu bloğu + soru) → öz-damıtılmış
    kısa grounded cümle. Blok yalnız TEK olgu taşır; hedef o olgudan sapamaz."""
    src = os.path.join("data", "train", "tanim-temiz.jsonl")
    rows = []
    with open(src, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    random.seed(0)                              # tekrarlanabilir örnek
    random.shuffle(lines)
    for rec in lines[:limit]:
        kavram, hedef = rec.get("kavram", "").strip(), rec.get("hedef", "").strip()
        if not kavram or not hedef:
            continue
        # Tek-olgu blok — facts_block biçimiyle birebir (inference'la aynı girdi).
        block = f"[1] {kavram.lower()} → {hedef.lower()}"
        question = f"{kavram} nedir"
        target = generate.answer(question, block)      # ÖZ-DAMITIM (Qwen üretir)
        if not target.strip():
            continue
        rows.append({"messages": [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": f"OLGULAR:\n{block}\n\nSORU: {question}"},
            {"role": "assistant", "content": target},
        ]})
    return rows


def _refusal_rows(limit):
    """Bilinmeyen sorular → 'bilmiyorum' (model kendi dilinde). Girdi olgusuz."""
    src = os.path.join("data", "train", "tanim-temiz.jsonl")
    rows = []
    with open(src, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    random.seed(1)
    random.shuffle(lines)
    for rec in lines[:limit]:
        kavram = rec.get("kavram", "").strip()
        if not kavram:
            continue
        question = f"{kavram} nedir"
        target = generate.refusal(question)
        if not target.strip():
            continue
        rows.append({"messages": [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content":
             f"SORU: {question}\n\n(Belleğinde bu konuda kayıtlı olgu YOK. "
             "Uydurma; bilmediğini söyle.)"},
            {"role": "assistant", "content": target},
        ]})
    return rows


def _identity_rows():
    """Kimlik olgusu graftan tohumlanır; birkaç kimlik sorusu öz-damıtılır."""
    m = Memory()
    m.self_key = m.identify("#self")
    gate = Gate(m)
    lmm = link.resolve(m, "lmm", {}, create=True)
    ruz = link.resolve(m, "rüzgar", {}, create=True)
    mk = link.resolve(m, "üretici", {}, create=True)
    gate.admit(lmm, mk, ruz, "#operator", OPERATOR)
    recs = retrieve.gather(m, lmm)
    idb = "\n".join(f"{link.label_of(m, r.subject)} "
                    f"{link.label_of(m, r.predicate)} → "
                    f"{link.label_of(m, r.value)}" for r in recs)
    rows = []
    for q in ["sen kimsin", "seni kim yaptı", "who made you", "seni kim üretti"]:
        target = generate.chat(q, idb)
        rows.append({"messages": [
            {"role": "system", "content": prompts.CHAT_SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": target},
        ]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400,
                    help="grounding+refusal için kavram sayısı (her biri)")
    ap.add_argument("--out", default="data/train/lora.jsonl")
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    rows = (_grounding_rows(args.limit) + _refusal_rows(args.limit // 2)
            + _identity_rows())
    random.seed(2)
    random.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} örnek → {args.out}")


if __name__ == "__main__":
    main()
