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
import re

# DİL FİLTRESİ: self-distillation, sistemin şu anki OYNAK çıktılarını da yakalar
# (Çince/İngilizce sızıntı). Bu örnekleri eğitim verisinden ATARIZ ki LoRA temiz
# hedef-dil davranışını öğrensin. Bu bir VERİ TEMİZLİĞİ süzgeci — çalışma-zamanı
# dil üretimi değil (condition-5 ihlali değil).
_CJK = re.compile(r"[　-鿿가-힯぀-ヿ]")
_EN = re.compile(r"\b(the|and|is|are|was|were|of|to|this|that|with|not|for|"
                 r"you|your|according|source)\b", re.I)


def _tr_ok(text):
    """Türkçe-girdi hedefi temiz mi (Çince yok, ağır İngilizce sızıntı yok)."""
    if not text or not text.strip():
        return False
    if _CJK.search(text):
        return False
    return len(_EN.findall(text)) < 2


def _no_cjk(text):
    """Kimlik satırları TR ya da EN olabilir — yalnız CJK sızıntısını ele."""
    return bool(text and text.strip()) and not _CJK.search(text)

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
        if not _tr_ok(target):                          # DİL FİLTRESİ
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
        if not _tr_ok(target):                          # DİL FİLTRESİ
            continue
        rows.append({"messages": [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content":
             f"SORU: {question}\n\n(Belleğinde bu konuda kayıtlı olgu YOK. "
             "Uydurma; bilmediğini söyle.)"},
            {"role": "assistant", "content": target},
        ]})
    return rows


# Kimlik soruları — ÇOK DİLLİ, ÇOK BİÇİMLİ. Kimlik gibi spesifik davranış az
# örnekle oturmaz; geniş tut. Elle CEVAP değil, elle SORU listesi (kullanıcı
# girdisi) — hedefi yine Qwen üretir (condition-5: cevap kalıbı yok).
_IDENTITY_Q = [
    "sen kimsin", "kimsin sen", "sen nesin", "adın ne", "kendini tanıt",
    "sen kimsin?", "senin adın ne",
    "seni kim yaptı", "seni kim üretti", "seni kim geliştirdi", "üreticin kim",
    "kim yarattı seni", "seni kim yazdı", "seni kim yaptı?", "yapımcın kim",
    "who are you", "what are you", "who made you", "who created you",
    "who built you", "what is your name",
    "wer bist du", "wer hat dich gemacht",
]


def _identity_rows(samples=2):
    """Kimlik: graftan tohumla, SORULARI Qwen'e sor, yalnız DOĞRU cevapları hasat
    et (üreticiyi/kimliği söyleyen; 'bilmiyorum' ya da yanlış olanları at). Böylece
    az-ama-temiz kimlik verisi çıkar — 'seni kim yaptı' tutarlılığını bu çözer."""
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
    for q in _IDENTITY_Q:
        for _ in range(samples):
            target = generate.chat(q, idb)
            if not _no_cjk(target):
                continue
            low = target.lower()
            # DOĞRULUK filtresi: cevap üreticiyi (rüzgar) ya da kimliği (lmm)
            # anmalı — yoksa 'bilmiyorum'/yanlış, eğitime girmesin.
            if "rüzgar" not in low and "lmm" not in low:
                continue
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
