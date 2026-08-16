"""UYKU-KONSOLİDASYONU — hasat: sohbet günlükleri → LoRA eğitim verisi.

Beyin analojisinin motoru: gün içinde graf (hipokampus) hızlı öğrenir; bu araç
o birikimi periyodik olarak MOTORA (korteks/ağırlık) taşınacak veriye çevirir.
Döngü: yaşa → hasat et (bu dosya) → eğit (train.py --resume) → değiştir (lora/).

GÜVENLİ damıtım ilkesi — modelin KENDİ ham çıktısı asla hedef olmaz (hata
büyütür, v2 dersi). Hasat edilen tek şey KAPI-ONAYLI çiftlerdir:
    kullanıcı cümlesi → o tur kapının GERÇEKTEN kabul ettiği üçlüler
(chat.py her turun "learned" alanına yazar). Yani hedefler grafın kendisinden
gelir — condition-5 temiz, elle yazılmış dil yok.

Ayrıca ASK örnekleri türetir ("{özne} nedir" → ASK) ki WRITE/ASK ayrımı
tazelensin — özne etiketi gerçek veriden gelir, cümle şablonu değil olgu.

Kullanım (tetik İNSAN-onaylı; otomatik eğitim başlatmaz — guardrail):
    python3.11 -m lmm.finetune.consolidate                # logs/ → data/train/consolidate.jsonl
    python3.11 -m lmm.finetune.consolidate --min-turns 50 # az birikimse çık
Çıktı 0 değilse önerilen eğitim komutunu basar.
"""
import argparse
import glob
import json
import os

from lmm import prompts


def harvest(log_dir="logs", state_path=None):
    """Günlüklerden kapı-onaylı (cümle, üçlüler) çiftlerini topla.

    FİLİGRAN (code-review bulgusu #2): işlenen dosya+satır sayısı state'te
    tutulur; sonraki hasat yalnız YENİ turları alır. Yoksa ilk günlerin turları
    her döngüde yeniden eğitime girer, eski örnekler katlanarak ağır basardı
    (sessiz overfit). DÖNEN: (pairs, new_state) — new_state'i ÇAĞIRAN, çıktıyı
    gerçekten yazdıktan SONRA kaydeder (min-turns'te çıkarsa filigran ilerlemez,
    turlar kaybolmaz). state_path=None → filigransız (test)."""
    state = {}
    if state_path and os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    pairs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "lmm-*.jsonl"))):
        name = os.path.basename(path)
        done = state.get(name, 0)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[done:]:
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                continue
            learned = turn.get("learned") or []
            message = (turn.get("in") or "").strip()
            if not (message and learned):
                continue            # o tur kapıdan bir şey geçmedi → hasat yok
            triples = [[s, p, v] for s, p, v in
                       (t[:3] for t in learned if len(t) >= 3) if s and v]
            if triples:
                pairs.append((message, triples))
        state[name] = len(lines)
    return pairs, state


def commit_state(state_path, state):
    """Filigranı kalıcıla — YALNIZ hasat çıktısı gerçekten yazıldıktan sonra."""
    if not state_path:
        return
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f)


def rows_from(pairs, ask_ratio=0.4):
    """(cümle, üçlüler) → make_extract_data ile AYNI formatta SFT satırları."""
    rows = []
    acc = 0.0                # birikimli oran: ask_ratio GERÇEKTEN o oran olsun
    for message, triples in pairs:
        rows.append({"messages": [
            {"role": "system", "content": prompts.EXTRACT_SYSTEM},
            {"role": "user", "content": message},
            {"role": "assistant", "content": json.dumps(
                {"kind": "WRITE", "triples": triples}, ensure_ascii=False)},
        ]})
        acc += ask_ratio
        if acc >= 1.0:
            acc -= 1.0
            subject = triples[0][0]
            rows.append({"messages": [
                {"role": "system", "content": prompts.EXTRACT_SYSTEM},
                {"role": "user", "content": f"{subject} nedir"},
                {"role": "assistant", "content": json.dumps(
                    {"kind": "ASK", "triples": [[subject, "", ""]]},
                    ensure_ascii=False)},
            ]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="data/train/consolidate.jsonl")
    ap.add_argument("--min-turns", type=int, default=20,
                    help="bundan az hasat varsa eğitime değmez — çık")
    ap.add_argument("--state", default="data/train/.consolidated.json",
                    help="filigran: işlenmiş günlük satırları (tekrar hasat yok)")
    args = ap.parse_args()

    pairs, state = harvest(args.logs, state_path=args.state)
    if len(pairs) < args.min_turns:
        print(f"hasat {len(pairs)} tur < {args.min_turns} — birikim az, "
              f"eğitime değmez (döngü: yaşamaya devam; filigran İLERLEMEDİ)")
        return
    rows = rows_from(pairs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    commit_state(args.state, state)
    print(f"{len(rows)} örnek ({len(pairs)} kapı-onaylı tur) → {args.out}")
    print("önerilen (İNSAN başlatır — guardrail):")
    print(f"  python3.11 -m lmm.finetune.train --data {args.out} "
          f"--out models/lmm/lora_next --epochs 1 --rank 64 --mlp")


if __name__ == "__main__":
    main()
