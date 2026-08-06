"""Niyet ayrıştırıcısının asıl eğitimi — iki omurga, düzgün sınav, en iyi tur.

Bu betik, gecenin bütün ölçümlerinin vardığı yer. Elle tasarlanmış özellik
uzayının tavanı dört yoldan doğrulandı (bkz. `docs/DENEMELER.md`); temsili
öğrenmekten başka yol kalmadı.

İlk denemeler bir şeyi acı biçimde gösterdi: **ölçüt küçükse ölçüm yalan
söyler.** On altı elle yazılmış soruyla %75 ile %81 arasındaki fark
ayırt edilemez. Burada sınav gerçek sorulardan ayrılıyor ve yüzlerce.

Ve ikinci ders, kendi çıktımızdan: 4.850 örnekle dördüncü turda eğitim kaybı
0,212'ye inerken sınav %68,5'ten %56,3'e düştü. Ders ezberi. O yüzden burada
en iyi tur saklanıyor, son tur değil.

İki omurga yarışıyor:

    çekirdek    kendi modelimiz — 16,8M, bizim derlemimizle eğitilmiş
    hazır       ytu-ce-cosmos Türkçe GPT-2 — 355M, çok daha fazla veriyle

Kendi çekirdeğimiz yaklaşıyorsa onu kullanmak açıkça daha iyi: yirmi kat
küçük ve her satırı bizim. Ölçüm karar verir.

Ve mimari sınır duruyor: bu ağ neyin DOĞRU olduğuna hiç karar vermez, yalnızca
"bu cümle ne soruyor" der. Doğruluk grafta ve epistemik kapıda kalır.

Kullanım:
    python3 scripts/niyet_egitim.py --omurga cekirdek --tur 8
    python3 scripts/niyet_egitim.py --omurga hazir --tur 4
"""
import collections
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from core.intent import _where                              # noqa: E402
CORE_CHECKPOINT = _where("core")
PIECES = os.path.join(ROOT, "data", "tr-parcalayici.model")
READY = os.path.join(ROOT, "core", "turkce-cekirdek")
OUTPUT = os.path.join(ROOT, "core", "niyet")

OUTSIDE = "DISARIDA"
LONGEST = 32
HELD_PER_CLASS = 40     # sınıf başına ayrılan sınav; küçük sınıflarda hepsi değil


def sources():
    """Etiketli veri, büyüğü varsa o. Üretilmiş cümleler yalnızca destek."""
    big = os.path.join(ROOT, "data", "tr-gercek-niyet-buyuk.txt")
    small = os.path.join(ROOT, "data", "tr-gercek-niyet.txt")
    made = os.path.join(ROOT, "data", "tr-niyet.txt")
    real = big if os.path.exists(big) else small
    return real, made


def load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if "\t" in line:
                kind, sentence = line.rstrip("\n").split("\t", 1)
                rows.append((sentence.strip(), kind))
    return rows


def split(real, made):
    """Sınav GERÇEK ve KAPSAM İÇİ sorulardan ayrılır.

    Kapsam dışı sorular sınavda ölçüye girmiyor: onları doğru bilmek kolay
    (çoğunluk sınıfı) ve skoru şişirir. Ölçmek istediğimiz şey, sorulan bir
    soruyu doğru ANLAMAK.
    """
    by_kind = collections.defaultdict(list)
    for sentence, kind in real:
        by_kind[kind].append((sentence, kind))
    held, train = [], []
    for kind, rows in by_kind.items():
        random.Random(7).shuffle(rows)
        if kind == OUTSIDE:
            train.extend(rows)
            continue
        cut = min(HELD_PER_CLASS, len(rows) // 3)
        held.extend(rows[:cut])
        train.extend(rows[cut:])
    return train + made, held


def main(argv):
    import torch
    import torch.nn as nn

    backbone = (argv[argv.index("--omurga") + 1] if "--omurga" in argv
                else "cekirdek")
    rounds = int(argv[argv.index("--tur") + 1]) if "--tur" in argv else 8
    batch = int(argv[argv.index("--yigin") + 1]) if "--yigin" in argv else 32
    rate = float(argv[argv.index("--hiz") + 1]) if "--hiz" in argv else None

    real_path, made_path = sources()
    if "--veri" in argv:
        real_path = os.path.join(ROOT, argv[argv.index("--veri") + 1]) \
            if not os.path.isabs(argv[argv.index("--veri") + 1]) \
            else argv[argv.index("--veri") + 1]
        made_path = None
    real = load(real_path)
    made = load(made_path) if made_path else []
    train, held = split(real, made)
    labels = sorted({kind for _, kind in train + held})
    index = {kind: i for i, kind in enumerate(labels)}
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"veri: {os.path.basename(real_path)} — eğitim {len(train)}, "
          f"SINAV {len(held)} (gerçek, kapsam içi)")
    print(f"{len(labels)} sınıf, aygıt {device}, omurga {backbone}")

    # Sınıf ağırlığı: DIŞARIDA baskın ve dengesizlik bir kez çöküşe yol açtı
    # (hepsine "dışarıda" diyen bir model %25 verdi).
    counts = collections.Counter(kind for _, kind in train)
    most = max(counts.values())
    weights = torch.tensor([most / counts.get(kind, 1) for kind in labels],
                           dtype=torch.float32).to(device)

    if backbone == "hazir":
        from transformers import (AutoTokenizer,
                                  AutoModelForSequenceClassification)
        tokenizer = AutoTokenizer.from_pretrained(READY)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            READY, num_labels=len(labels))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)

        def encode(pairs):
            found = tokenizer([s for s, _ in pairs], truncation=True,
                              max_length=LONGEST, padding="max_length",
                              return_tensors="pt")
            return (found["input_ids"], found["attention_mask"],
                    torch.tensor([index[k] for _, k in pairs]))

        def logits_of(ids, mask):
            return model(input_ids=ids, attention_mask=mask).logits
        rate = rate or 2e-5
    else:
        import sentencepiece as spm
        from core.model import Config, Core
        pieces = spm.SentencePieceProcessor(model_file=PIECES)
        saved = torch.load(CORE_CHECKPOINT, map_location="cpu",
                           weights_only=False)
        config = (Config(**saved["config"]) if isinstance(saved["config"], dict)
                  else saved["config"])
        core = Core(config)
        core.load_state_dict({k: v for k, v in saved["model"].items()
                              if not k.startswith("position")}, strict=False)

        class Classifier(nn.Module):
            """Çekirdeğin temsili + tek doğrusal baş, maskeli ortalama havuz."""

            def __init__(self, core, classes):
                super().__init__()
                self.core = core
                self.drop = nn.Dropout(0.1)
                self.head = nn.Linear(core.config.dimensions, classes)

            def forward(self, tokens, mask):
                x = self.core.drop(self.core.token(tokens))
                for block in self.core.blocks:
                    x = block(x)
                x = self.core.final(x)
                weight = mask.unsqueeze(-1).float()
                pooled = (x * weight).sum(1) / weight.sum(1).clamp(min=1)
                return self.head(self.drop(pooled))

        model = Classifier(core, len(labels)).to(device)
        print(f"çekirdek {core.size/1e6:.1f}M parametre")

        def encode(pairs):
            ids, masks, wanted = [], [], []
            for sentence, kind in pairs:
                piece = pieces.encode(sentence)[:LONGEST]
                pad = LONGEST - len(piece)
                ids.append(piece + [0] * pad)
                masks.append([1] * len(piece) + [0] * pad)
                wanted.append(index[kind])
            return torch.tensor(ids), torch.tensor(masks), torch.tensor(wanted)

        def logits_of(ids, mask):
            return model(ids, mask)
        rate = rate or 3e-4

    optimiser = torch.optim.AdamW(model.parameters(), lr=rate)
    loss_of = nn.CrossEntropyLoss(weight=weights)
    ids, masks, wanted = encode(train)
    order = list(range(len(train)))
    best, best_at = 0.0, 0
    started = time.time()

    def score(pairs):
        model.eval()
        right = 0
        confusion = collections.Counter()
        with torch.no_grad():
            for start in range(0, len(pairs), batch):
                piece = pairs[start:start + batch]
                a, b, c = encode(piece)
                guess = logits_of(a.to(device), b.to(device)).argmax(-1).cpu()
                for got, want, (sentence, _) in zip(guess.tolist(),
                                                    c.tolist(), piece):
                    right += got == want
                    if got != want:
                        confusion[(labels[want], labels[got])] += 1
        return right / max(len(pairs), 1), confusion

    for round_at in range(rounds):
        model.train()
        random.Random(round_at).shuffle(order)
        total = steps = 0
        for start in range(0, len(order), batch):
            chunk = order[start:start + batch]
            out = logits_of(ids[chunk].to(device), masks[chunk].to(device))
            loss = loss_of(out, wanted[chunk].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()
            total += loss.item()
            steps += 1
        accuracy, confusion = score(held)
        mark = ""
        if accuracy > best:
            best, best_at = accuracy, round_at + 1
            os.makedirs(OUTPUT, exist_ok=True)
            torch.save({"model": model.state_dict(), "labels": labels,
                        "backbone": backbone},
                       os.path.join(OUTPUT, f"niyet-{backbone}.pt"))
            mark = "  <- saklandı"
        print(f"  TUR {round_at+1}: kayıp {total/steps:.3f}  "
              f"SINAV %{accuracy*100:.1f}{mark}  ({time.time()-started:.0f} sn)",
              flush=True)

    print(f"\nen iyi: tur {best_at}, sınav %{best*100:.1f}")
    _, confusion = score(held)
    if confusion:
        print("  en çok karışanlar:")
        for (want, got), count in confusion.most_common(6):
            print(f"    {want} -> {got}  x{count}")
    with open(os.path.join(OUTPUT, "siniflar.json"), "w",
              encoding="utf-8") as out:
        json.dump(labels, out, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
