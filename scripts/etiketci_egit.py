"""Kavram ve hedefi BULMAYI öğretir — ek soymadan, kalıpsız, listesiz.

Bugünkü okuyucu ikiye bölünmüş durumda: eğitilmiş ağ niyeti söylüyor, ama
cümlenin hangi parçasının kavram olduğunu kurallı ayrıştırıcı buluyor. O
ayrıştırıcı `lmm/turkish.py`'deki 416 elle yazılmış ögeyi okuyor ve o dosyanın
yaşamasının tek sebebi bu — ölçüldü, listelerin hepsi çalışma sırasında yoğun
biçimde okunuyor (`oblique_suffixes` tek sınavda 118.685 kez).

Bir dil modelinde o dosyanın karşılığı yok. Burada da olmaması için, bulma
işinin öğrenilmesi gerekiyor.

    girdi    "Kartal, yırtıcı bir kuş türüdür."
    çıktı     KKKKKK·············HHH··········     harf başına etiket
    okunan   kavram=Kartal · hedef=kuş

HARF DÜZEYİNDE. Kelime ya da alt-kelime parçalama yok: o da yazılması ya da
ayrıca öğrenilmesi gereken bir şey olurdu. Türkçe eklemeli ve ekler harf
dizileri — harf düzeyi hem en saf hem bu dile en uygunu. Sözlük, derlemin
kendi harflerinden çıkıyor; hiçbir yerde bildirilmiyor.

ÖLÇÜT KURALLI AYRIŞTIRICI. Bu ağ ancak onu geçerse `turkish.py` silinebilir.
Ölçüm aynı 200 gerçek soruda, yan yana.

Kullanım:
    python3 scripts/etiketci_egit.py <okuma.jsonl> [...] [--tur 3] [--boyut 256]
"""
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tagger import rows_from, spans_from, OUT, CONCEPT, TARGET  # noqa: E402

# Bağlam. 2048 harf — bugünkü verinin ortancası 78 ve en uzunu 160, yani şu an
# fazlasıyla geniş. Bilerek: uzun cümleyi ANLAMAK için mimarinin buna hazır
# olması gerekiyor ve konum kodlaması göreli (RoPE), yani genişletmek sonradan
# yeniden eğitim istemiyor. Dolgu maliyeti yığın içinde en uzun örneğe göre
# kesiliyor, sabit 2048 değil.
LONGEST = 2048
HELD_OUT = 3000         # sınav için ayrılan, eğitimde hiç görülmeyen


def alphabet_of(rows):
    """Derlemin kendi harfleri. Hiçbir yerde bildirilmiyor, sayılıyor."""
    seen = set()
    for sentence, _, _ in rows:
        seen.update(sentence)
    return {letter: at + 1 for at, letter in enumerate(sorted(seen))}


def encode(sentence, marks, letters, width=None):
    """Harf kimlikleri ve etiketler, YIĞININ en uzununa göre dolgulanmış.

    Sabit 2048'e dolgulamak işlemin çoğunu boşluğa harcıyordu: ortanca cümle
    78 harf, yani %96'sı dolgu. Yığın başına kesmek aynı sonucu veriyor.
    """
    width = width or LONGEST
    ids = [letters.get(ch, 0) for ch in sentence[:width]]
    tags = list(marks[:width])
    pad = width - len(ids)
    return ids + [0] * pad, tags + [-100] * pad, len(ids)


def main(argv):
    paths = [a for a in argv[1:] if a.endswith(".jsonl")]
    if not paths:
        print(__doc__.strip().splitlines()[-1])
        return 1
    rounds = int(argv[argv.index("--tur") + 1]) if "--tur" in argv else 12
    width = int(argv[argv.index("--boyut") + 1]) if "--boyut" in argv else 256
    out = (argv[argv.index("--yaz") + 1] if "--yaz" in argv
           else "models/etiketci")

    import torch
    from torch import nn
    from core.model import Config, Core

    rows = rows_from(paths)
    random.Random(7).shuffle(rows)
    exam, rows = rows[:HELD_OUT], rows[HELD_OUT:]
    # En iyi tur saklanıyor, son tur değil. Bu projede bir kez ölçüldü:
    # 4.850 örnekle dördüncü turda eğitim kaybı düşerken sınav %68,5'ten
    # %56,3'e indi — ders ezberi. Son turu saklamak o düşüşü kaydeder.
    best = 0.0
    letters = alphabet_of(rows)
    print(f"  {len(rows):,} eğitim · {len(exam):,} sınav · "
          f"{len(letters)} harf (sayıldı, bildirilmedi)")

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    config = Config(vocabulary=len(letters) + 1, dimensions=width,
                    layers=4, heads=8, context=LONGEST)

    class Tagger(nn.Module):
        """Çekirdek + harf başına üç sınıf. Yeni mimari yok, var olanın başlığı."""

        def __init__(self, config):
            super().__init__()
            self.core = Core(config)
            self.head = nn.Linear(config.dimensions, 3)

        def forward(self, ids):
            x = self.core.token(ids)
            for block in self.core.blocks:
                # ÇİFT YÖNLÜ: bir cümleyi anlamak için sondaki eki görmek
                # gerekiyor. Üretimde nedensellik şart, anlamada tersi.
                x = block(x, causal=False)
            return self.head(self.core.final(x))

    model = Tagger(config).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"  {total / 1e6:.1f}M parametre · {device}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_of = nn.CrossEntropyLoss(ignore_index=-100)
    batch = 64
    started = time.time()
    for turn in range(rounds):
        model.train()
        random.Random(turn).shuffle(rows)
        running, seen = 0.0, 0
        for at in range(0, len(rows), batch):
            chunk = rows[at:at + batch]
            width = min(LONGEST, max(len(s) for s, _, _ in chunk))
            ids, tags = [], []
            for sentence, marks, _ in chunk:
                one, two, _ = encode(sentence, marks, letters, width)
                ids.append(one)
                tags.append(two)
            ids = torch.tensor(ids, device=device)
            tags = torch.tensor(tags, device=device)
            logits = model(ids)
            loss = loss_of(logits.reshape(-1, 3), tags.reshape(-1))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running += float(loss)
            seen += 1
            if seen % 400 == 0:
                done = at + batch
                print(f"    tur {turn + 1} · {done:,}/{len(rows):,} · "
                      f"kayıp {running / seen:.4f} · "
                      f"{time.time() - started:.0f} sn", flush=True)
        # Sınav: etiket doğruluğu değil, ÇIKARILAN OLGU doğruluğu.
        model.eval()
        right = both = 0
        with torch.no_grad():
            for at in range(0, len(exam), 128):
                chunk = exam[at:at + 128]
                width = min(LONGEST, max(len(s) for s, _, _ in chunk))
                ids = torch.tensor([encode(s, m, letters, width)[0]
                                    for s, m, _ in chunk], device=device)
                guess = model(ids).argmax(-1).tolist()
                for (sentence, marks, _), line in zip(chunk, guess):
                    want_c, want_t = spans_from(sentence, marks)
                    got_c, got_t = spans_from(sentence, line[:len(sentence)])
                    if got_c.lower() == want_c.lower():
                        right += 1
                    if (got_c.lower() == want_c.lower()
                            and got_t.lower() == want_t.lower()):
                        both += 1
        print(f"  tur {turn + 1} · kayıp {running / max(seen, 1):.4f} · "
              f"KAVRAM %{right / len(exam) * 100:.1f} · "
              f"KAVRAM+HEDEF %{both / len(exam) * 100:.1f}", flush=True)
        score = both / len(exam)
        if score >= best:
            best = score
            os.makedirs(out, exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "config": config.to_dict(),
                        "letters": letters, "tur": turn + 1, "isabet": score},
                       os.path.join(out, "etiketci.pt"))
            print("      (en iyi, saklandı)", flush=True)
    print(f"\n  -> {out}/etiketci.pt  ({time.time() - started:.0f} sn)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
