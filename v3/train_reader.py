"""Okuyucu ağının eğitimi — küçük ölçekte, hattı görmek için.

Bu betik iki şeyi birden öğretir çünkü ikisi aynı okumanın iki yüzüdür:

    işlem türü      cümle ne yapmak istiyor (PASS · ASK · WRITE)
    harf rolleri    hangi harfler özne, yüklem, değer

Eğitimden ÖNCE üç denetim var ve üçü de geçmeden eğitim başlamaz. Sebebi
ölçülmüş: bir gece, veri eksikken saatlerce koşan bir eğitim ve sonunda
kullanılamaz bir model. Denetimler ucuz, saatler pahalı.

    VERİ       beklenen bantta mı, iki sınıf da var mı
    HİZALAMA   etiketli örneklerde gerçekten rol işaretlenmiş mi
    EZBER      tek yığın 120 adımda ezberlenebiliyor mu — ezberleyemeyen
               hat bozuktur ve ona saat verilmez

En iyi tur saklanır, son tur değil: eski mimaride ölçüldü, kayıp düşerken
sınav %68,5'ten %56,3'e inmişti — ders ezberi. Sabır dolunca kendi durur.

Kullanım:
    python3 -m v3.train_reader [--sayi 20000] [--tur 8] [--boyut 192]
"""
import glob
import os
import random
import sys
import time

from v3 import dataset as ds

HELD_OUT = 800          # sınav — eğitimde hiç görülmeyen
LONGEST = 256


def encode(text, roles, letters, width):
    ids = [letters.get(ch, 0) for ch in text[:width]]
    tags = list(roles[:width])
    pad = width - len(ids)
    return ids + [0] * pad, tags + [-100] * pad


def main(argv):
    count = int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv else 20000
    rounds = int(argv[argv.index("--tur") + 1]) if "--tur" in argv else 8
    size = int(argv[argv.index("--boyut") + 1]) if "--boyut" in argv else 192
    out = argv[argv.index("--yaz") + 1] if "--yaz" in argv else "models/v3"

    import torch
    from torch import nn

    facts = sorted(glob.glob("data/train/facts/*.jsonl"))
    dialogue = sorted(glob.glob("data/raw/dialogue/*.gz"))
    rows = ds.build(facts, dialogue, most_facts=count,
                    most_dialogue=max(1, count // 3))

    # DENETİM 1 — veri
    kinds = {}
    for _, kind, _ in rows:
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"  {len(rows):,} örnek · dağılım {kinds}")
    assert len(rows) > 1000, "VERİ EKSİK"
    assert len(kinds) >= 2, "TEK SINIF — okuyucu ayrım öğrenemez"
    print("  DENETİM 1 GEÇTİ")

    # DENETİM 2 — hizalama
    marked = sum(1 for _, kind, roles in rows
                 if kind == ds.WRITE and ds.SUBJECT in roles)
    writes = kinds.get(ds.WRITE, 0)
    share = marked / max(writes, 1)
    print(f"  öznesi işaretli WRITE: {marked:,}/{writes:,}  %{share*100:.0f}")
    assert share > 0.9, "HİZALAMA BOZUK"
    print("  DENETİM 2 GEÇTİ")

    letters = ds.alphabet(rows)
    exam, rows = rows[:HELD_OUT], rows[HELD_OUT:]
    print(f"  {len(rows):,} eğitim · {len(exam):,} sınav · {len(letters)} harf")

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.token = nn.Embedding(len(letters) + 1, size)
            self.place = nn.Embedding(LONGEST + 1, size)
            block = nn.TransformerEncoderLayer(
                size, 8, size * 4, batch_first=True, norm_first=True,
                dropout=0.1)
            self.body = nn.TransformerEncoder(block, 4)
            self.final = nn.LayerNorm(size)
            self.kind_head = nn.Linear(size, 3)
            self.role_head = nn.Linear(size, 4)

        def forward(self, ids):
            steps = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            x = self.token(ids) + self.place(steps)
            x = self.final(self.body(x))
            return self.kind_head(x.mean(dim=1)), self.role_head(x)

    model = Net().to(device)
    print(f"  {sum(p.numel() for p in model.parameters())/1e6:.1f}M · {device}")

    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
    kind_loss = nn.CrossEntropyLoss()
    role_loss = nn.CrossEntropyLoss(ignore_index=-100)

    def batch_of(chunk):
        width = min(LONGEST, max(len(one[0]) for one in chunk))
        ids, tags, kinds_ = [], [], []
        for text, kind, roles in chunk:
            one, two = encode(text, roles, letters, width)
            ids.append(one)
            tags.append(two)
            kinds_.append(kind)
        return (torch.tensor(ids, device=device),
                torch.tensor(tags, device=device),
                torch.tensor(kinds_, device=device))

    # DENETİM 3 — ezber
    probe = rows[:32]
    ids, tags, kinds_ = batch_of(probe)
    first = last = None
    for step in range(120):
        kind_out, role_out = model(ids)
        loss = (kind_loss(kind_out, kinds_)
                + role_loss(role_out.reshape(-1, 4), tags.reshape(-1)))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if first is None:
            first = float(loss)
        last = float(loss)
    ok = last < first * 0.6
    print(f"  DENETİM 3: kayıp {first:.3f} -> {last:.3f}  "
          f"{'GEÇTİ' if ok else 'BAŞARISIZ'}")
    assert ok, "HAT BOZUK — tek yığın ezberlenemedi"

    model = Net().to(device)       # denetim ezberi eğitime taşınmasın
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)

    best, since, started = 0.0, 0, time.time()
    for turn in range(rounds):
        model.train()
        random.Random(turn).shuffle(rows)
        for at in range(0, len(rows), 32):
            chunk = rows[at:at + 32]
            ids, tags, kinds_ = batch_of(chunk)
            kind_out, role_out = model(ids)
            loss = (kind_loss(kind_out, kinds_)
                    + role_loss(role_out.reshape(-1, 4), tags.reshape(-1)))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        model.eval()
        right_kind = right_span = 0
        with torch.no_grad():
            for at in range(0, len(exam), 64):
                chunk = exam[at:at + 64]
                ids, tags, kinds_ = batch_of(chunk)
                kind_out, role_out = model(ids)
                guess_kind = kind_out.argmax(-1).tolist()
                guess_role = role_out.argmax(-1).tolist()
                for (text, kind, roles), gk, gr in zip(chunk, guess_kind,
                                                       guess_role):
                    if gk == kind:
                        right_kind += 1
                    if kind != ds.WRITE:
                        continue
                    from v3.reader import spans
                    want = spans(text, roles)
                    got = spans(text, gr[:len(text)])
                    if want[0] and got[0] == want[0]:
                        right_span += 1
        writes_in_exam = sum(1 for _, kind, _ in exam if kind == ds.WRITE)
        kind_score = right_kind / len(exam)
        span_score = right_span / max(writes_in_exam, 1)
        print(f"  tur {turn+1} · İŞLEM %{kind_score*100:.1f} · "
              f"ÖZNE %{span_score*100:.1f} · {time.time()-started:.0f} sn",
              flush=True)

        score = (kind_score + span_score) / 2
        if score >= best:
            best, since = score, 0
            os.makedirs(out, exist_ok=True)
            torch.save({"model": model.state_dict(), "letters": letters,
                        "size": size, "layers": 4, "heads": 8,
                        "width": LONGEST, "score": score,
                        "round": turn + 1},
                       os.path.join(out, "reader.pt"))
            print("     (en iyi, saklandı)", flush=True)
        else:
            since += 1
            if since >= 4:
                print("  SABIR doldu.")
                break

    print(f"\n  -> {out}/reader.pt  ({time.time()-started:.0f} sn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
