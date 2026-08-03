"""Gömme vektörlerini derlemden sayarak kurar ve modele yazar.

`lmm/vectors.py` yazılmıştı, ölçülmüştü ve kullanılmıyordu. Ölçüm 2.105
cümlelik oyuncak bir derlemle bile rastgelenin iki katını veriyor:

    sınıf      kelime   komşusu aynı sınıf   rastgele olsa
    nitelik       264              %24            %17
    kavram        550              %61            %34

Elimizde 808 MB derlem var ve hiç kullanılmadı.

Neden bu GPU'suz kalabiliyor: Levy ve Goldberg 2014'te word2vec'in aslında
kaydırılmış bir PMI matrisini örtük olarak çarpanlarına ayırdığını gösterdi.
Gradyan inişi anlamın geldiği yer hiç olmadı — sayarak ve doğrusal cebirle de
aynı geometriye varılıyor. Burada olan bu: sayım, PPMI ağırlığı, alt uzay
yinelemesi. Gradyan yok, rastgelelik yok, bağımlılık yok.

SINIR — MUTLAK. Dağılımsal benzerlik zıtları ayıramaz: "neden oldu" ile
"engelledi" aynı çevrede geçer, ve bu projede ölçüldü — 1288 olguda "sıcak"
ile "soğuk" birlikte geçiyor. Geometri ADAY bulur; kararı epistemik kapı
verir. Bir vektör hiçbir zaman bir olgu yazmamalı.

Kullanım:
    python3 scripts/gomme_kur.py [--veri data/tr-metin.txt] [--cümle 200000]
                                 [--boyut 96] [--yaz models/gomme.json]
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.intuition import tokenize                          # noqa: E402
from lmm.memory import Memory                               # noqa: E402
from lmm.vectors import Vectors                             # noqa: E402

# Vikipedi dökümünde tablo, kod ve listeler var; bunlar dağılım hakkında
# yanlış şey öğretir. Cümle olmayanı elemek için harf oranına bakılıyor —
# dile özel bir liste değil, bir sayım.
LETTERS = 0.75
SHORTEST, LONGEST = 30, 300


def sentences(path, count):
    """Derlemden cümle gibi duran satırlar."""
    found = []
    for line in open(path, encoding="utf-8"):
        text = line.strip()
        if not SHORTEST <= len(text) <= LONGEST:
            continue
        letters = sum(1 for ch in text if ch.isalpha() or ch == " ")
        if letters / len(text) < LETTERS:
            continue
        words = tokenize(text)
        if len(words) < 4:
            continue
        found.append(words)
        if len(found) >= count:
            break
    return found


def main(argv):
    path = (argv[argv.index("--veri") + 1] if "--veri" in argv
            else "data/tr-metin.txt")
    count = (int(argv[argv.index("--cümle") + 1]) if "--cümle" in argv
             else 200000)
    size = int(argv[argv.index("--boyut") + 1]) if "--boyut" in argv else 96
    out = (argv[argv.index("--yaz") + 1] if "--yaz" in argv
           else "models/gomme.json")
    if not os.path.exists(path):
        print(f"yok: {path}")
        return 1

    started = time.time()
    lines = sentences(path, count)
    print(f"{path} — {len(lines):,} cümle, {time.time() - started:.0f} sn")

    started = time.time()
    vectors = Vectors(dimensions=size)
    vectors.learn(lines)
    print(f"{len(vectors.vectors):,} kelime, {size} boyut, "
          f"{time.time() - started:.0f} sn  (GPU yok, gradyan yok)")

    # Kaba bir akıl sağlığı denetimi: grafın bildiği birkaç kavramın komşuları
    # anlamlı mı. Bu bir ölçüt değil, gözle bakma — asıl ölçüm
    # `scripts/gomme_olcum.py`.
    memory = Memory.load("models/graph/birlesik.lmm")
    known = set(memory.concepts())
    for name in ("kuş", "hastalık", "şehir", "ilaç"):
        if name in vectors.vectors:
            close = [word for word, _ in vectors.similar(name, 6)]
            print(f"    {name:<10} {close}")

    with open(out, "w", encoding="utf-8") as handle:
        json.dump(vectors.to_dict(), handle, ensure_ascii=False)
    print(f"\n  -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"  grafın kavramlarından {sum(1 for w in vectors.vectors if w in known):,}"
          f" tanesinin vektörü var")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
