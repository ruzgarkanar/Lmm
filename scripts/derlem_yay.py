"""Derlemin TAMAMINA yayılmış cümle seçimi.

`derlem_sec.py` dosyanın başından okuyor ve yeterince cümle bulunca duruyor.
Ölçüldü ve seçim tek konuya çakıldı: 1.500 cümlenin hepsi Cengiz Han ve
Moğolistan'dı, çünkü ilk 15.633 satır o belgeydi. Bir derlemden öğrenmek,
derlemin BAŞINDAN öğrenmek değildir.

Adımlı tarama bunu düzeltiyor: her N satırdan biri okunuyor, yani seçim
derlemin her yerinden geliyor. Süzgeçler `derlem_sec`ten alınıyor —
çoğaltılmıyor, çünkü iki kopya er geç ayrışır.

Kullanım:
    python3 scripts/derlem_yay.py --sayi 10000 [--veri data/tr-metin.txt]
                                  [--adım 0] [--yaz secilen.txt]
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import frequency                                   # noqa: E402
from lmm.intuition import tokenize                          # noqa: E402
from lmm.turkish import TurkishMorphology                   # noqa: E402
from lmm.verbs import is_structural                         # noqa: E402


# Örneklenen kaç satırdan biri cümle veriyor — ölçüldü, tahmin değil.
YIELD = 6


def _selector():
    """`derlem_sec` süzgeçleri — kopyalanmadan, olduğu yerden."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "derlem_sec", os.path.join(here, "derlem_sec.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pick(path, want, stride, seen_sentences=()):
    select = _selector()
    counts, verbs = frequency.counts(), frequency.verbs()
    morphology = TurkishMorphology()
    found, seen, skipped = [], 0, set(seen_sentences)
    for line in open(path, encoding="utf-8", errors="ignore"):
        seen += 1
        if seen % stride:
            continue
        for text in select._sentences(line):
            if text in skipped or not select.shaped(text):
                continue
            opening = select.OPENING.match(text)
            if not opening:
                continue
            head = opening.group(1).lower().split()[-1]
            if is_structural(head, counts) or head in verbs:
                continue
            words = tokenize(text)
            if not any(select._predicative(word, morphology, verbs)
                       for word in words[1:]):
                continue
            found.append(text)
            skipped.add(text)
            break                       # belge başına bir cümle: çeşitlilik
        if len(found) >= want:
            break
    return found, seen


def main(argv):
    path = (argv[argv.index("--veri") + 1] if "--veri" in argv
            else "data/tr-metin.txt")
    want = int(argv[argv.index("--sayi") + 1]) if "--sayi" in argv else 10000
    out = argv[argv.index("--yaz") + 1] if "--yaz" in argv else None
    if not os.path.exists(path):
        print(f"yok: {path}")
        return 1
    # Adım, derlemin boyutundan VE ölçülen verimden hesaplanıyor: istenen
    # sayı ne olursa olsun tarama dosyanın SONUNA kadar yayılsın. Sabit adım,
    # küçük istekte dosyanın başında kalırdı.
    #
    # Katsayı ölçüm: adım 83 ile 30.240 satır örneklendi ve 5.220 cümle
    # çıktı — örneklenen satırların ancak altıda biri süzgeçleri geçiyor.
    # Üçle çarpmak yetmiyordu, istenenin yarısı geliyordu.
    total = sum(1 for _ in open(path, encoding="utf-8", errors="ignore"))
    stride = (int(argv[argv.index("--adım") + 1]) if "--adım" in argv
              else max(1, total // max(want * YIELD, 1)))
    found, seen = pick(path, want, stride)
    print(f"{path} — {total:,} satır, adım {stride}")
    print(f"  {seen:,} satır tarandı -> {len(found):,} cümle")
    for text in found[:5]:
        print(f"    · {text[:96]}")
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            handle.write("\n".join(found))
        print(f"\n  -> {out}")
    else:
        print("\n  (yazmak için --yaz ekle)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
