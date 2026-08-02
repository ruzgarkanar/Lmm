"""Derleyici gerçekten okuyor mu? Aynı metin, iki yöntem, tek sayı.

Ölçülen şey "100 cümlede kaç doğrulanmış olgu". Karşılaştırma dürüst olsun diye
iki yöntem de aynı cümleleri görür ve derleyicinin kabul ettiği her olgu, kaynak
cümleye çapalanma denetiminden geçmiştir — yani sayı, modelin ürettiği değil,
denetimi geçen olgu sayısıdır.

Herkese açık metinle çalışır; gizli bir belge dışarı gönderilmez.

Kullanım: python3 scripts/derleyici_olcum.py [--cumle 100]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.compiler import (compile_text, model_reader, local_reader,  # noqa: E402
                          CompileError,
                          NOT_IN_DOCUMENT, NOT_IN_ANCHOR)
from lmm.memory import Memory                            # noqa: E402
from lmm.reading import read_text, sentences             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT = os.path.join(ROOT, "data", "tr-metin.txt")


def sample(count):
    """Ansiklopedinin ortasından düz nesir — tanım cümlesi avantajı olmasın."""
    found = []
    with open(TEXT, encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index < 400:             # ilk maddelerin açılışlarını atla
                continue
            for sentence in sentences(line):
                if 40 <= len(sentence) <= 220:
                    found.append(sentence)
                    if len(found) >= count:
                        return found
    return found


def existing_parser(chosen):
    memory = Memory()
    report = read_text(" ".join(s + "." for s in chosen), memory, "olcum.txt")
    return len(report.learned)


def main(argv):
    count = int(argv[argv.index("--cumle") + 1]) if "--cumle" in argv else 100
    chosen = sample(count)
    print(f"{len(chosen)} cümle, Wikipedia düz nesri (herkese açık)\n")

    ours = existing_parser(chosen)
    print(f"  MEVCUT AYRIŞTIRICI   {ours} olgu   "
          f"(100 cümlede {ours/len(chosen)*100:.1f})")

    document = " ".join(chosen)
    started = time.time()
    reader = (local_reader(argv[argv.index("--yerel") + 1])
              if "--yerel" in argv else model_reader())
    try:
        report = compile_text(chosen, document, reader)
    except CompileError as error:
        print(f"\n  DERLEYİCİ çalışmadı: {error}")
        return 1
    elapsed = time.time() - started
    accepted = len(report.accepted)
    print(f"  DERLEYİCİ            {accepted} olgu   "
          f"(100 cümlede {accepted/len(chosen)*100:.1f})   {elapsed:.0f} sn")
    print(f"\n  model {report.total} aday üretti, "
          f"{accepted} tanesi çapayı geçti "
          f"(%{accepted/max(report.total,1)*100:.0f})")
    for reason, n in sorted(report.reasons().items(), key=lambda x: -x[1]):
        print(f"    reddedildi: {reason} — {n}")
    print("\n  kabul edilenlerden örnekler:")
    for candidate in report.accepted[:10]:
        print(f"    {candidate.concept} | {candidate.relation} | {candidate.target}")
    if report.refused:
        print("\n  reddedilenlerden örnekler (çapası tutmayanlar):")
        for candidate, reason in report.refused[:5]:
            print(f"    {candidate}  <- {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
