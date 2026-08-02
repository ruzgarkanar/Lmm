"""Derlemden çıkarılan fiiller — çalışma anında yüklenen liste.

Fiil keşfi ölçülmüş bir yöntem: bir gövde hem olumlu hem olumsuz geniş zamanda
geçiyorsa fiildir ("uçar" ve "uçamaz"). Elle yazılmış liste yok, sayım var.

Ama keşif her oturumda yeniden yapılamaz — sıklık tablosunu taramak saniyeler
sürüyor. Sonucu dosyaya almak, sıklık tablosunda yaptığımızın aynısı.

Neye yarıyor: söyleyiş öğrenirken okumanın cümleye uyup uymadığını denetlemek.
Oturumun kendi sözlüğü yalnızca ÖĞRETİLMİŞ fiilleri biliyor (206 tane) ve
"gerekir" gibi bir kelimeyi tanımıyor — o yüzden denetim boşa çıkıyordu.

Kullanım: python3 scripts/fiil_cikar.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import frequency                                   # noqa: E402
from lmm.verbs import discover                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(ROOT, "data", "tr-fiiller.txt")
LEAST = 20


def main(argv):
    least = int(argv[argv.index("--en-az") + 1]) if "--en-az" in argv else LEAST
    words = frequency.counts()
    if not words:
        print("önce: python3 scripts/sıklık_çıkar.py")
        return 1
    found = list(discover(words, minimum=least))
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for infinitive, positive, negative in sorted(found):
            out.write(f"{infinitive}\t{positive}\t{negative}\n")
    print(f"{len(found)} fiil -> {OUTPUT} "
          f"({os.path.getsize(OUTPUT)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
