"""Kelime sıklık tablosu — derlemden bir kez çıkarılır, sonra hep yüklenir.

`case_of` durum ekini kurala göre çözmüyor, **derleme sorarak** çözüyor: "banka"
mı yoksa "bank"a mı — kök kelimenin kendisi kadar sık geçiyorsa ek gerçektir.
Bu, projenin en iyi işleyen fikirlerinden biri ve bugüne kadar yalnız toplu
okuma betiklerinde vardı: her çalıştırmada derlemin başından 300 milyon karakter
yeniden sayılıyordu.

Çalışma anında ise sayaç hiç yoktu. Sonucu ölçüldü ve ağırdı: soruşturma
"Penguenler güney yarım kürede yaşar" cümlesinin öznesini `kürede` sanıyordu,
çünkü tek bir yazının içinde "küre" hiç geçmiyor ve bulunma eki tanınamıyor.

Tablo bir model değil, bir sözlük: satır satır okunabilir, kimse eğitmiyor,
kimse ayarlamıyor. Sayılar derlemden geliyor ve derlem değişirse yeniden
sayılır.

Kullanım: python3 scripts/sıklık_çıkar.py [--karakter 3000000000] [--en-az 10]
"""
import collections
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "data", "tr-web.txt")
OUTPUT = os.path.join(ROOT, "data", "tr-siklik.txt")
CHARACTERS = 3_000_000_000
LEAST = 10
WORD = re.compile(r"[a-zçğıöşü]+")


def main(argv):
    limit = (int(argv[argv.index("--karakter") + 1]) if "--karakter" in argv
             else CHARACTERS)
    least = int(argv[argv.index("--en-az") + 1]) if "--en-az" in argv else LEAST
    if not os.path.exists(SOURCE):
        print(f"eksik: {SOURCE}")
        return 1

    words = collections.Counter()
    got = 0
    started = time.time()
    with open(SOURCE, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            words.update(WORD.findall(line.lower()))
            got += len(line)
            if got >= limit:
                break
            if got % (200_000_000) < len(line):
                print(f"  {got/1e9:.1f}G karakter, {len(words)/1e6:.1f}M kelime,"
                      f" {time.time()-started:.0f} sn", flush=True)

    kept = [(n, w) for w, n in words.items() if n >= least]
    kept.sort(reverse=True)
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for count, word in kept:
            out.write(f"{word}\t{count}\n")
    print(f"\n{len(words)} ayrı kelime görüldü, {len(kept)} tanesi >= {least}")
    print(f"  {OUTPUT}  {os.path.getsize(OUTPUT)/2**20:.1f} MB"
          f"  ({(time.time()-started)/60:.0f} dk)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
