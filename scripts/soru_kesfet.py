"""Soru ekinin çekimli hâllerini derlemden keşfeder.

İnsanlar soruyu çıplak ekle sormuyor. "bahseder misin", "anlatır mısınız",
"gider miyim" — soru eki kişi eki alıyor ve ayrıştırıcı bunları hiç
tanımıyordu. Ölçüldü: "bana penguenlerden bahseder misin" cümlesi dört
kavram olarak etiketleniyor, yani soru olduğu tamamen görünmez oluyordu.

Liste yazmak kolay olurdu ve yanlış olurdu — bu projede kural, dilin
makinesini keşfetmek. Burada da öyle yapılıyor, iki işareti birleştirerek:

    çekirdek     kelime bilinen bir soru ekiyle başlamalı (mı/mi/mu/mü —
                 kapalı sınıf, dört kelime, zaten bildirilmiş)
    dağılım      çekimli fiilden hemen sonra gelmeye meyilli olmalı, çünkü
                 soru ekinin işi tam olarak budur

Tek başına hiçbiri yetmiyor. Önek "milyon", "mustafa", "mümkün" gibi sıradan
kelimeleri de yakalıyor; fiil-sonrası dağılım ise "fakat", "çünkü", "ancak"
gibi bağlaçları. İkisi birlikte temiz ayırıyor — gerçek paradigma 11-45 kat,
gürültü 4 katın altında.

Çıktı bir liste değil bir ölçüm sonucu: satırında kaç kat ve kaç kez geçtiği
duruyor, derlem değişirse yeniden çıkarılır.

Kullanım: python3 scripts/soru_kesfet.py [--karakter 150000000]
"""
import collections
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import frequency                                   # noqa: E402
from lmm.intuition import QUESTION_PARTICLES                # noqa: E402
from lmm.lexicon import Lexicon                             # noqa: E402
from lmm.verbs import discover                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "data", "tr-web.txt")
OUTPUT = os.path.join(ROOT, "data", "tr-soru.txt")
CHARACTERS = 150_000_000
WORD = re.compile(r"[a-zçğıöşü]+")

# Ayrımın ölçüldüğü yer: gerçek paradigma 11-45 kat, en yakın gürültü 4,1.
# Eşik aradaki boşlukta duruyor, bir uca yaslanmıyor.
LEAST_RATIO = 10.0
LEAST_COUNT = 40
LONGEST = 9         # "mülkiyetinin" 22 kat alıyor ama soru eki değil, uzun


def main(argv):
    limit = (int(argv[argv.index("--karakter") + 1]) if "--karakter" in argv
             else CHARACTERS)
    if not os.path.exists(SOURCE):
        print(f"eksik: {SOURCE}")
        return 1
    words = frequency.counts()
    if not words:
        print("önce: python3 scripts/sıklık_çıkar.py")
        return 1

    lexicon = Lexicon()
    for infinitive, positive, negative in discover(words, minimum=20):
        lexicon.learn_verb(infinitive, positive, negative)
    print(f"{len(lexicon.verbs)} fiil derlemden", flush=True)

    after = collections.Counter()
    every = collections.Counter()
    positions = got = 0
    started = time.time()
    with open(SOURCE, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            tokens = WORD.findall(line.lower())
            every.update(tokens)
            for index in range(1, len(tokens)):
                if lexicon.knows(tokens[index - 1]):
                    after[tokens[index]] += 1
                    positions += 1
            got += len(line)
            if got >= limit:
                break
    total = sum(every.values())
    print(f"{positions:,} fiil-sonrası konum, {time.time()-started:.0f} sn",
          flush=True)

    found = []
    for word, count in after.items():
        if count < LEAST_COUNT or len(word) > LONGEST:
            continue
        if word in QUESTION_PARTICLES:
            continue
        if not any(word.startswith(seed) for seed in QUESTION_PARTICLES):
            continue
        expected = positions * every[word] / total
        ratio = count / expected if expected else 0.0
        if ratio >= LEAST_RATIO:
            found.append((ratio, count, word))

    found.sort(reverse=True)
    with open(OUTPUT, "w", encoding="utf-8") as out:
        for ratio, count, word in found:
            out.write(f"{word}\t{ratio:.1f}\t{count}\n")
    print(f"\n{len(found)} çekimli soru eki -> {OUTPUT}")
    for ratio, count, word in found:
        print(f"  {word:<11} {ratio:5.1f}x  ({count:,} kez)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
