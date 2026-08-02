"""Ansiklopedik tanımları grafa okur — kaynağın yapısını kullanarak.

Grafı otomatik büyütmenin altı yolu denendi ve altısı da çöktü
(`docs/DENEMELER.md`). Ortak sebep: ham metinde tanım ile mecaz, görüş, anlatı
birbirine karışıyor ve hiçbir sayım onları ayıramıyor.

Burada fark kaynakta: **ansiklopedik tanım kipi.** "Baklava, ... bir tatlıdır"
cümlesinin yapısı sabit ve o yapı bilgi taşıyor:

    konu başta      virgülden önceki ilk ad, yazının başlığıdır
    tür sonda       koşaçlı ad öbeği türü verir

Okuyucumuz özneyi cümlenin bütününden seçmeye çalışırken yanılıyordu —
"Alaçam, Çanakkale'nin bir köyüdür" cümlesinden `çanakkale` çıkarıyordu. Oysa
özne zaten belli: **kaynağın kendi düzeni söylüyor.** Onu okuyucuya sormak,
bilineni tahmin etmek olurdu.

Türün iyelik eki soyuluyor ("tatlısı" -> "tatlı") ve ek olup olmadığı yine
derleme sorularak karara bağlanıyor — `case_of`'un yıllardır yaptığı testin
aynısı: gövde kelimenin kendisi kadar sık geçiyorsa ek gerçektir.

Yer taslakları atılıyor: Türkçe Vikipedi'nin çoğu köy/mahalle taslağı ve
onlar bir bilgi grafı değil, bir yer sözlüğü kurar.

Kullanım:
    python3 scripts/tanim_derle.py [--sinir 100000] [--cikti tanimlar.lmm]
"""
import collections
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm import frequency                                   # noqa: E402
from lmm.frames import read, to_fact                        # noqa: E402
from lmm.intuition import _MORPHOLOGY, tokenize             # noqa: E402
from lmm.memory import Memory, Edge                         # noqa: E402
from lmm.verbs import discover, is_structural               # noqa: E402
import lmm.verbs                                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(ROOT, "data", "tr-tanimlar.txt")
OUTPUT = os.path.join(ROOT, "models", "graph", "tanimlar.lmm")
SOURCE_NAME = "vikipedi-tanım"

# Yer taslağı: Türkçe Vikipedi'nin en kalabalık yazı türü ve bilgi grafına
# hiçbir şey katmıyor — 35 bin kavramlı bir grafın %70'ini kaplayıp ilişki
# keşfini ölçülemez hâle getirmişti.
PLACE = re.compile(r"\b(köy|mahalle|belde|kasaba|ilçe|bucak|semt)\w*\b", re.I)
# Tanım kipi: yazının konusu başta durur. İlk yazışta düzenli ifade TEK
# kelimelik başlık bekliyordu ve kaynağın YARISINI sessizce atıyordu —
# "Basketbol ya da sepettopu, ...", "Sosyoloji veya toplum bilimi, ...",
# "Ferrari S.p.A., ...". Türkçe ansiklopedide başlıklar sık sık iki adlı ve
# adın içinde nokta olabiliyor.
#
# Sınır koymak kaçınılmaz ama SESSİZ sınır koymak değil: burada ne atıldığı
# sayılıyor ve raporda görünüyor.
TITLE = re.compile(
    r"^([A-ZÇĞİÖŞÜ][\wçğıöşüÇĞİÖŞÜ'’.]{1,30}"      # ilk ad (nokta olabilir)
    r"(?:\s+[\wçğıöşüÇĞİÖŞÜ'’.]{2,20}){0,2})"       # en çok iki kelime daha
    r"\s*(?:,|\(|\s+(?:ya\s+da|veya|ayrıca)\s)")

SHORTEST, LONGEST = 40, 300
POSSESSIVE = tuple(sorted(_MORPHOLOGY.possessive_suffixes, key=len,
                          reverse=True))


def bare(word, counts):
    """İyelik eki soyulmuş hâli: "tatlısı" -> "tatlı".

    Ek mi kelimenin parçası mı — derlem söylüyor. "kedi" kelimesinden "ked"
    çıkarmak yanlış olurdu ve sıklık testi tam bunu engelliyor.
    """
    for ending in POSSESSIVE:
        if not word.endswith(ending) or len(word) - len(ending) < 3:
            continue
        stem = word[: -len(ending)]
        if counts.get(stem, 0) >= counts.get(word, 0):
            return stem
    return word


class _SubjectFromSource:
    """Özne kaynaktan geldiği için okuyucunun özne testini gevşetir.

    Ölçüldü: okuyucunun reddettiklerinin %71,5'i tek bir kapıda düşüyor —
    `is_noun(subject, least=2)`, yani özne derlemde en az iki durum eki
    ailesiyle görülmüş olmalı. Nadir özel adlar bunu geçemiyor ve tertemiz
    cümleler kayboluyor:

        "Isparta, Türkiye'nin bir ilidir."       <- 5 kelime, düşüyor
        "Safranbolu, Karabük ilinin bir ilçesidir."

    Kapı genel metinde HAKLI: orada öznenin gerçekten özne olduğunu bilmiyoruz.
    Ama burada biliyoruz — ansiklopedik tanımda konu başta durur ve onu zaten
    başlıktan alıyoruz. Okuyucudan istediğimiz tek şey TÜR.

    Genel gevşetme denendi ve gürültü getirdi (`least=1` ile gelen olguların
    çoğu köy taslağı, bir kısmı yanlış öznelli). Gevşetme yalnız kaynağın
    yapısını bildiğimiz yerde meşru.
    """

    def __enter__(self):
        self.kept = lmm.verbs.is_noun
        lmm.verbs.is_noun = lambda word, seen, least=1: self.kept(word, seen,
                                                                  least=1)
        return self

    def __exit__(self, *_):
        lmm.verbs.is_noun = self.kept


def facts(path, counts, graded, lexicon, limit, verbs=()):
    """Tanım cümlelerinden (kavram, ilişki, hedef) — konu kaynaktan."""
    found, read_lines = [], 0
    with _SubjectFromSource(), open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not SHORTEST <= len(text) <= LONGEST or PLACE.search(text):
                continue
            title = TITLE.match(text)
            if not title:
                continue            # tanım kipinde değil
            read_lines += 1
            concept = title.group(1).lower().rstrip("'’")
            # Kapalı sınıf sözcük kavram olamaz. Bu denetim olmadan "Bir kase,
            # ..." cümlesinden `bir` kavramı doğuyordu ve bir belirtecin kavram
            # olması "X bir Y" kalıplarının hepsini bozuyor — sohbet sınavı
            # büyümeyle 37/40'tan 36/40'a düşmüştü ve sebebi buydu.
            #
            # Kirlilik sessiz: graf büyürken kimse fark etmiyor, ayrıştırıcı
            # aylar sonra bozuluyor.
            head = concept.split()[-1] if " " in concept else concept
            # Durum eki taşıyan bir baş, cümlenin ÖZNESİ değil ÇERÇEVESİDİR:
            # "Matematikte, bir grup ... yapıdır" cümlesinde konu `grup`,
            # `matematikte` yalnız hangi alandan söz edildiğini söylüyor.
            # Kalıp cümle başındaki büyük harfli kelimeyi başlık sandığı için
            # zarf özne yazılıyordu. Ölçüldü: üretim grafında 28 böyle düğüm,
            # 48 olgu — ve her birinde GERÇEK özne kayıp. Oran küçük (%0.3)
            # ama ölçekle çarpılıyor: 16 milyon olguda 48 bin çöp düğüm.
            #
            # Sınama sözcük listesiyle değil morfolojiyle: `is_oblique` bu
            # dosyada değil dilin kendi organında yaşıyor ve yorumu zaten bu
            # kusuru anlatıyor ("bir bulunma eki ad öbeğine sızdı").
            if _MORPHOLOGY.is_oblique(head):
                continue
            if is_structural(head, counts) or is_structural(concept, counts):
                continue
            # Fiil mastarı da kavram olamaz. "demek" yapısal sözcük sayılmıyor
            # (sıralaması düşük) ama bir fiildir ve kavram olunca "kuş ne
            # demek" sorusunu bozuyordu — cümlenin dilbilgisi parçası, konusu
            # değil.
            if head in verbs or lexicon.knows(head):
                continue
            frame = read(tokenize(text.split(".")[0]), counts, lexicon)
            if frame is None:
                continue
            fact = to_fact(frame, lexicon, counts, graded)
            if fact is None:
                continue
            # Özne KAYNAKTAN geliyor, okuyucudan değil: ansiklopedik tanımda
            # konu başta durur ve bunu tahmin etmek bilineni tahmin etmektir.
            found.append((concept, fact[1], bare(fact[2], counts)))
            if read_lines >= limit:
                break
    return found, read_lines


def main(argv):
    limit = (int(argv[argv.index("--sinir") + 1]) if "--sinir" in argv
             else 100000)
    output = argv[argv.index("--cikti") + 1] if "--cikti" in argv else OUTPUT
    if not os.path.exists(SOURCE):
        print(f"eksik: {SOURCE}")
        return 1

    counts = frequency.counts()
    graded = frequency.grades()
    memory = Memory()
    verbs = set()
    for infinitive, positive, negative in discover(counts, minimum=20):
        memory.learn_word(infinitive, positive, negative)
        verbs.add(infinitive)
    print(f"{len(memory.vocabulary)} fiil, {len(counts):,} kelime sayacı")

    started = time.time()
    found, read_lines = facts(SOURCE, counts, graded, memory.lexicon,
                              limit, verbs)
    print(f"{read_lines:,} tanım okundu, {len(found):,} olgu "
          f"({time.time()-started:.0f} sn)")

    verbs = {i for i, _ in memory.lexicon.verbs.values()}
    written = clashed = dropped = 0
    for concept, relation, target in found:
        if len(concept) < 3 or len(target) < 3 or concept == target:
            dropped += 1
            continue
        # HEDEF de kapalı sınıf olamaz. Okuyucu gerçek türü bulamadığında
        # elindeki son kelimeyi veriyor ve "homer jay simpson --type--> bir"
        # gibi kayıtlar doğuyor. Bir belirteç ya da bağlaç, hiçbir şeyin türü
        # değildir; bunları yazmak grafı sessizce çürütür.
        if is_structural(target, counts) or target in verbs:
            dropped += 1
            continue
        try:
            memory.write(Edge(concept, relation, target, source=SOURCE_NAME))
            written += 1
        except Exception:                                   # noqa: BLE001
            clashed += 1
    memory.save(output)
    kinds = collections.Counter(r for _, r, _ in found)
    print(f"\n{written:,} olgu yazıldı, {clashed:,} çakıştı, "
          f"{dropped:,} elendi (yapısal sözcük ya da çok kısa)")
    print(f"  {len(memory.concepts()):,} kavram")
    print(f"  ilişkiler: {dict(kinds.most_common(5))}")
    print(f"  100 tanımda {written/max(read_lines,1)*100:.1f} olgu")
    print(f"  -> {output}  ({os.path.getsize(output)/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
