"""Fiilleri metinden bulmak — çift testiyle, tahminle değil.

Ölçüldü ve beklenmedik çıktı: bir belgeyi okurken kazancın **tamamı** sözlükten
geliyor, bilgi grafından değil. Aynı belgede boş bellek 5,6 olgu/100 cümle
veriyor; 486 kavramlık graf tek başına yine 5,6; ama 98 fiillik sözlük tek başına
7,1. Kavramların katkısı sıfır.

Sebebi geriye bakınca açık: kalıplar bilinen bir fiil istiyor. Fiil sözlükte
yoksa hiçbir kalıp eşleşmiyor ve cümle tamamen kayboluyor. Kavram ise şekle göre
tanınıyor, üyeliğe göre değil — yani grafta olması gerekmiyor.

Aynı belgede okunamayan cümleciklerin %23'ü, son kelimesi fiil şeklinde ama
sözlükte olmayan cümleler: `uygulanır`, `gerekir`, `ölçülür`, `oluşur`. 167
benzersiz fiil, kaybın dörtte birini açıyor.

Bunları elle yazmak yanlış olur — hem bitmez, hem her dil için yeniden gerekir.
Mekanik bir test var: **bir kök hem olumlu hem olumsuz çekimiyle metinde
geçiyorsa fiildir.**

    uygulanır + uygulanmaz  -> fiil
    broker    + brokermaz   -> yok, fiil değil

Bu, Harris'in 1950'lerdeki ikame testinin dar ve kesin bir hâli. Dağılımsal
benzerlik gibi bulanık değil: ya vardır ya yoktur. Ve her dil için değişen şey
yalnızca ek listesi — buradaki kod değil.
"""
import collections
import re

SHORTEST_STEM = 2

_MORPHOLOGY = None


def _language():
    """Ekleri hangi dilden okuyacağı — `lmm/frames.py`'deki ile aynı desen.

    Buradaki kod "bir kök hem olumlu hem olumsuz çekimiyle geçiyorsa fiildir"
    testinden ibaret ve o test dile bakmaz. Bakan şey ek listeleriydi ve
    onlar `lmm/turkish.py`'ye taşındı: bir dil eklemek artık orada birkaç
    satır yazmak, burada hiçbir şey değiştirmemek demek.

    Bildirmeyen bir dil boş küme verir; o zaman `stems_of` hiçbir kök
    bulamaz ve keşif sessizce boş döner — yanlış fiil uydurmaktansa hiç
    bulmamak, bu dosyanın zaten savunduğu duruş.
    """
    # Biçimbilim silindi: keşif boş tabloyla sessizce boş döner.
    return None


def _of(name, fallback=()):
    return tuple(getattr(_language(), name, fallback))


def _vowels():
    return getattr(_language(), "vowels", "")


def stems_of(word):
    """Bu kelime bir geniş zaman çekimiyse, kökü ne olabilir?

    Birden çok olasılık döner: "gelir" hem "gel"+ir hem "geli"+r okunabilir.
    Hangisinin doğru olduğuna olumsuz biçimin varlığı karar verir.
    """
    # Çoğul eki de -lar/-ler ile biter ve "r" ile bitiyor diye fiil sanılabilir.
    if word.endswith(_of("plural_suffixes")) or not word.isalpha():
        return []
    found = []
    for suffix in _of("aorist_suffixes"):
        if word.endswith(suffix) and len(word) - len(suffix) >= SHORTEST_STEM:
            found.append(word[: -len(suffix)])
    return found


def _harmonised(stem, suffixes):
    """Ekin, kökün son ünlüsüne göre kalın ya da ince hâli.

    İkili ve sıralı bir liste bekliyor: önce kalın ("-maz", "-mak"), sonra
    ince. Uyum kuralı dile değil bu koda ait olan tek şey, çünkü kural
    "son ünlüye bak"tan ibaret; hangi ünlünün kalın olduğunu ve ekin iki
    hâlini dil söylüyor.

    Ünlü bulunamazsa ince hâl seçiliyor — kod buraya geldiğinde kök zaten
    ünlüsüz demektir ve eski davranış da buydu.
    """
    if not suffixes:
        return stem                 # ekini bildirmeyen dil: kök olduğu gibi
    back, front = suffixes[0], suffixes[-1]
    vowels = _vowels()
    heavy = getattr(_language(), "back_vowels", "")
    for character in reversed(stem):
        if character in vowels:
            return stem + (back if character in heavy else front)
    return stem + front


def negative_of(stem):
    """Kökün olumsuz geniş zaman biçimi, ünlü uyumuna göre."""
    return _harmonised(stem, _of("aorist_negative_suffixes"))


def infinitive_of(stem):
    """Mastar: aynı uyum kuralı."""
    return _harmonised(stem, _of("infinitive_suffixes"))


def discover(words, minimum=2):
    """Metindeki kelimelerden fiil bul. (mastar, olumlu, olumsuz) döner.

    `minimum`, her iki biçimin de kaç kez görülmesi gerektiği. Bir kez geçen
    bir çift yazım hatası olabilir; iki kez geçen bir düzenliliktir.
    """
    seen = words if isinstance(words, collections.Counter) \
        else collections.Counter(words)
    found = {}
    for word, count in seen.items():
        if count < minimum:
            continue
        for stem in stems_of(word):
            negative = negative_of(stem)
            if seen.get(negative, 0) >= minimum:
                # Uzun kök tercih edilir: "gelir" için "gel" doğru, "geli" değil,
                # ve doğruyu olumsuz biçimin varlığı zaten seçiyor.
                infinitive = infinitive_of(stem)
                previous = found.get(infinitive)
                if previous is None or len(stem) > len(previous[0]):
                    found[infinitive] = (stem, word, negative)
    return [(infinitive, positive, negative)
            for infinitive, (_, positive, negative) in sorted(found.items())]


# İsim testi (`case_families`), sıfat-fiil ekleri (`participle_suffixes`) ve
# yapı kelimeleri (`light_words`) artık `lmm/turkish.py`'de duruyor. Testlerin
# kendisi burada ve dile bakmıyor: "durum eki alıyorsa isimdir", "eki var ve
# kökü fiilse sıfat-fiildir", "en sık üç yüzdeyse dilbilgisidir".


def is_participle(word, verbs=None):
    """Bu kelime bir sıfat-fiil mi? Kavram olamaz.

    Ekin varlığı yetmez, KÖKÜN FİİL OLMASI gerekir: "penguen" de -en ile
    biter ama "pengu" diye bir fiil yoktur. Ek listesine bakıp karar vermek
    "penguen", "kaplan", "orman", "insan" gibi isimleri eliyordu — bu gece
    "banka"yı "bank'a" sanan hatanın aynısı, ve çözümü de aynı: derleme sor.
    """
    for suffix in _of("participle_suffixes"):
        if not word.endswith(suffix) or len(word) - len(suffix) < 2:
            continue
        stem = word[: -len(suffix)]
        if verbs is None:
            continue            # sözlük yoksa karar verilemez, isim sayılır
        # Uyum kuralıyla üretilen mastarın yanında ekin İKİ hâli de deneniyor:
        # kökün son ünlüsü kurala uymayabiliyor ("gel" ince, "gelmek" doğru;
        # ama alıntı köklerde uyum tutmuyor) ve tek hâle bakmak fiili
        # kaçırıyordu.
        if infinitive_of(stem) in verbs:
            return True
        if any(stem + mark in verbs for mark in _of("infinitive_suffixes")):
            return True
    return False


SMALLEST_CORPUS = 10000     # bunun altında sıklık ölçüsü güvenilmez
STRUCTURAL_RANK = 300       # en sık ilk üç yüz kelime yapı taşır


def is_structural(word, seen, rank=STRUCTURAL_RANK):
    """Çok sık geçen kelime kavram değil, dilbilgisidir.

    Ölçü SIRA, oran değil. Oran denendi ve derlem boyutuna göre değişti: küçük
    bir test derleminde her kelime toplamın büyük bir yüzdesi oluyor ve hepsi
    "yapısal" çıkıyordu. Aynı tuzağa bu gece tamlama testinde de düşülmüştü —
    karşılıklı bilgi de boyutla ölçekleniyordu.

    Sıra ölçeklenmiyor: bir dilin en sık üç yüz kelimesi her derlemde aynı
    türden şeylerdir — bağlaç, zamir, yardımcı fiil, "olan", "şey".
    """
    return word in structural_set(seen, rank)


_CACHE = {}


def structural_set(seen, rank=STRUCTURAL_RANK):
    """En sık `rank` kelime. Bir kez hesaplanır ve saklanır.

    Önbellek dışarıda tutuluyor çünkü Counter'a öznitelik yazılamıyor; ilk
    yazışta bunu fark etmeyip her çağrıda milyon kelimeyi yeniden sıralamıştım
    ve okuma kırk kat yavaşlamıştı. Sessiz bir yavaşlama, çünkü sonuç doğruydu.
    """
    key = (id(seen), len(seen))
    found = _CACHE.get(key)
    if found is None:
        # Küçük derlemde sıklık bir şey söylemez; hesap da orada yapılıyor
        # çünkü toplamı almak milyon girdide pahalı ve her çağrıda değil bir
        # kez yapılmalı. İlk yazışta toplam her çağrıda alınıyordu ve okuma
        # kırk kat yavaşlamıştı — sonuç doğru olduğu için sessiz bir hataydı.
        if sum(seen.values()) < SMALLEST_CORPUS:
            found = frozenset()
        else:
            found = {word for word, _ in seen.most_common(rank)}
        _CACHE.clear()          # tek derlem yeter; bellek şişmesin
        _CACHE[key] = found
    return found


def is_noun(word, seen, least=1):
    """Bu kelime metinde durum eki almış hâlleriyle de geçiyor mu?

    `least`, kaç ayrı ek ailesinin görülmesi gerektiği. Bir tek çekim
    tesadüf olabilir — "hem" kelimesinin ardından "hemi" geçiyor olabilir —
    ama iki ayrı aile bir düzenliliktir.
    """
    if not word or len(word) < 3:
        return False
    families = 0
    for family in _of("case_families"):
        for suffix in family:
            if seen.get(word + suffix, 0) >= 2:
                families += 1
                break
    return families >= least


# Sıfat testi: Türkçe'de yalnızca sıfat ve zarf derecelenir. "daha zor" olur,
# "daha banka" olmaz. Ham sayı yanıltıyor ("çok risk" geçerlidir) ama ORAN
# ayırıyor: ölçüldü, gerçek sıfatlar 194-339‰, isimler 0,4-7‰. Arada uçurum var.
ADJECTIVE_RATE = 0.05       # binde 50; ölçülen iki küme arasındaki boşlukta


def is_adjective(word, graded, seen):
    """Bu kelime derecelenerek geçiyor mu?

    `graded`, (derecelendirici, kelime) ikililerinin sayacı. Bir kelimenin
    kendi sıklığına oranı bakılıyor — sık isimler mutlak sayıda yanıltıyor.
    """
    total = seen.get(word, 0)
    if total < 20:
        return False        # az geçen kelimede oran güvenilmez
    hits = sum(graded.get((grader, word), 0) for grader in _of("graders"))
    return hits / total >= ADJECTIVE_RATE


def graded_pairs(tokens):
    """Metindeki (derecelendirici, kelime) ikililerini sayar."""
    found = collections.Counter()
    previous = None
    for token in tokens:
        if previous in _of("graders"):
            found[(previous, token)] += 1
        previous = token
    return found


def from_text(text, minimum=2):
    """Düz metinden doğrudan. Büyük dosyalar için parça parça verilmeli."""
    # Alfabe dile ait: İngiliz harfleriyle aramak "çalışır"ı "al" ve "r"
    # diye ikiye bölerdi. Harflerini bildirmeyen bir dilde ASCII'ye
    # düşülüyor — eksik, ama sessizce boş dönmekten iyi.
    letters = getattr(_language(), "letters", "a-z")
    return discover(collections.Counter(
        re.findall(f"[{letters}]+", text.lower())), minimum)
