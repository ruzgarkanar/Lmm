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

# Türkçe geniş zaman: olumlu -r/-ar/-er/-ır/-ir/-ur/-ür, olumsuz -maz/-mez.
# Bir dil eklemek bu iki listeyi yazmaktır, kodu değiştirmek değil.
POSITIVE = ("ar", "er", "ır", "ir", "ur", "ür", "r")
NEGATIVE = ("maz", "mez")
BACK = "aıou"

# Çoğul eki de -lar/-ler ile biter ve "r" ile bitiyor diye fiil sanılabilir.
NOT_VERBS = ("lar", "ler")
SHORTEST_STEM = 2


def stems_of(word):
    """Bu kelime bir geniş zaman çekimiyse, kökü ne olabilir?

    Birden çok olasılık döner: "gelir" hem "gel"+ir hem "geli"+r okunabilir.
    Hangisinin doğru olduğuna olumsuz biçimin varlığı karar verir.
    """
    if word.endswith(NOT_VERBS) or not word.isalpha():
        return []
    found = []
    for suffix in POSITIVE:
        if word.endswith(suffix) and len(word) - len(suffix) >= SHORTEST_STEM:
            found.append(word[: -len(suffix)])
    return found


def negative_of(stem):
    """Kökün olumsuz geniş zaman biçimi, ünlü uyumuna göre."""
    for character in reversed(stem):
        if character in "aeıioöuü":
            return stem + ("maz" if character in BACK else "mez")
    return stem + "mez"


def infinitive_of(stem):
    """Mastar: aynı uyum kuralı."""
    for character in reversed(stem):
        if character in "aeıioöuü":
            return stem + ("mak" if character in BACK else "mek")
    return stem + "mek"


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


# İsim testi: bir kelime durum eki alıyorsa isimdir. "risk" isimdir çünkü
# "riski", "riske", "riskten" de geçiyor; "henüz" değildir çünkü hiçbiri
# geçmiyor. Fiil testinin aynı ilkesi — türü kural değil, ekler söyler.
NOUN_MARKS = (("i", "ı", "u", "ü"), ("e", "a"), ("de", "da", "te", "ta"),
              ("den", "dan", "ten", "tan"), ("in", "ın", "un", "ün"),
              ("ler", "lar"))


# Sıfat-fiil ekleri: fiilden sıfat yapar. "olan", "gereken", "olduğu" bir
# kavram değil, bir yüklemin sıfatlaşmış hâli — ama durum eki aldıkları için
# isim testinden geçiyorlardı ve grafın en zengin "kavramları" oldular.
PARTICIPLES = ("an", "en", "dığı", "diği", "duğu", "düğü", "tığı", "tiği",
               "acak", "ecek", "esi", "ası", "mış", "miş", "muş", "müş")

# Yapı kelimeleri: dilbilgisi taşırlar, kavram değildirler. Kapalı ve kısa bir
# liste — ama asıl eleme sıklıkla yapılıyor (bkz. is_structural).
LIGHT = frozenset((
    "şey", "zaman", "kendi", "taraf", "yer", "hâl", "hal", "durum", "konu",
    "biri", "kimse", "yan", "yanı", "üzere", "kadar", "gibi", "göre",
))


def is_participle(word, verbs=None):
    """Bu kelime bir sıfat-fiil mi? Kavram olamaz.

    Ekin varlığı yetmez, KÖKÜN FİİL OLMASI gerekir: "penguen" de -en ile
    biter ama "pengu" diye bir fiil yoktur. Ek listesine bakıp karar vermek
    "penguen", "kaplan", "orman", "insan" gibi isimleri eliyordu — bu gece
    "banka"yı "bank'a" sanan hatanın aynısı, ve çözümü de aynı: derleme sor.
    """
    for suffix in PARTICIPLES:
        if not word.endswith(suffix) or len(word) - len(suffix) < 2:
            continue
        stem = word[: -len(suffix)]
        if verbs is None:
            continue            # sözlük yoksa karar verilemez, isim sayılır
        if infinitive_of(stem) in verbs or stem + "mak" in verbs \
                or stem + "mek" in verbs:
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
    if word in LIGHT:
        return True
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
    for family in NOUN_MARKS:
        for suffix in family:
            if seen.get(word + suffix, 0) >= 2:
                families += 1
                break
    return families >= least


# Sıfat testi: Türkçe'de yalnızca sıfat ve zarf derecelenir. "daha zor" olur,
# "daha banka" olmaz. Ham sayı yanıltıyor ("çok risk" geçerlidir) ama ORAN
# ayırıyor: ölçüldü, gerçek sıfatlar 194-339‰, isimler 0,4-7‰. Arada uçurum var.
GRADERS = ("daha", "en", "çok", "pek", "oldukça")
ADJECTIVE_RATE = 0.05       # binde 50; ölçülen iki küme arasındaki boşlukta


def is_adjective(word, graded, seen):
    """Bu kelime derecelenerek geçiyor mu?

    `graded`, (derecelendirici, kelime) ikililerinin sayacı. Bir kelimenin
    kendi sıklığına oranı bakılıyor — sık isimler mutlak sayıda yanıltıyor.
    """
    total = seen.get(word, 0)
    if total < 20:
        return False        # az geçen kelimede oran güvenilmez
    hits = sum(graded.get((grader, word), 0) for grader in GRADERS)
    return hits / total >= ADJECTIVE_RATE


def graded_pairs(tokens):
    """Metindeki (derecelendirici, kelime) ikililerini sayar."""
    found = collections.Counter()
    previous = None
    for token in tokens:
        if previous in GRADERS:
            found[(previous, token)] += 1
        previous = token
    return found


def from_text(text, minimum=2):
    """Düz metinden doğrudan. Büyük dosyalar için parça parça verilmeli."""
    return discover(collections.Counter(
        re.findall(r"[a-zçğıöşü]+", text.lower())), minimum)
