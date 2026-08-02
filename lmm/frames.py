"""Cümleyi durum ekleriyle okumak — sözcük sırasıyla değil.

Bugüne kadarki kalıplarımız İngilizce şeklindeydi: `[KAVRAM, FİİL]`, yani rolü
**konumdan** okuyorlar. İngilizce için doğru — "dog bites man" ile "man bites
dog" farklıdır ve fark sıradadır. Türkçe için yanlış: "köpek adamı ısırdı" ile
"adamı köpek ısırdı" aynı şeydir, çünkü rol **ekte** taşınır.

Ölçüldü ve teşhis kesin: belgedeki cümleciklerin %72,4'ü dört kelimeden uzun,
yani hiçbir sözcüksel kalıp onlara yetişemez; kalan %27,6'nın ancak dörtte biri
yakalanıyor. Aynı derlemde ise **altı ek sınıfı yüklemlerin ~%75'ini** kapsıyor.
Kırk sözcüksel kalıp %7 veriyor; altı ek sınıfı üç kat fazlasını vaat ediyor.

Yaklaşımın örneği ClausIE (2013): İngilizce için **yedi** cümle kalıbı yazdı ve
binlerce kalıplı sistemleri iki-üç kat geçti. Kazanç kalıp sayısından değil,
kalıbın durduğu **katmandan** geliyor.

Buradaki yapı iki parçalı:

    durum eki  -> hangi kelime özne, hangisi nesne, hangisi yer
    yüklem eki -> cümle ne yapıyor: bildiriyor mu, niteliyor mu, anlatıyor mu

İkisi de veri. Başka bir dil için ek listeleri değişir; buradaki kod değişmez.
"""

# Durum ekleri: bir kelimenin cümledeki rolünü söyleyen şey.
# Sıra önemli — uzun ek önce denenmeli, yoksa "evden" içinde "de" bulunur.
NOMINATIVE = "yalın"
ACCUSATIVE = "belirtme"
DATIVE = "yönelme"
LOCATIVE = "bulunma"
ABLATIVE = "ayrılma"
GENITIVE = "tamlayan"
INSTRUMENTAL = "vasıta"

CASES = (
    (("nden", "ndan", "den", "dan", "ten", "tan"), ABLATIVE),
    (("nde", "nda", "de", "da", "te", "ta"), LOCATIVE),
    (("nın", "nin", "nun", "nün", "ın", "in", "un", "ün"), GENITIVE),
    (("yle", "yla", "ile"), INSTRUMENTAL),
    (("ye", "ya", "e", "a"), DATIVE),
    (("yı", "yi", "yu", "yü", "nı", "ni", "nu", "nü",
      "ı", "i", "u", "ü"), ACCUSATIVE),
)

# Yüklem ekleri. Ölçüldü: ilk altısı yüklemlerin ~%75'ini kapsıyor, ve
# ilk ikisi tek başına yarısını. Ansiklopedik nesir bu dağılımda cömert —
# tanım ve niteleme cümleleri, yani grafı besleyen tür, yoğun kısımda.
COPULA = "koşaç"            # -dır: "bir kuştur", "beyazdır"     %17
AORIST = "geniş"            # -r:   "uçar"                        %14
PAST = "geçmiş"             # -dı:  "kuruldu"                     %32
PERFECT = "bitmiş"          # -mıştır: "kurulmuştur"              %14
PROGRESSIVE = "sürmekte"    # -maktadır: "kullanılmaktadır"        %8
ABILITY = "yeterlik"        # -abilir: "kullanılabilir"            %2
NEGATIVE = "olumsuz"        # -maz:  "uçamaz" — kutup, ayrı bir zaman değil
OBLIGATION = "gereklilik"   # -malı: "şifrelenmeli" — yapılması gereken
CONDITION = "koşul"         # -sa:   "yağarsa" — tek başına olgu değil

PREDICATES = (
    # Kip ekleri koşaçtan ÖNCE denenmeli: "yapılmalıdır" kelimesi koşaç kuralına
    # takılıp "yapılmalı" + koşaç diye okunuyordu, oysa gereklilik kipidir.
    (("malıdır", "melidir", "malı", "meli"), OBLIGATION),
    (("sa", "se", "ysa", "yse"), CONDITION),
    (("maktadır", "mektedir"), PROGRESSIVE),
    (("mıştır", "miştir", "muştur", "müştür",
      "mıştı", "mişti", "muştu", "müştü"), PERFECT),
    (("abilir", "ebilir"), ABILITY),
    (("tur", "tır", "dur", "dır", "tür", "tir", "dür", "dir"), COPULA),
    (("dı", "di", "du", "dü", "tı", "ti", "tu", "tü"), PAST),
    (("ar", "er", "ır", "ir", "ur", "ür", "r"), AORIST),
)

SHORTEST_STEM = 2
# Koşaç, zaman ekinin üstüne de gelir: "uçacak-tır", "uçmakta-dır".
COPULA_ON_TENSE = ("dır", "dir", "dur", "dür", "tır", "tir", "tur", "tür")
# Çoğul eki de -r ile biter; geniş zaman sanılmasın.
NOT_PREDICATES = ("lar", "ler")
PLURAL = ("lar", "ler")


def case_of(word, seen=None):
    """(kök, durum). Ek yoksa durum yalındır ve kelime olduğu gibi kalır.

    Türkçe'de her ek belirsizdir: "banka" ya bir kelimedir ya "bank"a yönelmedir,
    "işlemi" ya belirtmedir ya iyeliktir. Kurala bakarak karar verilemez.

    Karar derleme sorularak veriliyor, fiil keşfindeki testin aynısıyla: **kök,
    kelimenin kendisi kadar sık geçiyorsa ek gerçektir.** "müşteriye" nadirdir,
    "müşteri" sıktır -> ek. "banka" sıktır, "bank" nadirdir -> ek değil, kelime.

    Sayaç verilmezse eski davranış sürer, yani en uzun ek soyulur. Bu, sayaç
    olmayan yerlerde (testler, küçük bellekler) sistemin çalışmaya devam
    etmesini sağlıyor.
    """
    for suffixes, case in CASES:
        for suffix in suffixes:
            if not word.endswith(suffix) or len(word) - len(suffix) < SHORTEST_STEM:
                continue
            stem = word[: -len(suffix)]
            if seen is not None and seen.get(stem, 0) < seen.get(word, 0):
                continue        # kelimenin kendisi daha sık: ek değil
            return stem, case
    return word, NOMINATIVE


def _from_lexicon(word, lexicon, depth=3):
    """Bu yüklemi sözlük tanıyor mu — çekim katmanlarını soyarak.

    Ölçüt `predicate_of`'un soyma yolunu İZLEMELİ, yoksa çerçeve doğru
    kurulup olgu reddediliyor: "penguen uçacaktır" çerçevesi "penguen uçar"
    ile birebir aynı çıkıyor ama `attested` yanlış olduğu için `to_fact`
    atıyordu. İki yerde iki ayrı ölçüt olması, sessiz uyumsuzluk üretiyor.
    """
    if depth <= 0 or not word:
        return False
    if lexicon.reading(word) is not None:
        return True
    for ending in NOT_PREDICATES + COPULA_ON_TENSE:
        if word.endswith(ending) and len(word) > len(ending) + 1:
            if _from_lexicon(word[: -len(ending)], lexicon, depth - 1):
                return True
    return False


def predicate_of(word, lexicon=None):
    """(kök, zaman). Yüklem eki yoksa None.

    Yalnızca cümlenin son kelimesine sorulmalı: Türkçe'de yüklem sondadır ve
    ortadaki bir kelimede aynı harfler durum eki olabilir.

    Sözlük varsa önce ona sorulur, çünkü koşaç ekleri fiil çekimleriyle
    çakışıyor: "üretir" kelimesini kural "üre + tir" diye koşaç okuyordu, oysa
    "üret + ir" geniş zamandır. Aynı harfler, iki ayrı yapı; kural ayıramaz,
    sözlük ayırır.
    """
    if word.endswith(NOT_PREDICATES):
        # Çoğul eki de -r ile bitiyor ve geniş zaman sanılmasın diye bu kapı
        # kondu. Ama kapı fazla genişti: Türkçe'de yüklem de çoğul çekim alır
        # ("kuşlar UÇARLAR") ve o cümleler hiç okunamıyordu — derlemde
        # ölçüldü, 200.000 cümlenin %3,39'u böyle bitiyor.
        #
        # Ayrım sözlükte: "uçarlar" çoğul eki atılınca "uçar" oluyor ve sözlük
        # onu tanıyor; "kuşlar" atılınca "kuş" oluyor ve sözlük tanımıyor.
        # Yani kural değil sözlük ayırıyor — bu dosyanın her yerinde olduğu gibi.
        if lexicon is not None:
            for ending in NOT_PREDICATES:
                if not word.endswith(ending):
                    continue
                stem = word[: -len(ending)]
                reading = lexicon.reading(stem)
                if reading is not None:
                    return reading[0], (AORIST if reading[1] else NEGATIVE)
        return None, None
    if lexicon is not None:
        reading = lexicon.reading(word)
        if reading is not None:
            # Sözlük kutbu da söylüyor ve onu atmak olgunun tersini üretiyordu:
            # "penguen uçamaz" cümlesinden "penguen can uçmak" çıkıyordu. Bir
            # olgunun tersi, olgusuzluktan kötüdür.
            return reading[0], (AORIST if reading[1] else NEGATIVE)
    for suffixes, tense in PREDICATES:
        for suffix in suffixes:
            if not word.endswith(suffix):
                continue
            if len(word) - len(suffix) < SHORTEST_STEM:
                continue
            stem = word[: -len(suffix)]
            # Koşaç bir zaman ekinin ÜSTÜNE binebilir: "uçacak-tır",
            # "uçmakta-dır". Koşaç dalı önce eşleştiği için bunlar hep
            # koşaç okunuyordu ve altındaki zaman görünmüyordu — geri okuma
            # kapısında delik açan buydu: gelecek zamanda söylenen bir çelişki
            # yakalanmıyordu. Altta bir yüklem varsa o kazanır.
            if tense == COPULA:
                under = predicate_of(stem, lexicon)
                if under[0] is not None:
                    return under
            return stem, tense
    return None, None


NOT_A_NODE = frozenset((
    "değil", "yok", "var", "bir", "birer", "bu", "şu", "o", "her", "tek",
    "çok", "az", "daha", "en", "ve", "ile", "de", "da", "ki", "gibi", "göre",
    "kadar", "için", "ise", "ne", "hem", "ya", "veya", "ancak", "ama",
))


def _node(word):
    """Bu kelime bir kavram ya da hedef olabilir mi?"""
    if not word or len(word) < 3 or word in NOT_A_NODE:
        return False
    return word.replace("'", "").isalpha()


class Frame:
    """Bir cümlenin rol dökümü: kim, neyi, nerede, ne yapıyor."""

    def __init__(self, tokens):
        self.tokens = tokens
        self.roles = {}         # durum -> kelime kökü
        self.stem = None        # yüklemin kökü
        self.tense = None       # yüklemin ek sınıfı
        self.complement = None  # koşaçtan önceki kelime: "bir KUŞtur"
        self.attested = False   # yüklem sözlükte var mı, yoksa kural mı üretti

    @property
    def subject(self):
        return self.roles.get(NOMINATIVE)

    @property
    def object(self):
        return self.roles.get(ACCUSATIVE)

    def __repr__(self):
        parts = ", ".join(f"{case}:{word}" for case, word in self.roles.items())
        return f"<{parts} | {self.stem}·{self.tense}>"


def read(tokens, seen=None, lexicon=None, graded=None):
    """Cümleyi çerçeveye çevirir. Yüklemi yoksa None.

    Türkçe'de özne düşebilir; düştüğünde de cümle okunabilir olmalı — o zaman
    özne None kalır ve çağıran bağlamdan tamamlar.
    """
    if len(tokens) < 2:
        return None
    stem, tense = predicate_of(tokens[-1], lexicon)
    if tense is None:
        return None
    frame = Frame(tokens)
    frame.stem, frame.tense = stem, tense
    # Yüklem sözlükten mi geldi, kuraldan mı? Kuraldan gelen geniş zaman
    # kökleri güvenilmez: "riskti" kelimesinden "ti" çıkıyordu. Sözlük 734
    # madenden çıkarılmış fiili biliyor; tanımadığı bir çekim büyük ihtimalle
    # fiil değil.
    # "Sözlükten mi geldi" ölçütü, çoğul çekimli yüklemi de saymalı: sözlük
    # "uçarlar"ı doğrudan tanımıyor ama çoğul eki soyunca tanıyor ve
    # `predicate_of` tam olarak bunu yapıyor. Ölçüt burada dar kaldığı için
    # "kuşlar uçarlar" çerçevesi doğru kuruluyor, sonra `to_fact` onu
    # "kuraldan geldi" sanıp atıyordu — cümlelerin %3,39'u böyle kayboluyordu.
    frame.attested = (lexicon is not None
                      and _from_lexicon(tokens[-1], lexicon))
    # Koşaç bir eylem değil bir eşitlemedir: "bir kuştur" cümlesinde yüklemin
    # kökü zaten tümleçtir. Diğer zamanlarda tümleç ayrı bir kelimedir.
    if tense == COPULA:
        frame.complement = stem
    elif len(tokens) >= 2:
        frame.complement = tokens[-2]

    # Çoğul eki kavramı değiştirmez: "kuşlar uçar" cümlesinin öznesi "kuş"tur.
    # Soyulmadığında graf "kuş" ile "kuşlar"ı iki ayrı kavram sanıyordu.
    parsed = []
    for word in tokens[:-1]:
        root, case = case_of(word, seen)
        if case == NOMINATIVE and root.endswith(PLURAL) and len(root) > 5:
            root = root[:-3]
        parsed.append((root, case))
    for root, case in parsed:
        if case != NOMINATIVE:
            frame.roles.setdefault(case, root)
    subject = _subject(parsed, seen, graded)
    if subject:
        frame.roles[NOMINATIVE] = subject
    return frame


def _subject(parsed, seen, graded):
    """Yalın kelimelerden hangisi özne.

    Kanıt varsa **ilk** yalın ad alınır: Türkçe'de özne cümlenin başında durur.
    Uzun süre "son yalın kelime" alınıyordu ve bu bir yamaydı — sıfatların özne
    olmasını engellemek için konmuştu. Bedeli ölçüldü: yükleme bitişik yalın
    dizi genelde özne değil TÜMLEÇtir, o yüzden "Penguen perde ayaklı deniz
    kuşudur" cümlesinin öznesi `deniz` çıkıyordu.

    24 ansiklopedi girişi, ~480 çerçeve. Ölçü elle etiket istemiyor: bir yazının
    cümlelerinin çoğu başlığı hakkındadır, yani çıkan özne başlığa eşit olmalı.

        son yalın kelime                  %6,2
        yalın dizinin başı ya da sonu     %15,9
        ilk yalın ad                      %21,5
        ilk yalın ad, sıfat atlanarak     %22,0

    Sıfat ve dilbilgisi kelimeleri atlanıyor, ikisi de sayımla: sıfat
    derecelenerek geçer ("daha basit" olur, "daha banka" olmaz), dilbilgisi
    kelimesi derlemin en sık üç yüzündedir. Hiçbir aday geçmezse atlananın
    ilki yine de öznedir — öznesiz bir çerçeve hiç yoktan kötüdür.

    Sayaç yoksa karar verilemez ve konum sezgisine dönülür: yalın dizinin sonu.
    Bu, kanıtsız çalışan yerlerin (testler, küçük bellekler) davranışını
    korumak için değil yalnız — niteleyici ile özneyi ayıran şey gerçekten de
    sayımdır, ve o yoksa dil hakkında tahmin yürütmek gerekir.
    """
    from lmm.verbs import is_adjective, is_structural
    nominatives = [root for root, case in parsed if case == NOMINATIVE
                   and _node(root)]
    if not nominatives:
        return None
    if seen is None:
        return nominatives[-1] if len(nominatives) > 1 else nominatives[0]
    skipped = None
    for root in nominatives:
        if is_structural(root, seen):
            skipped = skipped or root
            continue
        if graded is not None and is_adjective(root, graded, seen):
            skipped = skipped or root
            continue
        return root
    return skipped


# Çerçeveden olguya: hangi rol örüntüsü hangi ilişkiyi verir.
# ClausIE'nin yedi kalıbının Türkçe karşılığı — sayı değil, katman önemli.
def to_fact(frame, lexicon=None, seen=None, graded=None):
    """(kavram, ilişki, hedef) ya da çıkarılamıyorsa None.

    Koşaç bir eşitlemedir: özne ile tümleç arasında tür ya da nitelik bağı.
    Diğer zamanlar eylemdir: özne bir şey yapar, nesnesi varsa ona yapar.
    """
    subject = frame.subject
    if not _node(subject) or not frame.stem:
        return None
    # Özne bir isim olmalı. Zarflar ve sıfatlar cümlenin başında durabilir ve
    # yalın görünürler; "henüz | can | yapılmak" böyle çıkıyordu. Türü kural
    # değil ekler söylüyor: isim durum eki alır, zarf almaz.
    if seen is not None:
        from lmm.verbs import (is_adjective, is_noun, is_participle,
                               is_structural)
        known = getattr(lexicon, "_infinitives", None) if lexicon else None
        if known is None and lexicon is not None:
            known = {i for i, _ in lexicon.verbs.values()}
            try:
                lexicon._infinitives = known
            except AttributeError:
                pass
        if is_participle(subject, known) or is_structural(subject, seen):
            return None
        if not is_noun(subject, seen, least=2):
            return None
        # Sıfatlar da durum eki alır ("zoru", "zora"), o yüzden isim testinden
        # geçiyorlar ve özne sanılıyorlardı: "zor | can | olgunlaşmak".
        # Derecelenme onları ayırıyor.
        if graded is not None and is_adjective(subject, graded, seen):
            return None
    if frame.tense == COPULA:
        if not _node(frame.stem):
            return None
        # "kartal bir kuştur" -> tür;  "kar beyazdır" -> nitelik.
        # Ayrımı "bir" belirteci yapar: tür bir sınıfa yerleştirir.
        relation = "type" if "bir" in frame.tokens else "property"
        return subject, relation, frame.stem
    # Koşul cümlesi tek başına bir olgu değil, bir kuralın önkoşuludur:
    # "yağmur yağarsa" doğru ya da yanlış olamaz. Bunu olgu saymak, henüz
    # temsil edemediğimiz bir şeyi varmış gibi göstermek olur.
    if frame.tense == CONDITION:
        return None
    if frame.tense == OBLIGATION and frame.stem:
        # Kök mastara çevrilir: "kısıtlan" bir kelime değil, "kısıtlanmak"
        # kelimedir. Graf kavramları tam hâlleriyle tutmalı, yoksa aynı eylem
        # her çekimde ayrı bir düğüm olur.
        from lmm.verbs import infinitive_of
        return subject, "must", infinitive_of(frame.stem)
    if frame.tense in (AORIST, NEGATIVE) and frame.stem:
        if not frame.attested:
            return None         # kuraldan üretilmiş kök, sözlükte yok
        return subject, "can" if frame.tense == AORIST else "cannot", frame.stem
    if lexicon is not None:
        reading = lexicon.reading(frame.tokens[-1])
        if reading is not None:
            return subject, "can", reading[0]
    return None
