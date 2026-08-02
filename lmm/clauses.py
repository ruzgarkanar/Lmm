"""Uzun cümleyi okunabilir parçalara bölmek.

Ölçtük: en uzun kalıbımız 4 kelime, gerçek cümlelerin %78'i 5 kelime ve üzeri.
Bu yüzden sistem genel düzyazıda 100 cümlede 3 olgu çıkarıyordu — kalıp eksikliği
değil, mimari bir sınır. "Cümlenin tamamını anlayamazsam hiçbir şey almam"
demek, uzun cümleyi baştan kaybetmek demek.

Pencere kaydırma denendi ve çöp üretti: "sonra type karar", "ve type bankanın
bu". Kör bir bölme dilin yapısını görmez.

Buradaki bölme dilin kendi işaretlerini kullanıyor:

    bağlaç      "ve", "ama", "çünkü" — cümleciklerin arasına girer
    virgül      aynı işi noktalamayla yapar
    özne düşmesi Türkçe'de bağlaçtan sonra özne tekrarlanmaz; geri konur

Ölçüldü: bankacılık dokümanında 3,1 -> 8,8 olgu/100 cümle, Wikipedia nesrinde
2,9 -> 10,8. Hiçbir dil modeli kullanılmadan.

Bunların hiçbiri Türkçe'ye özel bir *mekanizma* değil — bağlaç listesi ve özne
kuralı veridir. Başka bir dil için o iki şey değişir, buradaki kod değişmez.
"""
import re

# Cümlecikleri birbirine bağlayan kelimeler. Kapalı bir sınıf: sayıca sabit,
# yeni üye almaz, ve bir dil için bir kez yazılır.
JOINERS = ("ve", "veya", "ya da", "ancak", "fakat", "ama", "çünkü", "ayrıca",
           "yani", "oysa", "hâlbuki", "halbuki", "lakin", "ne var ki")

# Ayracın türü korunuyor, çünkü ikisi farklı iş yapıyor: virgül bir özneyi de
# ayırabilir ("Penguen, ... yaşar"), bağlaç ayıramaz. Bu ayrım olmadan
# "kuşlar uçar ve balıklar yüzer" cümlesinde "kuşlar uçar" özne sanılıyordu.
BOUNDARY = re.compile(
    r"(,|;|\s+(?:" + "|".join(re.escape(word) for word in JOINERS) + r")\s+)")

# Türkçe yan cümlesini bağlaçla değil EKLE kurar. Ölçüldü: cümlelerin %74,4'ü
# bunlardan en az birini taşıyor, ve bağlaç bölmesi onlara hiç dokunmuyordu.
# Bu, kalıplarımızın İngilizce şeklinde olmasının doğrudan sonucu — İngilizce
# rolleri sözcük sırasıyla taşır, Türkçe eklerle.
#
# Her biri bir yan cümlenin SONUNU işaretler: "kaydı alıp raporlar" iki
# yüklemdir. Ek, fiili çekimsiz bıraktığı için sözlük onu tanımaz; bölerken
# çekimli hâline geri çevriliyor ve cümlecik okunabilir hâle geliyor.
SUBORDINATORS = (
    ("ip", "ıp", "up", "üp"),               # gelip, alıp
    ("erek", "arak"),                       # koşarak
    ("ince", "ınca", "unca", "ünce"),       # gelince
    ("madan", "meden"),                     # görmeden
    ("dıkça", "dikçe", "dukça", "dükçe"),   # gördükçe
    ("ken",),                               # bakarken
)

SHORTEST_CLAUSE = 2     # tek kelime bir cümlecik değildir
LONGEST_SUBJECT = 2     # "kredi başvurusu" özne olur, yarım cümle olmaz


def _finite(word, lexicon=None):
    """Yan cümle ekini söküp fiili çekimli hâline döndürür.

    "alıp" -> "alır". Ek fiili çekimsiz bırakıyor ve sözlük onu tanımıyor;
    bölmek tek başına yetmez, cümleciğin okunabilmesi için yüklemin geri
    kurulması gerekiyor.

    Çekim TAHMİN EDİLMİYOR. Türkçe geniş zaman düzensizdir — "al" kökü "alır"
    verir, "bak" kökü "bakar"; kurala bağlamak "alar" gibi olmayan kelimeler
    üretiyordu. Doğrusu sözlüğe sormak, ve sözlük artık metinden çıkarılmış 734
    fiili içeriyor. Sözlükte yoksa cümlecik bölünmez: yanlış bir yüklem
    uydurmaktansa cümleyi olduğu gibi bırakmak.
    """
    from lmm.verbs import infinitive_of
    for family in SUBORDINATORS:
        for suffix in family:
            if not word.endswith(suffix) or len(word) - len(suffix) < 2:
                continue
            stem = word[: -len(suffix)]
            if suffix == "ken":
                return stem if stem.endswith(("ar", "er", "ır", "ir")) else None
            # "doğrulayarak" -> kök "doğrula", araya kaynaştırma "y" girmiş.
            if stem.endswith("y") and len(stem) > 2:
                stem = stem[:-1]
            if lexicon is None:
                return None
            # surface() bilinmeyen fiilde mastarın kendisini döndürüyor;
            # "doğrulayarak" -> "doğrulamak" gibi çekimsiz bir yüklem çıkıyordu.
            # Sözlükte gerçekten var mı, onu sormak gerekiyor.
            infinitive = infinitive_of(stem)
            form = lexicon.surface(infinitive, True)
            return form if form != infinitive else None
    return None


def with_subordinators(sentence, lexicon=None):
    """Yan cümle eki taşıyan kelimelerden böler, yüklemi geri kurar."""
    words = sentence.split()
    pieces, current = [], []
    for word in words:
        current.append(word)
        finite = _finite(word.lower().strip(",;"), lexicon)
        if finite is not None and len(current) >= SHORTEST_CLAUSE:
            current[-1] = finite
            pieces.append(" ".join(current))
            current = []
    if current:
        # Kalan kuyruk tek kelimeyse atmak ana yüklemi kaybettiriyordu:
        # "veri şifrelenmeden çıkmaz" cümlesinde "çıkmaz" düşüyordu. Özneyi
        # ilk parçadan taşımak, bölmede zaten kullandığımız kural.
        tail = " ".join(current)
        if len(current) < SHORTEST_CLAUSE and pieces:
            tail = f"{pieces[0].split()[0]} {tail}"
        pieces.append(tail)
    return [piece for piece in pieces if len(piece.split()) >= SHORTEST_CLAUSE]


def split(sentence):
    """Cümleyi, her biri tek başına okunabilir parçalara böler.

    Özne yalnızca ilk parçada geçtiğinde sonrakilere taşınır. "Sistem, her
    işlemi kaydeder ve şüpheli hareketleri raporlar" cümlesinde ikinci yüklemin
    öznesi de sistemdir; bunu geri koymadan parça okunamaz.
    """
    # Sonda boşta kalan bağlaç ("kuşlar uçar ve") ayrıştırmayı bozar; onu
    # bırakmak, olmayan bir cümleciği varmış gibi göstermek olur.
    sentence = re.sub(r"\s+(?:" + "|".join(JOINERS) + r")\s*$", "",
                      sentence.strip())
    pieces = [piece.strip() for piece in BOUNDARY.split(sentence)]
    parts, separators = pieces[0::2], pieces[1::2]
    parts = [part for part in parts if part]
    if not parts:
        return []
    subject = ""
    # Baştaki kısa parça, ARDINDAN VİRGÜL geliyorsa bir cümlecik değil öznedir:
    # "Penguen, soğuk yerlerde yaşar" — "Penguen" atılırsa özne "soğuk" sanılır.
    # Ardından bağlaç geliyorsa öyle değildir: "kuşlar uçar ve ..." cümlesinde
    # "kuşlar uçar" zaten tam bir cümleciktir.
    if (len(parts) > 1 and separators and separators[0] in (",", ";")
            and len(parts[0].split()) <= LONGEST_SUBJECT):
        subject = parts.pop(0)
    # Özne GERİ KONMADAN eleme yapılıyordu ve bu, Türkçe'nin en sık kurulumunu
    # sessizce siliyordu: ikinci yüklem öznesi düştüğü için tek kelime kalıyor
    # ("sistem her işlemi kaydeder VE RAPORLAR") ve `SHORTEST_CLAUSE` eşiğine
    # takılıp atılıyordu. Ölçüldü: bağlaçla bölünen 90.400 cümlenin 6.675'inde
    # (%7,4) ikinci iddia hiç okunmuyordu.
    #
    # Doğrusu önce özneyi koymak, sonra elemek: "raporlar" tek kelimedir ama
    # "sistem raporlar" bir cümledir.
    if not subject:
        first = next((p for p in parts
                      if len(p.split()) >= SHORTEST_CLAUSE), None)
        if first is None:
            return []
        subject = first.split()[0]
    head = subject.split()[0].lower()
    restored = [part if part.split()[0].lower() == head
                else f"{subject} {part}"
                for part in parts if part.split()]
    return [clause for clause in restored
            if len(clause.split()) >= SHORTEST_CLAUSE]


def readable(sentence, longest=4, lexicon=None):
    """Cümle zaten kısaysa bağlaçtan bölme; ama yan cümle eki her zaman bölünür.

    Bölmek bedava değil: parçalar özne taşırken hata yapılabilir. Kalıpların
    okuyabildiği bir cümleyi bağlaçtan bölmek kazançsız risk. Ama yan cümle eki
    farklı — dört kelimelik "Sistem kaydı alıp raporlar" iki yüklem taşır ve
    uzunluk korumasına takılıp bölünmeden kalıyordu.
    """
    parts = ([sentence] if len(sentence.split()) <= longest
             else (split(sentence) or [sentence]))
    found = []
    for part in parts:
        found.extend(with_subordinators(part, lexicon) or [part])
    return found if found else [sentence]
