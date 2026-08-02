"""Bilgi derlemesi: dil modeli bir okuyucudur, bir otorite değil.

RAG'de dil modeli sorgu anında devrededir, yani her cevap uydurabilen bir şeyden
geçer. Burada dil modeli **bir kez**, çevrimdışı, belge okunurken kullanılır;
çıktısı denetlenir; sonra ortadan kalkar. Çalışma anında graf yalnız kalır ve
CPU'da çalışır.

Bunun işe yaraması tek bir soruya bağlı: *hakkında hiçbir şey bilmediğin bir
iddiayı nasıl doğrularsın?* Epistemik kapı çelişkiyi yakalar ama bir firmanın
belgesini okurken bildiği şey sıfırdır — her şey yenidir, dolayısıyla hiçbir şey
çelişmez ve kapı sessiz kalır.

Buradaki cevap, dil modeline güvenmek yerine onu **kaynak metne çapalamak**:

    olgu tek başına gelemez; onu söyleyen cümleyle birlikte gelir
    o cümle belgede gerçekten geçmelidir      -> yoksa uydurulmuş
    iddianın parçaları o cümlede geçmelidir   -> yoksa cümleden çıkmıyor

İkisi de makineyle denetlenir ve ikisi de dil modelinin iyi niyetine bağlı
değildir. Uydurma, modele güvenilerek değil, metne çapalanarak elenir.

Kimlik bilgileri yalnızca ortam değişkenlerinden okunur; hiçbir anahtar bu
depoya yazılmaz.
"""
import json
import os
import re
import urllib.error
import urllib.request

from lmm.intuition import lower

# Bu ikisi ÖLÇÜLMEDİ ve öyle olduğu yazsın. İkisi de yalnızca ağ ekonomisini
# ayarlıyor — hangi olgunun kabul edildiğine etkileri yok, o iş aşağıdaki
# denetimlerin. Bir gün ölçülürse ölçüsü "aynı belgede kaç olgu, kaç saniye".
TIMEOUT = 90
BATCH = 12              # tek istekte gönderilen cümle sayısı

BRIEF = """Sana bir belgeden cümleler verilecek. Her cümlenin SÖYLEDİĞİ olguları
çıkar. Cümlenin söylemediği hiçbir şeyi yazma.

Her satır tam olarak şu biçimde:
KAVRAM | İLİŞKİ | HEDEF | KAYNAK CÜMLE

İLİŞKİ, şu dördünden EN ÖZGÜL olanı olmalı:
  type     : X bir Y türüdür        (linux | type | işletim sistemi)
  can      : X şunu yapar/yapabilir (sunucu | can | çalıştırmak)
  property : X şu niteliktedir      (linux | property | özgür)
  has      : X'in parçası/sahibi    (bilgisayar | has | işlemci)

"has" SON ÇAREDİR. Bir olgu type, can veya property ile anlatılabiliyorsa onu
kullan. Emin değilsen o olguyu hiç yazma.

KAVRAM ve HEDEF birer kavram olmalı:
  - en fazla 3 kelime, tercihen 1
  - sayı, yüzde, tarih OLMAZ
  - liste OLMAZ ("gnome, kde ve xfce" yerine hiçbir şey yazma)
  - başka cümlelerde de geçebilecek, yeniden kullanılabilir bir ad olmalı
  - eylemler mastar halinde ("çalıştırmak", "çalıştırıyor" değil)

KAYNAK CÜMLE, olgunun çıktığı cümlenin TAM METNİ — kelimesi kelimesine,
kısaltmadan, düzeltmeden.

Cümleden sağlam bir olgu çıkmıyorsa o cümle için hiçbir satır yazma. Az ve
doğru, çok ve şişkinden iyidir. Açıklama, başlık, numara yazma."""


class CompileError(Exception):
    pass


class Candidate:
    def __init__(self, concept, relation, target, anchor):
        self.concept = concept
        self.relation = relation
        self.target = target
        self.anchor = anchor        # olguyu söylediği iddia edilen cümle
        self.refusal = None         # neden reddedildi

    def __repr__(self):
        return f"{self.concept} {self.relation} {self.target}"


# Reddedilme sebepleri. Sayılabilir olmaları, derleyicinin kendisini ölçmemizi
# sağlar — hangi denetimin ne kadar iş yaptığı görünür kalsın.
NO_ANCHOR = "kaynak cümle verilmedi"
NOT_IN_DOCUMENT = "kaynak cümle belgede yok"
NOT_IN_ANCHOR = "iddia kaynak cümlede geçmiyor"
BAD_RELATION = "tanınmayan ilişki"
EMPTY = "boş alan"

RELATIONS = {"type", "can", "property", "has"}

# Bir grafın değeri tekrar kullanımdan gelir. "gnome, kde, xfce gibi gelişmiş
# masaüstü teknolojileri" metinde gerçekten geçiyor olabilir ama bir daha asla
# başka bir olguda görünmez; düğüm olarak ölü ağırlıktır ve grafı şişirir.
# Çapa "metne dayanıyor" der; hijyen "kavram olarak kullanılabilir" der. İkisi
# ayrı sorulardır ve ikisi de sorulmalı.
LONGEST_NODE = 3        # kelime
SHORTEST_NODE = 3       # harf

# Çapa bu kadar karakterden kısaysa bir cümle değildir. Sayı isimsiz duruyordu;
# ölçüldü (`data/tr-metin.txt`, 3.594 cümle): 10 karakterin altındaki 99 parça
# (%2,8) istisnasız sayı, kısaltma ya da kırıntı — "13", "XIII", "Dr", ") idi".
# Yani eşik bir olgu kaybettirmiyor, yalnız cümle olmayanı eliyor.
SHORTEST_ANCHOR = 10    # harf
UNUSABLE = "düğüm olarak kullanılamaz"
NOT_AN_ACTION = "yapılan bir eylem değil"
WRONG_WAY = "tür iddiası cümlede bu yönde değil"


def normalise(text):
    """Karşılaştırma için: küçük harf, tek boşluk, noktalama yok.

    Sıra önemli: önce noktalama silinir, sonra boşluk sıkıştırılır. Tersi
    yapılınca "Kredi, bir ürün" -> "kredi  bir ürün" oluyordu ve çift boşluk
    çapayı belgede bulunamaz hale getiriyordu — yani doğru bir olgu, yalnızca
    virgülün yeri yüzünden reddedilirdi.
    """
    stripped = re.sub(r"[^\wçğıöşü ]+", " ", lower(text))
    return re.sub(r"\s+", " ", stripped).strip()


def parse(reply):
    """Modelin düz metin cevabından adaylar. Bozuk satırlar sessizce atlanır."""
    found = []
    for line in reply.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        found.append(Candidate(lower(parts[0]), parts[1].lower(),
                               lower(parts[2]), parts[3]))
    return found


def verify(candidate, document):
    """Çapalama denetimi. Geçerse True, kalırsa sebebi yazılır.

    Belge bir kez normalleştirilip verilmelidir; her aday için yeniden
    normalleştirmek büyük belgelerde işi ikinci dereceye çıkarır.
    """
    if not all((candidate.concept, candidate.relation, candidate.target)):
        candidate.refusal = EMPTY
        return False
    if candidate.relation not in RELATIONS:
        candidate.refusal = BAD_RELATION
        return False
    if not candidate.anchor or len(candidate.anchor) < SHORTEST_ANCHOR:
        candidate.refusal = NO_ANCHOR
        return False
    anchor = normalise(candidate.anchor)
    if anchor not in document:
        candidate.refusal = NOT_IN_DOCUMENT
        return False
    # İddianın kendisi o cümleden çıkmalı. Türkçe ekli bir dil olduğu için tam
    # eşleşme aranmaz; kökün cümlede geçmesi yeterli, ama geçmesi şart.
    for part in (candidate.concept, candidate.target):
        if not _appears(normalise(part), anchor):
            candidate.refusal = NOT_IN_ANCHOR
            return False
    for part in (candidate.concept, candidate.target):
        if not usable(part):
            candidate.refusal = UNUSABLE
            return False
    if candidate.relation == "type" and not states_a_type(candidate, anchor):
        candidate.refusal = WRONG_WAY
        return False
    if candidate.relation in ("can", "cannot") and not _an_action(candidate.target):
        candidate.refusal = NOT_AN_ACTION
        return False
    return True


# Türkçe'de "-mak/-mek" mastardır, "-ma/-me" EYLEM ADIdır: "rastlanmak" bir
# eylem, "rastlanma" bir addır. Ayrım kaçınca graf "ahtapot --can--> rastlanma"
# gibi kayıtlar alıyor ve cevap "ahtapot rastlanma" diye çıkıyordu.
INFINITIVE = ("mak", "mek")
# Mastar bundan kısaysa mastar değil, ekin kendisidir. Ölçüldü
# (`data/tr-fiiller.txt`, 807 mastar): dört harf ve altında olan SIFIR tane —
# en kısası "acmak", beş harf. Yani eşik hiçbir gerçek fiili elemiyor, ama
# "amak", "omek" gibi ayrıştırma kazalarını eliyor.
SHORTEST_INFINITIVE = 4


def _an_action(target):
    """CAN/CANNOT hedefi bir eylem mi — mastar biçiminde mi."""
    if not target:
        return False
    last = target.split()[-1]
    return last.endswith(INFINITIVE) and len(last) > SHORTEST_INFINITIVE


def states_a_type(candidate, anchor):
    """Cümle gerçekten "X bir Y'dir" diyor mu, yoksa "X Y" bir tamlama mı?

    Çapa denetimi "linux | type | dağıtım"ı kabul ediyordu, çünkü iki kelime de
    "Linux dağıtımları ..." cümlesinde geçiyor. Ama o cümle bir tür iddiası
    değil, bir isim tamlaması — ve tür ilişkisi grafın omurgası olduğu için ters
    yazılmış bir tanesi altındaki her şeyi bozar.

    Türkçe'de tür iddiası iki biçimde kurulur ve ikisi de kabul edilir:

        belirtme       "Kartal BİR kuştur."
        ara söz        "Vaşak, kedigiller familyasından ... hayvan türlerinin
                        ortak adı." — kavramdan sonra virgül, tanım sonda

    İlk yazışta yalnız birincisi aranıyordu ve ikincisi Türkçe ansiklopedi
    tanımlarının olağan biçimi olduğu için ölçülebilir bir kayıptı: `vaşak →
    hayvan türü`, `volkanoloji → bilim dalı`, `yanardağ → coğrafi yer şekli`
    üçü de doğruyken reddediliyordu.

    Konumlar kök eşleşmesiyle bulunuyor, düz arama ile değil: `türü` kelimesi
    metinde `türlerinin` diye geçiyor ve düz arama onu hiç bulamıyordu — yani
    denetim reddettiğini sandığı şeyi değil, bulamadığı şeyi reddediyordu.
    """
    # Virgül HAM çapadan okunuyor. `normalise` noktalamayı siliyor ve ilk
    # yazışta denetim normalleştirilmiş metne bakıyordu: aradığı işaret oraya
    # hiç ulaşmıyordu, yani kural sessizce hep "hayır" diyordu.
    raw = (candidate.anchor or anchor).split()
    words = [lower(word).strip(".,;:!?\"'()") for word in raw]
    concept = normalise(candidate.concept).split()
    target = normalise(candidate.target).split()
    here = _phrase_at(concept, words)
    there = _phrase_at(target, words)
    if here < 0 or there < 0:
        return False
    if there < here:
        return False            # "kuş türleri arasında penguen" — tür iddiası değil
    between = words[here + len(concept):there]
    if "bir" in between or "birer" in between:
        return True
    # Ara sözle tanım: kavramın hemen ardından virgül. Bitişiklik ise tam
    # olarak reddedilmesi gereken durum — "linux dağıtımları" bir isim
    # tamlamasıdır, tür iddiası değil. İlk yazdığımda bitişikliği izin olarak
    # yazmışım ve denetim tersine çalışmıştı: yanlışı geçirip doğruyu eliyordu.
    after = here + len(concept) - 1
    return bool(between) and after < len(raw) and raw[after].endswith(",")


def _phrase_at(parts, words):
    """Öbeğin kelime sırası, kök eşleşmesiyle. Bulunamazsa -1."""
    if not parts:
        return -1
    for start in range(len(words) - len(parts) + 1):
        if all(_same_root(part, words[start + offset])
               for offset, part in enumerate(parts)):
            return start
    return -1


def usable(node):
    """Bu, başka olgularda da geçebilecek bir kavram mı?"""
    words = node.split()
    if not words or len(words) > LONGEST_NODE:
        return False
    if len(node.replace(" ", "")) < SHORTEST_NODE:
        return False
    # Sayı, oran ve tarih bir kavram değil; bir ölçüdür. Ölçüler kendi
    # alanlarını hak eder, düğüm olarak değil.
    if any(character.isdigit() for character in node) or "%" in node:
        return False
    # Her kelime gerçek bir kelime olmalı. "tool broker /" üç kelime sayılıp
    # geçiyordu; bir eğik çizgi kavram olamaz ve o düğüm bir daha hiçbir olguda
    # görünmez.
    for word in words:
        if not word.replace("'", "").replace("-", "").isalpha():
            return False
    return True


# Olumsuzluk bir ektir ve çapa denetiminin yakalaması gereken asıl şey odur:
# "uçar" ile "uçamaz" ortak kök paylaşır ama zıt şey söyler. Kök eşleşmesi
# gevşetilirken bu kapı açık bırakılamaz.
# Kutup kelimenin TAMAMINDAN okunur, ortak kökten sonrasından değil. İlk
# yazışta sonrasına bakılıyordu ve sessizce açık kalıyordu: "uçamamak" ile
# "uçar" ortak kökü "uça" olarak buluyor, olumsuzluk eki önekin içinde kalıyor
# ve iki taraf da "olumsuz değil" görünüyordu.
NEGATIVE = ("maz", "mez", "amaz", "emez", "mıyor", "miyor", "muyor", "müyor",
            "madı", "medi", "mamış", "memiş", "mayan", "meyen",
            "mamak", "memek", "mama", "meme", "masın", "mesin")
SHORTEST_ROOT = 2       # Türkçe kökleri kısadır: "uç", "ev", "el", "su"

# İddia tarafında kaç harflik ek farkına izin verilir. Sayı gerekçesiz duruyordu
# ve ölçülünce GEVŞEK çıktı (`data/tr-fiiller.txt`, 807 mastar; mastar ile
# çekimli hâli eşleşmeli, farklı iki fiilin çekimi eşleşmemeli):
#
#     izin   doğru eşleşme        4000 rastgele çiftte YANLIŞ eşleşme
#      3        806/807 (%99,9)                 5
#      5        806/807 (%99,9)                26
#      8        806/807 (%99,9)                53
#
# Yani 3'ten sonrası tek bir doğru eşleşme kazandırmıyor, yanlış eşleşmeyi
# beşe katlıyor. Değer DEĞİŞTİRİLMEDİ — davranışı değiştirmek bu işin konusu
# değildi ve gerçek belgede kaç olguya dokunduğu ayrıca ölçülmeli. Ama sayının
# yanında artık ne aldığı ve ne ödettiği yazıyor.
LONGEST_SUFFIX = 5


def _appears(word, sentence):
    """Kelime, eklerine rağmen cümlede var mı?

    Türkçe'de aynı kelime `tür` / `türü` / `türlerinin` diye değişir; yüzey
    biçimi aramak sistematik kayıp veriyordu. Ölçüldü: "Vaşak, ... yabani
    hayvan türlerinin ortak adı" cümlesinden çıkan `vaşak → hayvan türü`
    iddiası, doğru olduğu hâlde reddediliyordu — çünkü dört harften kısa
    kelimeler tam eşleşme istiyordu ve Türkçe kökleri çoğu zaman üç harf.

    Gevşetmenin bir riski var ve bilinen bir risk: `uçar` ile `uçamaz` ortak
    kök paylaşır. O yüzden kök eşleşmesi olumsuzluk ekine bakıyor — köke
    eklenen parça bir tarafta olumsuzsa diğerinde değilse, eşleşme sayılmaz.
    Bu, sistemin en tehlikeli hata biçimine karşı konmuş bir kapı: doğru
    kavram, ters iddia.
    """
    if not word:
        return False
    if word in sentence:
        return True
    words = sentence.split()
    for piece in word.split():
        if not any(_same_root(piece, other) for other in words):
            return False
    return True


def _same_root(piece, other):
    """İki kelime aynı kökten mi — ekleri farklı olabilir, kutbu olamaz.

    Ortak kök kısa tutuluyor çünkü Türkçe kökleri kısa; bunun bedeli, uzun bir
    çekimin kısa bir parçayla eşleşebilmesi. Bilinçli bir denge: dilin kendisi
    böyle ve asıl korunması gereken şey uzunluk değil **kutup**.
    """
    if piece == other:
        return True
    shared = 0
    for left, right in zip(piece, other):
        if left != right:
            break
        shared += 1
    if shared < SHORTEST_ROOT or len(piece) - shared > LONGEST_SUFFIX:
        return False
    return _negated(piece) == _negated(other)


def _negated(word):
    """Kelime olumsuz mu — ek olarak, kelimenin sonundan okunur.

    "yapmaz" olumsuz, "yapmak" değil; ikisi de "ma" taşır ve fark sondadır.
    """
    return any(word.rstrip(".,;:!?").endswith(mark) for mark in NEGATIVE)


class Report:
    def __init__(self):
        self.accepted = []
        self.refused = []       # (candidate, reason)
        self.anchors = {}       # (kavram, ilişki, hedef) -> {kaynak cümleler}

    @property
    def total(self):
        return len(self.accepted) + len(self.refused)

    def reasons(self):
        counts = {}
        for _, reason in self.refused:
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def record(self, candidate):
        self.accepted.append(candidate)
        key = (candidate.concept, candidate.relation, candidate.target)
        self.anchors.setdefault(key, set()).add(normalise(candidate.anchor))

    def facts(self):
        """Benzersiz olgular, kaç ayrı cümlenin söylediğiyle birlikte.

        Aynı olguyu iki farklı cümle söylüyorsa bu bir tekrar değil, bir
        doğrulamadır: tek okumanın hatası olma ihtimali düşer. Bellek zaten
        `corroborate` ile bunu ifade edebiliyor; eksik olan derleyicinin bu
        sayıyı taşımasıydı.
        """
        return sorted(((key, len(anchors)) for key, anchors in self.anchors.items()),
                      key=lambda pair: -pair[1])


def compile_text(sentences, document, ask, report=None, batch=BATCH):
    """Cümleleri okuyucuya verir, dönenleri çapaya karşı denetler.

    `ask` bir işlevdir: cümle listesi alır, düz metin cevap döndürür. Böylece
    testler ağa çıkmadan çalışır ve derleyici hangi modelin kullanıldığını
    bilmez.
    """
    report = report or Report()
    reference = normalise(document)
    for start in range(0, len(sentences), batch):
        chunk = sentences[start:start + batch]
        try:
            reply = ask(chunk)
        except CompileError:
            raise
        for candidate in parse(reply):
            if verify(candidate, reference):
                report.record(candidate)
            else:
                report.refused.append((candidate, candidate.refusal))
    return report


# --- gerçek bir modele bağlanma -----------------------------------------

def _config():
    """Bağlantı ayarı tek yerde durur. Hasat zaten Azure ile OpenAI arasındaki
    farkı çözüyordu; ikinci bir kopya, ikinci bir yanlış yapılandırma demekti."""
    from lmm.harvest import _config as harvest_config, HarvestError
    try:
        url, headers, _ = harvest_config()
    except HarvestError as error:
        raise CompileError(str(error))
    return url, headers


LOCAL_URL = "http://localhost:11434/api/chat"


def local_reader(model="gemma3:4b", url=LOCAL_URL, temperature=0.0, brief=None):
    """Aynı okuyucu, ama makinenin kendi üstünde çalışan bir model.

    Bu, derleme anındaki son dış bağımlılığı kaldırıyor. Belgeyi okuyan model
    firmanın kendi sunucusunda; hiçbir metin dışarı çıkmıyor, hiçbir API
    anahtarı gerekmiyor, ve çalışma anında zaten dil modeli yoktu.

    Denetim değişmiyor: yerel modelin söyledikleri de kaynak metne çapalanıyor.
    Okuyucunun nerede çalıştığı, ona ne kadar güvenildiğini değiştirmez.
    """
    def ask(chunk):
        body = {"model": model, "stream": False,
                "options": {"temperature": temperature},
                "messages": [{"role": "system", "content": brief or BRIEF},
                             {"role": "user", "content": "\n".join(chunk)}]}
        request = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT * 3) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise CompileError(f"HTTP {error.code}: {error.read()[:200].decode()}")
        except urllib.error.URLError as error:
            raise CompileError(f"yerel model yok ({error.reason}) — "
                               f"'ollama serve' çalışıyor mu?")
        return payload["message"]["content"]
    return ask


def model_reader(url=None, headers=None, temperature=0.0, brief=None):
    """Bir dil modelini okuyucu olarak veren işlev.

    `brief` verilmezse olgu çıkarma istemi kullanılır. Bu varsayılan bir kez
    sessizce yanlış işe yaradı: niyet sınıflandırması için çağrıldığında model
    hâlâ "olgu çıkar" talimatı alıyordu ve hiçbir şey döndürmüyordu. Okuyucu
    genel bir araç; ne okuyacağı çağıranın işi.
    """
    def ask(chunk):
        target, head = (url, headers) if url else _config()
        body = {"messages": [{"role": "system", "content": brief or BRIEF},
                             {"role": "user", "content": "\n".join(chunk)}],
                "temperature": temperature}
        request = urllib.request.Request(
            target, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **head})
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as error:
            raise CompileError(f"HTTP {error.code}: {error.read()[:200].decode()}")
        except urllib.error.URLError as error:
            raise CompileError(f"bağlanılamadı: {error.reason}")
        return payload["choices"][0]["message"]["content"]
    return ask
