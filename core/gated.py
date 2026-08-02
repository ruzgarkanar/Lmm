"""Kapının çekirdeği kısıtladığı yer — LMM'i LMM yapan katman.

RETRO bir dil modelinin dışarıdan metin çekebileceğini gösterdi. AlphaGeometry
bir dil modelinin sembolik bir doğrulayıcıyla çalışabileceğini gösterdi. İkisinde
de eksik olan şey aynı: dil modeli hâlâ **ne söyleyeceğine kendi karar veriyor**.
Çekilen metin bir öneri, doğrulayıcı bir denetçi; üretimin kendisi serbest.

Burada değişen bu. Çekirdek bir cümleyi ancak belleğin onayladığı içerik
kelimeleriyle kurabilir; onaylanmamış bir içerik kelimesi üretim sırasında
erişilemez durumdadır — olasılığı düşük değil, **eksi sonsuz**. "Penguen uçar"
cümlesi kurulamıyor çünkü o cümleyi kuracak parçalar o an açık değil.

İki kelime türü ayrılır:

    içerik      kavram, nitelik, eylem — bunlar olgudur, kapıdan geçer
    bağlayıcı   "bir", "ve", "çünkü", ekler — bunlar olgu taşımaz, serbesttir

Bu ayrım keyfi değil, dilbilimin kapalı/açık sınıf ayrımı: bağlayıcılar sayıca
sabittir ve yeni üye almaz, bu yüzden içlerinden yeni bir iddia çıkamaz. Bir dil
için bir kez yazılır ve büyümez.

Doğrulanabilir özellik şudur: onaylanmamış hiçbir içerik kelimesi çıktıda
görünemez. Bu bir eğilim değil, arama uzayının şekli.
"""
import torch

# Türkçe'nin kapalı sınıfı: olgu taşımayan, yeni üye almayan kelimeler.
GLUE = (
    "bir", "birer", "ve", "ile", "ya", "veya", "ama", "fakat", "çünkü", "ki",
    "de", "da", "ise", "hem", "ne", "değil", "yok", "var", "için", "gibi",
    "göre", "kadar", "sonra", "önce", "en", "çok", "az", "daha", "bu", "şu",
    "o", "her", "bazı", "hiç", "tüm", "bütün", "evet", "hayır", "ancak",
    "yalnız", "sadece", "böyle", "şöyle", "öyle", "yani", "ayrıca", "ise",
    # Sınıflandırma kelimeleri: içerik taşımazlar, yapı kurarlar. "bir kuş
    # TÜRÜDÜR" cümlesindeki iddia kuştur, tür değil.
    "tür", "türü", "türüdür", "çeşit", "çeşidi", "cins", "cinsi", "olan",
    "olarak", "biri", "biridir", "şey", "şeydir",
)
PUNCTUATION = (".", ",", "!", "?", ";", ":", "-", "(", ")", "'")


class GatedVoice:
    """Belleğin onayladığını akıcı söyleyen, onaylamadığını söyleyemeyen ses."""

    def __init__(self, core, pieces, device="cpu", glue=GLUE):
        self.core = core
        self.pieces = pieces
        self.device = device
        self.glue_pieces = self._pieces_for(glue + PUNCTUATION)
        self.always = self.glue_pieces | self._structural()

    def _structural(self):
        """Yalnızca cümlenin bitebilmesi için gerekenler.

        Burada bir kez tek harfli TÜM parçalar serbest bırakılmıştı; gerekçesi
        "harf harf kelime kurmak pratikte olmaz" idi ve yanlıştı. 16.000
        parçalık gerçek bir sözlükte tek harfler her kelimeyi kurar ve model
        tam olarak bunu yaptı: onaylanmamış "iğere" harf harf üretildi.
        Sahte parçalayıcıyla yazılmış test bunu göremedi çünkü içinde yalnızca
        iki tek-harf parçası vardı.

        Ders, kapının kendisinden daha genel: bir yasak, yasaklananın etrafından
        dolaşan bir yol bırakıyorsa yasak değildir. Şimdi izinli küme yalnızca
        onaylanmış içerik, kapalı sınıf ve noktalamadan oluşuyor.
        """
        return {self.pieces.eos_id()}

    def _pieces_for(self, words):
        found = set()
        for word in words:
            for form in (word, " " + word):
                found.update(self.pieces.encode(form))
        return found

    def permit(self, content):
        """İzin verilen parça kümesi: bağlayıcılar + yalnızca bu içerik."""
        return sorted(self.always | self._pieces_for(content))

    def say(self, content, opening="", length=40, temperature=0.7,
            unrestricted=False):
        """İçerik kelimeleriyle bir cümle kurar.

        `unrestricted` verildiğinde çekirdek tüm sözlüğünü kullanır; denetim
        üretimden sonra, cümle düzeyinde yapılır. Parça düzeyindeki kısıtlama
        güvenli ama dilsizdi — ölçüldü ve öyle çıktı.
        """
        allowed = (None if unrestricted
                   else torch.tensor(self.permit(content), device=self.device))
        seed = self.pieces.encode(opening) or [self.pieces.bos_id()]
        tokens = torch.tensor([seed], device=self.device)
        produced = self.core.continue_from(tokens, length=length,
                                           temperature=temperature,
                                           allowed=allowed)
        return self.pieces.decode(produced[0].tolist()).strip()

    def escaped(self, text, content):
        """Çıktıda, izin verilmemiş bir içerik kelimesi var mı?

        Testin kendisi. Boş liste dönerse kapı tutmuştur.

        Buradaki ilk sürüm önekin ilk üç harfini karşılaştırıyordu ve bu, denetimi
        tam da en çok gerektiği yerde çürütüyordu: "uçar" ile "uçamaz" ilk üç
        harfte aynıdır, yani olumsuz bir olgu olumlu iddiaya lisans veriyordu.
        Şimdi ilişki tam kelime üzerinden kuruluyor — üretilen kelime izinli
        kelimenin ekli hali olabilir, ama ondan sapan bir kelime olamaz.
        """
        permitted = set(GLUE)
        for word in content:                    # "hayvanlar.txt" -> ikisi de
            permitted.update(words_in(word))
        loose = []
        for stripped in words_in(text):
            if len(stripped) <= 1:
                continue
            if not any(_licensed(stripped, allowed) for allowed in permitted):
                loose.append(stripped)
        return loose


def words_in(text):
    """Metni kelimelere ayırır — denetimle üretim aynı bölmeyi kullanmalı.

    Kaynak adı "hayvanlar.txt" içerik listesinde tek parça, çıktı denetiminde
    iki parça sayılıyordu ve doğru bir cevap kendi kaynağı yüzünden "sızdı"
    diye işaretleniyordu. İki taraf aynı işlevi çağırırsa bu olmaz.
    """
    found = []
    for raw in text.lower().replace(".", " ").replace(",", " ").split():
        word = raw.strip("'\"()-:;!?")
        if word:
            found.append(word)
    return found


# Ünsüz yumuşaması: "ağaç" + ünlü -> "ağacı". Kapı bunu bilmediği için doğru
# üretilmiş cümleleri kaçak sayıyordu — `grammar.py` ve `wording.py` aynı
# tabloyu zaten kullanıyor, burada eksikti.
SOFTENS = {"t": "d", "k": "ğ", "p": "b", "ç": "c"}
# "renk" -> "rengi": n'den sonraki k, ğ değil g olur. Tek harfe bakan tablo
# bunu kaçırıyordu.
SOFTENS_AFTER_N = {"k": "g"}

# ÖNEK olarak lisans verebilme hakkı, kelimenin uzunluğuna değil TÜRÜNE bağlı.
#
# Önce uzunluğa bakıldı (`SHORTEST_PREFIX = 4`) ve sızıntı kapandı: izinli
# kümedeki `o`, `de`, `ne`, `ve`, `bu`, `en`, `tür` bağlaçları yüzünden
# `denizaltı`, `nefret`, `vergi`, `türkiye` onaysız geçiyordu — en sık 20.000
# kelimenin %15,1'i.
#
# Ama uzunluk yanlış ölçüt: `kuş`, `göz`, `dil`, `ses` de üç harfli ve onlar
# GERÇEK kavram. Ölçüldü — 314 kavram (%1,9) kendi çekimini onaylatamaz oldu.
#
# Doğru ölçüt zaten elimizde: bağlaç listesi bildirilmiş bir kapalı sınıf.
# Bağlaç yalnız TAM eşleşmeyle lisans verir; içerik kelimesi uzunluğu ne
# olursa olsun kendi çekimlerini onaylatır.
GLUE_WORDS = frozenset(GLUE)


def _softened(word):
    """Son ünsüzü yumuşamış hâli, yoksa None."""
    if not word:
        return None
    if len(word) > 1 and word[-2] == "n":
        soft = SOFTENS_AFTER_N.get(word[-1])
        if soft:
            return word[:-1] + soft
    soft = SOFTENS.get(word[-1])
    return word[:-1] + soft if soft else None


def _licensed(produced, allowed):
    """Üretilen kelime, izinli kelimenin bir çekimi mi?

    "penguen" -> "penguenin" olur (ek eklenmiş).
    "hızlıdır" -> "hızlı" olur (ek düşmüş, kök korunmuş).
    "ağaç" -> "ağacı" olur (ünsüz yumuşaması).
    "uçamaz" -> "uçar" OLMAZ; ikisi ayrı iddialardır.
    "de" -> "denizaltı" OLMAZ; kısa kelime önek olarak lisans veremez.
    """
    if not allowed or not produced:
        return False
    if produced == allowed:
        return True
    if allowed in GLUE_WORDS:
        return False        # bağlaç yalnız kendisini lisanslar
    if produced.startswith(allowed):
        return True
    soft = _softened(allowed)
    if soft and produced.startswith(soft):
        return True
    return (produced not in GLUE_WORDS
            and allowed.startswith(produced)
            and len(produced) >= len(allowed) - 3)


def approved(memory, concepts, reasoning=None):
    """Bu kavramlar hakkında grafın bildiği HER ŞEY.

    Onay kümesi başta yalnızca o anki cevaptı ve fazla dardı: model "kartal
    sıcakkanlı bir KUŞTUR" dediğinde kapı "kuş"u kaçak saydı — oysa "kartal bir
    kuştur" grafta yazılı. Model uydurmamıştı, grafın başka bir olgusunu
    kullanmıştı.

    Doğru sınır şu: sistem, sorulan kavramlar hakkında **bildiği** her şeyi
    söyleyebilir. Bilmediğini söyleyemez. Kaynak her hâlükârda graf.
    """
    # Atalar da dahil: "zürafa bir memelidir" ve "memeli bir hayvandır"
    # biliniyorsa, sistem zürafaya hayvan diyebilir. Kalıtım zaten muhakemenin
    # yaptığı şey; onay kümesinin ondan dar olması, sistemin bildiğini
    # söylemesini engelliyordu.
    reach = list(concepts)
    if reasoning is not None:
        for concept in concepts:
            reach.extend(reasoning.ancestors(concept))
    words = set()
    for concept in reach:
        for edge in memory.query(concept):
            for part in (edge.concept, edge.relation, edge.target, edge.object):
                if part:
                    words.update(part.lower().split())
    return words


def content_of(chain):
    """Bir muhakeme zincirinden içerik kelimelerini çıkarır.

    Zincir zaten kapıdan geçmiş, kaynağı bilinen bir cevaptır; buradaki iş onu
    çekirdeğin kullanabileceği kelime listesine indirgemek.
    """
    found = []
    for line in chain:
        for word in line.replace("(", " ").replace(")", " ").split():
            cleaned = word.strip(".,;:'\"").lower()
            if cleaned and cleaned not in found:
                found.append(cleaned)
    return found
