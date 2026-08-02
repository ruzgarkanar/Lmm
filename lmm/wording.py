"""Söyleyiş öğrenmek: anlaşılmayan bir cümleden kalıcı bir kalıp çıkarmak.

Bu, projenin kendi tezinin **dile** uygulanmış hâli. Bilgi için zaten böyle
çalışıyor: bilmiyorsa gidip okuyor, denetliyor, grafa künyesiyle yazıyor, bir
daha sormuyor (`lmm/inquiry.py`). Dil için yapılmıyordu — ayrıştırıcı 48 elle
yazılmış kalıptı ve hiç büyümüyordu.

Ölçümler bu boşluğu tek başına bıraktı. Bu gece dört şey elendi:

    bilgi eksik değil          niyet elle verilince graf her şeyi cevaplıyor
    derinlik eksik değil       500 sıçramalık kompozisyon, 0,21 ms
    instruction tuning gerekmez  o aşamanın işini yapı görüyor
    kelime eksik değil         eş anlamlılık graftan okunuyor ve çalışıyor

Geriye tek şey kaldı: cümlenin **yapısını** tanımak. Damıtma denendi ve gerçek
işi düşürdü (%72 -> %56), çünkü üreticinin dağılımına uydu — konuşan insanın
diline değil.

Buradaki yol başka: **anlaşılmayan cümle geldiğinde bir kez sor, doğrula, yaz.**
Yakınsadığı yer genel bir dağılım değil, bu insanın konuşma biçimi.

Üç kapı var ve üçü de olgular için zaten kurulmuş olanların aynısı:

    kaynak      dil modeli hangi kelimenin ne olduğunu söyler — bir otorite
                değil, bir okuyucu
    denetim     çıkan niyet grafta GERÇEKTEN cevaplanabiliyor mu; uydurma bir
                okuma cevap üretemez, dolayısıyla kalıp da yazılmaz
    künye       her kalıp hangi cümleden öğrenildiğini taşır, silinebilir

Kalıp uydurma riski buranın asıl tehlikesi ve denetim tam ona karşı: bir kalıp,
işe yaradığı görülmeden yazılmıyor.
"""
import json
import re

from lmm.grammar import Pattern, learn_pattern
from lmm.intuition import (ASK, ASK_ABILITIES, ASK_DESCRIBE, ASK_PROPERTIES,
                           ASK_WHY, TEACH, tokenize)
from lmm.relations import CAN, CANNOT, HAS_PROPERTY, IS_A

# Modelin seçebileceği okumalar. Kapalı bir küme — model niyet icat edemez,
# yalnızca var olanlardan birini gösterebilir.
KINDS = {
    "TANIM_SOR": (ASK, IS_A),
    "YETENEK_SOR": (ASK, CAN),
    "NITELIK_SOR": (ASK, HAS_PROPERTY),
    "YETENEKLERI_SOR": (ASK_ABILITIES, None),
    "NITELIKLERI_SOR": (ASK_PROPERTIES, None),
    "ANLAT": (ASK_DESCRIBE, None),
    "NEDEN_SOR": (ASK_WHY, CAN),
}

BRIEF = """Sana anlaşılmayan bir Türkçe cümle verilecek. Görevin, cümlenin NE
İSTEDİĞİNİ ve hangi kelimenin NEYİ gösterdiğini söylemek.

Okuma türleri — yalnız bunlardan birini seç:
  TANIM_SOR         "kartal nedir"              bir şeyin ne olduğu soruluyor
  YETENEK_SOR       "penguen uçar mı"           belli bir şeyi yapıp yapmadığı
  NITELIK_SOR       "kartal hızlı mı"           belli bir niteliği olup olmadığı
  YETENEKLERI_SOR   "kartal ne yapabilir"       yapabildiklerinin tamamı
  NITELIKLERI_SOR   "kartal nasıldır"           niteliklerinin tamamı
  ANLAT             "penguen anlat"             serbest tarif isteniyor
  NEDEN_SOR         "penguen neden uçamaz"      sebebi soruluyor

Cevabını tek satırda, şu biçimde ver:
  TÜR | kavramın_kaç_ıncı_kelime_olduğu | hedefin_kaç_ıncı_kelime_olduğu

Kelimeler 0'dan sayılır. Hedef yoksa -1 yaz.

Örnek:  "bana penguenlerden bahseder misin"
        kelimeler: 0=bana 1=penguenlerden 2=bahseder 3=misin
        cevap: ANLAT | 1 | -1

Başka hiçbir şey yazma."""

# Kaç kelimeye kadar kalıp öğrenilir. 7 idi ve gerekçesi "daha uzun cümleden
# çıkan kalıp yalnız kendisiyle eşleşir" idi. Ölçüldü ve gerekçe yanlış yerde:
# kalıbın genelleşmesi 3-5 kelime arasında çöküyor, 7'de değil. 7 kelimede
# yeniden kullanım %1,57, 8 kelimede %0,95 — aynı büyüklük sınıfı.
#
# Yani sınır işe yaramayan kalıpları ENGELLEMİYORDU, yalnızca gerçek web
# sorularının yarısından öğrenmeyi reddediyordu. 10 ile kapsam %50'den %68'e
# çıkıyor; risk düşük, çünkü uzun kalıp daha özgüldür — yanlış eşleşme değil,
# hiç eşleşmeme üretir.
LONGEST = 10


class Reading:
    """Bir cümlenin okunuşu: ne isteniyor ve hangi kelime ne."""

    def __init__(self, kind, relation, concept_at, target_at):
        self.kind = kind
        self.relation = relation
        self.concept_at = concept_at
        self.target_at = target_at


def parse_reply(reply, count):
    """Modelin cevabından okuma. Biçim bozuksa None — tahmin edilmez."""
    for line in (reply or "").splitlines():
        parts = [piece.strip() for piece in line.split("|")]
        if len(parts) != 3 or parts[0].upper() not in KINDS:
            continue
        try:
            concept_at, target_at = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if not 0 <= concept_at < count:
            continue
        if target_at >= count:
            continue
        kind, relation = KINDS[parts[0].upper()]
        return Reading(kind, relation, concept_at,
                       target_at if target_at >= 0 else None)
    return None


def ask_model(sentence, reader):
    """Modele bir kez sorar. Cevabı ayrıştırılamazsa None."""
    tokens = tokenize(sentence)
    numbered = " ".join(f"{i}={t}" for i, t in enumerate(tokens))
    reply = reader([f"cümle: {sentence}\nkelimeler: {numbered}"])
    return parse_reply(reply, len(tokens))


def answerable(reading, tokens, memory, gate, morphology=None):
    """Bu okuma grafta GERÇEKTEN cevaplanıyor mu.

    Denetim burada. Uydurma bir okuma cevap üretemez — kavramı grafta yoktur,
    ya da kapı "bilmiyorum" der. O yüzden kalıp, işe yaradığı görülmeden
    yazılmıyor; olgular için çapa denetiminin yaptığı işin aynısı.
    """
    from lmm.intuition import Intent
    from lmm.phrasing import is_a_refusal
    concept = _concept_at(tokens, reading.concept_at, memory, morphology)
    if concept is None:
        return False, None
    target = tokens[reading.target_at] if reading.target_at is not None else None
    said = gate.answer(Intent(reading.kind, concept, reading.relation, target))
    if is_a_refusal(said):
        return False, None
    return True, said


def _concept_at(tokens, index, memory, morphology=None):
    """Kavramı yalın hâline indirir. Grafta yoksa None.

    "penguenlerden" grafta durmuyor, "penguen" duruyor. Durum eki soyulmadan
    her okuma reddediliyordu — kapı çalışıyordu ama yanlış sebeple.

    Biçimbilim dışarıdan veriliyor, bellekten okunmuyor: `Memory` nesnesinin
    `morphology` diye bir özniteliği YOK ve `getattr(memory, "morphology", None)`
    her yerde sessizce None dönüyordu. Aynı sessiz hata `cli.py` içinde de
    duruyordu — hiç fark edilmemiş, çünkü sonuç yalnızca "biraz daha az
    anlıyor" oluyordu.
    """
    word = tokens[index]
    known = set(memory.concepts())
    for candidate in _forms(word, morphology, known):
        if candidate in known:
            return candidate
    return None


PEELS = 4       # "penguenlerden" iki ek taşıyor; dördü fazlasıyla yeter


def _forms(word, morphology, known=()):
    """Bir kelimenin denenmeye değer hâlleri, en az soyulandan çoğa.

    Tek geçiş yetmiyor: "penguenlerden" hem çoğul hem çıkma eki taşıyor ve
    ikisi ayrı ayrı soyuluyor. İlk yazışta tek geçiş yapılıyordu ve "penguen"e
    hiç ulaşılamıyordu — kapı her okumayı reddediyordu, ama yanlış sebeple.

    Soymak katman katman yapılıyor ve her ara hâl de aday: hangi ekin gerçek
    olduğunu burada bilemeyiz, grafta hangisinin durduğuna bakarız.
    """
    from lmm.frames import case_of
    seen, edge = [word], [word]
    for _ in range(PEELS):
        following = []
        for form in edge:
            candidates = []
            if morphology is not None:
                candidates.append(morphology.strip_plural(form))
                # İyelik eki: "penguenin özellikleri neler" sorusunda kavram
                # bulunamıyor, ilk bulunan başka düğüm alınıyordu. Biçimbilim
                # dile göre değişir, o yüzden varlığı sorulmadan çağrılmıyor.
                genitive = getattr(morphology, "strip_genitive", None)
                if genitive is not None:
                    candidates.append(genitive(form, known))
            candidates.append(case_of(form)[0])
            for candidate in candidates:
                if candidate and candidate != form and candidate not in seen:
                    seen.append(candidate)
                    following.append(candidate)
        if not following:
            break
        edge = following
    return seen


def concept_in(tokens, memory, morphology=None, focus=None):
    """Cümledeki kavramı yapısal olarak bulur: grafta duran ilk kelime.

    Ağ yalnızca niyet TÜRÜNÜ söylüyor, hangi kelimenin kavram olduğunu değil.
    Onu uydurmak yerine grafa soruyoruz — cümlede grafta karşılığı olan ilk
    kelime kavramdır. Hiçbiri yoksa None, ve o zaman okuma kullanılmaz.
    """
    from lmm.relations import SAME_AS
    # Zamir varsa sohbetin odağına düşülür: "bunu bana anlatır mısın"
    # sorusunun kavramı az önce konuşulan şeydir. Odak yoksa uydurulmuyor.
    if focus:
        from lmm.intuition import PRONOUNS
        for index, token in enumerate(tokens):
            if token in PRONOUNS:
                return index, focus
    best = (None, None, -1)
    for index in range(len(tokens)):
        found = _concept_at(tokens, index, memory, morphology)
        if found is None:
            continue
        # Hakkında OLGU olan kavram tercih ediliyor. Eş anlamlılık düğümleri
        # de grafta duruyor ("özellik = nitelik") ve ilk bulunanı almak
        # "penguenin özellikleri neler" sorusunun kavramını `özellik` yapıyordu.
        # Bir kavram, hakkında bir şey bilinen şeydir.
        weight = sum(1 for edge in memory.query(found)
                     if edge.relation != SAME_AS)
        if weight > best[2]:
            best = (index, found, weight)
        if weight and best[1] == found:
            break
    return best[0], best[1]


def request_words(kind, patterns):
    """Bu niyetin kalıplarında geçen sabit sözcükler — "anlat", "yapabilir"."""
    from lmm.grammar import SLOTS
    found = set()
    for pattern in patterns:
        if pattern.kind != kind:
            continue
        for token in pattern.tokens:
            if token not in SLOTS:
                found.add(token)
    return found


def accounted_for(kind, tokens, lexicon, patterns, meanings=None, consumed=()):
    """Cümledeki her bilinen fiil bu okumayla açıklanıyor mu.

    Bu, kalıcı hafızayı koruyan kapı ve gerçek bir kusurdan doğdu: denetim
    "graf bu okumayı cevaplayabiliyor mu" diye soruyordu, "okuma cümleye
    uyuyor mu" diye değil. Sonuç ölçüldü — "bir kuşun uçabilmesi için ne
    gerekir" cümlesi "kuş ne yapabilir" diye okundu, graf o okumayı
    cevapladı, denetim geçti ve **yanlış kalıp kalıcı yazıldı.**

    Ayıran işaret cümlenin fiillerinde:

        bahseder misin    ASK_DESCRIBE kalıpları "anlat" içeriyor ve grafta
                          "anlat = bahset" yazıyor           -> açıklanıyor
        ne gerekir        ASK_ABILITIES kalıpları "yapabilir" içeriyor;
                          "gerekmek" hiçbirine bağlı değil   -> açıklanmıyor

    Yani sistem, envanterinde karşılığı olmayan bir soruyu yaklaşık bir
    ilişkiye çevirmek yerine geri çekiliyor. Yanlış kural yazmak, kural
    yazamamaktan pahalıdır — hafıza bu mimarinin tek varlığı.

    `consumed`: okumanın KENDİ kullandığı kelimelerin yerleri. Bunlar tanım
    gereği açıklanmıştır ve sorulmamalıdır. Yokken kapı iki doğru okumayı
    kesiyordu:

        kar gerçekten beyaz mıdır   `kar` cümlenin KAVRAMI, ama derlem onu bir
                                    fiil çekimi sanıyor ("karmak") ve kapı
                                    "açıklanmayan fiil" diye reddediyordu
        penguen uçar mı             `uçar` cümlenin HEDEFİ; istek sözcüğü
                                    olmadığı için açıklanmamış sayılıyordu

    Ayrım şu: kapı, okumanın DIŞINDA kalan fiilleri sorgulamak için var —
    "ne gerekir"deki `gerekir` gibi. Okumanın içindekini sorgulamak, kendi
    cevabını reddetmek demek.
    """
    from lmm import frequency
    asked = request_words(kind, patterns)
    wider = frequency.verbs()
    for index, token in enumerate(tokens):
        if index in consumed:
            continue
        # Oturumun sözlüğü yalnızca öğretilmiş fiilleri bilir; burada sorulan
        # şey daha geniş: "bu kelime Türkçe'de bir fiil mi?" Cevabı derlem
        # veriyor. Bu ayrım olmadan denetim boşa çıkıyordu — "gerekir"
        # tanınmadığı için cümlede hiç fiil yok sanılıyordu.
        known = lexicon.reading(token) is not None or _a_verb(token, wider)
        if not known:
            continue
        if not _explained(token, asked, meanings):
            return False
    return True


def _a_verb(token, wider):
    """Derlem bu kelimeyi bir fiil çekimi olarak biliyor mu."""
    if token in wider:
        return True
    for surface in wider:
        if len(surface) > 3 and _starts_with(token, surface[:-1]):
            return True
    return False


# Ünsüz yumuşaması: gövde sonundaki sert ünsüz, ünlüyle başlayan ek alınca
# yumuşar. "bahset" + "er" -> "bahsed"er. Düz önek karşılaştırması bunu
# kaçırıyor ve doğru okumayı reddediyordu — dilin kendi kuralı, bir istisna
# değil. `lmm/turkish.py` aynı eşlemeyi ters yönde zaten kullanıyor.
SOFTENS = {"t": "d", "k": "ğ", "p": "b", "ç": "c"}


def _starts_with(token, stem):
    """Kelime bu gövdeyle başlıyor mu — ünsüz yumuşaması sayılarak."""
    if token.startswith(stem):
        return True
    if not stem:
        return False
    soft = SOFTENS.get(stem[-1])
    return bool(soft) and token.startswith(stem[:-1] + soft)


def _explained(token, asked, meanings=None):
    """Bu fiil, niyetin kendi sözcüklerinden biriyle açıklanıyor mu."""
    for word in asked:
        if _starts_with(token, word):
            return True
        for other in (meanings(word) if meanings else ()):
            if _starts_with(token, other):
                return True
    return False


def learn(sentence, reading, lexicon, morphology):
    """Okumadan yeniden kullanılabilir bir kalıp. Uzun cümleden kalıp çıkmaz."""
    tokens = tokenize(sentence)
    if not 1 < len(tokens) <= LONGEST:
        return None
    return learn_pattern(tokens, reading.kind, reading.relation,
                         reading.concept_at, reading.target_at,
                         lexicon, morphology)
