"""Turkish surface language: the lexicon and every user-facing wording.

All Turkish lives here. The other organs speak in concepts and relations, and
this module is the only place that turns them into sentences — so the internal
labels never leak into what the system says, and a second language would mean
swapping this file alone.
"""
from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                           LACKS_PROPERTY, HAS_PART, LACKS_PART, PLACE, SOURCE,
                           ALL, MOST, SOME, NO)
from lmm.lexicon import ACTIVE

from lmm.trust import INFERENCE, DISTILLED_PREFIX

_BACK_UNROUNDED = "aı"
_FRONT_UNROUNDED = "ei"
_BACK_ROUNDED = "ou"
_VOICELESS = "fstkçşhp"


def _harmony_vowel(word):
    """The vowel four-way harmony picks for a suffix following this word."""
    vowels = [c for c in word if c in "aeıioöuü"]
    last_vowel = vowels[-1] if vowels else "a"
    if last_vowel in _BACK_UNROUNDED:
        return "ı"
    if last_vowel in _FRONT_UNROUNDED:
        return "i"
    if last_vowel in _BACK_ROUNDED:
        return "u"
    return "ü"


def copula(word):
    """The -DIr suffix, obeying vowel harmony and consonant assimilation."""
    consonant = "t" if word and word[-1] in _VOICELESS else "d"
    return consonant + _harmony_vowel(word) + "r"


def case_form(word, role):
    """Put the case ending back on when speaking: kutup + yer -> "kutupta".

    Two-way harmony for the vowel, voicing for the consonant — the same rules
    the copula follows, because they are the language's rules and not this
    suffix's.
    """
    if role not in (PLACE, SOURCE):
        return word
    vowels = [c for c in word if c in "aeıioöuü"]
    last = vowels[-1] if vowels else "a"
    vowel = "a" if last in "aıou" else "e"
    consonant = "t" if word and word[-1] in _VOICELESS else "d"
    suffix = consonant + vowel + ("n" if role == SOURCE else "")
    return word + suffix


def clitic_da(word):
    """The separate "da"/"de", which harmonises two ways, not four."""
    vowels = [c for c in word if c in "aeıioöuü"]
    last = vowels[-1] if vowels else "a"
    return "da" if last in "aıou" else "de"


def question_particle(word):
    """The mı/mi/mu/mü that follows this word."""
    return "m" + _harmony_vowel(word)


def verb_form(infinitive, positive, lexicon=None):
    """"uçmak", True -> "uçar"; "uçmak", False -> "uçamaz"."""
    return (lexicon or ACTIVE).surface(infinitive, positive)


def is_a_clause(concept, target):
    """penguen, kuş -> "penguen bir kuştur"."""
    return f"{concept} bir {target}{copula(target)}"


def is_not_a_clause(concept, target):
    """penguen, memeli -> "penguen bir memeli değildir"."""
    return f"{concept} bir {target} değildir"


def denied(concept, target, reasons=()):
    """Tür sorusuna "hayır", dayandığı kanıtı da göstererek.

    Gerekçe süs değil sınır: bu cümle ancak kayıtlı bir ret varsa kurulabilsin
    diye kapı gerekçeyi getirmek zorunda. Gerekçesiz "hayır" yalnız doğrudan
    reddin kendisidir.
    """
    clause = is_not_a_clause(concept, target)
    if not reasons:
        return f"hayır, {clause}."
    return f"hayır, {clause}, çünkü {listing(list(reasons))}."


def property_clause(concept, prop, positive=True, object=None, role=None):
    """kar, beyaz -> "kar beyazdır"; with a second concept, "kartal serçeden
    büyüktür"."""
    middle = f"{case_form(object, role)} " if object else ""
    if positive:
        return f"{concept} {middle}{prop}{copula(prop)}"
    return f"{concept} {middle}{prop} değildir"


_SOFTENS = {"k": "ğ", "p": "b", "ç": "c", "t": "d"}


def genitive(word):
    """The possessor's ending: kuş -> kuşun, kedi -> kedinin, balık -> balığın.

    Harmony picks the vowel, whether the word ends in one picks the -n-, and a
    final voiceless consonant softens before it.
    """
    if not word:
        return word
    stem = word
    if stem[-1] in _SOFTENS and len(stem) > 2:
        stem = stem[:-1] + _SOFTENS[stem[-1]]
    buffer = "n" if word[-1] in "aeıioöuü" else ""
    return stem + buffer + _harmony_vowel(word) + "n"


def possessed(word):
    """Sahip olunanın eki: kanat -> kanadı, kedi -> kedisi, göz -> gözü.

    `genitive()`'in eşi ve bugüne kadar eksikti. Graf iyelikli biçimi tutuyor
    ("kuşun KANADI var") ama çevrimdışı okuyucu çıplak adı veriyor ("kanat");
    ikisi eşleşmeyince olgu geri okunamıyor ve reddediliyordu. Yani kusur
    okuyucuda değil, sistemin kendi ağzında: söylediği biçimi ÜRETEMİYORDU,
    yalnız kayıtta hazır bulunca söyleyebiliyordu.

    Kurallar `genitive()` ile aynı ve zaten burada: ünlü uyumu vokali seçer,
    ünlüyle biten kelime kaynaştırma alır, son sert ünsüz yumuşar.
    """
    if not word:
        return word
    stem = word
    if stem[-1] in _SOFTENS and len(stem) > 2:
        stem = stem[:-1] + _SOFTENS[stem[-1]]
    buffer = "s" if word[-1] in "aeıioöuü" else ""
    return stem + buffer + _harmony_vowel(word)


def aorist(infinitive, positive=True):
    """Mastardan geniş zaman: "tırmanmak" -> "tırmanır" / "tırmanmaz".

    SON ÇARE. Önce sözlüğe, sonra derleme bakılıyor; ikisi de bilmiyorsa
    cümle mastarla kuruluyordu ve bozuk çıkıyordu: "karakter tırmanmak".
    Bozuk cümle geri okunamıyor, dolayısıyla doğru bir olgu kapıda
    reddediliyordu — sistemin kendi ağzı, kendi bilgisini eliyordu.

    Türetme dilin bildirdiği eklerden: ünlüyle biten gövde yalnız -r alır,
    ünsüzle biten uyuma göre geniş ünlü + r. Düzensiz fiiller (gitmek ->
    gider) bu kuralla yanlış çıkar; o yüzden son çare, ilk yol değil. Yanlış
    bir çekim yine de mastardan iyidir: mastar cümleyi hiç kurdurmuyor.
    """
    # BİLEŞİK FİİL: "devam etmek", "yer almak". Çekim yalnız SON kelimede
    # olur, baştaki ad olduğu gibi durur. Ölçüldü — okuma pilotunda 36 olgu
    # tam buradan kayboldu: "devam etir", "ileri sürer" yerine bozuk biçimler
    # çıkıyor ve bozuk cümle geri okunamıyordu.
    if " " in infinitive:
        head, _, last = infinitive.rpartition(" ")
        return f"{head} {aorist(last, positive)}"
    stem = infinitive
    for ending in ("mak", "mek"):
        if stem.endswith(ending) and len(stem) > len(ending) + 1:
            stem = stem[: -len(ending)]
            break
    if not positive:
        return stem + ("maz" if _harmony_vowel(stem) in "aıou" else "mez")
    # Olumsuzda ek ÜNSÜZLE başlıyor, yumuşama olmuyor ("etmez"); olumluda
    # ünlüyle başlıyor ve gövde yumuşuyor ("eder").
    stem = _voicing_stems().get(stem, stem)
    if stem and stem[-1] in "aeıioöuü":
        return stem + "r"
    # Tek heceli gövde GENİŞ ünlü alır, çok heceli DAR: "uç" -> uçar ama
    # "tırman" -> tırmanır. İlk yazışta bu ayrım yoktu ve tek heceliler
    # "uçur", "yüzür" diye çıkıyordu. Hece sayısı ünlü sayısıdır.
    vowels = [letter for letter in stem if letter in "aeıioöuü"]
    if len(vowels) <= 1 and stem not in _narrow_stems():
        return stem + ("a" if _harmony_vowel(stem) in "aıou" else "e") + "r"
    return stem + _harmony_vowel(stem) + "r"


def _voicing_stems():
    """Ünlü önünde son ünsüzü yumuşayan gövdeler — dilden soruluyor."""
    from lmm.turkish import TurkishMorphology
    return getattr(TurkishMorphology, "voicing_aorist_stems", {})


def _narrow_stems():
    """Tek heceli olduğu hâlde dar ünlü alan gövdeler — dilden soruluyor."""
    from lmm.turkish import TurkishMorphology
    return getattr(TurkishMorphology, "narrow_aorist_stems", ())


def is_a_step(concept, ancestor):
    """Kalıtım zincirinin bir basamağı: "penguen bir kuş".

    `lmm/reasoning.py` bunu üç yerde kendi içinde kuruyordu ve Türkçe " bir "
    bağlacı motorun ortasında duruyordu. Muhakeme hangi dile baktığını
    bilmemeli — ikinci bir dil, zinciri de kendi bağlacıyla kurar.
    """
    return f"{concept} bir {ancestor}"


def own_inference(said):
    """Kalıtılan bir tahminin künyesi: "... (kendi çıkarımım)".

    Çıkarım da bir tahmindir ve öyle söylenmeli — ama SÖYLENİŞİ dile ait.
    """
    return f"{said} (kendi çıkarımım)"


def part_clause(concept, part, positive=True, object=None, role=None):
    """kuş, kanadı -> "kuşun kanadı var" / "kuşun kanadı yok"."""
    return f"{genitive(concept)} {part} {'var' if positive else 'yok'}"


def property_question(concept, prop, object=None, role=None):
    """kar, beyaz -> "kar beyaz mı?"; with a comparison, "kuş serçeden büyük mü?"."""
    middle = f"{case_form(object, role)} " if object else ""
    return f"{concept} {middle}{prop} {question_particle(prop)}?"


def part_question(concept, part):
    """kuş, kanadı -> "kuşun kanadı var mı?"."""
    return f"{genitive(concept)} {part} var mı?"


def property_summary(concept, properties):
    """kar, [(beyaz, True), (sıcak, False)] -> "kar beyazdır ama sıcak değildir"."""
    positives = [p for p, is_so in properties if is_so]
    negatives = [p for p, is_so in properties if not is_so]
    parts = []
    if positives:
        parts.append(f"{listing(positives)}{copula(positives[-1])}")
    if negatives:
        parts.append(f"{listing(negatives)} değildir")
    return f"{concept} " + " ama ".join(parts)


def listing(items):
    """["kuş", "serçe", "kartal"] -> "kuş, serçe ve kartal"."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " ve " + items[-1]


def who_clause(concepts, action, positive=True):
    """["kuş", "serçe"], uçmak -> "kuş ve serçe uçar"."""
    return f"{listing(concepts)} {verb_form(action, positive)}"


def ability_summary(concept, abilities):
    """penguen, [(yüzmek, True), (uçmak, False)] -> "penguen yüzer ve uçamaz".

    Üçlü yerine dörtlü de gelebilir: (eylem, kutup, nesne, rol). Nesne varsa
    söylenir — "kalp pompalar" değil "kalp kanı pompalar". Bir cümlenin
    LLM'inki gibi okunmasını sağlayan şey akıcı sözcük seçimi değil, eylemin
    NEYE yapıldığının orada durması.
    """
    clauses = []
    for ability in abilities:
        action, positive = ability[0], ability[1]
        object = ability[2] if len(ability) > 2 else None
        role = ability[3] if len(ability) > 3 else None
        middle = f"{case_form(object, role)} " if object else ""
        clauses.append(f"{middle}{verb_form(action, positive)}")
    return f"{concept} {listing(clauses)}"


def ability_clause(concept, action, positive, object=None, role=None):
    """penguen, uçmak, False -> "penguen uçamaz"; with a second concept,
    "kedi fare yakalar" or "penguen kutupta yaşar" depending on its role."""
    if object:
        return f"{concept} {case_form(object, role)} {verb_form(action, positive)}"
    return f"{concept} {verb_form(action, positive)}"


def definition_question(concept):
    """penguen -> "penguen nedir?"."""
    return f"{concept} nedir?"


def ability_question(concept, action, object=None, role=None):
    """penguen, yüzmek -> "penguen yüzer mi?"; with a second concept,
    "penguen kutupta yaşar mı?"."""
    verb = verb_form(action, True)
    middle = f"{case_form(object, role)} " if object else ""
    return f"{concept} {middle}{verb} {question_particle(verb)}?"


# One sample sentence per shape, used to guess at what a teacher meant.
SHAPE_EXAMPLES = {
    "TEACH_TYPE": "penguen bir kuştur",
    "TEACH_NOT_TYPE": "penguen bir memeli değildir",
    "TEACH_ABILITY": "kuşlar uçar",
    "TEACH_PROPERTY": "kuşlar tüylüdür",
    "TEACH_NOT_PROPERTY": "penguen tüylü değildir",
    "ASK_DEFINITION": "penguen nedir",
    "ASK_ABILITY": "penguen uçar mı",
    "ASK_PROPERTY": "penguen tüylü mü",
    "ASK_WHO": "kimler uçar",
    "ASK_ABILITIES": "penguen ne yapabilir",
    "ASK_PROPERTIES": "penguen nasıldır",
    "ASK_WHY": "penguen neden uçamaz",
}

# Örnek cümleler GRAFTAN kuruluyor, elle yazılmıyor. Eskiden sabit sekiz
# penguen cümlesi vardı ve sistem 486 kavram bilirken hep onları gösteriyordu —
# kendini olduğundan aptal gösteren bir hata mesajı. Şimdi gerçekten bildiği
# bir kavramla örnek veriyor.
SHAPE_MOULDS = {
    "ASK_DEFINITION": "{} nedir",
    "ASK_ABILITY": "{} {} mı",
    "ASK_PROPERTY": "{} {} mı",
    "ASK_ABILITIES": "{} ne yapabilir",
    "ASK_PROPERTIES": "{} nasıldır",
    "ASK_DESCRIBE": "{} anlat",
    "ASK_WHY": "{} neden {}",
    "TEACH_TYPE": "{} bir {}dır",
}


def _example(kind, memory):
    """Grafın gerçekten bildiği bir kavramla örnek cümle. Yoksa None."""
    mould = SHAPE_MOULDS.get(kind)
    if not mould or memory is None:
        return None
    from lmm.relations import CAN, HAS_PROPERTY, IS_A
    wanted = {"ASK_ABILITY": CAN, "ASK_WHY": CAN,
              "ASK_PROPERTY": HAS_PROPERTY, "TEACH_TYPE": IS_A}.get(kind)
    for edge in memory.edges:
        if wanted and edge.relation != wanted:
            continue
        if mould.count("{}") == 2:
            if not edge.target:
                continue
            # Fiilin MASTARI değil çekimi yazılır: "kuş uçmak mı" değil
            # "kuş uçar mı". Sözlük zaten yüzey biçimini biliyor.
            target = edge.target
            if wanted == CAN and getattr(memory, "lexicon", None):
                target = memory.lexicon.surface(edge.target, True)
            made = mould.format(edge.concept, target)
            return _harmonised(made)
        return mould.format(edge.concept)
    return None


# Soru ekinin ünlüsü son heceye uyar. Örnek cümle üretirken "tüylü mı"
# yazmak, sistemin kendi dilini bilmediğini gösterir.
BACK, FRONT = "aıou", "eiöü"


def _harmonised(sentence):
    """Sondaki soru ekini son ünlüye uydurur."""
    words = sentence.split()
    if len(words) < 2 or words[-1] not in ("mı", "mi", "mu", "mü"):
        return sentence
    vowels = [c for c in words[-2] if c in BACK + FRONT]
    if not vowels:
        return sentence
    last = vowels[-1]
    if last in "aı":
        words[-1] = "mı"
    elif last in "ei":
        words[-1] = "mi"
    elif last in "ou":
        words[-1] = "mu"
    else:
        words[-1] = "mü"
    return " ".join(words)


def known_shapes(memory=None, count=6):
    """Sistemin gerçekten cevaplayabildiği örnek cümleler."""
    found = []
    for kind in SHAPE_MOULDS:
        made = _example(kind, memory)
        if made:
            found.append(f"'{made}'")
        if len(found) >= count:
            break
    if found:
        return ", ".join(found)
    return "'<kavram> nedir', '<kavram> ne yapabilir', '<kavram> nasıldır'"


def not_understood(resembles=None, memory=None, spotted=(), long=False):
    """What to say when nothing parsed — with a guess, if there is one.

    Uzun bir cümleye örnek kalıp saymak ("şunları sorabilirsin: 'kuş tüylü
    mü'...") ölçümde en utandırıcı cevaptı: şeker hastası hamileler sorusuna
    kuş örneği. Örnekler keşfeden kullanıcıya yardım eder; gerçek bir soruyu
    anlamamışsak doğru hamle LLM'inkiyle aynı — AÇIKLAMA İSTEMEK, ders vermek
    değil. Cümlede tanınan bir kavram varsa o da söylenir: en azından neyi
    yakaladığımız belli olur.
    """
    example = _example(resembles, memory) or SHAPE_EXAMPLES.get(resembles)
    # Şekil örneği yalnız KISA cümlede yardım eder: "kartal uçarmı" yazana
    # "'kuş uçar mı' gibi mi?" demek yol gösterir. Beş kelimelik gerçek bir
    # soruya kuş örneği vermek ölçümdeki en utandırıcı cevaptı.
    if example and not long:
        return (f"bunu anlamadım. '{example}' gibi bir şey mi demek istedin? "
                f"kelimelerinden birini bilmiyor olabilirim.")
    # İki yeni söyleyiş de "anlamadım" taşıyor — `is_a_refusal` cevabı
    # ret olarak tanımalı, yoksa aday yolu ve öğrenme denemeleri açılmıyor.
    if spotted:
        return (f"bu cümleyi tam anlamadım ama {listing(list(spotted))} "
                f"hakkında bir şeyler biliyorum. sorunu tek cümleyle, düz "
                f"bir soru olarak sorar mısın?")
    if long:
        return ("bu cümleyi tam anlamadım. tek cümlelik, düz bir soru "
                "olarak sorar mısın?")
    return f"bunu anlamadım. şunları sorabilirsin: {known_shapes(memory)}."


def acknowledged(*_):
    """Tepkiye tepki: kısa, ve top yine karşıda."""
    return "öyle. devam edelim mi — başka ne sorayım dersen buradayım."


def dismissed(*_):
    """Konu kapandı; küslük yok."""
    return "tamam, geçtim. başka bir şey sor istersen."


def challenged(*_):
    """"Yanlışsın" içeriksiz bir itiraz: doğrusunu istemek tek dürüst cevap.

    Düzeltme mekanizması zaten var — kullanıcı doğrusunu tek cümleyle yazarsa
    kayıt düzeltiliyor. Buradaki iş yalnız o kapıyı göstermek.
    """
    return ("olabilir — neyi yanlış bildiğimi söyler misin? doğrusunu tek "
            "cümleyle yazarsan kaydımı düzeltirim.")


def directive_set(kind):
    """Yönerge alındı; tek cümleyle onay."""
    said = {"short": "tamam, bundan sonra kısa cevap veririm.",
            "long": "tamam, bundan sonra ayrıntılı anlatırım.",
            "plain": "tamam, kaynakları artık göstermem.",
            "cited": "tamam, cevaplarda kaynağı da söylerim."}
    return said.get(kind, "tamam.")


import re as _re

_CITATION = _re.compile(r"\s*\((?:doğrudan bilgi, )?kaynak(?:larım)?: [^)]*\)")


def directed(said, directives):
    """Oturumun yönergelerini söylenmiş cevaba uygular.

    LLM'de prompt üretimin İÇİNİ şartlar; burada üretim graftan geliyor ve
    zaten doğru, şartlanan yalnız SÖYLEYİŞ: uzunluk ve kaynak gösterimi.
    Kısaltma cümle sınırından yapılır, cümlenin içi kesilmez — kesilen bilgi
    kaybolmuyor, "devamını anlat" ile alınabilir.
    """
    if not said or not directives:
        return said
    if directives.get("sources") is False:
        said = _CITATION.sub("", said)
    if directives.get("length") == "short":
        pieces = said.split(". ")
        if len(pieces) > 2:
            said = ". ".join(pieces[:2]).rstrip(".") + "."
    return said


def which_reading(word, readings):
    """Two ways to read a possessor, and no way to choose without being told."""
    return (f"'{word}' iki türlü okunabilir: {listing(list(readings))}. "
            f"Hangisini kastettiğini bilmiyorum — önce onu bana tanıtır mısın?")


def teach_me_the_word(word):
    """Asking for vocabulary the way it asks for anything else it lacks."""
    return (f"'{word}' kelimesini bilmiyorum. şöyle öğretebilirsin: "
            f"kelime: <mastar> = {word} / <olumsuzu>")


def learned_word(infinitive, positive, negative):
    return f"kelimeyi öğrendim: {positive} / {negative} ({infinitive})."


def wondering(question):
    """How the system voices a gap it noticed in itself."""
    return f"bu arada, bunu hiç öğrenmedim: {question}"


# Sistemin "cevap veremedim" dediği hâllerin ortak işareti. Tek bir yerde
# duruyor çünkü iki yer bunu bilmek zorunda: söyleyişi öğrenen organ, bir
# okumanın gerçekten işe yarayıp yaramadığını buradan anlıyor. Bunlar kullanıcı
# metnine bakan anahtar kelimeler değil — KENDİ çıktımızın imzası.
REFUSALS = ("bilmiyorum", "anlamadım", "öğrenmedim", "duymadım",
            "öğrenmem lazım", "öğretir misin")


def is_a_refusal(said):
    """Bu cevap aslında bir cevap mı, yoksa cevap verememe mi?"""
    lowered = (said or "").lower()
    return not said or any(mark in lowered for mark in REFUSALS)


def compared(first, second, shared, only_first, only_second, kinds=None):
    """İki kavramın karşılaştırması: ortak yan, sonra ayrım.

    İlişki adları ham hâlleriyle yazılmıyor — "uçmak (cannot)" bir cevap değil
    bir döküm. İlişki kaydı zaten her ilişkinin okunabilir etiketini taşıyor
    (`lmm/kinds.py`), o kullanılıyor.

    Hiçbir şey bulunamazsa uydurulmuyor: karşılaştırma da bir iddiadır.
    """
    parts = []
    if shared:
        parts.append(f"ikisi de {' ve '.join(shared)} ile ilgili")
    for concept, traits in ((first, only_first), (second, only_second)):
        if traits:
            parts.append(f"{concept}: " + ", ".join(
                _trait(relation, target, kinds) for relation, target in traits))
    if not parts:
        return f"{first} ile {second} arasında kayıtlı bir fark bulamadım."
    return capitalize("; ".join(parts) + ".")


def _trait(relation, target, kinds=None):
    """Bir kaydı okunur hâle getirir: "uçamaz", "hızlı", "tüye sahip"."""
    if relation == "property":
        return target
    label = None
    if kinds is not None:
        kind = kinds.by_name.get(relation)
        label = kind.label if kind else None
    return f"{target} {label}" if label else f"{target} ({relation})"


def certainty(edges, kinds=None):
    """"emin misin" — güven ve kanıtın kendisi.

    Yeni bilgi gerektirmiyor: güven, kaynak ve kaç tanığın söylediği zaten her
    kenarda duruyor. Yalnızca sorulmuyordu.

    Bir LLM bu soruyu cevaplayamaz — ne güveni sayılabilir ne kaynağı vardır.
    Bizim için bedava, çünkü kayıt zaten künyeli.
    """
    if not edges:
        return "henüz bir şey söylemedim ki."
    edge = edges[0]
    witnesses = len(getattr(edge, "sources", None) or [edge.source])
    parts = [f"güvenim %{edge.confidence * 100:.0f}"]
    if witnesses > 1:
        parts.append(f"{witnesses} ayrı kaynak söylüyor")
    else:
        parts.append(f"tek kaynak: {edge.source}")
    if edge.is_exception:
        parts.append("bu bir istisna, kalıtımı bozuyor")
    if getattr(edge, "disputed", False):
        parts.append("tartışmalı — kaynaklar çelişiyor")
    return capitalize(", ".join(parts) + ".")


def whence(edges):
    """"nereden biliyorsun" — künyenin kendisi.

    Bu sorunun cevaplanabilmesi projenin dördüncü şartının somut hâli: her
    olgu nereden geldiğini taşır ve insan gidip bakabilir.
    """
    if not edges:
        return "henüz bir şey söylemedim ki."
    seen = []
    for edge in edges:
        for source in (getattr(edge, "sources", None) or [edge.source]):
            if source not in seen:
                seen.append(source)
    if len(seen) == 1:
        return f"kaynağım: {seen[0]}."
    return capitalize(f"kaynaklarım: {listing(seen)}.")


def no_opinion(concept=None):
    """Görüş sorusu. Anlamadım demekle görüşüm yok demek aynı şey değil.

    Görüşün dayanağı olmaz; dayanaksız cümle kurmayan bir sistemin görüşü de
    olamaz. Bunu söylemek bir eksiklik itirafı değil, mimarinin sonucu.
    """
    if concept:
        return (f"bu bir görüş sorusu ve benim görüşüm yok. {concept} hakkında "
                f"bildiklerimi sorabilirsin.")
    return "bu bir görüş sorusu ve benim görüşüm yok — yalnız bildiklerimi söylerim."


def inventory(memory, count=8):
    """Sistemin gerçekten ne bildiği — sayarak, uydurmadan.

    "ne biliyorsun" sorusunun cevabı bir tanıtım metni değil, bir DÖKÜM
    olmalı. Sistem 486 kavram bilirken "bunu anlamadım" diyordu; şimdi
    saydığını söylüyor.
    """
    import collections
    from lmm.relations import IS_A, SAME_AS
    concepts = [c for c in memory.concepts()]
    kinds = collections.Counter(
        e.target for e in memory.edges if e.relation == IS_A)
    sources = collections.Counter(
        e.source for e in memory.edges if e.relation != SAME_AS)
    if not concepts:
        return "henüz hiçbir şey bilmiyorum. bana bir şey öğretebilirsin."
    parts = [f"{len(memory.edges)} bilgi, {len(concepts)} kavram"]
    if kinds:
        top = ", ".join(f"{t} ({n})" for t, n in kinds.most_common(count // 2))
        parts.append(f"en çok bildiğim türler: {top}")
    if sources:
        where = ", ".join(f"{s} ({n})" for s, n in sources.most_common(3))
        parts.append(f"kaynaklarım: {where}")
    return capitalize(". ".join(parts) + ".")


def nothing_more(concept):
    """Söylenecek bir şey kalmadığında. Uydurmak yerine bitirmek."""
    if not concept:
        return "henüz bir şeyden konuşmadık, önce bir şey sor."
    return f"{concept} hakkında bildiğim başka bir şey yok."


def talked_about(topics):
    """Sohbette nelerden konuşulduğu. Hiçbiri yoksa uydurulmuyor."""
    if not topics:
        return "henüz bir şey konuşmadık."
    if len(topics) == 1:
        return f"{topics[0]} hakkında konuşuyorduk."
    return capitalize(", ".join(topics[:-1]) + f" ve {topics[-1]} "
                      "hakkında konuşuyorduk.")


def learned_wording(sentence):
    """Yeni bir söyleyiş öğrendiğini söylemesi.

    Söylenmesi önemli: sistem sessizce kural biriktirmemeli. Öğrendiği şey
    görünür olmalı ki yanlışsa silinebilsin.
    """
    return f"(bu söyleyişi de öğrendim: \"{sentence}\")"


def went_and_read(source, learned, refused=0):
    """Bilmediğini fark edip gidip okuduğunu söylemesi.

    Künye söyleniyor çünkü asıl mesele o: "öğrendim" demek kolay, nereden
    öğrendiğini gösterebilmek zor. Reddedilen varsa o da söyleniyor — kaynağın
    her dediğini almadığı, sistemin en çok gösterilmesi gereken davranışı.
    """
    if not learned:
        return f"bilmiyordum, {source} sayfasını okudum ama bir şey çıkaramadım."
    said = f"bilmiyordum, {source} sayfasını okudum: {learned} bilgi öğrendim"
    if refused:
        said += f" ({refused} iddiayı almadım)"
    return said + "."


def attribution(source):
    """Where a fact came from, said plainly."""
    if source == INFERENCE:
        return "kendi çıkarımım"
    if source.startswith(DISTILLED_PREFIX):
        return f"bir dil modelinden, {source[len(DISTILLED_PREFIX):]}"
    return f"doğrudan bilgi, kaynak: {source}"


def _origin(source):
    if source == INFERENCE:
        return "çıkarımla varsaymıştım"
    if source.startswith(DISTILLED_PREFIX):
        return "bir dil modelinden almıştım"
    return f"{source} kaynağından öğrenmiştim"


def generalising(examples, statement):
    """Announcing a pattern it spotted on its own."""
    return f"şunu fark ettim: {listing(examples)} — sanırım {statement}."


def corrected(statement, source):
    """Yielding a weaker source to a stronger one, naming what it gave up."""
    return f"bunu {_origin(source)}, senin sözünü üstün tutuyorum: {statement}."


def computed(expression, value):
    """A result the system produced, with the working as its justification."""
    return f"{expression} = {value} (hesapladım)."


def cannot_compute(reason):
    return f"bunu hesaplayamadım: {reason}."


def did_you_mean(suggestions):
    """The particle harmonises with the last word, like every other suffix."""
    words = list(suggestions)
    return f"yoksa {listing(words)} {question_particle(words[-1])} demek istedin?"


def disagreement(statement, others):
    """Two sources of equal standing disagree, and that is what gets said."""
    return (f"kaynaklar anlaşmıyor: {listing(list(others))} tersini söylüyor. "
            f"'{statement}' iddiasını da tartışmalı olarak kaydettim.")


# Bir nicelik cevabında kaç üye sayılır. Üretim grafında "bazı kuşlar uçar mı"
# sorusu 26 kuş sayıyordu: teknik olarak doğru ama cevap değil, döküm. Kesme
# gizli değil — kalan sayıyla söyleniyor, yani hiçbir şey saklanmıyor.
MOST_LISTED = 5


def _some_of(items):
    """Uzun üye listesini kısaltır ve kalanı sayarak söyler."""
    items = list(items)
    if len(items) <= MOST_LISTED:
        return listing(items)
    return listing(items[:MOST_LISTED]
                   + [f"{len(items) - MOST_LISTED} tanesi daha"])


def _members(counted, verb, others, contrast):
    """Bilinen üyeler, varsa karşı örnekleriyle — nicelik cevabının gövdesi."""
    if not counted:
        return f"{_some_of(others)} {contrast}"
    if others:
        return f"{_some_of(counted)} {verb} ama {_some_of(others)} {contrast}"
    return f"{_some_of(counted)} {verb}"


def _unknown(concept, counted, verb, others, contrast):
    """Bilmemek, ama bilineni de söyleyerek."""
    if not counted and not others:
        return f"{concept} hakkında bunu bilmiyorum."
    return f"bunu bilmiyorum; bildiğim {_members(counted, verb, others, contrast)}."


def how_many(concept, target, positive, yes, no, rule, quantifier=ALL):
    """Sorulan niceliği bulunan kanıtla karşılaştırır.

    Nicelik ayrıştırıcıdan buraya kadar geliyordu ama hiç okunmuyordu: "bazı
    kuşlar uçar mı", "her kuş uçar mı" ve "hiçbir kuş uçar mı" üçü de aynı
    cevabı alıyordu — sonuncusuna "evet" deniyordu. Oysa aynı kanıt üçüne
    farklı cevap verir: tek bir örnek "bazı"yı KANITLAR, tek bir karşı örnek
    "her"i ÇÜRÜTÜR, tek bir örnek de "hiçbiri"ni çürütür.

    İkinci düzeltme kapalı dünya: "bildiğim üyelerin hepsi uçuyor" ile "hepsi
    uçar" aynı şey değildir, çünkü üye kümesi kapalı değil — yarın bir üye
    daha öğrenilebilir. Eskiden iki kartal ve bir serçe görüp "evet, hepsi"
    deniyordu. Evrensel cevabı ancak türün KENDİ kaydı (kural) verebilir;
    üyeler yalnız çürütmeye ve varlığa yeter.
    """
    verb = verb_form(target, positive)
    contrast = verb_form(target, not positive)
    counted = list(yes if positive else no)   # sorulan yöndeki bilinen üyeler
    others = list(no if positive else yes)    # karşı örnekler
    # Kural sorulan yönde mi, tersine mi, yok mu — üç ayrı durum.
    ruled = None if rule is None else bool(rule) == positive
    body = _members(counted, verb, others, contrast)
    if quantifier == SOME:
        if counted:
            return f"evet, {body}."
        if ruled is True:
            return f"evet, kural olarak {concept} {verb}."
        if ruled is False:
            return f"hayır, kural olarak {concept} {contrast}."
        return _unknown(concept, counted, verb, others, contrast)
    if quantifier == NO:
        if counted:
            return f"hayır, {body}."
        if ruled is True:
            return f"hayır, kural olarak {concept} {verb}."
        if ruled is False:
            return f"evet, kural olarak {concept} {contrast}."
        return _unknown(concept, counted, verb, others, contrast)
    if quantifier == MOST:
        if ruled is True:
            return f"evet, kural olarak {concept} {verb}" + (
                f", ama {listing(others)} {contrast}." if others else ".")
        if ruled is False:
            return f"hayır, kural olarak {concept} {contrast}" + (
                f", ama {listing(counted)} {verb}." if counted else ".")
        if counted and len(counted) > len(others):
            return f"bildiğim kadarıyla evet, {body}."
        return _unknown(concept, counted, verb, others, contrast)
    if others:            # "her": tek karşı örnek yeter
        return f"hayır, {_members(others, contrast, counted, verb)}."
    if ruled is True:
        return f"evet, kural olarak {concept} {verb}."
    if ruled is False:
        return f"hayır, kural olarak {concept} {contrast}."
    if counted:
        return (f"bildiğim {listing(counted)} {verb}, ama {concept} "
                f"hakkında hepsini kapsayan bir kural bilmiyorum.")
    return f"{concept} hakkında bunu bilmiyorum."


def branch_frozen(branch):
    """A branch is refusing too much of what arrives, so it stops growing."""
    return (f"'{branch}' dalında son zamanlarda çok fazla çelişki çıktı, "
            f"o yüzden makine kaynaklarına kapattım. Sen öğretebilirsin; "
            f"gözden geçirip açmak istersen söyle.")


def disputed_note():
    return "kaynaklar bu konuda anlaşmıyor"


def dont_know(concept, suggestions=()):
    """Not knowing, plus the known words it might have been — as a question.

    The suggestion never becomes an answer: the system still says it does not
    know, and a guess that has to be confirmed cannot turn into a belief.
    """
    plain = f"bilmiyorum. {concept} hakkında bunu bana öğretir misin?"
    if not suggestions:
        return plain
    return f"{plain} {did_you_mean(suggestions)}"


def forgotten(concept, count):
    """Silindi — kaç kayıt gittiği söylenerek.

    Bir LLM'in ağırlıklarından bilgi silinemez. Burada silinebiliyor ve KAÇ
    kaydın gittiğini söylemek, silmenin gerçekten olduğunun kanıtı.
    """
    if not count:
        return f"{concept} hakkında zaten bir şey bilmiyordum."
    return f"{concept} hakkında bildiğim {count} şeyi unuttum."


def likely_traits(concept, guesses, family):
    """Bilinmiyor ama KARDEŞLERİNDEN çıkıyor — tahmin olduğu söylenerek.

    Bir olgu değil bir tahmin, ve gerekçesi cevapla BİRLİKTE veriliyor:
    "çünkü bildiğim X'lerin çoğu öyle". Gerekçesi olmayan tahmin uydurmadır;
    gerekçesi olan tahmin, insanın da yaptığı şeydir.

    Grafa yazılmıyor — bu gece ölçüldü, yazılan tahmin hafızayı kirletiyor.
    """
    return (f"{concept} hakkında bunu öğrenmedim, ama muhtemelen "
            f"{listing(guesses)} — çünkü bildiğim {family}lerin çoğu öyle.")


def related_instead(concept, neighbours):
    """Bilmiyorum, ama ANLAMCA yakın şunları biliyorum.

    Yazım benzerliğinden ayrı bir şey: "kus" ile "kuş" harf komşusu, "glokom"
    ile "katarakt" ANLAM komşusu. İkincisini yalnız dağılım verebilir.

    Söylenen şey bir iddia DEĞİL: sorulan kavram hakkında hiçbir şey
    söylenmiyor, yalnız elde ne olduğu gösteriliyor. Geometri aday bulur,
    kararı kapı verir — ve burada kapı zaten "bilmiyorum" demiş durumda.
    """
    return (f"{concept} hakkında bir şey bilmiyorum. "
            f"şunları biliyorum, ilgili olabilir: {listing(neighbours)}.")


def need_first(question, suggestions=()):
    """It has a goal but no ground to stand on yet."""
    plain = ("bunu bilmiyorum. cevaplayabilmem için önce şunu öğrenmem lazım: "
             f"{question}")
    if not suggestions:
        return plain
    return f"{plain} {did_you_mean(suggestions)}"


def climbing(concept, ancestor, question):
    """It has a goal and knows exactly which fact would unlock it."""
    return f"bunu bilmiyorum ama {concept} bir {ancestor}. {question}"


def now_i_can(answer):
    """Returning to the question that was waiting."""
    return f"şimdi ilk soruna dönebilirim: {answer}"


def capitalize(text):
    """Turkish capitalisation: "içer" starts a sentence as "İçer", not "Icer"."""
    if not text:
        return text
    first = "İ" if text[0] == "i" else text[0].upper()
    return first + text[1:]


def predicate(relation, target, object=None, role=None):
    """A clause with the subject left out, the way Turkish drops it.

    "uçar", "tüylüdür" — so a paragraph can say "kuş olduğu için uçar ve
    tüylüdür" instead of repeating the name in every clause.
    """
    if relation == IS_A:
        return f"bir {target}{copula(target)}"
    if relation == NOT_A:
        return f"bir {target} değildir"
    if relation == HAS_PART:
        return f"{target} var"
    if relation == LACKS_PART:
        return f"{target} yok"
    middle = f"{case_form(object, role)} " if object else ""
    if relation == HAS_PROPERTY:
        return f"{middle}{target}{copula(target)}"
    if relation == LACKS_PROPERTY:
        return f"{middle}{target} değildir"
    return f"{middle}{verb_form(target, relation == CAN)}"


# Kendi cümle biçimi olan ilişkiler. Gerisi ilişki kaydının etiketiyle
# söyleniyor.
KNOWN_CLAUSES = (IS_A, NOT_A, HAS_PROPERTY, LACKS_PROPERTY, HAS_PART,
                 LACKS_PART, CAN, CANNOT)


def describe(concept, relation, target, object=None, role=None, kinds=None):
    """A fact stated as a Turkish sentence, whatever its relation.

    Tanınmayan bir ilişki YETENEK sanılıyordu ve sessizce bozuk cümle
    üretiyordu: "kuş kanat gerektirir" bilgisi "öğrendim: kuş kanat" diye
    onaylanıyordu. Oysa ilişki kaydı her ilişkinin okunabilir etiketini zaten
    tutuyor (`lmm/kinds.py`) — projenin "ilişkiler veri, kod değil" iddiasının
    kaçırdığı yer burasıydı. Yeni bir ilişki eklendiğinde artık bu dosya
    değişmiyor.
    """
    if kinds is not None and relation not in KNOWN_CLAUSES:
        kind = kinds.by_name.get(relation)
        if kind is not None:
            return f"{concept} {target} {kind.label}"
    if relation == IS_A:
        return is_a_clause(concept, target)
    if relation == NOT_A:
        return is_not_a_clause(concept, target)
    if relation in (HAS_PROPERTY, LACKS_PROPERTY):
        return property_clause(concept, target, relation == HAS_PROPERTY,
                               object, role)
    if relation in (HAS_PART, LACKS_PART):
        return part_clause(concept, target, relation == HAS_PART)
    return ability_clause(concept, target, relation == CAN, object, role)


def which_meaning(name, readings):
    """Bir ad birden çok şeye işaret ediyorsa, hepsini say ve sor.

    Belirsizlik bir bilgisizlik değil; sistemin iki şeyi birden bilmesidir.
    Birini seçip emin görünmek, bildiğinden azını söylemek olur.
    """
    listed = listing([f"bir {reading}" for reading in readings])
    return (f"{name} birden fazla şeye işaret ediyor: {listed} olabilir. "
            f"hangisini soruyorsun?")


def sentences(clauses):
    """Birkaç yargıyı tek paragrafa dizer: büyük harf, nokta, ayırma."""
    return capitalize(". ".join(clauses) + ".")


# ---------------------------------------------------------------------------
# KAPININ SÖYLEYİŞİ
#
# Kapı (`lmm/gate.py`) neyi bildiğini biliyor; onu nasıl söyleyeceğini
# bilmemeli. Ölçüldü: kapıda 39 Türkçe dizgi sabiti vardı — ', çünkü ',
# 'hayır', ' karşılaştırması bilmiyorum.' — yani ikinci bir dil, kapıyı
# değiştirmeyi gerektiriyordu. Oysa kapının kararı dilden bağımsız: "biliyorum
# / bilmiyorum / şu kanıtla". Aşağıdakiler yalnızca o kararların Türkçesi.
# ---------------------------------------------------------------------------


def because(known, chain):
    """Evet/hayır ve gerekçesi. Kapının en sık kurduğu cümle.

    Gerekçe cevabın süsü değil kendisi: "evet" tek başına bir LLM cevabıdır,
    "evet, çünkü penguen bir kuş ve kuş uçar" bir kanıt zinciridir.
    """
    return f"{'evet' if known else 'hayır'}, çünkü {' ve '.join(chain)}."


def because_chain(chain):
    """Yalnız gerekçe — "neden" sorusunda evet/hayır zaten sorulmuyor."""
    return "çünkü " + " ve ".join(chain) + "."


def actually(clause):
    """Sorunun öncülü yanlışsa, cevap vermeden önce onu düzeltmek.

    "penguen neden uçar" sorusuna gerekçe uydurmak yerine öncülü çürütmek —
    bir dil modelinin en kolay yanıldığı yer.
    """
    return f"aslında {clause}."


def affirmed(clause):
    """Kanıtı zaten cümle olan bir "evet"."""
    return f"evet, {clause}."


def inherited_clause(clause, concept, ancestor=None):
    """Cevap kalıtımdan geliyorsa nereden geldiğini de söyler."""
    if ancestor is None:
        return clause + "."
    return f"{clause}, çünkü {concept} bir {ancestor}."


def definition_answer(concept, target, source, sure=True):
    """Tanım, künyesiyle. Güven düşükse cümlenin kendisi bunu söylüyor."""
    answer = f"{is_a_clause(concept, target)} ({attribution(source)})."
    return answer if sure else "emin değilim ama " + answer


def requirements(concept, needed):
    """"bir kuşun uçabilmesi için ne gerekir" — kayıtlı önkoşullar."""
    return f"{concept} için {listing(needed)} gerekir."


def unknown_requirement(concept):
    """Önkoşulu tahmin etmek cevabı uydurmakla aynı şey; onun yerine sormak."""
    return f"{concept} için ne gerektiğini bilmiyorum. bana öğretir misin?"


def more_so(winner, loser, trait):
    """Kayıtlı bir üstünlük: "kartal, çünkü kartal serçeden hızlıdır"."""
    return f"{winner}, çünkü {winner} {loser}den {trait}dır."


def both_but_unranked(trait):
    """İkisinde de nitelik var ama sıralama kayıtlı değil — uydurulmuyor."""
    return f"ikisi de {trait}, ama hangisinin daha {trait} olduğunu bilmiyorum."


def only_one_has(known_one, other, trait):
    """Biri hakkında bilinen var, öteki hakkında yok. İkisi de söyleniyor."""
    return f"{known_one} {trait}, {other} için bunu bilmiyorum."


def no_comparison(first, second, trait):
    return f"{first} ile {second} arasında {trait} karşılaştırması bilmiyorum."


def never_learned_properties(concept):
    return f"{concept} nasıldır, bunu hiç öğrenmedim."


def properties_answer(concept, properties):
    return property_summary(concept, properties) + "."


def never_learned_abilities(concept):
    return f"{concept} ne yapabilir, bunu hiç öğrenmedim."


def abilities_answer(concept, abilities):
    return ability_summary(concept, abilities) + "."


def never_heard_action(action):
    """Fiili hiç duymamak ile onu kimsenin yapmadığını bilmek ayrı şeyler."""
    return f"{verb_form(action, True)} diye bir şeyi hiç duymadım."


def nobody_does(action, positive):
    return f"bildiğim hiçbir şeyin {verb_form(action, positive)}ini öğrenmedim."


def who_answer(concepts, action, positive):
    return who_clause(concepts, action, positive) + "."


def never_heard_property(prop):
    return f"'{prop}' diye bir niteliği hiç duymadım."


def nobody_is(prop):
    return f"bildiğim hiçbir şeyin {prop} olduğunu öğrenmedim."


def who_is_answer(concepts, prop):
    return f"{listing(concepts)} {prop}{copula(prop)}."


# ---------------------------------------------------------------------------
# İKİ ÖZNELİ CEVABIN BİRLEŞTİRİLMESİ
#
# `lmm/coordination.py` cümleyi bölmeyi biliyor — o bir yapı işi ve dilden
# bağımsız. Ama iki cevabın TEK cümlede nasıl birleşeceği dile bağlı: "evet,
# ikisi de" Türkçenin kısaltması, İngilizcenin değil. Bölme orada kaldı,
# söyleyiş buraya geldi.
# ---------------------------------------------------------------------------

# Bir cevabın hangi yönde olduğunu, cevabın KENDİ açılışından anlıyoruz.
# Bu bir kullanıcı metni taraması değil: bu cümleleri yukarıdaki işlevler
# kurdu, yani kendi çıktımızın imzasına bakıyoruz — `REFUSALS` ile aynı fikir.
_YES, _NO = "evet", "hayır"


def combined(answers):
    """İki cevabı tek cümlede birleştirir — aynıysa kısaltarak.

    Farklıysa ikisi de söyleniyor, çünkü farkın kendisi cevabın parçası.
    """
    first, second = answers[0], answers[1]
    if first.startswith(_YES) and second.startswith(_YES):
        return f"evet, ikisi de. {first} {second}"
    if first.startswith(_NO) and second.startswith(_NO):
        return f"hayır, ikisi de değil. {first} {second}"
    return f"{first} Ama {second}"


# ---------------------------------------------------------------------------
# SOSYAL ALIŞVERİŞİN CEVAPLARI
#
# `lmm/social.py` hangi alışverişin sorulduğunu tanır; ne söyleneceğini bilmez.
# Sayılar bellekten geliyor, uydurulmuyor — bilinmiyorsa "birçok" deniyor,
# çünkü bir sayı vermek onu bilmek demektir.
# ---------------------------------------------------------------------------


def _counted(amount):
    return "birçok" if amount is None else amount


def greeted(facts=None, concepts=None):
    return "merhaba. bildiğim şeyleri sorabilirsin."


def farewelled(facts=None, concepts=None):
    return "görüşmek üzere. öğrendiklerim kayıtlı kalıyor."


def thanked(facts=None, concepts=None):
    return "rica ederim."


def wellbeing(facts=None, concepts=None):
    return (f"iyiyim. {_counted(facts)} bilgi ve "
            f"{_counted(concepts)} kavram tutuyorum.")


def introduced(facts=None, concepts=None):
    """Sistemin kendisi hakkında söyledikleri uydurma değil, kodun gerçeği."""
    return (f"ben LMM'im — yaşayan bellek modeli. bildiklerim ağırlıklarda "
            f"değil, okunabilir bir bellekte duruyor: şu an {_counted(facts)} "
            f"bilgi, {_counted(concepts)} kavram. bilmediğimi uyduramam.")


def what_i_can_do(facts=None, concepts=None):
    return ("bildiğim şeyleri sorabilirsin, bana yeni bilgi öğretebilirsin, "
            "yanlışımı tek cümleyle düzeltebilirsin. her cevabımda kaynağımı "
            "söylerim, bilmediğimde de bilmediğimi.")


# ---------------------------------------------------------------------------
# SOHBET ARAYÜZÜNÜN SÖYLEDİKLERİ
#
# `lmm/cli.py` oturumu kurar; ne söyleyeceğini bilmez. Açılış satırları,
# yardım ekranı ve onay cevapları da kullanıcıya söylenen Türkçedir — kapının
# cevapları kadar. Ölçüldü: arayüzde 51 Türkçe dizgi sabiti vardı.
# ---------------------------------------------------------------------------


def exception_learned():
    """Bekleyen çelişki istisna olarak kabul edildi."""
    return "öğrendim (istisna olarak işledim)."


def not_learned():
    """Öğretmen reddetti; kaydedilmedi ve bu açıkça söyleniyor."""
    return "tamam, öğrenmedim."


def status(facts, concepts, words, kinds):
    """Bir satırda sistemin sayılabilir hâli — tanıtım değil, döküm."""
    return (f"{facts} bilgi · {concepts} kavram · "
            f"{words} öğrenilmiş kelime · {kinds} ilişki türü")


def opened(path):
    return f"LMM — yaşayan bellek: {path}"


def inquiry_open():
    return "  soruşturma açık: bilmediğim bir şey sorulursa gidip okurum."


def intent_network(backbone, ready):
    """Ağ isteğe bağlı; yokluğu da söyleniyor çünkü davranışı değiştiriyor."""
    state = f"açık ({backbone})" if ready else "yok"
    return f"  niyet ağı: {state} — kalıp yetmediğinde devreye giriyor."


def wording_open():
    return ("  söyleyiş öğrenme: açık"
            " — anlaşılmayan cümleden kalıcı kalıp çıkarır.")


def dictionary_open():
    return "  sözlük: açık — tanımadığı sözcüğün eş anlamlısını arar."


def voice_state(ready):
    return (f"  akıcı ağız: {'açık' if ready else 'YÜKLENEMEDİ'}"
            " — cevap çekirdeğe söyletilir, geri okunup denetlenir.")


def help_hint():
    return "  'yardım' yazarsan ne söyleyebileceğini gösteririm."


def saved_and_gone():
    return "bellek kaydedildi. hoşça kal."


def help_text():
    """Yardım ekranı. Sabit değil işlev, çünkü ikinci bir dil bunu değiştirir.

    Örnekler elle yazılı ve bilerek: bu ekran sistemin ne bildiğini değil,
    hangi CÜMLE BİÇİMLERİNİ anladığını gösteriyor. Grafa göre değişmemesi
    gereken tek örnek listesi bu — `known_shapes` grafın bildiğini gösterir,
    burası dilin kendisini.
    """
    return """
ÖĞRETMEK                          SORMAK
  penguen bir kuştur                penguen nedir
  penguen bir memeli değildir       penguen bir kuş mu
  kuşlar uçar                       penguen uçar mı
  penguen uçamaz                    kimler uçar
  kuşlar tüylüdür                   penguen tüylü mü
  bazı kuşlar uçmaz                 bazı kuşlar uçar mı
  kuşun kanadı var                  kuşun kanadı var mı
  penguen kutupta yaşar             penguen neden uçamaz
  kediler fare yakalar              penguen ne yapabilir
  kartal serçeden büyüktür          penguen nasıldır
  kelime: koşmak = koşar / koşamaz  penguen anlat
                                    17 çarpı 43 kaç

  Konuşmayı sürdürür: "peki yüzer mi", "anlat", "nasıldır"
  Komutlar: yardım · durum · çık
"""
