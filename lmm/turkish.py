"""Turkish, described in one place.

The suffixes, the function words and the sentence patterns of one language. The
parser, the grammar and the reasoning know nothing about any of it — they ask a
morphology object questions like "does this word carry a copula" and match slots
against a pattern list.

A second language means another module shaped like this one, not another parser.
"""
from lmm.grammar import (Pattern, Grammar, KAVRAM, TUR, NITELIK, SOZ, FIIL,
                         SORU, KIM, ROL, NICEL, SAHIP, NESNEL, FROM_VERB)
from lmm.relations import (REQUIRES, IS_A, NOT_A, CAN, HAS_PROPERTY, LACKS_PROPERTY,
                           HAS_PART, LACKS_PART, PLACE, SOURCE, ALL,
                           MOST, SOME, NO)

TEACH = "TEACH"
ASK = "ASK"
ASK_WHO = "ASK_WHO"
ASK_ABILITIES = "ASK_ABILITIES"
ASK_WHY = "ASK_WHY"
ASK_PROPERTIES = "ASK_PROPERTIES"
ASK_DESCRIBE = "ASK_DESCRIBE"
# İki kavramı karşılaştırma. Yayılım organı (`lmm/spreading.py`) kurulmuş ve
# test edilmişti ama hiçbir yerden çağrılmıyordu; cevaplayabildiği soru ise
# ölçümde başarısız olanlardan biriydi: "kartal ile penguen arasındaki fark ne".
ASK_COMPARE = "ASK_COMPARE"
# Sohbetin kendisi hakkında soru. Dünya hakkında bir soru değil, o yüzden
# cevabı graftan değil sohbet geçmişinden geliyor (`lmm/thread.py`).
ASK_THREAD = "ASK_THREAD"
# "hangisi daha hızlı, kartal mı penguen mi" — iki kavramı bir nitelikte
# sıralama. İlişki grafta zaten var (`kartal --hızlı--> [çıkış: serçe]`);
# eksik olan yalnızca soru biçimi ve sıralama işlemiydi.
ASK_WHICH_MORE = "ASK_WHICH_MORE"
# "bir kuşun uçabilmesi için ne gerekir" — önkoşul sorusu.
ASK_REQUIREMENT = "ASK_REQUIREMENT"
# "başka ne biliyorsun", "daha fazlasını söyle" — sohbetin kendisi hakkında,
# dünya hakkında değil. Cevabı graftan değil, o an konuşulan kavramın henüz
# söylenmemiş kısmından geliyor.
ASK_MORE = "ASK_MORE"
# "ne biliyorsun", "hafızanda ne var" — sistemin KENDİSİ hakkında soru.
# Yaşayan bir hafızanın ilk cevaplaması gereken soru bu ve cevaplayamıyordu:
# 486 kavram bilirken "bunu anlamadım" diyordu.
ASK_INVENTORY = "ASK_INVENTORY"
# Cevabın KENDİSİ hakkında sorular. Yeni bilgi gerektirmiyorlar: graf zaten
# güveni ve künyeyi tutuyor, yalnızca sorulmuyordu. Bir LLM bu soruları
# cevaplayamaz — kaynağı yoktur.
ASK_CERTAINTY = "ASK_CERTAINTY"      # "emin misin"
ASK_SOURCE = "ASK_SOURCE"            # "nereden biliyorsun"
# Görüş sorusu. Cevaplanmıyor ama ANLAŞILIYOR: "anlamadım" demekle "benim
# görüşüm yok" demek aynı şey değil.
ASK_OPINION = "ASK_OPINION"
ASK_HOW_MANY = "ASK_HOW_MANY"
ASK_WHERE = "ASK_WHERE"


class TurkishMorphology:
    question_particles = ("mı", "mi", "mu", "mü")
    # Soru sözcükleri kapalı bir sınıf: bir dilde birkaç tanedir ve hiçbir
    # sayım "kaç"ın soru sorduğunu göstermez — bu, dil hakkında bildirilmesi
    # gereken bilgi. Liste eksikti ve bedeli ölçüldü: "kartal kaç yaşında
    # yaşar" sorusu BİLDİRME sanılıp grafa `kartal kaç --can--> yaşamak`
    # diye yazılıyordu. Hafızayı kirletmek, bu mimarideki en pahalı hata.
    interrogatives = ("kim", "kimler", "ne", "neler", "kaç", "nasıl",
                      "hangi", "hangisi", "nere", "nerede", "nereye",
                      "nereden", "niye", "niçin", "neden", "kaçıncı")
    copula_suffixes = ("tur", "tır", "dur", "dır", "tür", "tir", "dür", "dir")
    plural_suffixes = ("lar", "ler")
    # Açılış sözcükleri: cümlenin başında durup hiçbir şey eklemeyenler.
    # "acaba penguen uçabilir mi" ile "penguen uçabilir mi" aynı soru.
    openers = ("peki", "ya", "hem", "acaba", "bir", "de", "da", "işte")
    # Bir kavram öbeğinin İÇİNE giremeyecek kelimeler. Hepsi kapalı sınıf ve
    # hepsi aynı işi görüyor: kavramı nitelemiyorlar, cümlenin yapısını
    # kuruyorlar. Bu bilgi DİLE aittir ve burada durur — `grammar.py` hangi
    # dile baktığını bilmemeli.
    from lmm.clauses import JOINERS as _JOINERS
    joiners = _JOINERS
    # Ayrı yazılan pekiştirme parçacıkları. Ek hâlleri bitişik yazıldığı için
    # karışmıyor: "kartalda" bulunma, "kartal da" pekiştirme.
    # Görüş isteyen açılışlar. Kapalı sınıf: bir dilde birkaç tanedir ve
    # hepsi aynı şeyi yapar — cümleyi bilgi sorusundan kanaat sorusuna çevirir.
    opinion_marks = ("sence", "bence", "sizce", "bizce", "kanaatince")
    clitics = ("da", "de", "dahi", "bile")
    # Çoğul gönderme: "ikisi de", "onlar", "bu ikisi" — sohbette az önce
    # geçen kavramlara işaret eder. Tekil zamirler (`o`, `bu`) zaten
    # `pronouns` içinde; bunlar İKİ şeye birden gönderdikleri için ayrı.
    plural_pronouns = ("ikisi", "ikiside", "onlar", "bunlar", "şunlar",
                       "hepsi", "her", "üçü")
    # Muhatap zamirleri. Bir soruda "bana"/"bize" cümlenin KONUSU değil,
    # kime söylendiğidir — ve LMM her zaman muhataptır. "bunu bana anlatır
    # mısın" ile "bunu anlatır mısın" aynı şeyi soruyor.
    addressees = ("bana", "bize", "sen", "siz", "sana", "size", "senin",
                  "sizin", "benim", "bizim", "ben", "biz")
    # Aynı şeyi soran kapalı sınıf sözcükler. Kalıpta biri yazılıyor, üçü de
    # eşleşiyor — "neden" için kalıp yazıp "niçin" için yazmamak, aynı soruyu
    # iki kez yazmak demekti. Hangi sözcüğün hangisiyle aynı şeyi sorduğu
    # DİLE ait bilgidir ve sayımla bulunamaz.
    groups = (
        ("neden", "niçin", "niye", "niden"),
        ("nasıl", "ne şekilde"),
        ("kim", "kimler"),
        ("ne", "neler"),
    )
    denials = ("değil", "değildir", "yok", "yoktur")
    # Bir SORUYA verilen onay ve ret. Cümle içindeki olumsuzluktan (`denials`)
    # ayrı: "yok" ikisinde de geçiyor ama biri yüklemi olumsuzluyor, öteki
    # sorulan şeyi reddediyor. Bunlar `lmm/cli.py`'de sabit duruyordu ve orası
    # dilin yaşadığı yer değil — ikinci bir dil eklenince oraya dokunmak
    # gerekirdi. Dil burada yaşar.
    affirmations = ("evet", "e", "ee", "aynen", "tabii", "tabi", "olur",
                    "öğren", "kaydet")
    refusals = ("hayır", "hayir", "yok", "olmaz", "istemiyorum", "boşver")
    postpositions = ("ile", "ila", "karşı", "göre", "kadar", "gibi", "için",
                     "rağmen", "beri", "dolayı")
    intensifiers = ("çok", "daha", "en", "pek", "oldukça", "gayet", "epey",
                    "hayli", "az", "biraz")
    correlatives = ("hem", "ya", "gerek", "kah", "kâh", "ister")
    # Endings that mark a word as playing its own part in the sentence rather
    # than belonging to the noun beside it: "serçeden" is a comparison, not half
    # of a compound. Discovery finds these families; until it supplies them,
    # they are listed.
    oblique_suffixes = ("den", "dan", "ten", "tan", "yle", "yla", "ile")
    # A case ending says what a second concept is doing in the sentence, so the
    # role is read off the word instead of guessed from where it sits.
    case_roles = (
        (("den", "dan", "ten", "tan"), SOURCE),
        (("de", "da", "te", "ta"), PLACE),
    )
    genitive_suffixes = ("nın", "nin", "nun", "nün", "ın", "in", "un", "ün")
    possessive_suffixes = ("sı", "si", "su", "sü", "ı", "i", "u", "ü")
    pronouns = ("o", "onu", "onun", "bu", "bunu", "şu", "şunu")
    # Qualitative, never numeric. Which words mark how much of a kind is meant.
    quantifiers = {"bütün": ALL, "tüm": ALL, "her": ALL, "bilcümle": ALL,
                   "çoğu": MOST, "ekseri": MOST,
                   "bazı": SOME, "kimi": SOME, "birtakım": SOME,
                   "hiçbir": NO, "hiç": NO}

    # A consonant softens before the suffix: balık + ın is "balığın". Undoing
    # that is part of reading the possessor back out.
    softened = {"ğ": "k", "b": "p", "c": "ç", "d": "t"}

    def genitive_readings(self, word):
        """Every way this word could be a possessor, longest stem first."""
        found = []
        for suffix in self.genitive_suffixes:
            if not (word.endswith(suffix) and len(word) > len(suffix) + 1):
                continue
            stem = word[: -len(suffix)]
            if suffix.startswith("n") != (stem[-1] in "aeıioöuü"):
                continue
            found.append(stem)
            hardened = stem[:-1] + self.softened.get(stem[-1], stem[-1])
            if hardened != stem:
                found.append(hardened)
        return sorted(set(found), key=len, reverse=True)

    accusative_suffixes = ("yı", "yi", "yu", "yü", "ı", "i", "u", "ü")

    def strip_accusative(self, word):
        return self._strip(word, self.accusative_suffixes)

    def strip_genitive(self, word, known=()):
        """"kuşun" -> "kuş", "balığın" -> "balık".

        The -nIn form follows a vowel and the -In form a consonant, so the
        candidates are checked against that rather than taken longest-first:
        "penguenin" is penguen + in, not pengue + nin.
        """
        candidates = []
        for suffix in self.genitive_suffixes:
            if not (word.endswith(suffix) and len(word) > len(suffix) + 1):
                continue
            stem = word[: -len(suffix)]
            after_vowel = stem[-1] in "aeıioöuü"
            if suffix.startswith("n") != after_vowel:
                continue                # the wrong form for what precedes it
            candidates.append(stem)
            hardened = stem[:-1] + self.softened.get(stem[-1], stem[-1])
            if hardened != stem:
                candidates.append(hardened)
        # "insanın" splits two ways and both obey the rule — insa+nın and
        # insan+ın — so only knowing the word decides. Where nothing decides,
        # nothing is returned: guessing here would write a concept that does
        # not exist and never say so.
        known = set(known)
        for candidate in candidates:
            if candidate in known:
                return candidate
        return candidates[0] if len(candidates) == 1 else word

    def role_of(self, word):
        """(stem, role) when the word carries a case ending, else (word, None)."""
        for suffixes, role in self.case_roles:
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 1:
                    return word[: -len(suffix)], role
        return word, None

    def is_oblique(self, word):
        """Does this word carry a case ending, so it stands on its own?

        Every ending that marks a role counts, not just the ones listed for
        instruments — a locative slipped into a noun phrase once and made
        "penguen kutupta" a concept.
        """
        if self._has(word, self.oblique_suffixes):
            return True
        return any(self._has(word, suffixes) for suffixes, _ in self.case_roles)

    def has_copula(self, word):
        return self._has(word, self.copula_suffixes)

    def strip_copula(self, word):
        return self._strip(word, self.copula_suffixes)

    def strip_plural(self, word):
        return self._strip(word, self.plural_suffixes)

    @staticmethod
    def _has(word, suffixes):
        return any(word.endswith(s) and len(word) > len(s) + 1 for s in suffixes)

    @staticmethod
    def _strip(word, suffixes):
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[: -len(suffix)]
        return word


# Order matters: "kim uçar" has the shape of "kuşlar uçar", so the question has
# to be tried before the lesson.
PATTERNS = [
    # "bazı kuşlar uçmaz" — an existence claim, not a rule about every bird.
    Pattern([NICEL, KAVRAM, FIIL, SORU], ASK_HOW_MANY, FROM_VERB, 1, 2,
            "bazı kuşlar uçar mı", quantifier=0),
    Pattern([NICEL, KAVRAM, FIIL], TEACH, FROM_VERB, 1, 2, "bazı kuşlar uçmaz",
            quantifier=0),
    Pattern([NICEL, KAVRAM, NITELIK], TEACH, HAS_PROPERTY, 1, 2,
            "bazı kuşlar tüylüdür", quantifier=0),
    # A sentence that leaves its subject out, carrying on from the last one.
    Pattern([FIIL, SORU], ASK, CAN, None, 0, "çıplak yetenek sorusu"),
    # Odak sorusu: ek fiilden ÖNCE gelir. "kartal MI uçuyor" — Türkçe'de bu
    # diziliş özneyi öne çıkarır ama sorulan şey aynıdır. "kartal da mı
    # uçuyor" cümlesi parçacık atıldıktan sonra buraya düşüyor.
    Pattern([KAVRAM, SORU, FIIL], ASK, FROM_VERB, 0, 2, "kartal mı uçuyor"),
    Pattern(["nedir"], ASK, IS_A, None, None, "çıplak tanım sorusu"),
    Pattern(["anlat"], ASK_DESCRIBE, None, None, None, "çıplak anlat"),
    Pattern(["başka", "ne", "biliyorsun"], ASK_MORE, None, None, None, "başka ne"),
    Pattern(["daha", "fazlasını", "söyle"], ASK_MORE, None, None, None, "daha fazla"),
    Pattern(["devam", "et"], ASK_MORE, None, None, None, "devam et"),
    Pattern(["hepsi", "bu", "kadar", SORU], ASK_MORE, None, None, None, "hepsi bu mu"),
    Pattern(["başka"], ASK_MORE, None, None, None, "çıplak başka"),
    Pattern(["emin", "misin"], ASK_CERTAINTY, None, None, None, "emin misin"),
    Pattern(["ne", "düşünüyorsun"], ASK_OPINION, None, None, None, "ne düşünüyorsun"),
    Pattern(["sen", "ne", "düşünüyorsun"], ASK_OPINION, None, None, None, "sen ne düşünüyorsun"),
    Pattern(["fikrin", "ne"], ASK_OPINION, None, None, None, "fikrin ne"),
    Pattern(["görüşün", "ne"], ASK_OPINION, None, None, None, "görüşün ne"),
    Pattern(["hangisi", "daha", SOZ], ASK_OPINION, None, None, None, "hangisi daha X"),
    Pattern(["emin", "misiniz"], ASK_CERTAINTY, None, None, None, "emin misiniz"),
    Pattern(["gerçekten", SORU], ASK_CERTAINTY, None, None, None, "gerçekten mi"),
    Pattern(["nereden", "biliyorsun"], ASK_SOURCE, None, None, None, "nereden biliyorsun"),
    Pattern(["kaynağın", "ne"], ASK_SOURCE, None, None, None, "kaynağın ne"),
    Pattern(["bunu", "nereden", "biliyorsun"], ASK_SOURCE, None, None, None, "bunu nereden"),
    Pattern(["nereden", "bildin"], ASK_SOURCE, None, None, None, "nereden bildin"),
    Pattern(["ne", "biliyorsun"], ASK_INVENTORY, None, None, None, "ne biliyorsun"),
    Pattern(["neler", "biliyorsun"], ASK_INVENTORY, None, None, None, "neler biliyorsun"),
    Pattern(["nelerden", "haberin", "var"], ASK_INVENTORY, None, None, None, "nelerden haberin"),
    Pattern(["hafızanda", "ne", "var"], ASK_INVENTORY, None, None, None, "hafızanda ne var"),
    Pattern(["hafızanda", "neler", "var"], ASK_INVENTORY, None, None, None, "hafızanda neler var"),
    Pattern(["neleri", "biliyorsun"], ASK_INVENTORY, None, None, None, "neleri biliyorsun"),
    # Önkoşul soruları. Kavram başta, fiil ortada.
    Pattern(["bir", KAVRAM, SOZ, "için", "ne", "gerekir"], ASK_REQUIREMENT,
            None, 1, 2, "bir X Y için ne gerekir"),
    Pattern([KAVRAM, SOZ, "için", "ne", "gerekir"], ASK_REQUIREMENT,
            None, 0, 1, "X Y için ne gerekir"),
    Pattern([KAVRAM, "için", "ne", "gerekir"], ASK_REQUIREMENT,
            None, 0, None, "X için ne gerekir"),
    # Öğretme: "uçmak kanat gerektirir"
    Pattern([KAVRAM, TUR, "gerektirir"], TEACH, REQUIRES, 0, 1,
            "X Y gerektirir"),
    Pattern([KAVRAM, "için", TUR, "gerekir"], TEACH, REQUIRES, 0, 2,
            "X için Y gerekir"),
    # "az önce ne konuşuyorduk" — sohbetin kendisi soruluyor.
    Pattern(["az", "önce", "ne", "konuşuyorduk"], ASK_THREAD, None, None, None,
            "az önce ne konuşuyorduk"),
    Pattern(["ne", "konuşuyorduk"], ASK_THREAD, None, None, None,
            "ne konuşuyorduk"),
    Pattern(["neden", "bahsediyorduk"], ASK_THREAD, None, None, None,
            "neden bahsediyorduk"),
    Pattern(["nelerden", "konuştuk"], ASK_THREAD, None, None, None,
            "nelerden konuştuk"),
    Pattern(["nasıldır"], ASK_PROPERTIES, None, None, None, "çıplak nasıl"),
    Pattern(["nasıl"], ASK_PROPERTIES, None, None, None, "çıplak nasıl 2"),
    Pattern(["ne", "yapabilir"], ASK_ABILITIES, None, None, None, "çıplak neler"),

    Pattern([KIM, FIIL], ASK_WHO, FROM_VERB, None, 1, "kimler uçar"),
    Pattern([KAVRAM, "ne", "yapabilir"], ASK_ABILITIES, None, 0, None, "ne yapabilir"),
    Pattern([KAVRAM, "ne", "yapar"], ASK_ABILITIES, None, 0, None, "ne yapar"),
    Pattern([KAVRAM, "neler", "yapar"], ASK_ABILITIES, None, 0, None, "neler yapar"),
    Pattern([KAVRAM, "nerede", FIIL], ASK_WHERE, FROM_VERB, 0, 2, "nerede yaşar"),
    Pattern([KAVRAM, "neden", FIIL], ASK_WHY, FROM_VERB, 0, 2, "neden uçar"),
    Pattern([KAVRAM, "neden", SOZ], ASK_WHY, HAS_PROPERTY, 0, 2, "neden beyaz"),
    Pattern([KAVRAM, "anlat"], ASK_DESCRIBE, None, 0, None, "anlat"),
    # Kibar biçim: "kartaldan bahseder misin", "penguenleri anlatır mısın".
    # Kalıptaki "anlat" grafta eş sayılan her sözcüğü eşleştiriyor.
    Pattern([KAVRAM, "anlat", SORU], ASK_DESCRIBE, None, 0, None, "anlat mı"),
    Pattern([ROL, "anlat", SORU], ASK_DESCRIBE, None, 0, None, "rolden anlat mı"),
    # "kartal ile penguen arasındaki fark ne" — iki kavram, aradaki bağ ya da
    # ayrım soruluyor. Hedef ikinci kavram; karşılaştırma kapıda yapılıyor.
    Pattern([KAVRAM, "ile", KAVRAM, "arasındaki", "fark", "ne"], ASK_COMPARE,
            None, 0, 2, "ile arasındaki fark"),
    # "penguen ile kartal aynı mı" — aynılık sorusu da bir karşılaştırmadır;
    # cevabı ortak yanları ve ayrıldıkları yeri göstermek.
    Pattern([KAVRAM, "ile", KAVRAM, "aynı", SORU], ASK_COMPARE, None, 0, 2,
            "ile aynı mı"),
    Pattern([KAVRAM, "ile", KAVRAM, "benzer", SORU], ASK_COMPARE, None, 0, 2,
            "ile benzer mi"),
    # Üstünlük soruları. Nitelik ortada, iki kavram sonda ya da başta.
    Pattern(["hangisi", "daha", SOZ, KAVRAM, SORU, KAVRAM, SORU],
            ASK_WHICH_MORE, None, 3, 2, "hangisi daha X A mı B mi", object=5),
    Pattern([KAVRAM, SORU, KAVRAM, SORU, "daha", SOZ], ASK_WHICH_MORE,
            None, 0, 5, "A mı B mi daha X", object=2),
    Pattern([KAVRAM, "ile", KAVRAM, "arasındaki", "fark", "nedir"], ASK_COMPARE,
            None, 0, 2, "ile arasındaki fark nedir"),
    Pattern([KAVRAM, "ile", KAVRAM, "farkı", "ne"], ASK_COMPARE, None, 0, 2,
            "farkı ne"),
    Pattern([KAVRAM, "ile", KAVRAM, "arasında", "ne", "fark", "var"],
            ASK_COMPARE, None, 0, 2, "arasında ne fark var"),
    Pattern([KAVRAM, "ile", KAVRAM, "arasındaki", "bağ", "ne"], ASK_COMPARE,
            None, 0, 2, "arasındaki bağ"),
    # More ways to ask for the same thing, gathered from what people wrote.
    Pattern([KAVRAM, "açıkla"], ASK_DESCRIBE, None, 0, None, "açıkla"),
    Pattern([NESNEL, "açıkla"], ASK_DESCRIBE, None, 0, None, "açıkla nesnel"),
    Pattern([NESNEL, "tarif", "et"], ASK_DESCRIBE, None, 0, None, "tarif et"),
    Pattern([NESNEL, "tanımla"], ASK_DESCRIBE, None, 0, None, "tanımla"),
    Pattern([KAVRAM, "hakkında", "bilgi", "ver"], ASK_DESCRIBE, None, 0, None,
            "hakkında bilgi ver"),
    Pattern([KAVRAM, "hakkında", "ne", "biliyorsun"], ASK_DESCRIBE, None, 0,
            None, "hakkında ne biliyorsun"),
    Pattern([KAVRAM, "kim"], ASK, IS_A, 0, None, "kim"),
    Pattern([KAVRAM, "kimdir"], ASK, IS_A, 0, None, "kimdir"),
    Pattern([KIM, SOZ], ASK_WHO, HAS_PROPERTY, None, 1, "kimler beyaz"),
    Pattern([KAVRAM, "anlatsana"], ASK_DESCRIBE, None, 0, None, "anlatsana"),
    Pattern([KAVRAM, "nasıldır"], ASK_PROPERTIES, None, 0, None, "nasıldır"),
    Pattern([KAVRAM, "nasıl"], ASK_PROPERTIES, None, 0, None, "nasıl"),

    Pattern([KAVRAM, "bir", SOZ, "değildir"], TEACH, NOT_A, 0, 2, "bir X değildir"),
    # "penguen bir kuş değil mi" — olumsuz kurulmuş bir DOĞRULAMA sorusu.
    # Türkçe'de bu kalıp olumsuzluk sormaz, teyit ister: cevabı "evet, bir
    # kuştur" olmalı. O yüzden ilişki IS_A, NOT_A değil.
    Pattern([KAVRAM, "bir", TUR, "değil", SORU], ASK, IS_A, 0, 2,
            "bir X değil mi"),
    Pattern([KAVRAM, TUR, "değil", SORU], ASK, HAS_PROPERTY, 0, 1,
            "X değil mi"),
    Pattern([KAVRAM, SOZ, "değildir"], TEACH, LACKS_PROPERTY, 0, 1, "X değildir"),
    Pattern([KAVRAM, "bir", TUR, SORU], ASK, IS_A, 0, 2, "bir X mı"),
    Pattern([KAVRAM, "nedir"], ASK, IS_A, 0, None, "nedir"),
    Pattern([KAVRAM, "ne"], ASK, IS_A, 0, None, "ne"),
    # Turkish puts the object before the verb. Which position it takes is a
    # fact about a language, so it sits in this list and nowhere else.
    # Possession arrived as a pattern and a registry entry — no engine change.
    Pattern([SAHIP, KAVRAM, "var", SORU], ASK, HAS_PART, 0, 1, "kanadı var mı"),
    Pattern([SAHIP, KAVRAM, "var"], TEACH, HAS_PART, 0, 1, "kanadı var"),
    Pattern([SAHIP, KAVRAM, "yok"], TEACH, LACKS_PART, 0, 1, "kanadı yok"),
    Pattern([KAVRAM, ROL, FIIL, SORU], ASK, CAN, 0, 2, "kutupta yaşar mı",
            object=1),
    Pattern([KAVRAM, ROL, SOZ, SORU], ASK, HAS_PROPERTY, 0, 2,
            "serçeden büyük mü", object=1),
    Pattern([KAVRAM, KAVRAM, FIIL, SORU], ASK, CAN, 0, 2, "fare yakalar mı",
            object=1),
    Pattern([KAVRAM, FIIL, SORU], ASK, CAN, 0, 1, "uçar mı"),
    Pattern([KAVRAM, "bir", TUR], TEACH, IS_A, 0, 2, "bir kuştur"),
    Pattern([KAVRAM, ROL, FIIL], TEACH, FROM_VERB, 0, 2, "kutupta yaşar",
            object=1),
    Pattern([KAVRAM, ROL, NITELIK], TEACH, HAS_PROPERTY, 0, 2,
            "serçeden büyüktür", object=1),
    Pattern([KAVRAM, KAVRAM, FIIL], TEACH, FROM_VERB, 0, 2, "fare yakalar",
            object=1),
    Pattern([KAVRAM, FIIL], TEACH, FROM_VERB, 0, 1, "uçar"),
    Pattern([KAVRAM, SOZ, SORU], ASK, HAS_PROPERTY, 0, 1, "beyaz mı"),
    Pattern([KAVRAM, NITELIK], TEACH, HAS_PROPERTY, 0, 1, "beyazdır"),
]


# One example each is enough for discovery to find the whole system: what a
# suffix *means* cannot be read off its shape, but everything else can.
ANCHORS = {"çoğul": ("kuş", "kuşlar"), "koşaç": ("kuş", "kuştur")}


def turkish(words=None, known=(), meanings=None):
    """The grammar. Given words, its suffix rules are discovered rather than read.

    Discovery needs enough words to see a pattern; below that it falls through
    to the declared lists, so a thin memory degrades instead of breaking.
    """
    morphology = TurkishMorphology()
    if words:
        from lmm.discovered import DiscoveredMorphology
        morphology = DiscoveredMorphology(morphology, words, ANCHORS)
    return Grammar(PATTERNS, morphology, known, meanings)
