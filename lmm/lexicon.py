"""The words the system knows.

Vocabulary used to be a constant in the source, which meant LMM could only ever
discuss what we had hardcoded. It is knowledge like any other: learned, stored
with the memory, and carried by packs. A domain extends the language by shipping
its verbs alongside its facts — no change to this codebase.

Each memory carries its own vocabulary, because a service holds more than one at
a time and two tenants must never teach each other words. The module keeps a
default for the common case of a single memory, and `use()` swaps in another for
the duration of a request.
"""
import contextvars

# Elle yazılmış çekirdek: surface -> (mastar, olumlu_mu).
#
# Bu liste artık başlangıç dağarcığının TAMAMI değil, DİBİ. Derlem varken
# üstüne 1555 fiil biniyor (bkz. `seed_verbs`); yokken yalnız bu kalır ve
# sistem çalışmaya devam eder.
#
# Yine de silinemez, ve nedeni ölçülebilir: derlem olumsuzu kuralla üretiyor
# ("uçmaz"), oysa bu dilde yetersizlik ayrı bir ektir ("uçamaz") ve sistemin
# KONUŞTUĞU biçim odur. Sıra bu yüzden önemli — `_forms` ilk gördüğü yüzeyi
# saklıyor, o da buradan gelmeli. Yalnız derleme bırakılsa "penguen uçamaz"
# yerine "penguen uçmaz" denirdi: anlam aynı değil, ve söyleyiş bozulurdu.
CORE_VERBS = {
    "uçar": ("uçmak", True), "uçamaz": ("uçmak", False),
    "yüzer": ("yüzmek", True), "yüzemez": ("yüzmek", False),
    "koşar": ("koşmak", True), "koşamaz": ("koşmak", False),
    "okur": ("okumak", True), "okuyamaz": ("okumak", False),
    "içer": ("içmek", True), "içemez": ("içmek", False),
    "konuşur": ("konuşmak", True), "konuşamaz": ("konuşmak", False),
    # Turkish separates "does not" from "cannot": uçmaz and uçamaz are both
    # heard, and both land on the same relation here.
    "uçmaz": ("uçmak", False), "yüzmez": ("yüzmek", False),
    "koşmaz": ("koşmak", False), "okumaz": ("okumak", False),
    "içmez": ("içmek", False), "konuşmaz": ("konuşmak", False),
}


# Çekim ekleri — zaman, olumsuzluk, kişi, yeterlik — `lmm/turkish.py`'ye
# taşındı. Burada kalan şey SOYMA işlemi ve o dile bakmıyor: en dıştan içe,
# katman katman, her katmanda sözlüğe sor. Ek listelerini bildirmeyen bir dil
# boş küme verir; o zaman yalnız doğrudan yazılmış biçimler tanınır — dar,
# ama yanlış değil.
_MORPHOLOGY = None


def _language():
    """İthal tembel: `turkish` modülü `grammar`'ı, o da bu modülü çekiyor.
    İlk soruda kuruluyor ve saklanıyor — `reading()` kelime başına çağrılıyor
    ve her seferinde modül aramak ölçülebilir bir masraf."""
    global _MORPHOLOGY
    if _MORPHOLOGY is None:
        from lmm.turkish import TurkishMorphology
        _MORPHOLOGY = TurkishMorphology()
    return _MORPHOLOGY


def _of(name):
    return tuple(getattr(_language(), name, ()))


def _without(word, suffixes):
    """Mastar ekini atar. Eki bildirmeyen dilde kelime olduğu gibi kalır."""
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix):
            return word[: -len(suffix)]
    return word


def seed_verbs():
    """Bir belleğin doğduğu andaki fiil dağarcığı.

    Elle yazılmış altı fiil, sistemin başlangıcını kendi kaynak kodunun
    genişliğiyle sınırlıyordu: sözlükte olmayan bir yüklem hiçbir kalıba
    uymuyor ve cümle tamamen kayboluyor (ölçümü `lmm/verbs.py` başında).
    Oysa fiiller zaten çıkarılmış durumda — `data/tr-fiiller.txt`, derlemden
    çift testiyle bulunmuş 807 mastar, 1614 çekim. Sözlüğün onları
    bilmemesi için bir sebep yok; onları yazıya dökmek içinse hiç yok.

    Derlem yoksa boş sözlük döner ve çekirdek tek başına kalır. Tablo bir
    kolaylık, bir bağımlılık değil — `lmm/frequency.py`'nin her yerinde
    olduğu gibi.
    """
    from lmm import frequency
    found = dict(CORE_VERBS)
    for surface, reading in frequency.verbs().items():
        found.setdefault(surface, reading)
    return found


def _bare_ability():
    """Yeterlilik ekinin ÇIPLAK hâli: "abilir" -> "abil".

    Türetiliyor, bildirilmiyor. Dil zaten `ability_suffixes` ile ("ebilir",
    "abilir") diyor ve geniş zaman ekini de ayrıca bildiriyor; ikisinin farkı
    çıplak yeterliliktir. Yeni bir liste yazmak aynı bilgiyi iki yerde tutmak
    olur ve iki kopya er geç ayrışır — bu projede bir kez ölçüldü, `openers`
    iki yerde tutulunca üç eksiltili takip sessizce anlaşılmıyordu.
    """
    from lmm.frames import AORIST
    aorist = ()
    for suffixes, tense in _of("predicate_suffixes"):
        if tense == AORIST:
            aorist = suffixes
    found = []
    for suffix in _of("ability_suffixes"):
        for ending in sorted(aorist, key=len, reverse=True):
            if suffix.endswith(ending) and len(suffix) > len(ending) + 1:
                found.append(suffix[: -len(ending)])
                break
    return tuple(found)


class Lexicon:
    def __init__(self, verbs=None):
        self.verbs = dict(seed_verbs() if verbs is None else verbs)
        # Several surfaces may mean the same thing — "uçmaz" and "uçamaz" both
        # deny flight — but only the first stays the one used for speaking.
        self._forms = {}
        for surface, value in self.verbs.items():
            self._forms.setdefault(value, surface)

    def knows(self, surface):
        return self.reading(surface) is not None

    def reading(self, surface):
        found = self._direct_or_ability(surface)
        if found is not None:
            return found
        return self._inflected(surface)

    def _inflected(self, surface, depth=2):
        """Geniş zaman dışındaki çekimler — gövdeyi bularak.

        Fiil keşfi geniş zaman çiftine dayanıyor ("uçar"/"uçamaz") ve bu
        ölçülmüş, iyi çalışan bir test. Ama insanlar öyle konuşmuyor: "penguen
        neden UÇAMIYOR" diye soruyor ve o biçim sözlükte yok. Sonucu ölçüldü —
        cümle nitelik sorusu sanılıp cevapsız kalıyordu.

        Her fiil için sekiz biçim saklamak yerine ek soyuluyor: sözlüğün
        `-ebilir` için zaten yaptığı şeyin aynısı, diğer zamanlara genişletilmiş.
        Olumsuzluk ekten okunuyor, tahmin edilmiyor.
        """
        # Kişi eki en dışta durur; soyulup altındaki çekime bakılıyor.
        if depth > 0:
            for ending in _of("person_suffixes"):
                if not surface.endswith(ending):
                    continue
                if len(surface) - len(ending) < 3:
                    continue
                inner = surface[: -len(ending)]
                found = self._direct_or_ability(inner) or \
                    self._inflected(inner, depth - 1)
                if found is not None:
                    return found
        for suffixes, negative in _of("tense_suffixes"):
            for suffix in suffixes:
                if not surface.endswith(suffix):
                    continue
                stem = surface[: -len(suffix)]
                if len(stem) < 2:
                    continue
                polarity = True
                for mark in _of("negation_marks"):
                    if stem.endswith(mark) and len(stem) > len(mark) + 1:
                        stem, polarity = stem[: -len(mark)], False
                        break
                found = self._stem_of(stem)
                if found is not None:
                    return found, polarity and negative
        return None

    def _stem_of(self, stem, depth=2):
        """Bilinen bir fiilin gövdesi mi — ünlü kaymasına izin vererek."""
        marks = _of("infinitive_suffixes")
        vowels = getattr(_language(), "vowels", "")
        for infinitive, positive in self.verbs.values():
            if not positive:
                continue
            root = _without(infinitive, marks)
            if root and (stem == root or stem.rstrip(vowels) == root):
                return infinitive
        # Yeterlilik katmanı zaman ekinin ALTINDA da durabilir: "uçabiliyor" =
        # uç + abil + iyor. Zaman soyulunca geriye `uçabil` kalıyor ve o hiçbir
        # mastarın gövdesi değil. Ölçüldü, ikisi de tek başına çalışıyordu:
        #
        #   uçabilir   -> uçmak      (yeterlilik yalnız)
        #   uçuyor     -> uçmak      (şimdiki zaman yalnız)
        #   uçabiliyor -> uçabiliyo  (birleşimi çözülmüyordu)
        #
        # "kartal uçabiliyor mu" en doğal sorulardan biri ve cevapsız
        # kalıyordu. Katman soyma özyinelemiyordu, eksik olan buydu.
        #
        # Ekin çıplak hâli BİLDİRİMDEN TÜRETİLİYOR, yeniden yazılmıyor:
        # `ability_suffixes` zaten ("ebilir","abilir") diyor ve geniş zaman eki
        # de bildirili; ikisinin farkı çıplak yeterliliktir. Yeni bir liste
        # eklemek, aynı bilgiyi iki yerde tutmak olurdu.
        if depth > 0:
            for mark in _bare_ability():
                if stem.endswith(mark) and len(stem) - len(mark) >= 2:
                    under = self._stem_of(stem[: -len(mark)].rstrip("y"),
                                          depth - 1)
                    if under is not None:
                        return under
        return None

    def _direct_or_ability(self, surface):
        """"uçamaz" -> ("uçmak", False); None when the word is unknown.

        Turkish also marks ability with -ebilir/-abilir, and "penguen yüzebilir"
        is how people actually say it. Rather than storing a third form for every
        verb, the suffix is peeled off and matched to a stem we already know.
        """
        direct = self.verbs.get(surface)
        if direct is not None:
            return direct
        for suffix in _of("ability_suffixes"):
            if surface.endswith(suffix) and len(surface) > len(suffix) + 1:
                stem = surface[: -len(suffix)].rstrip("y")
                for infinitive, positive in self.verbs.values():
                    if positive and infinitive.startswith(stem):
                        return infinitive, True
        return None

    def surface(self, infinitive, positive):
        """"uçmak", False -> "uçamaz". Bilmiyorsa derleme sorar.

        Mastara düşmek cümleyi bozuyordu: "kartal uçar, BÜYÜMEK, ÖLMEK, YEMEK"
        — biri çekimli, gerisi mastar. Sebebi oturumun sözlüğünün yalnız
        öğretilmiş fiilleri bilmesi; derlem 807 fiili çekimleriyle biliyor ve
        sorulması yeterliydi.
        """
        found = self._forms.get((infinitive, positive))
        if found is not None:
            return found
        from lmm import frequency
        for word, (name, is_positive) in frequency.verbs().items():
            if name == infinitive and is_positive == positive:
                return word
        # Son çare: mastardan TÜRET. Mastara düşmek cümleyi bozuyor
        # ("karakter tırmanmak") ve bozuk cümle geri okunamıyor — sistemin
        # kendi ağzı, kendi bilgisini eliyor. Yanlış bir çekim bile mastardan
        # iyidir, çünkü mastar cümleyi hiç kurdurmuyor.
        from lmm.phrasing import aorist
        return aorist(infinitive, positive)

    def learn_verb(self, infinitive, positive, negative, plain_negative=None):
        """Teach one verb in both polarities. Idempotent.

        A language may have more than one way to say no — Turkish has "uçmaz"
        beside "uçamaz" — so extra negative forms are accepted for hearing while
        the first stays the one used for speaking.
        """
        for surface, polarity in ((positive, True), (negative, False)):
            self.verbs[surface] = (infinitive, polarity)
            self._forms[(infinitive, polarity)] = surface
        if plain_negative:
            self.verbs[plain_negative] = (infinitive, False)

    def signature(self):
        """Identity of this vocabulary, for caching things derived from it."""
        return frozenset(self.verbs)


CORE = Lexicon()        # the default, for a process holding a single memory
_current = contextvars.ContextVar("lexicon", default=None)


def current():
    """The vocabulary in force right now."""
    return _current.get() or CORE


def use(lexicon):
    """Make a vocabulary current. Returns a token for restoring the previous."""
    return _current.set(lexicon)


def restore(token):
    _current.reset(token)


class _Active:
    """`ACTIVE.verbs` keeps working while meaning "whichever is current"."""

    def __getattr__(self, name):
        return getattr(current(), name)


ACTIVE = _Active()
