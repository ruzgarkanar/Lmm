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

# The starting vocabulary of the controlled world: surface -> (infinitive, is_positive)
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


# Geniş zaman dışındaki çekimler. Olumsuzluk ayrı bir ek olarak okunuyor;
# `negative` sütunu ekin KENDİSİNİN olumlu olup olmadığını söylüyor.
TENSES = (
    (("ıyor", "iyor", "uyor", "üyor", "yor"), True),     # şimdiki zaman
    (("acak", "ecek", "acağı", "eceği"), True),          # gelecek
    (("mış", "miş", "muş", "müş"), True),                # duyulan geçmiş
    (("dı", "di", "du", "dü", "tı", "ti", "tu", "tü"), True),   # görülen geçmiş
)
# Olumsuzluk ve yetersizlik: "uçmuyor", "uçamıyor". Ünlü uyumu yüzünden
# ek ünlüsü değişiyor, o yüzden hepsi sayılıyor — kapalı bir sınıf.
NEGATIVE_MARKS = ("amı", "emi", "amu", "emü", "mı", "mi", "mu", "mü", "ma", "me")

# Kişi ekleri. Fiil çekiminin en dış katmanı ve soyulmadan sözlük kelimeyi
# tanımıyordu: "söyleyebilirsin", "biliyorsun", "konuşuyoruz" hepsi
# "bilmiyorum" cevabı alıyordu. Aynı ayrımı soru ekinde çözmüştük
# (`misin` -> `mi`); fiilde çözmemişiz.
#
# Kapalı sınıf: bir dilde altı kişi vardır ve ünlü uyumuyla çoğalırlar.
PERSON = ("sınız", "siniz", "sunuz", "sünüz",
          "ım", "im", "um", "üm", "yım", "yim", "yum", "yüm",
          "sın", "sin", "sun", "sün",
          "ız", "iz", "uz", "üz", "yız", "yiz", "yuz", "yüz",
          "lar", "ler")

ABILITY_SUFFIXES = ("ebilir", "abilir")


class Lexicon:
    def __init__(self, verbs=None):
        self.verbs = dict(CORE_VERBS if verbs is None else verbs)
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
            for ending in PERSON:
                if not surface.endswith(ending):
                    continue
                if len(surface) - len(ending) < 3:
                    continue
                inner = surface[: -len(ending)]
                found = self._direct_or_ability(inner) or \
                    self._inflected(inner, depth - 1)
                if found is not None:
                    return found
        for suffixes, negative in TENSES:
            for suffix in suffixes:
                if not surface.endswith(suffix):
                    continue
                stem = surface[: -len(suffix)]
                if len(stem) < 2:
                    continue
                polarity = True
                for mark in NEGATIVE_MARKS:
                    if stem.endswith(mark) and len(stem) > len(mark) + 1:
                        stem, polarity = stem[: -len(mark)], False
                        break
                found = self._stem_of(stem)
                if found is not None:
                    return found, polarity and negative
        return None

    def _stem_of(self, stem):
        """Bilinen bir fiilin gövdesi mi — ünlü kaymasına izin vererek."""
        for infinitive, positive in self.verbs.values():
            if not positive:
                continue
            root = infinitive[:-3] if infinitive.endswith(("mak", "mek")) \
                else infinitive
            if root and (stem == root or stem.rstrip("aeıioöuü") == root):
                return infinitive
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
        for suffix in ABILITY_SUFFIXES:
            if surface.endswith(suffix) and len(surface) > len(suffix) + 1:
                stem = surface[: -len(suffix)].rstrip("y")
                for infinitive, positive in self.verbs.values():
                    if positive and infinitive.startswith(stem):
                        return infinitive, True
        return None

    def surface(self, infinitive, positive):
        """"uçmak", False -> "uçamaz"; falls back to the infinitive itself."""
        return self._forms.get((infinitive, positive), infinitive)

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
