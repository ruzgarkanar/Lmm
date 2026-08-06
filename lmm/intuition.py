"""Language Intuition Core: sentence -> Intent.

Kalıp dilbilgisi ve biçimbilim SİLİNDİ — sahibin kuralı: okuma eğitilmiş
ağlarda (`core/`), ayrık kural ve ek listesi sistemde durmayacak. Bu dosyada
kalan şey görev sözlüğü: niyet türlerinin adları, Intent taşıyıcısı ve saf
dizgi işlemleri (küçültme, bölme). `Intuition.understand` artık hiçbir cümleyi
kendisi okumuyor; her cümleye UNKNOWN der ve okuma `lmm/cli.py`'deki ağ
yollarından geçer.
"""
from lmm.relations import ALL
from lmm.language import LanguageOrgan

# Niyet türleri: görev sözlüğü, dil değil. Eğitim verisini kuran betiklerle
# ve ağın sınıf adlarıyla aynı.
TEACH = "TEACH"
ASK = "ASK"
ASK_WHO = "ASK_WHO"
ASK_ABILITIES = "ASK_ABILITIES"
ASK_WHY = "ASK_WHY"
ASK_PROPERTIES = "ASK_PROPERTIES"
ASK_DESCRIBE = "ASK_DESCRIBE"
ASK_COMPARE = "ASK_COMPARE"
ASK_THREAD = "ASK_THREAD"
ASK_WHICH_MORE = "ASK_WHICH_MORE"
ASK_REQUIREMENT = "ASK_REQUIREMENT"
ASK_MORE = "ASK_MORE"
ASK_INVENTORY = "ASK_INVENTORY"
ASK_CERTAINTY = "ASK_CERTAINTY"
ASK_SOURCE = "ASK_SOURCE"
ASK_OPINION = "ASK_OPINION"
ASK_HOW_MANY = "ASK_HOW_MANY"
ASK_WHERE = "ASK_WHERE"

UNKNOWN = "UNKNOWN"
UNKNOWN_WORD = "UNKNOWN_WORD"
AMBIGUOUS = "AMBIGUOUS"

PUNCTUATION = ".,!?;:\"'"

# Ek listeleri dille birlikte gitti. Boş küme: hiçbir şey soyulmaz.
QUESTION_PARTICLES = ()
COPULA_SUFFIXES = ()
PLURAL_SUFFIXES = ()
INTERROGATIVES = ()
PRONOUNS = ()
OUTER_SUFFIXES = ()

# Alfabe düzeltmesi kural değil harf eşlemesi: Python "İ"yi noktalı "i"ye
# indirir ve kelime sessizce başka kelime olur.
LOWERCASE_PAIRS = (("İ", "i"), ("I", "ı"))


class _Morphology:
    def __getattr__(self, name):
        # Silinen morfoloji tablolarını arayan her organ boş bulur ve
        # sessizce kapanır — dil bilgisi artık hiçbir tabloda değil.
        return ()

    """Biçimbilim yokken çağıranların düşeceği yumuşak zemin."""

    def has_copula(self, word):
        return False

    def strip_copula(self, word):
        return word

    def strip_plural(self, word):
        return word

    def strip_accusative(self, word):
        return word

    def strip_genitive(self, word, known=()):
        return word

    def role_of(self, word):
        return word, None


_MORPHOLOGY = _Morphology()


def lower(text):
    for upper, small in LOWERCASE_PAIRS:
        text = text.replace(upper, small)
    return text.lower()


def tokenize(sentence):
    cleaned = "".join(c for c in sentence if c not in PUNCTUATION)
    return lower(cleaned).split()


def _has_suffix(word, suffixes):
    return any(word.endswith(s) and len(word) > len(s) + 1 for s in suffixes)


def _strip_suffix(word, suffixes):
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)]
    return word


class Intent:
    def __init__(self, kind, concept=None, relation=None, target=None,
                 object=None, role=None, quantifier=ALL, confidence=1.0):
        self.kind = kind
        self.concept = concept
        self.relation = relation
        self.target = target
        self.object = object        # a second concept, if the sentence had one
        self.role = role            # what that concept is doing there
        self.quantifier = quantifier    # how much of the kind is meant
        self.confidence = confidence
        self.resembles = None       # shape the network saw, when nothing matched
        self.resemblance = 0.0


class _Grammar:
    """Dilbilgisi yok; çağıranlar boş tablo bulur ve yolları kapanır."""

    morphology = _MORPHOLOGY
    patterns = ()

    def known(self):
        return frozenset()


class Intuition(LanguageOrgan):
    """Okumayan okuyucu: her cümleye UNKNOWN.

    Cümleyi anlamak `lmm/cli.py`'deki eğitilmiş yolların işi
    (`_neural_reading`, `_neural_teaching`, etiketleyici). Burası yalnız
    arayüzü ayakta tutuyor — organ takılabilir kalır, kural taşımaz.
    """

    def __init__(self, network=None, lexicon=None, grammar=None, words=None,
                 known=None, meanings=None, memory=None, **_):
        self.network = network
        self.lexicon = lexicon
        self.memory = memory
        self.grammar = grammar or _Grammar()

    def understand(self, sentence):
        return Intent(UNKNOWN, confidence=0.0)
