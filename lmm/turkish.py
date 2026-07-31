"""Turkish, described in one place.

The suffixes, the function words and the sentence patterns of one language. The
parser, the grammar and the reasoning know nothing about any of it — they ask a
morphology object questions like "does this word carry a copula" and match slots
against a pattern list.

A second language means another module shaped like this one, not another parser.
"""
from lmm.grammar import (Pattern, Grammar, KAVRAM, TUR, NITELIK, SOZ, FIIL,
                         SORU, KIM, FROM_VERB)
from lmm.relations import IS_A, NOT_A, CAN, HAS_PROPERTY, LACKS_PROPERTY

TEACH = "TEACH"
ASK = "ASK"
ASK_WHO = "ASK_WHO"
ASK_ABILITIES = "ASK_ABILITIES"
ASK_WHY = "ASK_WHY"
ASK_PROPERTIES = "ASK_PROPERTIES"
ASK_DESCRIBE = "ASK_DESCRIBE"


class TurkishMorphology:
    question_particles = ("mı", "mi", "mu", "mü")
    interrogatives = ("kim", "kimler", "ne", "neler")
    copula_suffixes = ("tur", "tır", "dur", "dır", "tür", "tir", "dür", "dir")
    plural_suffixes = ("lar", "ler")
    openers = ("peki", "ya", "hem")
    # Endings that mark a word as playing its own part in the sentence rather
    # than belonging to the noun beside it: "serçeden" is a comparison, not half
    # of a compound. Discovery finds these families; until it supplies them,
    # they are listed.
    oblique_suffixes = ("den", "dan", "ten", "tan", "yle", "yla", "ile")
    pronouns = ("o", "onu", "onun", "bu", "bunu", "şu", "şunu")

    def is_oblique(self, word):
        """Does this word carry a case ending, so it stands on its own?"""
        return self._has(word, self.oblique_suffixes)

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
    # A sentence that leaves its subject out, carrying on from the last one.
    Pattern([FIIL, SORU], ASK, CAN, None, 0, "çıplak yetenek sorusu"),
    Pattern(["nedir"], ASK, IS_A, None, None, "çıplak tanım sorusu"),
    Pattern(["anlat"], ASK_DESCRIBE, None, None, None, "çıplak anlat"),
    Pattern(["nasıldır"], ASK_PROPERTIES, None, None, None, "çıplak nasıl"),
    Pattern(["nasıl"], ASK_PROPERTIES, None, None, None, "çıplak nasıl 2"),
    Pattern(["ne", "yapabilir"], ASK_ABILITIES, None, None, None, "çıplak neler"),

    Pattern([KIM, FIIL], ASK_WHO, FROM_VERB, None, 1, "kimler uçar"),
    Pattern([KAVRAM, "ne", "yapabilir"], ASK_ABILITIES, None, 0, None, "ne yapabilir"),
    Pattern([KAVRAM, "ne", "yapar"], ASK_ABILITIES, None, 0, None, "ne yapar"),
    Pattern([KAVRAM, "neler", "yapar"], ASK_ABILITIES, None, 0, None, "neler yapar"),
    Pattern([KAVRAM, "neden", FIIL], ASK_WHY, FROM_VERB, 0, 2, "neden uçar"),
    Pattern([KAVRAM, "neden", SOZ], ASK_WHY, HAS_PROPERTY, 0, 2, "neden beyaz"),
    Pattern([KAVRAM, "anlat"], ASK_DESCRIBE, None, 0, None, "anlat"),
    Pattern([KAVRAM, "anlatsana"], ASK_DESCRIBE, None, 0, None, "anlatsana"),
    Pattern([KAVRAM, "nasıldır"], ASK_PROPERTIES, None, 0, None, "nasıldır"),
    Pattern([KAVRAM, "nasıl"], ASK_PROPERTIES, None, 0, None, "nasıl"),

    Pattern([KAVRAM, "bir", SOZ, "değildir"], TEACH, NOT_A, 0, 2, "bir X değildir"),
    Pattern([KAVRAM, SOZ, "değildir"], TEACH, LACKS_PROPERTY, 0, 1, "X değildir"),
    Pattern([KAVRAM, "bir", TUR, SORU], ASK, IS_A, 0, 2, "bir X mı"),
    Pattern([KAVRAM, "nedir"], ASK, IS_A, 0, None, "nedir"),
    Pattern([KAVRAM, "ne"], ASK, IS_A, 0, None, "ne"),
    # Turkish puts the object before the verb. Which position it takes is a
    # fact about a language, so it sits in this list and nowhere else.
    Pattern([KAVRAM, KAVRAM, FIIL, SORU], ASK, CAN, 0, 2, "fare yakalar mı",
            object=1),
    Pattern([KAVRAM, FIIL, SORU], ASK, CAN, 0, 1, "uçar mı"),
    Pattern([KAVRAM, "bir", TUR], TEACH, IS_A, 0, 2, "bir kuştur"),
    Pattern([KAVRAM, KAVRAM, FIIL], TEACH, FROM_VERB, 0, 2, "fare yakalar",
            object=1),
    Pattern([KAVRAM, FIIL], TEACH, FROM_VERB, 0, 1, "uçar"),
    Pattern([KAVRAM, SOZ, SORU], ASK, HAS_PROPERTY, 0, 1, "beyaz mı"),
    Pattern([KAVRAM, NITELIK], TEACH, HAS_PROPERTY, 0, 1, "beyazdır"),
]


def turkish():
    return Grammar(PATTERNS, TurkishMorphology())
