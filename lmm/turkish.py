"""Turkish, described in one place.

The suffixes, the function words and the sentence patterns of one language. The
parser, the grammar and the reasoning know nothing about any of it — they ask a
morphology object questions like "does this word carry a copula" and match slots
against a pattern list.

A second language means another module shaped like this one, not another parser.
"""
from lmm.grammar import (Pattern, Grammar, KAVRAM, TUR, NITELIK, SOZ, FIIL,
                         SORU, KIM, ROL, NICEL, SAHIP, FROM_VERB)
from lmm.relations import (IS_A, NOT_A, CAN, HAS_PROPERTY, LACKS_PROPERTY,
                           HAS_PART, LACKS_PART, PLACE, SOURCE, ALL,
                           MOST, SOME, NO)

TEACH = "TEACH"
ASK = "ASK"
ASK_WHO = "ASK_WHO"
ASK_ABILITIES = "ASK_ABILITIES"
ASK_WHY = "ASK_WHY"
ASK_PROPERTIES = "ASK_PROPERTIES"
ASK_DESCRIBE = "ASK_DESCRIBE"
ASK_HOW_MANY = "ASK_HOW_MANY"


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


def turkish(words=None, known=()):
    """The grammar. Given words, its suffix rules are discovered rather than read.

    Discovery needs enough words to see a pattern; below that it falls through
    to the declared lists, so a thin memory degrades instead of breaking.
    """
    morphology = TurkishMorphology()
    if words:
        from lmm.discovered import DiscoveredMorphology
        morphology = DiscoveredMorphology(morphology, words, ANCHORS)
    return Grammar(PATTERNS, morphology, known)
