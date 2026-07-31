"""Language Intuition Core: sentence -> Intent.

Carries no knowledge of the world. Its only job is turning Turkish sentences into
structured intents; everything factual comes from memory. That split is why this
organ can stay small.
"""
from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                           LACKS_PROPERTY)
from lmm.phrasing import VERBS
from lmm.language import LanguageOrgan

PUNCTUATION = ".,!?;:\"'"

QUESTION_PARTICLES = ("mı", "mi", "mu", "mü")
COPULA_SUFFIXES = ("tur", "tır", "dur", "dır", "tür", "tir", "dür", "dir")
PLURAL_SUFFIXES = ("lar", "ler")
INTERROGATIVES = ("kim", "kimler", "ne", "neler")

TEACH = "TEACH"
ASK = "ASK"
ASK_WHO = "ASK_WHO"
ASK_ABILITIES = "ASK_ABILITIES"
ASK_WHY = "ASK_WHY"
ASK_PROPERTIES = "ASK_PROPERTIES"
UNKNOWN = "UNKNOWN"


def tokenize(sentence):
    cleaned = "".join(c for c in sentence if c not in PUNCTUATION)
    return cleaned.lower().split()


def _has_suffix(word, suffixes):
    return any(word.endswith(s) and len(word) > len(s) + 1 for s in suffixes)


def _strip_suffix(word, suffixes):
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)]
    return word


class Intent:
    def __init__(self, kind, concept=None, relation=None, target=None, confidence=1.0):
        self.kind = kind            # TEACH | ASK | UNKNOWN
        self.concept = concept
        self.relation = relation    # IS_A | CAN | CANNOT
        self.target = target
        self.confidence = confidence


class Intuition(LanguageOrgan):
    def __init__(self, network=None):
        self.network = network      # MiniNetwork supplies the confidence signal

    def understand(self, sentence):
        tokens = tokenize(sentence)
        intent = self._parse_pattern(tokens)
        if self.network is not None:
            _, confidence = self.network.predict(tokens)
            intent.confidence = confidence
        return intent

    def _parse_pattern(self, tokens):
        # Questions come first: "kim uçar" also fits the teaching shape
        # "X(lar) <verb>", and reading it as a lesson would be wrong.
        # "kimler <verb>" / "ne <verb>"
        if len(tokens) == 2 and tokens[0] in INTERROGATIVES and tokens[1] in VERBS:
            infinitive, positive = VERBS[tokens[1]]
            return Intent(ASK_WHO, relation=CAN if positive else CANNOT,
                          target=infinitive)
        # "X ne yapabilir"
        if len(tokens) == 3 and tokens[1] == "ne" and tokens[2] == "yapabilir":
            return Intent(ASK_ABILITIES, concept=tokens[0])
        # "X neden <verb>"
        if len(tokens) == 3 and tokens[1] == "neden" and tokens[2] in VERBS:
            infinitive, positive = VERBS[tokens[2]]
            return Intent(ASK_WHY, concept=_strip_suffix(tokens[0], PLURAL_SUFFIXES),
                          relation=CAN if positive else CANNOT, target=infinitive)
        # "X nasıldır"
        if len(tokens) == 2 and tokens[1] in ("nasıldır", "nasıl"):
            return Intent(ASK_PROPERTIES, concept=tokens[0])
        # "X bir Y değildir" (type) / "X Y değildir" (property)
        # The word "bir" is what separates being something from being like
        # something — Turkish already draws the line for us.
        if tokens and tokens[-1] == "değildir":
            body = tokens[:-1]
            if len(body) == 3 and body[1] == "bir":
                return Intent(TEACH, concept=body[0], relation=NOT_A, target=body[2])
            if len(body) == 2:
                return Intent(TEACH,
                              concept=_strip_suffix(body[0], PLURAL_SUFFIXES),
                              relation=LACKS_PROPERTY, target=body[1])
        # "X nedir"
        if len(tokens) == 2 and tokens[1] == "nedir":
            return Intent(ASK, concept=tokens[0], relation=IS_A)
        # "X <verb> mı"
        if len(tokens) == 3 and tokens[2] in QUESTION_PARTICLES and tokens[1] in VERBS:
            infinitive, _ = VERBS[tokens[1]]
            return Intent(ASK, concept=_strip_suffix(tokens[0], PLURAL_SUFFIXES),
                          relation=CAN, target=infinitive)
        # "X bir Y(dır)"
        if len(tokens) == 3 and tokens[1] == "bir":
            return Intent(TEACH, concept=tokens[0], relation=IS_A,
                          target=_strip_suffix(tokens[2], COPULA_SUFFIXES))
        # "X(lar) <verb>"
        if len(tokens) == 2 and tokens[1] in VERBS:
            infinitive, positive = VERBS[tokens[1]]
            return Intent(TEACH, concept=_strip_suffix(tokens[0], PLURAL_SUFFIXES),
                          relation=CAN if positive else CANNOT, target=infinitive)
        # "X Y mı" — a property question, once the verb reading is ruled out
        if len(tokens) == 3 and tokens[2] in QUESTION_PARTICLES:
            return Intent(ASK, concept=_strip_suffix(tokens[0], PLURAL_SUFFIXES),
                          relation=HAS_PROPERTY, target=tokens[1])
        # "X(lar) Y(dır)" — a property, since no "bir" made it a type
        if len(tokens) == 2 and _has_suffix(tokens[1], COPULA_SUFFIXES):
            return Intent(TEACH, concept=_strip_suffix(tokens[0], PLURAL_SUFFIXES),
                          relation=HAS_PROPERTY,
                          target=_strip_suffix(tokens[1], COPULA_SUFFIXES))
        return Intent(UNKNOWN, confidence=0.0)
