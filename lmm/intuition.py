"""Language Intuition Core: sentence -> Intent.

Carries no knowledge of the world. Its only job is turning Turkish sentences into
structured intents; everything factual comes from memory. That split is why this
organ can stay small.
"""
from lmm.memory import IS_A, NOT_A, CAN, CANNOT
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
UNKNOWN = "UNKNOWN"


def tokenize(sentence):
    cleaned = "".join(c for c in sentence if c not in PUNCTUATION)
    return cleaned.lower().split()


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
        # "X bir Y değildir" / "X Y değildir"
        if tokens and tokens[-1] == "değildir":
            body = tokens[:-1]
            if len(body) == 3 and body[1] == "bir":
                return Intent(TEACH, concept=body[0], relation=NOT_A, target=body[2])
            if len(body) == 2:
                return Intent(TEACH, concept=body[0], relation=NOT_A, target=body[1])
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
        return Intent(UNKNOWN, confidence=0.0)
