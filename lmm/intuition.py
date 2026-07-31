"""Language Intuition Core: sentence -> Intent.

Carries no knowledge of the world. Its only job is turning Turkish sentences into
structured intents; everything factual comes from memory. That split is why this
organ can stay small.
"""
from lmm.memory import IS_A, CAN, CANNOT

PUNCTUATION = ".,!?;:\"'"

# Verb forms of the controlled world: surface form -> (infinitive, is_positive)
VERBS = {
    "uçar": ("uçmak", True), "uçamaz": ("uçmak", False),
    "yüzer": ("yüzmek", True), "yüzemez": ("yüzmek", False),
    "koşar": ("koşmak", True), "koşamaz": ("koşmak", False),
    "okur": ("okumak", True), "okuyamaz": ("okumak", False),
    "içer": ("içmek", True), "içemez": ("içmek", False),
    "konuşur": ("konuşmak", True), "konuşamaz": ("konuşmak", False),
}
QUESTION_PARTICLES = ("mı", "mi", "mu", "mü")
COPULA_SUFFIXES = ("tur", "tır", "dur", "dır", "tür", "tir", "dür", "dir")
PLURAL_SUFFIXES = ("lar", "ler")

TEACH = "TEACH"
ASK = "ASK"
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


class Intuition:
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
