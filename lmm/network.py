"""Mini neural network: bag-of-words -> softmax over intent classes.

Written by hand, no libraries: the softmax, cross-entropy and gradient descent
from our own llm.docx walkthrough, running for real. Its output feeds the
system's confidence signal — when the network is unsure, LMM says so instead of
guessing, which is epistemic honesty applied to language itself.
"""
import math

from lmm.phrasing import copula

CLASSES = ["TEACH_TYPE", "TEACH_NOT_TYPE", "TEACH_ABILITY", "TEACH_PROPERTY",
           "TEACH_NOT_PROPERTY", "ASK_DEFINITION", "ASK_ABILITY", "ASK_PROPERTY",
           "ASK_WHO", "ASK_ABILITIES", "ASK_PROPERTIES", "ASK_WHY"]

_ENTITIES = ["kedi", "köpek", "kuş", "balık", "at", "penguen", "serçe", "çocuk"]
_TYPES = ["hayvan", "kuş", "canlı", "varlık"]
_VERBS = ["uçar", "yüzer", "koşar", "okur", "içer", "konuşur"]
_ADJECTIVES = ["beyaz", "siyah", "büyük", "küçük", "hızlı", "gizli"]
_INTERROGATIVES = ("kim", "kimler", "ne", "neler")
_QUESTION_PARTICLES = ("mı", "mi", "mu", "mü")


def training_data():
    """(tokens, class) examples generated from the controlled world's patterns."""
    examples = []
    for entity in _ENTITIES:
        for bare in _TYPES:
            # the copula comes from our own phrasing rules, not a hand-typed list
            examples.append(([entity, "bir", bare + copula(bare)], "TEACH_TYPE"))
            examples.append(([entity, "bir", bare, "değildir"], "TEACH_NOT_TYPE"))
        for index, verb in enumerate(_VERBS):
            examples.append(([entity + "lar", verb], "TEACH_ABILITY"))
            # rotate the particles so all four are learned without skewing classes
            particle = _QUESTION_PARTICLES[index % len(_QUESTION_PARTICLES)]
            examples.append(([entity, verb, particle], "ASK_ABILITY"))
        examples.append(([entity, "nedir"], "ASK_DEFINITION"))
        examples.append(([entity, "ne", "yapabilir"], "ASK_ABILITIES"))
        examples.append(([entity, "nasıldır"], "ASK_PROPERTIES"))
        for index, adjective in enumerate(_ADJECTIVES):
            examples.append(([entity, adjective + copula(adjective)],
                             "TEACH_PROPERTY"))
            examples.append(([entity, adjective, "değildir"],
                             "TEACH_NOT_PROPERTY"))
            particle = _QUESTION_PARTICLES[index % len(_QUESTION_PARTICLES)]
            examples.append(([entity, adjective, particle], "ASK_PROPERTY"))
    for verb in _VERBS:
        for interrogative in _INTERROGATIVES:
            examples.append(([interrogative, verb], "ASK_WHO"))
        for entity in _ENTITIES:
            examples.append(([entity, "neden", verb], "ASK_WHY"))
    return examples


class MiniNetwork:
    """Deliberately bias-free: a score is a sum of evidence, nothing else.

    A bias term is a standing opinion held in the absence of input, and it is
    what made an early version answer unseen gibberish with 92% confidence. With
    no bias, unknown tokens contribute zero, the distribution stays uniform, and
    low confidence propagates outward as "bunu anlamadım".
    """

    def __init__(self, classes):
        self.classes = list(classes)
        self.weights = {c: {} for c in self.classes}   # class -> {token: weight}

    def _logits(self, tokens):
        return {c: sum(self.weights[c].get(t, 0.0) for t in tokens)
                for c in self.classes}

    def _softmax(self, logits):
        largest = max(logits.values())
        exponentials = {c: math.exp(v - largest) for c, v in logits.items()}
        total = sum(exponentials.values())
        return {c: v / total for c, v in exponentials.items()}

    def predict(self, tokens):
        probabilities = self._softmax(self._logits(tokens))
        best = max(probabilities, key=probabilities.get)
        return best, probabilities[best]

    def train(self, examples, epochs=150, learning_rate=0.5):
        for _ in range(epochs):
            for tokens, correct in examples:
                probabilities = self._softmax(self._logits(tokens))
                for c in self.classes:
                    # derivative of softmax + cross-entropy w.r.t. the logit
                    gradient = probabilities[c] - (1.0 if c == correct else 0.0)
                    for token in tokens:
                        current = self.weights[c].get(token, 0.0)
                        self.weights[c][token] = current - learning_rate * gradient

    @staticmethod
    def default():
        """The trained seed network, built once per process.

        Training is deterministic — same data, same weights every time — so
        every session can share one network instead of paying for it again.
        """
        global _DEFAULT
        if _DEFAULT is None:
            _DEFAULT = MiniNetwork(CLASSES)
            _DEFAULT.train(training_data())
        return _DEFAULT


_DEFAULT = None
