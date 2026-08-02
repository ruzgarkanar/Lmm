"""Mini neural network: sentence shape -> softmax over intent classes.

Written by hand, no libraries: the softmax, cross-entropy and gradient descent
from our own llm.docx walkthrough, running for real.

It classifies *shapes*, not words. Before predicting, every token is reduced to
what it does in the sentence — a concept, a property, a verb, or a function word
like "bir". An earlier version learned the nouns themselves and then refused a
perfectly good sentence because the subject was new to it, which is exactly the
failure LMM exists to avoid: meeting something unfamiliar must be free. Only an
unfamiliar *shape* should give the system pause.

A pleasant consequence: learning new vocabulary never retrains anything. Words
change how a token is recognised, not what the shapes are.
"""
import math

from lmm.intuition import (QUESTION_PARTICLES, COPULA_SUFFIXES, PLURAL_SUFFIXES,
                           INTERROGATIVES, _MORPHOLOGY, _has_suffix)
from lmm.lexicon import ACTIVE

CLASSES = ["TEACH_TYPE", "TEACH_NOT_TYPE", "TEACH_ABILITY", "TEACH_PROPERTY",
           "TEACH_NOT_PROPERTY", "ASK_DEFINITION", "ASK_ABILITY", "ASK_PROPERTY",
           "ASK_WHO", "ASK_ABILITIES", "ASK_PROPERTIES", "ASK_WHY",
           "ASK_DESCRIBE"]

FUNCTION_WORDS = ({"bir", "değildir", "nedir", "neden", "nasıldır", "nasıl",
                   "yapabilir", "anlat", "nedir"} | set(QUESTION_PARTICLES) | set(INTERROGATIVES))

CONCEPT = "<kavram>"
CONCEPTS = "<kavram-çoğul>"
PROPERTY = "<nitelik>"
VERB = "<fiil>"
VERB_NEGATIVE = "<fiil-olumsuz>"


def _bare(token):
    """Soru ekiyse çıplak hâli, değilse koşaçsız hâli.

    Soru üç biçimde gelir ve üçü de aynı şeyi sorar: "uçar mı", "mutlu mudur",
    "bahseder misin". Sonuncusu hiç tanınmıyordu ve insanlar soruyu asıl öyle
    soruyor — ölçüldüğünde "bana penguenlerden bahseder misin" dört kavram
    olarak etiketleniyordu.
    """
    from lmm import asking
    found = asking.particle_of(token, _MORPHOLOGY)
    if found is not None:
        return found
    found = asking.interrogative_of(token, _MORPHOLOGY)
    if found is not None:
        return found
    return (_MORPHOLOGY.strip_copula(token) if _MORPHOLOGY.has_copula(token)
            else token)


def features(tokens, lexicon=None):
    """What each token *does*, tagged with where it sits and how long the whole is.

    Position and length matter: "kar beyaz mı" and "penguen bir kuş değildir"
    share tags but not shapes.
    """
    lexicon = lexicon or ACTIVE
    tagged = []
    for token in tokens:
        if token in FUNCTION_WORDS:
            tagged.append(token)
        # Soru eki koşaç alınca soru olmaktan çıkmaz: "mu" ile "mudur",
        # "neler" ile "nelerdir" aynı şeyi sorar. Bu ayrım kaçınca cümlenin
        # şekli bildirmeye benziyordu ve sonucu ağırdı: "sence penguenler
        # mutlu mudur" sorusu TEACH sayılıp grafa `sence penguenler mutlu
        # —property→ mu` diye yazılıyordu. Cevap uydurulmuyordu ama hafıza
        # kirleniyordu — ki hafıza bu projenin tek varlığı.
        #
        # Liste uzatılmıyor, ek soyuluyor: koşaç ekleri zaten türetilmiş ve
        # kapalı sınıf olduğu gibi duruyor.
        elif _bare(token) in FUNCTION_WORDS:
            tagged.append(_bare(token))
        elif lexicon.knows(token):
            _, positive = lexicon.reading(token)
            tagged.append(VERB if positive else VERB_NEGATIVE)
        elif _has_suffix(token, COPULA_SUFFIXES):
            tagged.append(PROPERTY)
        elif _has_suffix(token, PLURAL_SUFFIXES):
            tagged.append(CONCEPTS)
        else:
            tagged.append(CONCEPT)
    return [f"{index}:{tag}" for index, tag in enumerate(tagged)] + \
           [f"uzunluk:{len(tagged)}"]


def _shapes():
    """One entry per sentence shape the language accepts."""
    examples = [
        ([CONCEPT, "bir", PROPERTY], "TEACH_TYPE"),
        ([CONCEPT, "bir", CONCEPT], "TEACH_TYPE"),
        ([CONCEPT, "bir", CONCEPT, "değildir"], "TEACH_NOT_TYPE"),
        ([CONCEPT, "bir", PROPERTY, "değildir"], "TEACH_NOT_TYPE"),
        ([CONCEPTS, VERB], "TEACH_ABILITY"),
        ([CONCEPT, VERB], "TEACH_ABILITY"),
        ([CONCEPTS, VERB_NEGATIVE], "TEACH_ABILITY"),
        ([CONCEPT, VERB_NEGATIVE], "TEACH_ABILITY"),
        ([CONCEPT, PROPERTY], "TEACH_PROPERTY"),
        ([CONCEPTS, PROPERTY], "TEACH_PROPERTY"),
        ([CONCEPT, CONCEPT, "değildir"], "TEACH_NOT_PROPERTY"),
        ([CONCEPTS, CONCEPT, "değildir"], "TEACH_NOT_PROPERTY"),
        ([CONCEPT, "nedir"], "ASK_DEFINITION"),
        ([CONCEPT, "ne", "yapabilir"], "ASK_ABILITIES"),
        ([CONCEPT, "nasıldır"], "ASK_PROPERTIES"),
        ([CONCEPT, "nasıl"], "ASK_PROPERTIES"),
        ([CONCEPT, "anlat"], "ASK_DESCRIBE"),
        (["anlat"], "ASK_DESCRIBE"),
        (["nedir"], "ASK_DEFINITION"),
        (["nasıldır"], "ASK_PROPERTIES"),
        (["ne", "yapabilir"], "ASK_ABILITIES"),
    ]
    for particle in QUESTION_PARTICLES:
        examples.append(([CONCEPT, VERB, particle], "ASK_ABILITY"))
        examples.append(([CONCEPTS, VERB, particle], "ASK_ABILITY"))
        examples.append(([CONCEPT, CONCEPT, particle], "ASK_PROPERTY"))
        examples.append(([VERB, particle], "ASK_ABILITY"))
    for interrogative in INTERROGATIVES:
        examples.append(([interrogative, VERB], "ASK_WHO"))
        examples.append(([interrogative, VERB_NEGATIVE], "ASK_WHO"))
    for verb in (VERB, VERB_NEGATIVE):
        examples.append(([CONCEPT, "neden", verb], "ASK_WHY"))
        examples.append(([CONCEPTS, "neden", verb], "ASK_WHY"))
    return examples


def training_data():
    """(features, class) pairs — shapes already reduced to their feature form."""
    return [([f"{i}:{tag}" for i, tag in enumerate(tags)] +
             [f"uzunluk:{len(tags)}"], label)
            for tags, label in _shapes()]


class MiniNetwork:
    """Deliberately bias-free: a score is a sum of evidence, nothing else.

    A bias term is a standing opinion held in the absence of input, and it is
    what made an early version answer unseen gibberish with 92% confidence.
    """

    def __init__(self, classes):
        self.classes = list(classes)
        self.weights = {c: {} for c in self.classes}   # class -> {feature: weight}

    def _logits(self, encoded):
        return {c: sum(self.weights[c].get(f, 0.0) for f in encoded)
                for c in self.classes}

    def _softmax(self, logits):
        largest = max(logits.values())
        exponentials = {c: math.exp(v - largest) for c, v in logits.items()}
        total = sum(exponentials.values())
        return {c: v / total for c, v in exponentials.items()}

    def predict(self, tokens, lexicon=None):
        probabilities = self._softmax(self._logits(features(tokens, lexicon)))
        best = max(probabilities, key=probabilities.get)
        return best, probabilities[best]

    def train(self, examples, epochs=200, learning_rate=0.5):
        for _ in range(epochs):
            for encoded, correct in examples:
                probabilities = self._softmax(self._logits(encoded))
                for c in self.classes:
                    # derivative of softmax + cross-entropy w.r.t. the logit
                    gradient = probabilities[c] - (1.0 if c == correct else 0.0)
                    for feature in encoded:
                        current = self.weights[c].get(feature, 0.0)
                        self.weights[c][feature] = current - learning_rate * gradient

    @staticmethod
    def default():
        """The trained network, built once per process.

        Training is deterministic and shape-based, so one network serves every
        session and every vocabulary.
        """
        global _DEFAULT
        if _DEFAULT is None:
            _DEFAULT = MiniNetwork(CLASSES)
            _DEFAULT.train(training_data())
        return _DEFAULT


_DEFAULT = None
