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

# Şekil dilbilgisinin tanıdığı işlev sözcükleri. Bunlar Türkçedir ve bir
# motorun içinde durmamalı; asıl yerleri `lmm/turkish.py`. Tek bir tabloya
# indirildiler, çünkü aynı sözcükler dosyada iki kez yazılıydı — bir kez bu
# kümede, bir kez `_shapes()` içinde — ve iki kopya sessizce ayrışmıştı:
# kümede "nedir" iki kez geçiyor, "neden" ile "nasıl" ise zaten
# `INTERROGATIVES`'ten geliyordu, yani üç madde ölü yazıydı.
#
# Sayıldı: kapalı sınıflar (soru eki, soru sözcüğü) çıkarıldığında geriye
# dokuz sözcük kalıyor. Çok dilliliğe geçerken bu dosyada taşınacak yerin
# tamamı bu dokuz satır — geri kalan her şey şekil, dil değil.
ARTICLE = "bir"             # "penguen BİR kuştur"
NEGATION = "değildir"       # "penguen bir memeli DEĞİLDİR"
WHAT_IS = "nedir"           # "penguen NEDİR"
WHAT_LIKE = "nasıldır"      # "penguen NASILDIR"
HOW = "nasıl"               # "penguen NASIL"
WHICH = "ne"                # "penguen NE yapabilir"
CAN_DO = "yapabilir"        # "penguen ne YAPABİLİR"
WHY = "neden"               # "penguen NEDEN uçamaz"
TELL = "anlat"              # "penguen ANLAT"

FUNCTION_WORDS = ({ARTICLE, NEGATION, WHAT_IS, WHAT_LIKE, HOW, CAN_DO, WHY,
                   TELL} | set(QUESTION_PARTICLES) | set(INTERROGATIVES))

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
        ([CONCEPT, ARTICLE, PROPERTY], "TEACH_TYPE"),
        ([CONCEPT, ARTICLE, CONCEPT], "TEACH_TYPE"),
        ([CONCEPT, ARTICLE, CONCEPT, NEGATION], "TEACH_NOT_TYPE"),
        ([CONCEPT, ARTICLE, PROPERTY, NEGATION], "TEACH_NOT_TYPE"),
        ([CONCEPTS, VERB], "TEACH_ABILITY"),
        ([CONCEPT, VERB], "TEACH_ABILITY"),
        ([CONCEPTS, VERB_NEGATIVE], "TEACH_ABILITY"),
        ([CONCEPT, VERB_NEGATIVE], "TEACH_ABILITY"),
        ([CONCEPT, PROPERTY], "TEACH_PROPERTY"),
        ([CONCEPTS, PROPERTY], "TEACH_PROPERTY"),
        ([CONCEPT, CONCEPT, NEGATION], "TEACH_NOT_PROPERTY"),
        ([CONCEPTS, CONCEPT, NEGATION], "TEACH_NOT_PROPERTY"),
        ([CONCEPT, WHAT_IS], "ASK_DEFINITION"),
        ([CONCEPT, WHICH, CAN_DO], "ASK_ABILITIES"),
        ([CONCEPT, WHAT_LIKE], "ASK_PROPERTIES"),
        ([CONCEPT, HOW], "ASK_PROPERTIES"),
        ([CONCEPT, TELL], "ASK_DESCRIBE"),
        ([TELL], "ASK_DESCRIBE"),
        ([WHAT_IS], "ASK_DEFINITION"),
        ([WHAT_LIKE], "ASK_PROPERTIES"),
        ([WHICH, CAN_DO], "ASK_ABILITIES"),
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
        examples.append(([CONCEPT, WHY, verb], "ASK_WHY"))
        examples.append(([CONCEPTS, WHY, verb], "ASK_WHY"))
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

    # 200 tur ve 0,5 adım gerekçesiz duruyordu; ölçüldü (73 şekil, 13 sınıf):
    #
    #     tur   şekillerin doğrusu   en düşük güven   TANIMADIĞI şekilde güven
    #      10           73/73            0,647              ort. 0,658
    #      20           73/73            0,850              ort. 0,722
    #     200           73/73            0,985              ort. 0,855
    #
    # Yani doğruluk 10. turda oturuyor; kalan 190 tur yalnızca güven satın
    # alıyor. Ve satın aldığı güven bedava değil: ağ, hiç görmediği şekillerde
    # de kendine güveniyor — bu, sınıfın en başında "yanlılık terimi
    # atıldı" diye anlatılan hatanın aynısının başka kapıdan girmesi.
    #
    # Sayı DEĞİŞTİRİLMEDİ, çünkü tahmin bir karar değil bir ipucu:
    # `intuition` yalnızca hiçbir kalıp tutmadığında soruyor ve `cli`
    # 0,5'in altını atıyor. Ama o eşik, bu tabloya göre tanınmayan şekillerin
    # çoğunu da geçiriyor — eşik ile tur sayısı birlikte ayarlanmalı ve ikisi
    # ayrı dosyada. Ölçüm burada dursun ki karar veren onu görsün.
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
