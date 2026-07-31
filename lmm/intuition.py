"""Language Intuition Core: sentence -> Intent.

Carries no knowledge of the world, and no knowledge of Turkish either. It
tokenises, hands the tokens to a grammar, and turns whatever matched into an
intent. The patterns are a list and the morphology is a set of suffix rules —
both data, both replaceable, neither written into this file.

That is what makes a second language a second data module rather than a second
parser, and what lets the system be taught a new way of saying something while
it is running.
"""
from lmm.lexicon import ACTIVE
from lmm.language import LanguageOrgan
from lmm.turkish import (turkish, TurkishMorphology, TEACH, ASK, ASK_WHO,  # noqa: F401
                         ASK_ABILITIES, ASK_WHY, ASK_PROPERTIES, ASK_DESCRIBE)

PUNCTUATION = ".,!?;:\"'"

UNKNOWN = "UNKNOWN"
UNKNOWN_WORD = "UNKNOWN_WORD"

_MORPHOLOGY = TurkishMorphology()
QUESTION_PARTICLES = _MORPHOLOGY.question_particles
COPULA_SUFFIXES = _MORPHOLOGY.copula_suffixes
PLURAL_SUFFIXES = _MORPHOLOGY.plural_suffixes
INTERROGATIVES = _MORPHOLOGY.interrogatives
PRONOUNS = _MORPHOLOGY.pronouns


def lower(text):
    """Turkish lowercasing.

    Python maps "İ" to "i" plus a combining dot, not to "i", so "İnsanlar"
    silently becomes a different word from "insan". Every sentence starting
    with İ was quietly learned under a concept nobody could ever ask about.
    """
    return text.replace("İ", "i").replace("I", "ı").lower()


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
                 object=None, confidence=1.0):
        self.kind = kind
        self.concept = concept
        self.relation = relation
        self.target = target
        self.object = object        # what the action was done to, if anything
        self.confidence = confidence
        self.resembles = None       # shape the network saw, when nothing matched
        self.resemblance = 0.0


class Intuition(LanguageOrgan):
    def __init__(self, network=None, lexicon=None, grammar=None):
        self.network = network      # MiniNetwork supplies the resemblance hint
        self.lexicon = lexicon or ACTIVE
        self.grammar = grammar or turkish()

    def understand(self, sentence):
        """A matched pattern is the evidence; the network speaks when none matched.

        The grammar is deterministic, so overriding a successful match with a
        network score only ever invents doubt — an earlier version refused a
        perfectly good sentence because its subject was a word it had never met.
        """
        tokens = tokenize(sentence)
        morphology = self.grammar.morphology
        while tokens and tokens[0] in morphology.openers:
            tokens = tokens[1:]         # "peki uçar mı" is "uçar mı"

        pattern, captured = self.grammar.match(tokens, self.lexicon)
        if pattern is not None:
            kind, relation, concept, target, obj = self.grammar.read(pattern,
                                                                     captured)
            return Intent(kind, concept, relation, target, obj)

        # A subject followed by something that is neither a known verb nor a
        # property is almost certainly a verb nobody taught us. Saying so is
        # more useful than shrugging, and it is how vocabulary grows.
        if len(tokens) == 2 and not self.lexicon.knows(tokens[1]):
            return Intent(UNKNOWN_WORD,
                          concept=morphology.strip_plural(tokens[0]),
                          target=tokens[1])

        intent = Intent(UNKNOWN, confidence=0.0)
        if self.network is not None and tokens:
            intent.resembles, intent.resemblance = self.network.predict(
                tokens, self.lexicon)
        return intent
