"""Word order is data.

Turkish puts the object before the verb, English after it. Nothing in the
matcher knows either fact — both live in a pattern list, next to the suffix
rules of the language they belong to. This is the claim the whole grammar
refactor was for, so it is checked with a language the system was never written
for.
"""
import unittest

from lmm.grammar import Grammar, Pattern, KAVRAM, TUR, FIIL, FROM_VERB
from lmm.turkish import TEACH, ASK
from lmm.relations import IS_A, CAN
from lmm.lexicon import Lexicon
from lmm.intuition import Intuition
from lmm.memory import Memory, Edge
from lmm.reasoning import Reasoning


class EnglishMorphology:
    """English in a dozen lines: no copula suffix, no harmony, no case."""

    question_particles = ()
    interrogatives = ("who", "what")
    copula_suffixes = ()
    plural_suffixes = ("es", "s")
    oblique_suffixes = ()
    openers = ("so", "and")
    pronouns = ("it", "they")

    def is_oblique(self, word):
        return False

    def has_copula(self, word):
        return False

    def strip_copula(self, word):
        return word

    def strip_plural(self, word):
        for suffix in self.plural_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return word[: -len(suffix)]
        return word


def english_grammar():
    # Questions first, exactly as in Turkish: "does tabby catch mice" has the
    # shape of a lesson until the question pattern gets to look at it.
    return Grammar([
        Pattern(["does", KAVRAM, FIIL, KAVRAM], ASK, CAN, 1, 2, object=3),
        Pattern(["does", KAVRAM, FIIL], ASK, CAN, 1, 2),
        Pattern(["what", "is", KAVRAM], ASK, IS_A, 2),
        Pattern([KAVRAM, "is", "a", TUR], TEACH, IS_A, 0, 3),
        Pattern([KAVRAM, FIIL, KAVRAM], TEACH, FROM_VERB, 0, 1, object=2),
        Pattern([KAVRAM, FIIL], TEACH, FROM_VERB, 0, 1),
    ], EnglishMorphology())


def english_lexicon():
    return Lexicon({"catch": ("catching", True), "catches": ("catching", True),
                    "fly": ("flying", True), "flies": ("flying", True)})


class TestAGrammarForALanguageWeNeverWroteFor(unittest.TestCase):
    def setUp(self):
        self.intuition = Intuition(lexicon=english_lexicon(),
                                   grammar=english_grammar())

    def _read(self, sentence):
        intent = self.intuition.understand(sentence)
        return intent.kind, intent.relation, intent.concept, intent.target, \
            intent.object

    def test_the_object_comes_after_the_verb(self):
        self.assertEqual(self._read("cats catch mice"),
                         (TEACH, CAN, "cat", "catching", "mice"))

    def test_a_type_lesson(self):
        self.assertEqual(self._read("tabby is a cat")[:4],
                         (TEACH, IS_A, "tabby", "cat"))

    def test_a_question_is_not_read_as_a_lesson(self):
        """It once learned the concept "does tabby" from this sentence."""
        self.assertEqual(self._read("does tabby catch mice"),
                         (ASK, CAN, "tabby", "catching", "mice"))

    def test_intransitive_sentences_still_work(self):
        self.assertEqual(self._read("birds fly")[:4],
                         (TEACH, CAN, "bird", "flying"))

    def test_asking_what_something_is(self):
        self.assertEqual(self._read("what is tabby")[:3], (ASK, IS_A, "tabby"))


class TestReasoningDoesNotCareWhichLanguage(unittest.TestCase):
    """Inheritance over objects, on facts that arrived through English."""

    def setUp(self):
        self.memory = Memory()
        self.reasoning = Reasoning(self.memory)
        intuition = Intuition(lexicon=english_lexicon(), grammar=english_grammar())
        for sentence in ("cats catch mice", "tabby is a cat"):
            intent = intuition.understand(sentence)
            self.memory.write(Edge(intent.concept, intent.relation, intent.target,
                                   object=intent.object, source="sen"))

    def test_an_object_is_inherited_like_anything_else(self):
        known, chain = self.reasoning.can_do("tabby", "catching", "mice")
        self.assertTrue(known)
        self.assertIn("tabby bir cat", " ".join(chain))

    def test_a_different_object_is_not_assumed(self):
        self.assertIsNone(self.reasoning.can_do("tabby", "catching", "birds")[0])


if __name__ == "__main__":
    unittest.main()
