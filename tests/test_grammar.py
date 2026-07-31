"""Grammar as data: patterns that can be listed, matched, and learned.

Patterns used to be a chain of ifs in the parser, which meant the system could
learn any fact but not one new way of saying one. These check that the data-driven
grammar behaves exactly as the hand-written chain did, and that a pattern can now
be worked out from a single example.
"""
import unittest

from lmm.grammar import (Grammar, Pattern, learn_pattern, KAVRAM, TUR, NITELIK,
                         SOZ, FIIL, SORU, FROM_VERB)
from lmm.turkish import turkish, TurkishMorphology, TEACH, ASK, ASK_DESCRIBE
from lmm.relations import IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY
from lmm.lexicon import Lexicon
from lmm.intuition import Intuition


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.grammar = turkish()
        self.lexicon = Lexicon()

    def _read(self, sentence):
        tokens = sentence.split()
        pattern, captured = self.grammar.match(tokens, self.lexicon)
        self.assertIsNotNone(pattern, sentence)
        return self.grammar.read(pattern, captured)

    def test_a_type_lesson(self):
        self.assertEqual(self._read("penguen bir kuştur"),
                         (TEACH, IS_A, "penguen", "kuş"))

    def test_a_property_lesson(self):
        self.assertEqual(self._read("kuşlar tüylüdür"),
                         (TEACH, HAS_PROPERTY, "kuş", "tüylü"))

    def test_polarity_comes_from_the_verb(self):
        self.assertEqual(self._read("penguen uçamaz"),
                         (TEACH, CANNOT, "penguen", "uçmak"))
        self.assertEqual(self._read("kuşlar uçar"),
                         (TEACH, CAN, "kuş", "uçmak"))

    def test_a_denial(self):
        self.assertEqual(self._read("penguen bir memeli değildir"),
                         (TEACH, NOT_A, "penguen", "memeli"))

    def test_order_decides_between_two_shapes(self):
        """"kim uçar" and "kuşlar uçar" have the same shape."""
        kind, _, _, _ = self._read("kim uçar")
        self.assertEqual(kind, "ASK_WHO")

    def test_a_sentence_with_no_pattern_does_not_match(self):
        pattern, _ = self.grammar.match("bu cümle hiçbir kalıba uymaz".split(),
                                        self.lexicon)
        self.assertIsNone(pattern)


class TestSlots(unittest.TestCase):
    def setUp(self):
        self.grammar = Grammar([], TurkishMorphology())
        self.lexicon = Lexicon()

    def _capture(self, slot, token):
        return self.grammar._capture(slot, token, self.lexicon)

    def test_a_property_slot_demands_a_copula(self):
        self.assertEqual(self._capture(NITELIK, "beyazdır"), "beyaz")
        self.assertIsNone(self._capture(NITELIK, "beyaz"))

    def test_a_type_slot_accepts_either(self):
        self.assertEqual(self._capture(TUR, "kuştur"), "kuş")
        self.assertEqual(self._capture(TUR, "kuş"), "kuş")

    def test_a_concept_slot_drops_the_plural(self):
        self.assertEqual(self._capture(KAVRAM, "kuşlar"), "kuş")

    def test_a_verb_slot_reads_the_polarity(self):
        self.assertEqual(self._capture(FIIL, "uçamaz"), ("uçmak", False))
        self.assertIsNone(self._capture(FIIL, "zıplar"))

    def test_a_literal_must_match_exactly(self):
        self.assertEqual(self._capture("bir", "bir"), "bir")
        self.assertIsNone(self._capture("bir", "iki"))


class TestLearningAPattern(unittest.TestCase):
    """The point of the refactor: a new way of saying something can be taught."""

    def setUp(self):
        self.lexicon = Lexicon()
        self.morphology = TurkishMorphology()

    def test_a_pattern_is_worked_out_from_one_example(self):
        pattern = learn_pattern("penguen hakkında konuş".split(), ASK_DESCRIBE,
                                None, 0, None, self.lexicon, self.morphology)
        self.assertEqual(pattern.tokens, [KAVRAM, "hakkında", "konuş"])

    def test_the_learned_pattern_then_matches_new_sentences(self):
        grammar = turkish()
        grammar.add(learn_pattern("penguen hakkında konuş".split(), ASK_DESCRIBE,
                                  None, 0, None, self.lexicon, self.morphology),
                    first=True)
        intuition = Intuition(lexicon=self.lexicon, grammar=grammar)
        intent = intuition.understand("Kartal hakkında konuş")
        self.assertEqual(intent.kind, ASK_DESCRIBE)
        self.assertEqual(intent.concept, "kartal")

    def test_a_verb_in_the_example_becomes_a_verb_slot(self):
        pattern = learn_pattern("acaba kuşlar uçar mı".split(), ASK, CAN, 1, 2,
                                self.lexicon, self.morphology)
        self.assertEqual(pattern.tokens, ["acaba", KAVRAM, FIIL, SORU])


if __name__ == "__main__":
    unittest.main()
