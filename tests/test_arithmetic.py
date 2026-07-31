"""Arithmetic: computed, never recalled, never guessed.

A language model predicts the tokens of an answer, which is why it can be
confidently wrong about 17 × 43. A memory cannot help either — there are
infinitely many sums. So this is a third source next to what was taught and what
was inferred, and being deterministic it cannot be wrong.
"""
import os
import tempfile
import unittest

from lmm.arithmetic import (evaluate, normalise, looks_like_a_sum, tokenize,
                            Undecidable)
from lmm.cli import Session


class TestComputing(unittest.TestCase):
    def test_the_answers_are_exact(self):
        self.assertEqual(evaluate("17 çarpı 43"), 731)
        self.assertEqual(evaluate("12 artı 8 eksi 5"), 15)
        self.assertEqual(evaluate("2 üzeri 10"), 1024)

    def test_precedence_is_respected(self):
        self.assertEqual(evaluate("2 artı 3 çarpı 4"), 14)
        self.assertEqual(evaluate("2 çarpı 3 artı 4"), 10)

    def test_symbols_work_as_well_as_words(self):
        self.assertEqual(evaluate("17 * 43"), 731)
        self.assertEqual(evaluate("100 / 4"), 25)

    def test_whole_answers_stay_whole(self):
        self.assertIsInstance(evaluate("100 bölü 4"), int)
        self.assertNotIsInstance(evaluate("100 bölü 7"), int)

    def test_the_question_words_are_dropped_from_the_echo(self):
        self.assertEqual(normalise("17 çarpı 43 kaç eder"), "17 * 43")


class TestRefusing(unittest.TestCase):
    def test_division_by_zero_is_refused(self):
        with self.assertRaises(Undecidable):
            evaluate("5 bölü 0")

    def test_a_dangling_operator_is_refused(self):
        with self.assertRaises(Undecidable):
            evaluate("5 artı")

    def test_two_operators_in_a_row_are_refused(self):
        with self.assertRaises(Undecidable):
            evaluate("5 artı çarpı 3")

    def test_nothing_but_numbers_and_operators_is_accepted(self):
        """No text is ever evaluated — the words alone decide."""
        with self.assertRaises(Undecidable):
            tokenize("5 artı penguen")
        self.assertFalse(looks_like_a_sum("penguen bir kuştur"))
        self.assertFalse(looks_like_a_sum("__import__('os')"))


class TestInConversation(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))

    def test_a_sum_is_answered_with_its_working(self):
        answer = self.session.respond("17 çarpı 43 kaç")
        self.assertIn("731", answer)
        self.assertIn("hesapladım", answer)

    def test_a_result_is_not_stored_as_a_fact(self):
        """There are infinitely many sums; none of them belong in memory."""
        self.session.respond("17 çarpı 43 kaç")
        self.assertEqual(self.session.memory.edges, [])

    def test_sentences_are_untouched_by_it(self):
        self.assertIn("öğrendim", self.session.respond("penguen bir kuştur"))
        self.assertIn("kuş", self.session.respond("penguen nedir"))

    def test_what_it_cannot_settle_it_declines(self):
        self.assertIn("hesaplayamadım", self.session.respond("5 bölü 0"))


if __name__ == "__main__":
    unittest.main()
