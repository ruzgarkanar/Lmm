"""Nearness used to suggest, never to answer — the place an embedding would go.

An embedding decides two words are alike in a space nobody can inspect; when it
is wrong you get a fluent wrong answer and the only repair is retraining. Here a
guess can only ever become a question, so it cannot become a belief.
"""
import os
import tempfile
import unittest

from lmm.similarity import closeness, distance, nearest
from lmm.cli import Session


class TestCloseness(unittest.TestCase):
    def test_a_typo_is_close(self):
        self.assertGreater(closeness("kus", "kuş"), 0.8)
        self.assertGreater(closeness("pengueen", "penguen"), 0.8)

    def test_unrelated_words_are_not(self):
        self.assertLess(closeness("kar", "penguen"), 0.45)

    def test_synonyms_are_out_of_reach_and_that_is_the_point(self):
        """No string comparison connects these; a taught fact does."""
        self.assertLess(closeness("araba", "otomobil"), 0.45)

    def test_distance_gives_up_past_the_cap(self):
        self.assertEqual(distance("kuş", "kus"), 1)
        self.assertGreater(distance("kuş", "otomobil"), 2)


class TestNearest(unittest.TestCase):
    def test_it_ranks_the_likeliest_first(self):
        self.assertEqual(nearest("kus", ["penguen", "kuş", "kar"]), ["kuş"])

    def test_it_never_suggests_the_word_itself(self):
        self.assertEqual(nearest("kuş", ["kuş"]), [])

    def test_nothing_close_enough_means_no_suggestion(self):
        self.assertEqual(nearest("zürafa", ["penguen", "kar"]), [])


class TestInConversation(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))
        self.session.respond("kuşlar uçar")
        self.session.respond("penguen bir kuştur")
        self.session.respond("kar beyazdır")

    def test_a_typo_gets_a_question_not_an_answer(self):
        answer = self.session.respond("kus nedir")
        self.assertIn("bilmiyorum", answer)          # still does not know
        self.assertIn("kuş mu demek istedin", answer)

    def test_the_particle_harmonises_like_every_other_suffix(self):
        self.assertIn("kar mı demek istedin", self.session.respond("karr nedir"))
        self.assertIn("penguen mi demek istedin",
                      self.session.respond("pengueen nedir"))

    def test_an_unrelated_word_gets_no_guess(self):
        answer = self.session.respond("zürafa nedir")
        self.assertIn("bilmiyorum", answer)
        self.assertNotIn("demek istedin", answer)

    def test_a_suggestion_never_becomes_a_belief(self):
        """The whole rule: it asked, it did not decide."""
        self.session.respond("kus nedir")
        self.assertEqual(self.session.memory.query("kus"), [])
        self.assertIn("bilmiyorum", self.session.respond("kus uçar mı"))


if __name__ == "__main__":
    unittest.main()
