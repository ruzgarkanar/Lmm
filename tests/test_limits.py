"""What the system cannot learn, checked rather than assumed.

Every sentence in BEYOND_US is one a person would reasonably say and this design
cannot hold. The point of the file is that each fails *loudly* — the only outcome
worse than not learning something is learning it wrong in silence.

The list gets shorter as the fact grows. It began with eight entries; objects
took one, and case roles took two more.
"""
import os
import tempfile
import unittest

from lmm.relations import CAN, HAS_PROPERTY, OBJECT, PLACE, SOURCE
from lmm.cli import Session

BEYOND_US = [
    ("sahiplik", "kuşun kanadı var"),
    ("belirsiz nicelik", "bazı kuşlar uçmaz"),
    ("sayı", "insanın iki gözü var"),
    ("koşul", "yağmur yağarsa ıslanırsın"),
    ("sıra", "kuşlar yumurtadan çıkar sonra uçar"),
]


class TestItFailsLoudlyRatherThanQuietly(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))

    def test_nothing_beyond_the_model_is_written_to_memory(self):
        for name, sentence in BEYOND_US:
            with self.subTest(name):
                before = len(self.session.memory.edges)
                answer = self.session.respond(sentence)
                self.assertEqual(len(self.session.memory.edges), before, answer)

    def test_it_says_so_every_time(self):
        for name, sentence in BEYOND_US:
            with self.subTest(name):
                answer = self.session.respond(sentence)
                self.assertTrue("anlamadım" in answer or "bilmiyorum" in answer,
                                f"{sentence} -> {answer}")


class TestASecondConceptAndWhatItIsDoing(unittest.TestCase):
    """A case ending says what a second concept is doing, so the role is read
    off the word rather than guessed from where it sits."""

    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))
        self.session.respond("kelime: yaşamak = yaşar / yaşayamaz")
        self.session.respond("kelime: yakalamak = yakalar / yakalayamaz")
        for lesson in ("kediler fare yakalar", "penguen kutupta yaşar",
                       "kartal serçeden büyüktür", "penguen bir kuştur"):
            self.session.respond(lesson)

    def test_a_bare_second_concept_is_an_object(self):
        self.assertIsNotNone(self.session.memory.direct(
            "kedi", CAN, "yakalamak", "fare", OBJECT))

    def test_a_locative_marks_a_place(self):
        self.assertIsNotNone(self.session.memory.direct(
            "penguen", CAN, "yaşamak", "kutup", PLACE))

    def test_an_ablative_marks_a_comparison(self):
        self.assertIsNotNone(self.session.memory.direct(
            "kartal", HAS_PROPERTY, "büyük", "serçe", SOURCE))

    def test_the_ending_comes_back_when_it_speaks(self):
        self.assertIn("kutupta yaşar", self.session.respond("penguen anlat"))

    def test_it_answers_about_the_place_it_was_told(self):
        self.assertTrue(
            self.session.respond("penguen kutupta yaşar mı").startswith("evet"))

    def test_a_different_place_is_never_assumed(self):
        self.assertNotIn("evet", self.session.respond("penguen ormanda yaşar mı"))

    def test_it_answers_a_comparison(self):
        self.assertTrue(
            self.session.respond("kartal serçeden büyük mü").startswith("evet"))

    def test_a_role_is_inherited_with_everything_else(self):
        self.session.respond("imparator bir penguendir")
        answer = self.session.respond("imparator kutupta yaşar mı")
        self.assertTrue(answer.startswith("evet"))
        self.assertIn("imparator bir penguen", answer)


class TestPhrasesStillHoldTogether(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))

    def test_real_compounds_survive(self):
        self.session.respond("müşteri bakiyesi gizlidir")
        self.session.respond("tool broker bir güven kapısıdır")
        self.assertIsNotNone(self.session.memory.direct(
            "müşteri bakiyesi", HAS_PROPERTY, "gizli"))
        self.assertIsNotNone(self.session.memory.direct(
            "tool broker", "type", "güven kapısı"))

    def test_single_word_sentences_are_untouched(self):
        self.session.respond("kuşlar uçar")
        self.assertIsNotNone(self.session.memory.direct("kuş", CAN, "uçmak"))


if __name__ == "__main__":
    unittest.main()
