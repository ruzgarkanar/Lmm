"""What the system cannot learn, checked rather than assumed.

Every sentence below is one a person would reasonably say and this design cannot
hold. The point of the file is that each one fails *loudly* — the only outcome
worse than not learning something is learning it wrong in silence.
"""
import os
import tempfile
import unittest

from lmm.cli import Session

# "kediler fare yakalar" used to be on this list. It is not any more.
BEYOND_US = [
    ("sahiplik", "kuşun kanadı var"),
    ("yer", "penguen kutupta yaşar"),
    ("karşılaştırma", "kartal serçeden büyüktür"),
    ("belirsiz nicelik", "bazı kuşlar uçmaz"),
    ("sayı", "insanın iki gözü var"),
    ("koşul", "yağmur yağarsa ıslanırsın"),
    ("sıra", "kuşlar yumurtadan çıkar sonra uçar"),
]


class TestItFailsLoudlyRatherThanQuietly(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))

    def test_nothing_beyond_the_model_is_written_to_memory(self):
        """A sentence it cannot represent must leave no trace.

        "kartal serçeden büyüktür" once became the concept "kartal serçeden"
        being "büyük" — learned, stored, and never questioned.
        """
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


class TestCaseMarkingIsNotSwallowedByAPhrase(unittest.TestCase):
    """A case-marked word does its own job; it is not half of a compound."""

    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))

    def test_a_comparison_is_refused(self):
        self.session.respond("kartal serçeden büyüktür")
        self.assertEqual(self.session.memory.edges, [])

    def test_real_compounds_still_work(self):
        self.session.respond("müşteri bakiyesi gizlidir")
        self.session.respond("tool broker bir güven kapısıdır")
        self.assertIsNotNone(self.session.memory.direct(
            "müşteri bakiyesi", "property", "gizli"))
        self.assertIsNotNone(self.session.memory.direct(
            "tool broker", "type", "güven kapısı"))

    def test_single_word_sentences_are_untouched(self):
        self.session.respond("kuşlar uçar")
        self.assertIsNotNone(self.session.memory.direct("kuş", "can", "uçmak"))


class TestObjectsAreNoLongerBeyondUs(unittest.TestCase):
    """The fact became four parts instead of three, and four sentences moved
    off the list of things this cannot hold."""

    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))
        self.session.respond("kelime: yakalamak = yakalar / yakalayamaz")
        self.session.respond("kediler fare yakalar")
        self.session.respond("tekir bir kedidir")

    def test_the_object_is_part_of_the_fact(self):
        edge = self.session.memory.direct("kedi", "can", "yakalamak", "fare")
        self.assertIsNotNone(edge)

    def test_it_says_the_object_back(self):
        """Turkish drops the subject in a paragraph; the object stays."""
        self.assertIn("fare yakalar", self.session.respond("kedi anlat"))
        self.assertIn("fare yakalar", self.session.respond("tekir anlat"))

    def test_an_object_is_inherited(self):
        answer = self.session.respond("tekir fare yakalar mı")
        self.assertTrue(answer.startswith("evet"))
        self.assertIn("tekir bir kedi", answer)

    def test_a_different_object_is_never_assumed(self):
        answer = self.session.respond("kedi kuş yakalar mı")
        self.assertNotIn("evet", answer)


if __name__ == "__main__":
    unittest.main()
