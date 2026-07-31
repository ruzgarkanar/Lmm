"""How much of a kind a claim covers — qualitative, never numeric.

Cyc gave every assertion a number from 0 to 100 and abandoned it: people wrote
95 for one claim and 94 for another with no statistics behind either. Four words
carry what is actually needed, and the important one is that "bazı" does not
inherit. An existence claim that spread to every member would be a universal
wearing a disguise.
"""
import os
import tempfile
import unittest

from lmm.relations import ALL, MOST, SOME, NO, CAN, CANNOT
from lmm.memory import Memory, Edge, IS_A
from lmm.reasoning import Reasoning
from lmm.intuition import Intuition, TEACH, ASK_HOW_MANY
from lmm.cli import Session


class TestReadingQuantifiers(unittest.TestCase):
    def setUp(self):
        self.intuition = Intuition()

    def _read(self, sentence):
        intent = self.intuition.understand(sentence)
        return intent.kind, intent.concept, intent.relation, intent.quantifier

    def test_every_marker_is_recognised(self):
        self.assertEqual(self._read("bütün kuşlar uçar"),
                         (TEACH, "kuş", CAN, ALL))
        self.assertEqual(self._read("çoğu kuş uçar"), (TEACH, "kuş", CAN, MOST))
        self.assertEqual(self._read("bazı kuşlar uçmaz"),
                         (TEACH, "kuş", CANNOT, SOME))
        self.assertEqual(self._read("hiçbir kuş yüzmez"),
                         (TEACH, "kuş", CANNOT, NO))

    def test_an_unmarked_sentence_means_all(self):
        self.assertEqual(self._read("kuşlar uçar")[3], ALL)

    def test_a_quantifier_is_never_the_subject(self):
        """"bazı kuşlar uçmaz" was once read as the concept "bazı"."""
        self.assertEqual(self._read("bazı kuşlar uçmaz")[1], "kuş")

    def test_the_question_form_is_recognised(self):
        self.assertEqual(self._read("bazı kuşlar uçar mı")[0], ASK_HOW_MANY)

    def test_turkish_says_no_two_ways(self):
        """"uçmaz" and "uçamaz" both deny flight."""
        self.assertEqual(self._read("kuşlar uçmaz")[2], CANNOT)
        self.assertEqual(self._read("kuşlar uçamaz")[2], CANNOT)


class TestSomeDoesNotInherit(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))

    def test_an_existence_claim_says_nothing_about_a_member(self):
        self.session.respond("bazı kuşlar uçmaz")
        self.session.respond("penguen bir kuştur")
        self.assertIn("bilmiyorum", self.session.respond("penguen uçar mı"))

    def test_a_universal_still_inherits(self):
        self.session.respond("bütün kuşlar tüylüdür")
        self.session.respond("penguen bir kuştur")
        self.assertTrue(self.session.respond("penguen tüylü mü").startswith("evet"))

    def test_none_inherits_as_a_denial(self):
        self.session.respond("hiçbir kuş yüzmez")
        self.session.respond("penguen bir kuştur")
        self.assertTrue(self.session.respond("penguen yüzer mi").startswith("hayır"))


class TestSurveyingAKind(unittest.TestCase):
    """"Bazı kuşlar uçmaz" is answered by reading the members, not by storing it."""

    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))
        self.session.respond("kuşlar uçar")
        for bird in ("penguen", "devekuşu", "serçe"):
            self.session.respond(f"{bird} bir kuştur")
        for flightless in ("penguen", "devekuşu"):
            self.session.respond(f"{flightless} uçamaz")
            self.session.respond("evet")

    def test_some_do_and_some_do_not(self):
        answer = self.session.respond("bazı kuşlar uçar mı")
        self.assertIn("serçe uçar", answer)
        self.assertIn("penguen", answer)
        self.assertIn("devekuşu", answer)

    def test_the_negative_question_reads_the_exceptions(self):
        answer = self.session.respond("bazı kuşlar uçmaz mı")
        self.assertTrue(answer.startswith("evet"))
        self.assertIn("penguen", answer)

    def test_nothing_was_stored_to_answer_it(self):
        before = len(self.session.memory.edges)
        self.session.respond("bazı kuşlar uçar mı")
        self.assertEqual(len(self.session.memory.edges), before)


class TestPersistence(unittest.TestCase):
    def test_the_quantifier_survives_a_restart(self):
        path = os.path.join(tempfile.mkdtemp(), "memory.lmm")
        memory = Memory()
        memory.write(Edge("kuş", CANNOT, "uçmak", source="sen", quantifier=SOME))
        memory.save(path)
        self.assertEqual(Memory.load(path).edges[0].quantifier, SOME)


if __name__ == "__main__":
    unittest.main()
