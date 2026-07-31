"""The thread of a conversation — which is not a context window.

An LLM re-reads the whole conversation every turn and forgets whatever falls out
of a fixed budget. Here knowledge is already permanent, so nothing said can be
forgotten. What was missing was smaller and different: knowing what "peki yüzer
mi" is about.
"""
import os
import tempfile
import unittest

from lmm.intuition import Intuition, ASK, ASK_DESCRIBE, ASK_PROPERTIES
from lmm.relations import CAN, IS_A
from lmm.cli import Session


class TestElidedSentences(unittest.TestCase):
    def setUp(self):
        self.intuition = Intuition()

    def test_an_opener_is_not_part_of_the_question(self):
        intent = self.intuition.understand("peki uçar mı")
        self.assertEqual((intent.kind, intent.relation, intent.target),
                         (ASK, CAN, "uçmak"))
        self.assertIsNone(intent.concept)

    def test_bare_questions_leave_the_subject_out(self):
        self.assertEqual(self.intuition.understand("nedir").relation, IS_A)
        self.assertEqual(self.intuition.understand("anlat").kind, ASK_DESCRIBE)
        self.assertEqual(self.intuition.understand("nasıldır").kind,
                         ASK_PROPERTIES)

    def test_a_pronoun_stands_in_for_the_subject(self):
        self.assertEqual(self.intuition.understand("o uçar mı").concept, "o")


class TestConversationThread(unittest.TestCase):
    def setUp(self):
        path = os.path.join(tempfile.mkdtemp(), "memory.json")
        self.session = Session(path)
        for lesson in ["kuşlar uçar", "penguen bir kuştur", "penguen yüzer",
                       "kartal bir kuştur"]:
            self.session.respond(lesson)

    def test_a_bare_question_continues_about_the_same_thing(self):
        self.session.respond("penguen nedir")
        self.assertTrue(self.session.respond("peki yüzer mi").startswith("evet"))

    def test_a_pronoun_resolves_to_the_same_thing(self):
        self.session.respond("penguen nedir")
        self.assertIn("Penguen", self.session.respond("onu anlat"))

    def test_the_thread_moves_when_the_subject_does(self):
        self.session.respond("penguen nedir")
        self.session.respond("kartal nedir")
        self.assertIn("kartal", self.session.respond("anlat").lower())

    def test_teaching_also_sets_what_we_are_talking_about(self):
        self.session.respond("serçe bir kuştur")
        self.assertIn("Serçe", self.session.respond("anlat"))

    def test_without_a_thread_it_does_not_guess(self):
        fresh = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))
        self.assertIn("anlamadım", fresh.respond("peki uçar mı"))


class TestNoContextWindow(unittest.TestCase):
    def test_what_was_taught_first_is_still_known_after_a_long_conversation(self):
        """The property an LLM cannot have: the beginning never falls out."""
        path = os.path.join(tempfile.mkdtemp(), "memory.json")
        session = Session(path)
        session.respond("kuşlar uçar")
        session.respond("penguen bir kuştur")
        for index in range(300):            # a long way past any token budget
            session.respond(f"nesne{index} bir eşyadır")
        self.assertTrue(session.respond("penguen uçar mı").startswith("evet"))
        self.assertIn("kuş", session.respond("penguen nedir"))


if __name__ == "__main__":
    unittest.main()
