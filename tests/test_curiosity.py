"""Curiosity: the system noticing its own gaps and asking, unprompted."""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN
from lmm.reasoning import Reasoning
from lmm.curiosity import Curiosity
from lmm.cli import Session


class TestCuriosity(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.curiosity = Curiosity(self.memory, Reasoning(self.memory))

    def test_empty_memory_wonders_about_nothing(self):
        """No knowledge means no gaps — curiosity is grounded, not fabricated."""
        self.assertIsNone(self.curiosity.next_question())

    def test_asks_what_an_unplaced_concept_is(self):
        self.memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        question = self.curiosity.next_question()
        self.assertEqual(question.text, "kuş nedir?")

    def test_asks_whether_a_concept_can_do_a_known_action(self):
        self.memory.write(Edge("kuş", IS_A, "hayvan", source="sen"))
        self.memory.write(Edge("balık", IS_A, "hayvan", source="sen"))
        self.memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        texts = self._drain()
        self.assertIn("balık uçar mı?", texts)

    def test_question_particle_follows_vowel_harmony(self):
        self.memory.write(Edge("çocuk", IS_A, "canlı", source="sen"))
        self.memory.write(Edge("öğretmen", IS_A, "canlı", source="sen"))
        self.memory.write(Edge("çocuk", CAN, "okumak", source="sen"))
        self.assertIn("öğretmen okur mu?", self._drain())

    def test_never_asks_the_same_question_twice(self):
        self.memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        first = self.curiosity.next_question()
        remaining = self._drain()
        self.assertNotIn(first.text, remaining)

    def test_answered_gap_stops_being_a_gap(self):
        self.memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        self.assertEqual(self.curiosity.next_question().text, "kuş nedir?")
        self.memory.write(Edge("kuş", IS_A, "hayvan", source="sen"))
        fresh = Curiosity(self.memory, Reasoning(self.memory))
        fresh.memory.asked.clear()          # even with a clean slate of questions
        self.assertNotEqual(fresh.next_question().text, "kuş nedir?")

    def _drain(self, limit=40):
        texts = []
        for _ in range(limit):
            question = self.curiosity.next_question()
            if question is None:
                break
            texts.append(question.text)
        return texts


class TestCuriosityInConversation(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "memory.json")

    def test_wonders_aloud_after_learning(self):
        session = Session(self.path)
        reply = session.respond("kuşlar uçar")
        self.assertIn("öğrendim", reply)
        self.assertIn("kuş nedir?", reply)      # unprompted, from its own gap

    def test_teacher_can_answer_the_question_normally(self):
        session = Session(self.path)
        session.respond("kuşlar uçar")          # asks "kuş nedir?"
        reply = session.respond("kuş bir hayvandır")
        self.assertIn("öğrendim: kuş bir hayvandır", reply)
        self.assertIn("hayvan", session.respond("kuş nedir"))

    def test_questions_are_not_repeated_after_a_restart(self):
        """The full loop: notice, ask, learn, and never ask that again."""
        first = Session(self.path)
        asked = first.respond("kuşlar uçar")
        self.assertIn("kuş nedir?", asked)
        first.save()

        second = Session(self.path)
        later = second.respond("penguen bir kuştur")
        self.assertNotIn("kuş nedir?", later)


if __name__ == "__main__":
    unittest.main()
