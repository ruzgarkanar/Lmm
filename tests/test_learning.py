import unittest

from lmm.memory import Memory, IS_A, CAN, CANNOT
from lmm.reasoning import Reasoning
from lmm.learning import LearningLoop, LEARNED, REINFORCED, CONFLICT, REJECTED
from lmm.intuition import Intent, TEACH


class TestLearningLoop(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.loop = LearningLoop(self.memory, Reasoning(self.memory))

    def test_new_fact_is_learned(self):
        status, message, _ = self.loop.teach(
            Intent(TEACH, concept="penguen", relation=IS_A, target="kuş"))
        self.assertEqual(status, LEARNED)
        self.assertIn("öğrendim", message)
        self.assertIsNotNone(self.memory.direct("penguen", IS_A, "kuş"))

    def test_repetition_reinforces(self):
        intent = Intent(TEACH, concept="kedi", relation=IS_A, target="hayvan")
        self.loop.teach(intent)
        status, _, edge = self.loop.teach(intent)
        self.assertEqual(status, REINFORCED)
        self.assertGreater(edge.confidence, 0.6)

    def test_conflict_is_not_written_and_asks(self):
        self.loop.teach(Intent(TEACH, concept="kuş", relation=CAN, target="uçmak"))
        self.loop.teach(Intent(TEACH, concept="penguen", relation=IS_A, target="kuş"))
        status, message, _ = self.loop.teach(
            Intent(TEACH, concept="penguen", relation=CANNOT, target="uçmak"))
        self.assertEqual(status, CONFLICT)
        self.assertIn("çelişki", message)
        self.assertIsNone(self.memory.direct("penguen", CANNOT, "uçmak"))

    def test_confirmed_exception_is_written(self):
        self.loop.teach(Intent(TEACH, concept="kuş", relation=CAN, target="uçmak"))
        self.loop.teach(Intent(TEACH, concept="penguen", relation=IS_A, target="kuş"))
        _, _, edge = self.loop.teach(
            Intent(TEACH, concept="penguen", relation=CANNOT, target="uçmak"))
        self.loop.confirm_exception(edge)
        written = self.memory.direct("penguen", CANNOT, "uçmak")
        self.assertIsNotNone(written)
        self.assertTrue(written.is_exception)

    def test_cycle_is_rejected(self):
        self.loop.teach(Intent(TEACH, concept="a", relation=IS_A, target="b"))
        status, _, _ = self.loop.teach(
            Intent(TEACH, concept="b", relation=IS_A, target="a"))
        self.assertEqual(status, REJECTED)


if __name__ == "__main__":
    unittest.main()
