import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT
from lmm.reasoning import Reasoning
from lmm.gate import EpistemicGate
from lmm.intuition import Intent, ASK


class TestEpistemicGate(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.gate = EpistemicGate(self.memory, Reasoning(self.memory))

    def test_miss_says_dont_know(self):
        answer = self.gate.answer(Intent(ASK, concept="ejderha", relation=IS_A))
        self.assertIn("bilmiyorum", answer)
        self.assertIn("öğretir misin", answer)

    def test_ability_miss_says_dont_know(self):
        answer = self.gate.answer(Intent(ASK, concept="kedi", relation=CAN,
                                         target="uçmak"))
        self.assertIn("bilmiyorum", answer)

    def test_definition_answer_cites_source(self):
        self.memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        answer = self.gate.answer(Intent(ASK, concept="penguen", relation=IS_A))
        self.assertIn("kuş", answer)
        self.assertIn("sen", answer)

    def test_ability_answer_shows_chain(self):
        self.memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        self.memory.write(Edge("serçe", IS_A, "kuş", source="sen"))
        answer = self.gate.answer(Intent(ASK, concept="serçe", relation=CAN,
                                         target="uçmak"))
        self.assertTrue(answer.startswith("evet"))
        self.assertIn("çünkü", answer)

    def test_negative_answer(self):
        self.memory.write(Edge("penguen", CANNOT, "uçmak", source="sen"))
        answer = self.gate.answer(Intent(ASK, concept="penguen", relation=CAN,
                                         target="uçmak"))
        self.assertTrue(answer.startswith("hayır"))

    def test_low_confidence_hedges(self):
        self.memory.write(Edge("yarasa", IS_A, "kuş", source="sen", confidence=0.3))
        answer = self.gate.answer(Intent(ASK, concept="yarasa", relation=IS_A))
        self.assertIn("emin değilim", answer)


if __name__ == "__main__":
    unittest.main()
