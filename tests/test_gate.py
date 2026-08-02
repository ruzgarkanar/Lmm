import unittest

from lmm.memory import Memory, Edge, IS_A, NOT_A, CAN, CANNOT
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


class TestNegativeNeedsEvidence(unittest.TestCase):
    """Bilmemek ile olmadığını bilmek aynı şey değildir.

    Kapı, kavramın HERHANGİ bir tür bağı varsa aranan bağın yokluğunu olumsuz
    kanıtı sayıyordu. Sürekli okuyarak büyüyen bir grafta bu temelsiz ve
    canlı bir örnekte yakalandı: ansiklopediden "vaşak bir hayvan türüdür"
    okunduktan hemen sonra sistem "vaşak bir hayvan değildir" diyordu.

    Artık olumsuz üç kaynaktan biriyle söylenir: doğrudan ret, kalıtılan ret,
    ya da aynı hiyerarşide ayrı dalda olmak. Hiçbiri yoksa cevap bilmemektir.
    """

    def gate(self, facts, denials=()):
        memory = Memory()
        for concept, target in facts:
            memory.write(Edge(concept, IS_A, target, source="test"))
        for concept, target in denials:
            memory.write(Edge(concept, NOT_A, target, source="test"))
        return EpistemicGate(memory, Reasoning(memory))

    def ask(self, gate, concept, target):
        return gate.answer(Intent(ASK, concept, IS_A, target))

    def test_absence_of_a_path_is_not_a_denial(self):
        """Tek bir tür bağı, başka her şeyi olumsuzlamaya yetmez."""
        gate = self.gate([("vaşak", "hayvan türü")])
        self.assertIn("bilmiyorum", self.ask(gate, "vaşak", "hayvan"))

    def test_a_separate_branch_under_a_shared_root_is_evidence(self):
        """penguen ve memeli ikisi de hayvanın altında ve biri diğerinin atası
        değil — bu gerçek bir ayrılık kanıtı."""
        gate = self.gate([("penguen", "kuş"), ("kuş", "hayvan"),
                          ("memeli", "hayvan")])
        self.assertIn("değildir", self.ask(gate, "penguen", "memeli"))

    def test_a_denial_is_inherited(self):
        """"organ bir canlı değildir" + "kalp bir organdır" ⇒ kalp canlı değil."""
        gate = self.gate([("kalp", "organ")], denials=[("organ", "canlı")])
        answer = self.ask(gate, "kalp", "canlı")
        self.assertTrue(answer.startswith("hayır"))
        self.assertIn("çünkü", answer)      # gerekçesini göstermeli

    def test_a_direct_denial_still_answers_no(self):
        gate = self.gate([("kalp", "organ")], denials=[("kalp", "canlı")])
        self.assertTrue(self.ask(gate, "kalp", "canlı").startswith("hayır"))

    def test_what_is_known_is_still_answered_yes(self):
        gate = self.gate([("penguen", "kuş"), ("kuş", "hayvan")])
        self.assertTrue(self.ask(gate, "penguen", "hayvan").startswith("evet"))
