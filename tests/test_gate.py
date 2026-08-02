import unittest

from lmm.memory import Memory, Edge, IS_A, NOT_A, CAN, CANNOT
from lmm.reasoning import Reasoning
from lmm.gate import EpistemicGate
from lmm.intuition import Intent, ASK, ASK_HOW_MANY
from lmm.relations import ALL, SOME, NO


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

    İlk düzeltme yarım kaldı: "aynı hiyerarşide ayrı dal" da kanıt sayılmıştı
    ve bu, hiyerarşinin bir bölümleme olduğunu varsayar. Değil — "memeli bir
    hayvandır" ve "omurgalı bir hayvandır" örtüşen iki kategoridir, kardeş
    olmaları ayrı olmalarını göstermez.

    Artık olumsuz yalnız kayıtlı bir retten gelir: doğrudan ya da iki taraftan
    da kalıtılarak. Kayıt yoksa cevap bilmemektir.
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

    def test_a_separate_branch_under_a_shared_root_is_not_evidence(self):
        """Kardeş kategoriler örtüşebilir; ayrı dal ayrılığın kanıtı değildir.

        Ölçülen çürütme: memeli ve omurgalı ikisi de hayvanın altında, insan
        yalnız memelinin altında. Kapı "insan bir omurgalı değildir" diyordu.
        Doğru cevap bilmemek — insan gerçekten de bir omurgalı.
        """
        gate = self.gate([("hayvan", "canlı"), ("memeli", "hayvan"),
                          ("omurgalı", "hayvan"), ("insan", "memeli")])
        answer = self.ask(gate, "insan", "omurgalı")
        self.assertNotIn("değildir", answer)
        self.assertIn("bilmiyorum", answer)

    def test_a_denial_is_inherited_from_the_other_side_too(self):
        """"kuş bir memeli değildir" kaydı iki tarafa da iner.

        Kartal bir kuş, fare bir memeli; ret ikisinin arasındaki soruyu da
        cevaplar. Eskiden yalnız sorulan kavramın kendi tarafına bakılıyordu
        ve bu kanıt kullanılmadan duruyordu.
        """
        gate = self.gate([("kartal", "kuş"), ("fare", "memeli")],
                         denials=[("kuş", "memeli")])
        answer = self.ask(gate, "kartal", "fare")
        self.assertTrue(answer.startswith("hayır"))
        self.assertIn("kartal bir kuş", answer)     # gerekçe iki adımı da say
        self.assertIn("fare bir memeli", answer)

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


class TestTheQuantifierIsRead(unittest.TestCase):
    """Sorulan nicelik cevabı değiştirir; kapı onu okumuyordu.

    Ölçülen hali: "bazı kuşlar uçar mı", "her kuş uçar mı" ve "hiçbir kuş
    uçar mı" üçü de "evet, kartal ve serçe uçar" cevabını alıyordu — yani
    "hiçbiri mi?" sorusuna "evet" deniyordu ve penguen dururken "her" sorusu
    çürütülmüyordu.

    Aynı kanıt üç soruya üç ayrı cevap verir: tek örnek "bazı"yı kanıtlar,
    tek karşı örnek "her"i çürütür, tek örnek "hiçbiri"ni çürütür.
    """

    def gate(self, members, rule=None, exceptions=(), flying=()):
        memory = Memory()
        for member in members:
            memory.write(Edge(member, IS_A, "kuş", source="test"))
        if rule is not None:
            memory.write(Edge("kuş", CAN if rule else CANNOT, "uçmak",
                              source="test"))
        for member in flying:
            memory.write(Edge(member, CAN, "uçmak", source="test"))
        for member in exceptions:
            memory.write(Edge(member, CANNOT, "uçmak", source="test"))
        return EpistemicGate(memory, Reasoning(memory))

    def ask(self, gate, quantifier):
        return gate.answer(Intent(ASK_HOW_MANY, "kuş", CAN, "uçmak",
                                  quantifier=quantifier))

    def test_some_is_proved_by_one_member(self):
        gate = self.gate(["kartal", "serçe", "penguen"], rule=True,
                         exceptions=["penguen"])
        answer = self.ask(gate, SOME)
        self.assertTrue(answer.startswith("evet"))
        self.assertIn("kartal", answer)

    def test_every_is_refuted_by_one_exception(self):
        gate = self.gate(["kartal", "serçe", "penguen"], rule=True,
                         exceptions=["penguen"])
        answer = self.ask(gate, ALL)
        self.assertTrue(answer.startswith("hayır"))
        self.assertIn("penguen", answer)

    def test_none_is_refuted_by_one_member(self):
        gate = self.gate(["kartal", "serçe", "penguen"], rule=True,
                         exceptions=["penguen"])
        answer = self.ask(gate, NO)
        self.assertTrue(answer.startswith("hayır"))
        self.assertIn("kartal", answer)

    def test_every_rests_on_the_rule_not_on_the_members(self):
        """Bildiğim üyelerin hepsi uçuyor olması "hepsi uçar" demek değildir.

        Üye kümesi kapalı değil: yarın penguen öğrenilebilir. Eskiden iki üye
        görüp "evet, hepsi" deniyordu; evrensel cevabı ancak türün kendi
        kaydı verebilir.
        """
        gate = self.gate(["kartal", "serçe"], flying=["kartal", "serçe"])
        answer = self.ask(gate, ALL)
        self.assertNotIn("hepsi —", answer)
        self.assertFalse(answer.startswith("evet"))
        self.assertIn("kartal", answer)         # bildiğini yine de söylüyor

    def test_a_rule_carries_the_universal(self):
        gate = self.gate(["kartal", "serçe"], rule=True)
        self.assertTrue(self.ask(gate, ALL).startswith("evet"))
