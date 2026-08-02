import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT, HAS_PROPERTY
from lmm.reasoning import Reasoning
from lmm.trust import distilled_source


class TestInheritance(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.reasoning = Reasoning(self.memory)
        self.memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        self.memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        self.memory.write(Edge("serçe", IS_A, "kuş", source="sen"))

    def test_direct_knowledge(self):
        answer, chain = self.reasoning.can_do("kuş", "uçmak")
        self.assertTrue(answer)
        self.assertTrue(any("doğrudan" in step for step in chain))

    def test_inheritance(self):
        answer, chain = self.reasoning.can_do("serçe", "uçmak")
        self.assertTrue(answer)
        self.assertIn("serçe bir kuş", " ".join(chain))

    def test_exception_overrides_inheritance(self):
        self.memory.write(Edge("penguen", CANNOT, "uçmak", source="sen",
                               is_exception=True))
        answer, _ = self.reasoning.can_do("penguen", "uçmak")
        self.assertFalse(answer)                      # direct fact beat the parent
        sibling, _ = self.reasoning.can_do("serçe", "uçmak")
        self.assertTrue(sibling)                      # sibling untouched

    def test_unknown_returns_none(self):
        answer, chain = self.reasoning.can_do("penguen", "konuşmak")
        self.assertIsNone(answer)
        self.assertEqual(chain, [])

    def test_ancestor_chain(self):
        self.memory.write(Edge("kuş", IS_A, "hayvan"))
        self.assertEqual(self.reasoning.ancestors("penguen"), ["kuş", "hayvan"])


class TestConflict(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.reasoning = Reasoning(self.memory)
        self.memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        self.memory.write(Edge("penguen", IS_A, "kuş", source="sen"))

    def test_conflict_with_inherited_knowledge(self):
        candidate = Edge("penguen", CANNOT, "uçmak", source="sen")
        conflict = self.reasoning.find_conflict(candidate)
        self.assertIsNotNone(conflict)
        self.assertIn("kuş", conflict)   # explanation names the clashing chain

    def test_compatible_knowledge_is_not_a_conflict(self):
        candidate = Edge("penguen", CAN, "yüzmek", source="sen")
        self.assertIsNone(self.reasoning.find_conflict(candidate))

    def test_type_edges_skip_conflict_check(self):
        candidate = Edge("penguen", IS_A, "hayvan", source="sen")
        self.assertIsNone(self.reasoning.find_conflict(candidate))


def diamond(first, second, mammal_source="sen", flyer_source="sen"):
    """İki ata, zıt iki iddia — ve öğretme sırası dışarıdan verilebiliyor."""
    memory = Memory()
    for kind in (first, second):
        memory.write(Edge("yarasa", IS_A, kind, source="sen"))
    memory.write(Edge("memeli", CANNOT, "uçmak", source=mammal_source))
    memory.write(Edge("ucucu", CAN, "uçmak", source=flyer_source))
    return Reasoning(memory)


class TestDiamondInheritance(unittest.TestCase):
    """Aynı bilgiden iki farklı cevap çıkarmak, uydurmaktan farksızdır."""

    def test_the_teaching_order_does_not_decide(self):
        # Ölçüldü: `_resolve` ilk kaydı bulan atada dönüyordu ve `ancestors()`
        # aynı derinlikteki ataları `memory.edges` sırasıyla veriyordu, yani
        # "memeli önce" hayır, "uçucu önce" evet diyordu.
        one = diamond("memeli", "ucucu").can_do("yarasa", "uçmak")
        other = diamond("ucucu", "memeli").can_do("yarasa", "uçmak")
        self.assertEqual(one, other)

    def test_an_unarbitrated_clash_is_admitted_not_answered(self):
        answer, chain = diamond("memeli", "ucucu").can_do("yarasa", "uçmak")
        self.assertIsNone(answer)        # kimse kazanmadı
        self.assertIn("memeli", " ".join(chain))
        self.assertIn("ucucu", " ".join(chain))   # ikisi de zincirde duruyor

    def test_the_stronger_source_settles_it_whatever_the_order(self):
        # Muhakemenin kanıt gücünü OKUDUĞU yer: eskiden `_resolve` yalnız
        # sıraya bakıyordu, `grep -n confidence lmm/reasoning.py` bunu
        # gösteriyordu.
        model = distilled_source("claude")
        for first, second in (("memeli", "ucucu"), ("ucucu", "memeli")):
            reasoning = diamond(first, second, flyer_source=model)
            self.assertFalse(reasoning.can_do("yarasa", "uçmak")[0])

    def test_a_nearer_ancestor_still_wins_over_a_farther_clash(self):
        memory = Memory()
        memory.write(Edge("yarasa", IS_A, "memeli", source="sen"))
        memory.write(Edge("memeli", IS_A, "ucucu", source="sen"))
        memory.write(Edge("memeli", CANNOT, "uçmak", source="sen"))
        memory.write(Edge("ucucu", CAN, "uçmak", source="sen"))
        # Farklı basamaklar çatışma değil, istisnadır: yakın olan konuşur.
        self.assertFalse(Reasoning(memory).can_do("yarasa", "uçmak")[0])


class TestSenses(unittest.TestCase):
    """Aynı ada çökmüş iki şey, tek bir şey gibi yürünemez."""

    def setUp(self):
        self.memory = Memory()
        self.reasoning = Reasoning(self.memory)
        for edge in (Edge("penguen", IS_A, "kuş", source="sen"),
                     Edge("kuş", IS_A, "varlık", source="sen"),
                     # "varlık" burada iki şey: felsefi kavram ve bir dergi.
                     # İkisi de aynı güvende, aralarında hiçbir bağ yok.
                     Edge("varlık", IS_A, "kaynak", source="a.txt"),
                     Edge("varlık", IS_A, "dergisi", source="b.txt")):
            self.memory.write(edge)

    def test_an_unrelated_second_sense_is_not_walked_into(self):
        found = self.reasoning.ancestors("penguen")
        self.assertIn("varlık", found)          # oraya kadar yürünüyor
        self.assertNotIn("dergisi", found)      # ama hangi anlamdan girildiği
        self.assertNotIn("kaynak", found)       # bilinmiyor, ikisi de yürünmez

    def test_nothing_leaks_through_the_split(self):
        self.memory.write(Edge("kaynak", HAS_PROPERTY, "yararlı", source="sen"))
        self.assertIsNone(self.reasoning.has_property("penguen", "yararlı")[0])

    def test_related_types_are_one_sense_and_still_inherit(self):
        # Çok atalılık bir anlam ayrımı değil: memeli zaten bir hayvan.
        memory = Memory()
        for edge in (Edge("aslan", IS_A, "memeli", source="sen"),
                     Edge("aslan", IS_A, "hayvan", source="sen"),
                     Edge("memeli", IS_A, "hayvan", source="sen"),
                     Edge("kedigil", IS_A, "memeli", source="sen"),
                     Edge("kaplan", IS_A, "kedigil", source="sen"),
                     Edge("hayvan", HAS_PROPERTY, "canlı", source="sen")):
            memory.write(edge)
        reasoning = Reasoning(memory)
        self.assertIn("hayvan", reasoning.ancestors("kaplan"))
        self.assertTrue(reasoning.has_property("kaplan", "canlı")[0])

    def test_the_asked_concept_keeps_both_of_its_senses(self):
        # Kökte durmak ölçüldü ve pahalı: sınav %95,7'den %91,3'e düşüyordu.
        # Soru zaten o kavram hakkında; seçmeyi `gate` "hangi anlamda" diye
        # sorarak yapıyor.
        self.assertEqual(sorted(self.reasoning.ancestors("varlık")),
                         ["dergisi", "kaynak"])

    def test_lineage_reports_the_distance_of_each_ancestor(self):
        self.assertEqual(self.reasoning.lineage("penguen"),
                         [["kuş"], ["varlık"]])


if __name__ == "__main__":
    unittest.main()
