import json
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, CycleError, IS_A, CAN, CANNOT
from lmm.reasoning import Reasoning
from lmm.relations import ALL, MOST, SOME, NO


class TestMemory(unittest.TestCase):
    def test_write_and_query(self):
        m = Memory()
        m.write(Edge("penguen", IS_A, "kuş", source="sen"))
        result = m.query("penguen", IS_A)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].target, "kuş")
        self.assertEqual(result[0].source, "sen")

    def test_miss_returns_empty(self):
        m = Memory()
        self.assertEqual(m.query("ejderha", IS_A), [])

    def test_direct_lookup(self):
        m = Memory()
        m.write(Edge("kuş", CAN, "uçmak"))
        self.assertIsNotNone(m.direct("kuş", CAN, "uçmak"))
        self.assertIsNone(m.direct("kuş", CANNOT, "uçmak"))

    def test_the_same_source_twice_is_not_corroboration(self):
        """One mistake repeated is still one mistake."""
        m = Memory()
        first = m.write(Edge("kedi", IS_A, "hayvan", source="sen"))
        before = first.confidence
        second = m.write(Edge("kedi", IS_A, "hayvan", source="sen"))
        self.assertIs(first, second)  # no duplicate edge is added
        self.assertEqual(first.confidence, before)
        self.assertEqual(len(m.edges), 1)

    def test_an_independent_source_raises_confidence(self):
        m = Memory()
        edge = m.write(Edge("kedi", IS_A, "hayvan", source="ansiklopedi.txt"))
        before = edge.confidence
        m.write(Edge("kedi", IS_A, "hayvan", source="sozluk.txt"))
        self.assertGreater(edge.confidence, before)
        self.assertEqual(edge.sources, ["ansiklopedi.txt", "sozluk.txt"])

    def test_answers_cite_the_strongest_voice(self):
        m = Memory()
        edge = m.write(Edge("kedi", IS_A, "hayvan", source="llm:claude"))
        m.write(Edge("kedi", IS_A, "hayvan", source="sen"))
        self.assertEqual(edge.source, "sen")
        self.assertIn("llm:claude", edge.sources)

    def test_type_cycle_rejected(self):
        m = Memory()
        m.write(Edge("a", IS_A, "b"))
        m.write(Edge("b", IS_A, "c"))
        with self.assertRaises(CycleError):
            m.write(Edge("c", IS_A, "a"))


class TestQuantityIsPartOfIdentity(unittest.TestCase):
    """Nicelik kenar kimliğinin parçası olmasaydı iddialar birbirini yutardı.

    Ölçülen hata: "bazı kuşlar uçmaz" yazılıp ardından "hiçbir kuş uçmaz"
    söylendiğinde tek kenar kalıyor, ikinci cümle birincinin TANIĞI sayılıp
    güveni yükseltiyordu. Kimsenin söylemediği bir mutabakat.
    """

    def test_two_quantities_are_two_records(self):
        m = Memory()
        some = m.write(Edge("kuş", CANNOT, "uçmak", source="a", quantifier=SOME))
        every = m.write(Edge("kuş", CANNOT, "uçmak", source="b", quantifier=NO))
        self.assertIsNot(some, every)
        self.assertEqual(len(m.edges), 2)

    def test_a_different_quantity_is_not_a_witness(self):
        m = Memory()
        some = m.write(Edge("kuş", CANNOT, "uçmak", source="a", quantifier=SOME))
        before = some.confidence
        m.write(Edge("kuş", CANNOT, "uçmak", source="b", quantifier=NO))
        self.assertEqual(some.confidence, before)
        self.assertEqual(some.sources, ["a"])   # "b" başka bir şey söyledi

    def test_the_same_quantity_still_corroborates(self):
        """Ayırma, pekişmeyi bozmamalı: aynı iddiayı iki kaynak söylerse güven artar."""
        m = Memory()
        edge = m.write(Edge("kuş", CANNOT, "uçmak", source="a", quantifier=SOME))
        before = edge.confidence
        again = m.write(Edge("kuş", CANNOT, "uçmak", source="b", quantifier=SOME))
        self.assertIs(again, edge)
        self.assertGreater(edge.confidence, before)
        self.assertEqual(len(m.edges), 1)

    def test_an_exact_quantity_can_be_asked_for(self):
        m = Memory()
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=SOME))
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=NO))
        self.assertEqual(
            m.direct("kuş", CANNOT, "uçmak", quantifier=SOME).quantifier, SOME)
        self.assertIsNone(m.direct("kuş", CANNOT, "uçmak", quantifier=MOST))

    def test_both_quantities_may_stand_at_once(self):
        """Aynı kutupta nicelikler çelişmez — altbağlılık: hepsi -> çoğu -> bazı."""
        m = Memory()
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=SOME))
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=NO))
        self.assertEqual([e.quantifier for e in m.quantities("kuş", CANNOT, "uçmak")],
                         [SOME, NO])

    def test_the_widest_claim_answers_whichever_came_first(self):
        for order in ((SOME, ALL), (ALL, SOME)):
            m = Memory()
            for quantifier in order:
                m.write(Edge("kuş", CAN, "uçmak", quantifier=quantifier))
            self.assertEqual(m.direct("kuş", CAN, "uçmak").quantifier, ALL)

    def test_a_universal_arriving_later_reaches_the_members(self):
        """Asıl kazanç: "bazı" kaydı artık evrenseli sonsuza dek gölgelemiyor.

        Önce: "bazı kuşlar uçmaz" tek kayıt olduğu ve INHERITING dışında
        kaldığı için "penguen uçar mı" -> bilmiyorum. Sonra: evrensel iddia
        ayrı kayıt olarak duruyor ve iniyor.
        """
        m = Memory()
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=SOME))
        m.write(Edge("penguen", IS_A, "kuş"))
        self.assertIsNone(Reasoning(m).can_do("penguen", "uçmak")[0])
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=NO))
        self.assertIs(Reasoning(m).can_do("penguen", "uçmak")[0], False)

    def test_forgetting_clears_every_quantity(self):
        m = Memory()
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=SOME))
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=NO))
        self.assertEqual(m.forget("kuş"), 2)
        self.assertEqual(m.quantities("kuş", CANNOT, "uçmak"), [])
        self.assertIsNone(m.direct("kuş", CANNOT, "uçmak"))

    def test_both_quantities_survive_a_restart(self):
        m = Memory()
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=SOME))
        m.write(Edge("kuş", CANNOT, "uçmak", quantifier=NO))
        path = os.path.join(tempfile.mkdtemp(), "memory.lmm")
        m.save(path)
        reloaded = Memory.load(path)
        self.assertEqual(len(reloaded.edges), 2)
        self.assertEqual(reloaded.direct("kuş", CANNOT, "uçmak").quantifier, NO)


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "memory.json")

    def test_save_and_load(self):
        m = Memory()
        m.write(Edge("penguen", IS_A, "kuş", source="sen"))
        m.save(self.path)
        reloaded = Memory.load(self.path)          # simulates a restart
        result = reloaded.query("penguen", IS_A)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].target, "kuş")
        self.assertEqual(result[0].source, "sen")

    def test_a_file_written_before_quantities_still_loads(self):
        """Nicelik kimliğe girdi ama dosya biçimi değişmedi — eski model okunmalı.

        Ölçüldü: models/graph/birlesik.lmm hâlâ 16.164 olgu, 16.774 kavram.
        Niceliği hiç yazmamış bir kayıt evrensel sayılır; o zamanki tek okuma
        buydu, yani eski dosyanın anlamı değişmiyor.
        """
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump([{"concept": "kuş", "relation": CAN, "target": "uçmak",
                        "source": "sen"}], f)
        m = Memory.load(self.path)
        self.assertEqual(m.direct("kuş", CAN, "uçmak").quantifier, ALL)

    def test_missing_file_yields_empty_memory(self):
        m = Memory.load(os.path.join(self.directory, "nope.json"))
        self.assertEqual(m.edges, [])

    def test_corrupt_file_yields_empty_memory(self):
        with open(self.path, "w") as f:
            f.write("{corrupt json!!")
        m = Memory.load(self.path)
        self.assertEqual(m.edges, [])


if __name__ == "__main__":
    unittest.main()
