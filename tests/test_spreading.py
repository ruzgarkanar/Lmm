"""Grafta yayılım: bir kavramdan bütünü çağırmak.

Graf şimdiye kadar düz bir aramaydı — "kartal" sorulunca kartalın kayıtlarına
bakılıyor, orada bitiyordu. Oysa bilginin değeri komşuluklarda: kartal bir
kuştur, kuşlar uçar, uçan başka neler var.

Beyindeki karşılığı hipokampüsün örüntü tamamlaması; HippoRAG bu benzetmeyi
kurup çok adımlı sorularda %20 kazanç ölçtü.
"""
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, HAS_PROPERTY
from lmm.spreading import bridge, neighbours, related, spread


def world():
    memory = Memory()
    for concept, relation, target in (
            ("kartal", IS_A, "kuş"), ("penguen", IS_A, "kuş"),
            ("kuş", CAN, "uçmak"), ("penguen", CAN, "yüzmek"),
            ("kartal", CAN, "avlanmak"), ("penguen", HAS_PROPERTY, "siyah"),
            ("kömür", HAS_PROPERTY, "siyah"), ("kedi", IS_A, "memeli"),
            ("köpek", IS_A, "memeli"), ("kuş", IS_A, "hayvan"),
            ("memeli", IS_A, "hayvan")):
        memory.write(Edge(concept, relation, target, source="test"))
    return memory


class TestSpreading(unittest.TestCase):
    def setUp(self):
        self.memory = world()
        self.links = neighbours(self.memory)

    def test_a_direct_neighbour_is_reached(self):
        self.assertIn("kuş", related(self.memory, ["kartal"], links=self.links))

    def test_something_two_steps_away_is_reached(self):
        """"kartal" -> "kuş" -> "uçmak": düz arama bunu bulamaz."""
        self.assertIn("uçmak", related(self.memory, ["kartal"], links=self.links))

    def test_a_shared_property_connects_unrelated_things(self):
        """penguen ile kömür hiçbir kayıtta birlikte geçmiyor; ikisi de siyah."""
        self.assertIn("kömür", related(self.memory, ["penguen"], count=12,
                                       links=self.links))

    def test_the_seed_is_not_its_own_answer(self):
        self.assertNotIn("kartal", related(self.memory, ["kartal"],
                                           links=self.links))

    def test_interest_falls_with_distance(self):
        found = spread(self.memory, ["kartal"], self.links)
        self.assertGreater(found["kuş"], found.get("hayvan", 0))

    def test_an_unknown_seed_spreads_nothing(self):
        self.assertEqual(related(self.memory, ["ejderha"], links=self.links), [])

    def test_the_same_graph_always_answers_the_same(self):
        """Beraberlikte ad sırası: kararlılık, cevabın tekrarlanabilirliği."""
        first = related(self.memory, ["penguen"], links=self.links)
        second = related(self.memory, ["penguen"], links=self.links)
        self.assertEqual(first, second)


class TestBridging(unittest.TestCase):
    def setUp(self):
        self.memory = world()

    def test_it_finds_what_two_things_have_in_common(self):
        self.assertEqual(bridge(self.memory, "kartal", "penguen")[0], "kuş")

    def test_it_works_where_no_single_record_holds_both(self):
        found = bridge(self.memory, "kedi", "köpek")
        self.assertIn("memeli", found)

    def test_unconnected_things_have_no_bridge(self):
        self.assertEqual(bridge(self.memory, "kartal", "ejderha"), [])


if __name__ == "__main__":
    unittest.main()
