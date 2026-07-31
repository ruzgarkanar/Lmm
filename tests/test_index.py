"""The indexes must never disagree with the edges they summarise.

Everything in them is derivable from self.edges by a plain scan, so the test is
to do the scan and compare. An index that drifts is worse than no index: it
would make the system confidently answer from a stale view of its own memory.
"""
import time
import unittest

from lmm.memory import (Memory, Edge, IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                        LACKS_PROPERTY, TYPE_RELATIONS, ABILITY_RELATIONS,
                        PROPERTY_RELATIONS)


def scanned_concepts(memory):
    ordered = []
    for edge in memory.edges:
        target = edge.target if edge.relation in TYPE_RELATIONS else None
        for candidate in (edge.concept, target):
            if candidate is not None and candidate not in ordered:
                ordered.append(candidate)
    return ordered


def scanned_targets(memory, relations):
    ordered = []
    for edge in memory.edges:
        if edge.relation in relations and edge.target not in ordered:
            ordered.append(edge.target)
    return ordered


class TestIndexAgreesWithTheEdges(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        for edge in [Edge("kuş", IS_A, "hayvan"), Edge("kuş", CAN, "uçmak"),
                     Edge("kuş", HAS_PROPERTY, "tüylü"),
                     Edge("penguen", IS_A, "kuş"),
                     Edge("penguen", CANNOT, "uçmak"),
                     Edge("penguen", LACKS_PROPERTY, "hızlı"),
                     Edge("penguen", NOT_A, "memeli"),
                     Edge("balık", IS_A, "hayvan"), Edge("balık", CAN, "yüzmek")]:
            self.memory.write(edge)

    def test_query_matches_a_scan(self):
        for concept in self.memory.concepts():
            scanned = [e for e in self.memory.edges if e.concept == concept]
            self.assertEqual(self.memory.query(concept), scanned)

    def test_direct_matches_a_scan(self):
        for edge in self.memory.edges:
            self.assertIs(self.memory.direct(edge.concept, edge.relation,
                                             edge.target), edge)
        self.assertIsNone(self.memory.direct("kuş", CANNOT, "uçmak"))

    def test_listings_match_a_scan_and_keep_learning_order(self):
        self.assertEqual(self.memory.concepts(), scanned_concepts(self.memory))
        self.assertEqual(self.memory.actions(),
                         scanned_targets(self.memory, ABILITY_RELATIONS))
        self.assertEqual(self.memory.properties(),
                         scanned_targets(self.memory, PROPERTY_RELATIONS))

    def test_forgetting_leaves_no_trace_in_the_index(self):
        self.memory.forget("penguen")
        self.assertNotIn("penguen", self.memory.concepts())
        self.assertEqual(self.memory.query("penguen"), [])
        self.assertIsNone(self.memory.direct("penguen", NOT_A, "memeli"))
        self.assertEqual(self.memory.concepts(), scanned_concepts(self.memory))

    def test_a_reload_reproduces_the_same_index(self):
        import os
        import tempfile
        path = os.path.join(tempfile.mkdtemp(), "memory.json")
        self.memory.save(path)
        reloaded = Memory.load(path)
        self.assertEqual(reloaded.concepts(), self.memory.concepts())
        self.assertEqual(reloaded.actions(), self.memory.actions())
        self.assertIsNotNone(reloaded.direct("penguen", CANNOT, "uçmak"))


class TestItStaysFastAsItGrows(unittest.TestCase):
    def test_lookup_does_not_slow_down_with_size(self):
        """The point of the index: answering must not depend on how much is known."""
        small, large = Memory(), Memory()
        for memory, count in ((small, 200), (large, 40_000)):
            for i in range(count):
                memory.write(Edge(f"kavram{i}", IS_A, f"tur{i % 50}"))
            for i in range(50):
                memory.write(Edge(f"tur{i}", CAN, "uçmak"))

        def timed(memory):
            started = time.time()
            for i in range(200):
                memory.direct(f"kavram{i}", IS_A, f"tur{i % 50}")
            return time.time() - started

        self.assertLess(timed(large), timed(small) * 5 + 0.01)


if __name__ == "__main__":
    unittest.main()
