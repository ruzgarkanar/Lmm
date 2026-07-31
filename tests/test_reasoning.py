import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT
from lmm.reasoning import Reasoning


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


if __name__ == "__main__":
    unittest.main()
