import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, CycleError, IS_A, CAN, CANNOT


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

    def test_repeated_teaching_raises_confidence(self):
        m = Memory()
        first = m.write(Edge("kedi", IS_A, "hayvan", confidence=0.6))
        second = m.write(Edge("kedi", IS_A, "hayvan", confidence=0.6))
        self.assertIs(first, second)  # no duplicate edge is added
        self.assertAlmostEqual(first.confidence, 0.8)
        self.assertEqual(len(m.edges), 1)

    def test_type_cycle_rejected(self):
        m = Memory()
        m.write(Edge("a", IS_A, "b"))
        m.write(Edge("b", IS_A, "c"))
        with self.assertRaises(CycleError):
            m.write(Edge("c", IS_A, "a"))


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
