"""Memory packs: knowledge that can be shipped, read, merged and deleted."""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT
from lmm.reasoning import Reasoning
from lmm.pack import export_pack, read_pack, merge_pack
from lmm.cli import Session


class TestPackRoundTrip(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.pack_path = os.path.join(self.directory, "birds.json")

    def test_export_then_read_preserves_facts(self):
        memory = Memory()
        memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        export_pack(memory, self.pack_path, name="kuslar", version="1.0",
                    author="rüzgar")

        pack = read_pack(self.pack_path)
        self.assertEqual(pack.name, "kuslar")
        self.assertEqual(pack.author, "rüzgar")
        self.assertEqual(pack.provenance, "kuslar@1.0")
        self.assertEqual(len(pack.facts), 2)

    def test_pack_file_is_human_readable(self):
        """The whole point: you can read every belief before installing it."""
        memory = Memory()
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        export_pack(memory, self.pack_path, name="kuslar")
        with open(self.pack_path, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("penguen", text)
        self.assertIn("kuş", text)


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.pack_path = os.path.join(self.directory, "birds.json")
        source = Memory()
        source.write(Edge("kuş", CAN, "uçmak", source="sen"))
        source.write(Edge("penguen", IS_A, "kuş", source="sen"))
        export_pack(source, self.pack_path, name="kuslar", version="2.1")
        self.pack = read_pack(self.pack_path)

    def test_merge_into_empty_memory_records_provenance(self):
        memory = Memory()
        report = merge_pack(memory, self.pack)
        self.assertEqual(len(report.added), 2)
        self.assertTrue(report.clean)
        self.assertEqual(memory.direct("penguen", IS_A, "kuş").source, "kuslar@2.1")

    def test_known_facts_are_reinforced_not_duplicated(self):
        """A pack agreeing with a person is a second, independent voice."""
        memory = Memory()
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        before = memory.direct("penguen", IS_A, "kuş").confidence
        report = merge_pack(memory, self.pack)
        self.assertEqual(len(report.reinforced), 1)
        self.assertEqual(len(report.added), 1)
        edge = memory.direct("penguen", IS_A, "kuş")
        self.assertGreater(edge.confidence, before)
        self.assertEqual(len(edge.sources), 2)

    def test_declared_exceptions_travel_with_the_pack(self):
        """A pack's exceptions are its most valuable knowledge — they must survive.

        "penguen uçamaz" clashes with the pack's own "kuşlar uçar" by design.
        An early merge mistook that for a conflict and dropped exactly the fact
        the pack existed to carry.
        """
        source = Memory()
        source.write(Edge("kuş", CAN, "uçmak", source="sen"))
        source.write(Edge("penguen", IS_A, "kuş", source="sen"))
        source.write(Edge("penguen", CANNOT, "uçmak", source="sen", is_exception=True))
        path = os.path.join(self.directory, "with_exception.json")
        export_pack(source, path, name="kuslar")

        memory = Memory()
        report = merge_pack(memory, read_pack(path))
        self.assertTrue(report.clean)
        self.assertEqual(len(report.added), 3)

        reasoning = Reasoning(memory)
        self.assertFalse(reasoning.can_do("penguen", "uçmak")[0])   # exception held
        self.assertIsNone(reasoning.can_do("serçe", "uçmak")[0])    # nothing invented

    def test_exception_still_conflicts_with_a_direct_local_fact(self):
        """Two sources disagreeing head-on must still surface."""
        memory = Memory()
        memory.write(Edge("penguen", CAN, "uçmak", source="sen"))

        source = Memory()
        source.write(Edge("penguen", CANNOT, "uçmak", source="başkası",
                          is_exception=True))
        path = os.path.join(self.directory, "clash.json")
        export_pack(source, path, name="itiraz")

        report = merge_pack(memory, read_pack(path))
        self.assertEqual(len(report.conflicts), 1)
        self.assertIsNone(memory.direct("penguen", CANNOT, "uçmak"))

    def test_contradicting_pack_fact_is_reported_not_absorbed(self):
        """Two sources disagreeing must surface, never be averaged away."""
        memory = Memory()
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        memory.write(Edge("penguen", CANNOT, "uçmak", source="sen", is_exception=True))

        conflicting = Memory()
        conflicting.write(Edge("penguen", CAN, "uçmak", source="başkası"))
        path = os.path.join(self.directory, "wrong.json")
        export_pack(conflicting, path, name="yanlis")

        report = merge_pack(memory, read_pack(path))
        self.assertEqual(len(report.conflicts), 1)
        self.assertFalse(report.clean)
        # memory kept its own belief; nothing was silently overwritten
        self.assertIsNone(memory.direct("penguen", CAN, "uçmak"))


class TestForgetting(unittest.TestCase):
    def test_a_concept_can_be_truly_erased(self):
        """Selective deletion — the thing weight-based models cannot honour."""
        path = os.path.join(tempfile.mkdtemp(), "memory.json")
        session = Session(path)
        session.respond("kuşlar uçar")
        session.respond("penguen bir kuştur")
        self.assertTrue(session.respond("penguen uçar mı").startswith("evet"))

        removed = session.memory.forget("penguen")
        self.assertEqual(removed, 1)
        self.assertIn("bilmiyorum", session.respond("penguen nedir"))
        # unrelated knowledge is untouched
        session.respond("serçe bir kuştur")
        self.assertTrue(session.respond("serçe uçar mı").startswith("evet"))

    def test_forgetting_survives_a_restart(self):
        path = os.path.join(tempfile.mkdtemp(), "memory.json")
        first = Session(path)
        first.respond("penguen bir kuştur")
        first.memory.forget("penguen")
        first.save()

        second = Session(path)
        self.assertIn("bilmiyorum", second.respond("penguen nedir"))


if __name__ == "__main__":
    unittest.main()
