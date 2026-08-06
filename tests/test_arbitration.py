"""Weighing sources against each other, when rank alone cannot settle it.

Truth-discovery research compared a dozen elaborate schemes and found plain
majority voting almost impossible to beat, at a ninth to a hundredth of the
cost — and several of the clever ones were not reproducible run to run. So this
counts voices and keeps a record, and it does not model anything.

Where nothing separates two claims it follows Wikidata: keep both, mark the
disagreement, and never quietly pick a winner.
"""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, CAN, CANNOT, IS_A
from lmm.reasoning import Reasoning
from lmm.learning import LearningLoop, CORRECTED, DISPUTE, CONFLICT
from lmm.intuition import Intent
from lmm.intuition import TEACH
from lmm.trust import (arbitrate, reputation_of, note, CANDIDATE, INCUMBENT,
                       DISPUTED, TEACHER)
from lmm.cli import Session


def edge(relation, source, sources=None):
    return Edge("gemi", relation, "yüzmek", source=source, sources=sources)


class TestArbitrating(unittest.TestCase):
    def test_rank_decides_before_anything_else(self):
        self.assertEqual(arbitrate(edge(CAN, "kitap.txt"), edge(CANNOT, TEACHER)),
                         CANDIDATE)
        self.assertEqual(arbitrate(edge(CAN, TEACHER), edge(CANNOT, "kitap.txt")),
                         INCUMBENT)

    def test_more_independent_voices_win(self):
        many = edge(CAN, "a.txt", sources=["a.txt", "b.txt"])
        self.assertEqual(arbitrate(many, edge(CANNOT, "c.txt")), INCUMBENT)
        self.assertEqual(arbitrate(edge(CAN, "c.txt"), Edge(
            "gemi", CANNOT, "yüzmek", source="a.txt",
            sources=["a.txt", "b.txt"])), CANDIDATE)

    def test_a_record_breaks_a_tie(self):
        reputation = {}
        for _ in range(3):
            note(reputation, "güvenilir.txt", True)
        note(reputation, "şüpheli.txt", False)
        self.assertEqual(arbitrate(edge(CAN, "şüpheli.txt"),
                                   edge(CANNOT, "güvenilir.txt"), reputation),
                         CANDIDATE)

    def test_nothing_to_separate_them_means_disputed(self):
        self.assertEqual(arbitrate(edge(CAN, "a.txt"), edge(CANNOT, "b.txt")),
                         DISPUTED)

    def test_a_source_cannot_dispute_itself(self):
        """One document saying both is incoherent, not a disagreement."""
        self.assertEqual(arbitrate(edge(CAN, "a.txt"), edge(CANNOT, "a.txt")),
                         INCUMBENT)

    def test_an_unknown_source_starts_neither_trusted_nor_suspected(self):
        self.assertEqual(reputation_of({}, "yeni.txt"), 0.5)


class TestInTheLearningLoop(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.reasoning = Reasoning(self.memory)
        self.loop = LearningLoop(self.memory, self.reasoning)
        self.floats = Intent(TEACH, concept="gemi", relation=CAN, target="yüzmek")
        self.sinks = Intent(TEACH, concept="gemi", relation=CANNOT, target="yüzmek")

    def test_two_documents_disagreeing_are_both_kept_and_marked(self):
        self.loop.teach(self.floats, source="kitap-a.txt")
        status, message, _ = self.loop.teach(self.sinks, source="kitap-b.txt")
        self.assertEqual(status, DISPUTE)
        self.assertIn("kaynaklar anlaşmıyor", message)
        self.assertEqual(len(self.memory.edges), 2)
        self.assertTrue(all(e.disputed for e in self.memory.edges))

    def test_an_answer_admits_the_disagreement(self):
        self.loop.teach(self.floats, source="kitap-a.txt")
        self.loop.teach(self.sinks, source="kitap-b.txt")
        _, chain = self.reasoning.can_do("gemi", "yüzmek")
        self.assertIn("anlaşmıyor", " ".join(chain))

    def test_a_third_voice_settles_it(self):
        self.loop.teach(self.floats, source="kitap-a.txt")
        self.loop.teach(self.sinks, source="kitap-b.txt")
        status, _, _ = self.loop.teach(self.floats, source="kitap-c.txt")
        self.assertEqual(status, CORRECTED)
        known, chain = self.reasoning.can_do("gemi", "yüzmek")
        self.assertTrue(known)
        self.assertNotIn("anlaşmıyor", " ".join(chain))

    def test_the_record_follows_the_sources(self):
        self.loop.teach(self.floats, source="kitap-a.txt")
        self.loop.teach(self.sinks, source="kitap-b.txt")
        self.loop.teach(self.floats, source="kitap-c.txt")
        self.assertGreater(reputation_of(self.memory.reputation, "kitap-c.txt"),
                           reputation_of(self.memory.reputation, "kitap-b.txt"))

    def test_a_person_is_still_asked_rather_than_arbitrated(self):
        """Rank settles it for a person, so the conversation still happens."""
        self.loop.teach(Intent(TEACH, concept="kuş", relation=CAN, target="uçmak"))
        self.loop.teach(Intent(TEACH, concept="penguen", relation=IS_A,
                               target="kuş"))
        status, message, _ = self.loop.teach(
            Intent(TEACH, concept="penguen", relation=CANNOT, target="uçmak"))
        self.assertEqual(status, CONFLICT)
        self.assertIn("çelişki", message)

    def test_a_document_contradicting_itself_is_still_refused(self):
        self.loop.teach(Intent(TEACH, concept="kuş", relation=CAN, target="uçmak"),
                        source="belge.txt")
        self.loop.teach(Intent(TEACH, concept="penguen", relation=IS_A,
                               target="kuş"), source="belge.txt")
        status, _, _ = self.loop.teach(
            Intent(TEACH, concept="penguen", relation=CANNOT, target="uçmak"),
            source="belge.txt")
        self.assertEqual(status, CONFLICT)


class TestPersistence(unittest.TestCase):
    def test_the_record_and_the_dispute_survive_a_restart(self):
        path = os.path.join(tempfile.mkdtemp(), "memory.lmm")
        memory = Memory()
        loop = LearningLoop(memory, Reasoning(memory))
        floats = Intent(TEACH, concept="gemi", relation=CAN, target="yüzmek")
        sinks = Intent(TEACH, concept="gemi", relation=CANNOT, target="yüzmek")
        loop.teach(floats, source="kitap-a.txt")
        loop.teach(sinks, source="kitap-b.txt")
        memory.save(path)

        reloaded = Memory.load(path)
        self.assertTrue(all(e.disputed for e in reloaded.edges))
        self.assertEqual(reloaded.reputation, memory.reputation)


if __name__ == "__main__":
    unittest.main()
