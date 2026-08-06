"""Watching a branch go bad, which is how unsupervised learning actually fails.

NELL stayed above 90% precision on three quarters of its categories after six
months unsupervised — and fell to between 25% and 60% on the rest. The damage
was concentrated and quiet. A few minutes of review per category every few weeks
brought the whole thing back to about 87%.

So a branch that starts refusing much of what arrives stops growing on its own
and asks for a person. Nothing is deleted; it simply stops.
"""
import os
import tempfile
import unittest

from lmm.memory import Memory, IS_A, CAN, CANNOT
from lmm.reasoning import Reasoning
from lmm.learning import LearningLoop, FROZEN, LEARNED
from lmm.intuition import Intent
from lmm.intuition import TEACH
from lmm.drift import (branch_of, record, refusal_rate, frozen, frozen_branches,
                       thaw, may_write, MINIMUM)
from lmm.trust import TEACHER


class TestFindingTheBranch(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.reasoning = Reasoning(self.memory)
        for concept, parent in (("kuş", "hayvan"), ("penguen", "kuş")):
            self.memory.write(__import__("lmm.memory", fromlist=["Edge"]).Edge(
                concept, IS_A, parent, source=TEACHER))

    def test_a_branch_is_the_outermost_type(self):
        """Drift shows at the top: not one penguin, but everything under birds."""
        self.assertEqual(branch_of(self.reasoning, "penguen"), "hayvan")

    def test_an_unplaced_concept_is_its_own_branch(self):
        self.assertEqual(branch_of(self.reasoning, "zürafa"), "zürafa")


class TestFreezing(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()

    def test_a_short_run_of_refusals_is_not_drift(self):
        for _ in range(MINIMUM - 1):
            record(self.memory, "hayvan", False)
        self.assertFalse(frozen(self.memory, "hayvan"))

    def test_a_sustained_refusal_rate_freezes_the_branch(self):
        for _ in range(MINIMUM):
            record(self.memory, "hayvan", False)
        self.assertTrue(frozen(self.memory, "hayvan"))
        self.assertEqual(frozen_branches(self.memory), ["hayvan"])

    def test_a_healthy_branch_stays_open(self):
        for _ in range(20):
            record(self.memory, "hayvan", True)
        for _ in range(2):
            record(self.memory, "hayvan", False)
        self.assertFalse(frozen(self.memory, "hayvan"))
        self.assertLess(refusal_rate(self.memory, "hayvan"), 0.4)

    def test_review_reopens_it_and_starts_the_count_again(self):
        for _ in range(MINIMUM):
            record(self.memory, "hayvan", False)
        self.assertTrue(thaw(self.memory, "hayvan"))
        self.assertFalse(frozen(self.memory, "hayvan"))
        self.assertEqual(refusal_rate(self.memory, "hayvan"), 0.0)


class TestWhatFreezingChanges(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.reasoning = Reasoning(self.memory)
        self.loop = LearningLoop(self.memory, self.reasoning)
        self._teach("kuş", IS_A, "hayvan")
        self._teach("kuş", CAN, "uçmak")
        for bird in ("serçe", "kartal", "güvercin", "baykuş"):
            self._teach(bird, IS_A, "kuş")

    def _teach(self, concept, relation, target, source="belge.txt"):
        return self.loop.teach(Intent(TEACH, concept=concept, relation=relation,
                                      target=target), source=source)

    def _spoil(self):
        for bird in ("serçe", "kartal", "güvercin", "baykuş", "kuş"):
            self._teach(bird, CANNOT, "uçmak")

    def test_contradictions_pile_up_and_the_branch_closes(self):
        self.assertFalse(frozen(self.memory, "hayvan"))
        self._spoil()
        self.assertTrue(frozen(self.memory, "hayvan"))

    def test_a_machine_source_is_turned_away(self):
        self._spoil()
        status, message, _ = self._teach("kartal", CANNOT, "yüzmek",
                                         source="başka.txt")
        self.assertEqual(status, FROZEN)
        self.assertIn("hayvan", message)

    def test_a_person_may_always_teach(self):
        self._spoil()
        status, _, _ = self._teach("kartal", CANNOT, "yüzmek", source=TEACHER)
        self.assertEqual(status, LEARNED)

    def test_nothing_already_learned_is_doubted(self):
        self._spoil()
        self.assertTrue(self.reasoning.can_do("serçe", "uçmak")[0])

    def test_an_untouched_branch_keeps_growing(self):
        self._spoil()
        self._teach("taş", IS_A, "nesne")
        status, _, _ = self._teach("taş", CANNOT, "uçmak", source="başka.txt")
        self.assertEqual(status, LEARNED)


class TestPersistence(unittest.TestCase):
    def test_the_freeze_survives_a_restart(self):
        path = os.path.join(tempfile.mkdtemp(), "memory.lmm")
        memory = Memory()
        for _ in range(MINIMUM):
            record(memory, "hayvan", False)
        memory.save(path)
        self.assertEqual(frozen_branches(Memory.load(path)), ["hayvan"])


if __name__ == "__main__":
    unittest.main()
