"""Distilling at scale: a memory seeded from a model, then standing on its own.

These run against the packs in `packs/`, so they check the real pipeline rather
than a fixture: distil, audit, generalise, answer.
"""
import os
import tempfile
import unittest

from lmm.memory import Memory, IS_A, INFERRED
from lmm.reasoning import Reasoning
from lmm.distill import distill_file, generalise
from lmm.cli import Session

PACKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "packs")


class TestDistilledCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(tempfile.mkdtemp(), "memory.json")
        memory = Memory()
        cls.reports = [distill_file(os.path.join(PACKS, name), memory, "claude")
                       for name in ("tr-cekirdek.txt", "tr-cekirdek-2.txt")]
        cls.formed = generalise(memory)
        memory.save(cls.path)
        cls.memory = memory

    def test_the_core_is_substantial(self):
        self.assertGreater(len(self.memory.edges), 250)
        self.assertGreater(len(self.memory.concepts()), 100)
        self.assertGreater(len(self.memory.vocabulary), 20)

    def test_the_model_contradicted_itself_and_was_caught(self):
        """Birds fly, a penguin is a bird, a penguin cannot fly — all three said."""
        conflicts = [edge.concept for report in self.reports
                     for edge, _ in report.conflicts]
        self.assertIn("penguen", conflicts)
        self.assertIn("balina", conflicts)
        self.assertIn("bebek", conflicts)

    def test_nothing_contradictory_was_written(self):
        reasoning = Reasoning(self.memory)
        self.assertTrue(reasoning.can_do("penguen", "uçmak")[0])  # the rule stands
        self.assertIsNone(self.memory.direct("penguen", "cannot", "uçmak"))

    def test_it_formed_its_own_rules(self):
        self.assertGreater(len(self.formed), 10)
        inferred = [e for e in self.memory.edges if e.source == INFERRED]
        self.assertEqual(len(inferred), len(self.formed))

    def test_it_answers_across_several_levels_of_hierarchy(self):
        session = Session(self.path)
        answer = session.respond("kaplumbağa yer mi")
        self.assertTrue(answer.startswith("evet"))
        self.assertIn("canlı", answer)      # kaplumbağa > sürüngen > hayvan > canlı

    def test_derived_answers_outnumber_stated_ones(self):
        """The payoff: most of what it can answer, nobody wrote down."""
        reasoning = Reasoning(self.memory)
        stated = derived = 0
        for concept in self.memory.concepts():
            for action in self.memory.actions():
                if reasoning.can_do(concept, action)[0] is None:
                    continue
                if (self.memory.direct(concept, "can", action)
                        or self.memory.direct(concept, "cannot", action)):
                    stated += 1
                else:
                    derived += 1
        self.assertGreater(derived, stated)


class TestCorrectingAnOverGeneralisation(unittest.TestCase):
    """Induction over-reaches — ants are insects, and insects were said to fly.

    What matters is that the answer admits whose guess it is, and that one
    sentence fixes it for good.
    """

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(tempfile.mkdtemp(), "memory.json")
        memory = Memory()
        distill_file(os.path.join(PACKS, "tr-cekirdek.txt"), memory, "claude")
        generalise(memory)
        memory.save(cls.path)

    def setUp(self):
        self.session = Session(self.path)

    def test_an_inherited_guess_says_it_is_a_guess(self):
        answer = self.session.respond("karınca uçar mı")
        self.assertIn("kendi çıkarımım", answer)

    def test_one_sentence_corrects_it_permanently(self):
        self.session.respond("karınca uçar mı")
        correction = self.session.respond("karınca uçamaz")
        self.assertIn("çıkarımla varsaymıştım", correction)
        self.session.save()

        later = Session(self.session.path)
        self.assertTrue(later.respond("karınca uçar mı").startswith("hayır"))
        self.assertTrue(later.respond("arı uçar mı").startswith("evet"))


class TestTypeQuestions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(tempfile.mkdtemp(), "memory.json")
        memory = Memory()
        distill_file(os.path.join(PACKS, "tr-cekirdek-2.txt"), memory, "claude")
        memory.save(cls.path)

    def setUp(self):
        self.session = Session(self.path)

    def test_yes(self):
        self.assertTrue(self.session.respond("kalp bir organ mı").startswith("evet"))

    def test_no_from_a_stated_denial(self):
        self.assertIn("değildir", self.session.respond("kalp bir canlı mı"))

    def test_unknown_concept_is_admitted(self):
        self.assertIn("bilmiyorum", self.session.respond("zürafa bir hayvan mı"))


if __name__ == "__main__":
    unittest.main()
