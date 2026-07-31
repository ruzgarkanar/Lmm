"""Induction: knowledge nobody stated.

Chaining given facts is not learning. This is the part that answers about things
no one wrote down — done in the open, so it can be shown and corrected.
"""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT, HAS_PROPERTY, \
    LACKS_PROPERTY, INFERRED
from lmm.reasoning import Reasoning
from lmm.induction import Induction
from lmm.cli import Session


class TestProposing(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.reasoning = Reasoning(self.memory)
        self.induction = Induction(self.memory, self.reasoning)

    def _bird(self, name, flies=None):
        self.memory.write(Edge(name, IS_A, "kuş", source="sen"))
        if flies is not None:
            self.memory.write(Edge(name, CAN if flies else CANNOT, "uçmak",
                                   source="sen"))

    def test_two_agreeing_children_suggest_a_rule(self):
        self._bird("serçe", flies=True)
        self._bird("kartal", flies=True)
        hypothesis = self.induction.propose()
        self.assertEqual((hypothesis.concept, hypothesis.relation,
                          hypothesis.target), ("kuş", CAN, "uçmak"))
        self.assertEqual(sorted(hypothesis.examples), ["kartal", "serçe"])

    def test_one_example_is_not_a_pattern(self):
        self._bird("serçe", flies=True)
        self.assertIsNone(self.induction.propose())

    def test_a_counterexample_blocks_the_rule(self):
        self._bird("serçe", flies=True)
        self._bird("kartal", flies=True)
        self._bird("penguen", flies=False)
        self.assertIsNone(self.induction.propose())

    def test_it_generalises_in_the_negative_too(self):
        self.memory.write(Edge("kedi", IS_A, "hayvan", source="sen"))
        self.memory.write(Edge("köpek", IS_A, "hayvan", source="sen"))
        self.memory.write(Edge("kedi", CANNOT, "uçmak", source="sen"))
        self.memory.write(Edge("köpek", CANNOT, "uçmak", source="sen"))
        hypothesis = self.induction.propose()
        self.assertEqual((hypothesis.concept, hypothesis.relation), ("hayvan",
                                                                     CANNOT))

    def test_a_settled_type_is_left_alone(self):
        self._bird("serçe", flies=True)
        self._bird("kartal", flies=True)
        self.memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        self.assertIsNone(self.induction.propose())

    def test_properties_generalise_as_well(self):
        self.memory.write(Edge("serçe", IS_A, "kuş", source="sen"))
        self.memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        self.memory.write(Edge("serçe", HAS_PROPERTY, "tüylü", source="sen"))
        self.memory.write(Edge("kartal", HAS_PROPERTY, "tüylü", source="sen"))
        hypothesis = self.induction.propose()
        self.assertEqual((hypothesis.relation, hypothesis.target),
                         (HAS_PROPERTY, "tüylü"))

    def test_a_learned_hypothesis_is_marked_as_its_own(self):
        self._bird("serçe", flies=True)
        self._bird("kartal", flies=True)
        edge = self.induction.learn(self.induction.propose())
        self.assertEqual(edge.source, INFERRED)
        self.assertLess(edge.confidence, 0.5)   # low enough that answers hedge


class TestInductionInConversation(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))

    def _teach_two_birds(self):
        self.session.respond("serçe bir kuştur")
        self.session.respond("kartal bir kuştur")
        self.session.respond("serçe uçar")
        return self.session.respond("kartal uçar")

    def test_it_says_what_it_worked_out_and_why(self):
        reply = self._teach_two_birds()
        self.assertIn("fark ettim", reply)
        self.assertIn("serçe", reply)
        self.assertIn("kartal", reply)      # the examples that convinced it
        self.assertIn("kuş uçar", reply)

    def test_the_rule_reaches_a_concept_it_was_never_told_about(self):
        """The thing chaining alone could never do."""
        self._teach_two_birds()
        self.session.respond("güvercin bir kuştur")
        answer = self.session.respond("güvercin uçar mı")
        self.assertTrue(answer.startswith("evet"))

    def test_answers_from_its_own_guess_say_so(self):
        self._teach_two_birds()
        self.assertIn("kendi çıkarımım", self.session.respond("kuşlar uçar mı"))

    def test_a_teacher_outranks_its_own_guess_without_an_argument(self):
        self._teach_two_birds()
        self.session.respond("penguen bir kuştur")
        reply = self.session.respond("penguen uçamaz")
        self.assertIn("çıkarımla varsaymıştım", reply)
        self.assertNotIn("öğreneyim mi", reply)     # it does not push back
        self.assertTrue(self.session.respond("penguen uçar mı").startswith("hayır"))

    def test_taught_knowledge_is_still_defended(self):
        """Yielding applies to its own guesses only, not to what it was told."""
        self.session.respond("kuşlar uçar")         # a teacher said this
        self.session.respond("penguen bir kuştur")
        self.assertIn("çelişki", self.session.respond("penguen uçamaz"))


if __name__ == "__main__":
    unittest.main()
