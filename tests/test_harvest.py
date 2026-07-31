"""Harvesting: curiosity asks, a model answers, the audit decides.

The model is faked here — these check the loop, not the network.
"""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT
from lmm.trust import TEACHER
from lmm.reasoning import Reasoning
from lmm.harvest import harvest, agreed, open_questions


def steady(text):
    """A model that says the same thing every time."""
    return lambda questions, verbs: text


def wavering(*texts):
    """A model that changes its mind between samples."""
    answers = list(texts)

    def asker(questions, verbs):
        return answers.pop(0) if answers else texts[-1]
    return asker


class TestAgreement(unittest.TestCase):
    def test_only_what_every_sample_said_survives(self):
        kept = agreed(["kedi koşar.\nkedi uçar.", "kedi koşar.\nkedi uçamaz."])
        self.assertEqual(kept, "kedi koşar.")

    def test_order_of_the_first_sample_is_kept(self):
        kept = agreed(["a\nb\nc", "c\nb\na"])
        self.assertEqual(kept.splitlines(), ["a", "b", "c"])

    def test_total_disagreement_keeps_nothing(self):
        self.assertEqual(agreed(["kedi uçar.", "kedi uçamaz."]), "")


class TestHarvestLoop(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.memory.write(Edge("kuş", CAN, "uçmak", source=TEACHER))

    def test_the_questions_come_from_the_systems_own_gaps(self):
        questions = open_questions(self.memory, Reasoning(self.memory), 3)
        self.assertIn("kuş nedir?", questions)

    def test_consistent_answers_are_kept_and_marked(self):
        _, report = harvest(self.memory, count=2,
                            asker=steady("kuş bir hayvandır."),
                            model_name="sahte")
        self.assertEqual(len(report.learned), 1)
        self.assertEqual(self.memory.direct("kuş", IS_A, "hayvan").source,
                         "llm:sahte")

    def test_a_model_that_contradicts_itself_teaches_nothing(self):
        """One sample said one thing, the next said the opposite."""
        _, report = harvest(self.memory, count=2,
                            asker=wavering("kuş bir hayvandır.",
                                           "kuş bir nesnedir."),
                            model_name="sahte")
        self.assertEqual(len(report.learned), 0)
        self.assertEqual(self.memory.query("kuş", IS_A), [])

    def test_an_answer_that_fights_what_is_known_is_refused(self):
        self.memory.write(Edge("penguen", IS_A, "kuş", source=TEACHER))
        _, report = harvest(self.memory, count=4,
                            asker=steady("penguen uçamaz."),
                            model_name="sahte")
        self.assertEqual(len(report.conflicts), 1)
        self.assertIsNone(self.memory.direct("penguen", CANNOT, "uçmak"))

    def test_a_person_still_outranks_the_harvest(self):
        harvest(self.memory, count=2, asker=steady("kuş bir hayvandır."),
                model_name="sahte")
        from lmm.cli import Session
        path = os.path.join(tempfile.mkdtemp(), "memory.json")
        self.memory.save(path)
        session = Session(path)
        reply = session.respond("kuş bir hayvan değildir")
        self.assertIn("bir dil modelinden almıştım", reply)
        self.assertIn("senin sözünü üstün tutuyorum", reply)

    def test_nothing_left_to_wonder_about_ends_the_round(self):
        empty = Memory()
        questions, report = harvest(empty, count=4, asker=steady("x"),
                                    model_name="sahte")
        self.assertEqual(questions, [])
        self.assertIsNone(report)


if __name__ == "__main__":
    unittest.main()
