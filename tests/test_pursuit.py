"""Pursuit: holding a question and working out what would answer it.

An LLM agent guesses at which tool to call. This plan is read off the memory:
"a penguin is a bird, so tell me about birds" is a step the hierarchy dictates.
"""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN
from lmm.reasoning import Reasoning
from lmm.pursuit import Pursuit
from lmm.intuition import Intuition
from lmm.cli import Session


class TestPlanning(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.pursuit = Pursuit(self.memory, Reasoning(self.memory))
        self.ask = Intuition().understand

    def test_without_a_hierarchy_it_asks_what_the_thing_is(self):
        step = self.pursuit.next_step(self.ask("penguen uçar mı"))
        self.assertEqual(step.text, "penguen nedir?")

    def test_with_a_hierarchy_it_climbs_to_the_parent(self):
        self.memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        step = self.pursuit.next_step(self.ask("penguen uçar mı"))
        self.assertEqual(step.text, "kuş uçar mı?")

    def test_it_climbs_past_a_parent_that_does_not_help(self):
        self.memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        self.memory.write(Edge("kuş", IS_A, "hayvan", source="sen"))
        self.memory.write(Edge("kuş", CAN, "yüzmek", source="sen"))  # wrong ability
        step = self.pursuit.next_step(self.ask("penguen koşar mı"))
        self.assertEqual(step.text, "kuş koşar mı?")

    def test_an_answerable_question_needs_no_plan(self):
        self.memory.write(Edge("penguen", CAN, "uçmak", source="sen"))
        goal = self.ask("penguen uçar mı")
        self.assertTrue(self.pursuit.resolved(goal))
        self.assertIsNone(self.pursuit.next_step(goal))

    def test_property_goals_plan_the_same_way(self):
        self.memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        step = self.pursuit.next_step(self.ask("penguen tüylü mü"))
        self.assertEqual(step.text, "kuş tüylü mü?")


class TestPursuitInConversation(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))

    def test_the_full_loop_ends_at_the_original_question(self):
        opening = self.session.respond("penguen uçar mı")
        self.assertIn("penguen nedir?", opening)

        middle = self.session.respond("penguen bir kuştur")
        self.assertIn("kuş uçar mı?", middle)          # it knows what to ask next

        end = self.session.respond("kuşlar uçar")
        self.assertIn("ilk soruna dönebilirim", end)   # unprompted return
        self.assertIn("evet", end)
        self.assertIn("penguen bir kuş", end)          # with the chain that earned it

    def test_a_goal_outranks_idle_curiosity(self):
        reply = self.session.respond("penguen uçar mı")
        self.assertNotIn("bu arada", reply)
        follow = self.session.respond("penguen bir kuştur")
        self.assertNotIn("bu arada", follow)

    def test_a_new_question_replaces_the_goal(self):
        self.session.respond("penguen uçar mı")
        self.session.respond("kedi yüzer mi")
        reply = self.session.respond("kedi bir hayvandır")
        self.assertIn("hayvan yüzer mi?", reply)       # pursuing the newer goal

    def test_an_answered_question_clears_the_goal(self):
        self.session.respond("penguen uçar mı")
        self.session.respond("penguen uçamaz")         # answers it outright
        reply = self.session.respond("kedi bir hayvandır")
        self.assertNotIn("ilk soruna", reply)

    def test_a_step_is_not_repeated(self):
        self.session.respond("penguen uçar mı")
        self.session.respond("penguen bir kuştur")     # asks "kuş uçar mı?"
        reply = self.session.respond("serçe bir kuştur")
        self.assertNotIn("kuş uçar mı?", reply)


if __name__ == "__main__":
    unittest.main()
