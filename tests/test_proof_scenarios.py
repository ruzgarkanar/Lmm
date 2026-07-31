"""Proof scenarios: the behaviours that make LMM a different category from an LLM.

These are the claims of the design doc, executable. If one of these breaks, the
idea is broken — not just the code.
"""
import os
import tempfile
import time
import unittest

from lmm.cli import Session


class TestProofScenarios(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "memory.json")

    def _session(self):
        return Session(self.path)

    def test_1_permanence_survives_restart(self):
        """Pillar 1: learning is instant and permanent — no retraining."""
        first = self._session()
        first.respond("penguen bir kuştur")
        first.save()

        second = self._session()          # a genuinely fresh set of organs
        answer = second.respond("penguen nedir")
        self.assertIn("kuş", answer)
        self.assertNotIn("bilmiyorum", answer)   # it does not ask twice

    def test_2_epistemic_honesty_never_invents(self):
        """Pillar 2: with an empty memory there is no path to an answer."""
        session = self._session()
        for question in ["penguen nedir", "kedi koşar mı", "balık yüzer mi"]:
            self.assertIn("bilmiyorum", session.respond(question))

    def test_3_penguin_test_conflict_then_exception(self):
        """Pillar 4: inheritance, noticing a clash, and exception handling."""
        session = self._session()
        session.respond("kuşlar uçar")
        session.respond("penguen bir kuştur")
        self.assertTrue(session.respond("penguen uçar mı").startswith("evet"))

        clash = session.respond("penguen uçamaz")   # contradicts what it inferred
        self.assertIn("çelişki", clash)
        self.assertIn("penguen uçamaz", clash)      # it restates the claim plainly

        session.respond("evet")                     # teacher stands behind it
        self.assertTrue(session.respond("penguen uçar mı").startswith("hayır"))

        session.respond("serçe bir kuştur")         # the sibling is untouched
        self.assertTrue(session.respond("serçe uçar mı").startswith("evet"))

    def test_4_conflict_declined_leaves_memory_untouched(self):
        session = self._session()
        session.respond("kuşlar uçar")
        session.respond("penguen bir kuştur")
        session.respond("penguen uçamaz")
        self.assertIn("öğrenmedim", session.respond("hayır"))
        self.assertTrue(session.respond("penguen uçar mı").startswith("evet"))

    def test_5_answers_cite_their_source(self):
        session = self._session()
        session.respond("kedi bir hayvandır")
        self.assertIn("sen", session.respond("kedi nedir"))

    def test_6_unparsable_input_is_admitted_not_guessed(self):
        session = self._session()
        self.assertIn("anlamadım", session.respond("florp glorp zzz qqq"))

    def test_7_runs_instantly_on_ordinary_hardware(self):
        session = self._session()
        session.respond("kuşlar uçar")
        session.respond("penguen bir kuştur")
        started = time.time()
        for _ in range(100):
            session.respond("penguen uçar mı")
        self.assertLess(time.time() - started, 1.0)   # pure Python, no GPU


if __name__ == "__main__":
    unittest.main()
