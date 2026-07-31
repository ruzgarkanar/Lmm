"""The widened language: reverse questions, listing abilities, and "why".

The reasoning was always richer than the language could express. These are the
sentences that let a teacher reach it.
"""
import os
import tempfile
import unittest

from lmm.intuition import (Intuition, TEACH, ASK_WHO, ASK_ABILITIES, ASK_WHY,
                           Intent)
from lmm.memory import NOT_A, CAN, CANNOT
from lmm.cli import Session
from lmm.language import LanguageOrgan


class TestNewPatterns(unittest.TestCase):
    def setUp(self):
        self.intuition = Intuition()

    def test_who_question(self):
        intent = self.intuition.understand("kimler uçar?")
        self.assertEqual((intent.kind, intent.relation, intent.target),
                         (ASK_WHO, CAN, "uçmak"))

    def test_who_question_negative(self):
        intent = self.intuition.understand("kimler uçamaz?")
        self.assertEqual((intent.kind, intent.relation), (ASK_WHO, CANNOT))

    def test_abilities_question(self):
        intent = self.intuition.understand("penguen ne yapabilir?")
        self.assertEqual((intent.kind, intent.concept),
                         (ASK_ABILITIES, "penguen"))

    def test_why_question(self):
        intent = self.intuition.understand("penguen neden uçamaz?")
        self.assertEqual((intent.kind, intent.concept, intent.relation,
                          intent.target), (ASK_WHY, "penguen", CANNOT, "uçmak"))

    def test_negative_type_statement(self):
        intent = self.intuition.understand("penguen bir memeli değildir.")
        self.assertEqual((intent.kind, intent.concept, intent.relation,
                          intent.target), (TEACH, "penguen", NOT_A, "memeli"))

    def test_who_is_not_mistaken_for_a_lesson(self):
        """"kim uçar" has the same shape as "kuşlar uçar" — it must not teach."""
        self.assertEqual(self.intuition.understand("kim uçar?").kind, ASK_WHO)


class TestConversation(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))
        for lesson in ["kuşlar uçar", "penguen bir kuştur", "serçe bir kuştur",
                       "balıklar yüzer", "balık bir hayvandır"]:
            self.session.respond(lesson)
        self.session.respond("penguen uçamaz")
        self.session.respond("evet")            # confirm the exception

    def test_who_can_answers_from_inheritance(self):
        answer = self.session.respond("kimler uçar")
        self.assertIn("kuş", answer)
        self.assertIn("serçe", answer)
        self.assertNotIn("penguen", answer)     # the exception is respected

    def test_who_cannot(self):
        self.assertIn("penguen", self.session.respond("kimler uçamaz"))

    def test_unknown_action_is_admitted(self):
        self.assertIn("duymadım", self.session.respond("kimler konuşur"))

    def test_listing_abilities(self):
        answer = self.session.respond("penguen ne yapabilir")
        self.assertIn("uçamaz", answer)

    def test_abilities_of_an_unknown_concept(self):
        self.assertIn("öğrenmedim", self.session.respond("taş ne yapabilir"))

    def test_why_shows_the_reasoning_chain(self):
        answer = self.session.respond("serçe neden uçar")
        self.assertTrue(answer.startswith("çünkü"))
        self.assertIn("serçe bir kuş", answer)

    def test_why_corrects_a_false_premise(self):
        answer = self.session.respond("penguen neden uçar")
        self.assertIn("aslında", answer)
        self.assertIn("penguen uçamaz", answer)

    def test_negative_type_conflicts_with_the_hierarchy(self):
        answer = self.session.respond("penguen bir kuş değildir")
        self.assertIn("çelişki", answer)


class TestPluggableLanguage(unittest.TestCase):
    """The seam that lets someone swap in a different language layer."""

    class ShoutingOrgan(LanguageOrgan):
        """A toy organ: understands only "PENGUEN!" and nothing else."""

        def understand(self, sentence):
            if sentence.strip() == "PENGUEN!":
                return Intent(TEACH, concept="penguen", relation=CAN,
                              target="yüzmek", confidence=1.0)
            return Intent("UNKNOWN", confidence=0.0)

    def test_a_custom_organ_drives_the_same_organs(self):
        path = os.path.join(tempfile.mkdtemp(), "memory.json")
        session = Session(path, language=self.ShoutingOrgan())
        self.assertIn("öğrendim", session.respond("PENGUEN!"))
        session.save()

        # what it learned is ordinary knowledge, reachable through normal Turkish
        normal = Session(path)
        self.assertTrue(normal.respond("penguen yüzer mi").startswith("evet"))


if __name__ == "__main__":
    unittest.main()
