"""Trust: which source wins when two of them disagree.

A memory fed by people, documents, language models and its own generalisations
needs a ranking, or every disagreement becomes a coin toss.
"""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT
from lmm.trust import (level, outranks, confidence_for, distilled_source,
                       HUMAN, DOCUMENT, DISTILLED, INFERRED, TEACHER, INFERENCE)
from lmm.distill import distill_text, split_words
from lmm.cli import Session


class TestRanking(unittest.TestCase):
    def test_sources_are_ranked(self):
        self.assertEqual(level(TEACHER), HUMAN)
        self.assertEqual(level("hayvanlar.txt"), DOCUMENT)
        self.assertEqual(level(distilled_source("claude")), DISTILLED)
        self.assertEqual(level(INFERENCE), INFERRED)

    def test_a_person_outranks_everything(self):
        for weaker in ("hayvanlar.txt", distilled_source("claude"), INFERENCE):
            self.assertTrue(outranks(TEACHER, weaker))

    def test_equals_do_not_outrank_each_other(self):
        self.assertFalse(outranks(distilled_source("a"), distilled_source("b")))
        self.assertFalse(outranks(TEACHER, TEACHER))

    def test_weaker_sources_carry_less_confidence(self):
        self.assertGreater(confidence_for(TEACHER),
                           confidence_for(distilled_source("claude")))
        self.assertGreater(confidence_for(distilled_source("claude")),
                           confidence_for(INFERENCE))


class TestCorrectionByRank(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "memory.json")

    def test_a_person_overrides_a_model_without_an_argument(self):
        memory = Memory()
        distill_text("Kuşlar uçar. Penguen bir kuştur.", memory, "claude")
        memory.save(self.path)

        session = Session(self.path)
        correction = session.respond("penguen uçamaz")
        self.assertIn("bir dil modelinden almıştım", correction)
        self.assertIn("senin sözünü üstün tutuyorum", correction)
        self.assertTrue(session.respond("penguen uçar mı").startswith("hayır"))

    def test_two_equal_sources_are_not_settled_by_guessing(self):
        """A model contradicting a model has to be raised, not resolved."""
        memory = Memory()
        report = distill_text("Kuşlar uçar. Penguen bir kuştur. Penguen uçamaz.",
                              memory, "claude")
        self.assertEqual(len(report.conflicts), 1)
        self.assertIsNone(memory.direct("penguen", CANNOT, "uçmak"))

    def test_taught_knowledge_is_not_overridden_by_a_model(self):
        memory = Memory()
        memory.write(Edge("penguen", IS_A, "kuş", source=TEACHER))
        memory.write(Edge("penguen", CANNOT, "uçmak", source=TEACHER))
        report = distill_text("Penguen uçar.", memory, "claude")
        self.assertEqual(len(report.conflicts), 1)      # the model is refused
        self.assertIsNone(memory.direct("penguen", CAN, "uçmak"))


class TestDistilling(unittest.TestCase):
    def test_word_declarations_are_separated_from_sentences(self):
        words, rest = split_words("kelime: koşmak = koşar / koşamaz\nKedi koşar.")
        self.assertEqual(words, [("koşmak", "koşar", "koşamaz")])
        self.assertNotIn("kelime:", rest)

    def test_malformed_declarations_are_ignored(self):
        words, _ = split_words("kelime: bozuk satır\nKedi koşar.")
        self.assertEqual(words, [])

    def test_distilled_facts_are_marked_as_machine_sourced(self):
        memory = Memory()
        distill_text("Penguen bir kuştur.", memory, "claude")
        edge = memory.direct("penguen", IS_A, "kuş")
        self.assertEqual(edge.source, "llm:claude")
        self.assertLess(edge.confidence, confidence_for(TEACHER))

    def test_a_pack_can_teach_the_words_its_facts_need(self):
        memory = Memory()
        distill_text("kelime: hesaplamak = hesaplar / hesaplayamaz\n"
                     "Bilgisayar hesaplar.", memory, "claude")
        self.assertIsNotNone(memory.direct("bilgisayar", CAN, "hesaplamak"))

    def test_answers_name_the_model_they_came_from(self):
        path = os.path.join(tempfile.mkdtemp(), "memory.json")
        memory = Memory()
        distill_text("Elma tatlıdır.", memory, "claude")
        memory.save(path)
        answer = Session(path).respond("elma tatlı mı")
        self.assertIn("bir dil modelinden", answer)


class TestTurkishCasing(unittest.TestCase):
    def test_capital_i_is_lowercased_the_turkish_way(self):
        """Python maps İ to "i" plus a combining dot, quietly forking the word.

        Every sentence starting with İ was learned under a concept nobody could
        ask about, and it hid a real contradiction in a distilled pack.
        """
        memory = Memory()
        distill_text("İnsan bir canlıdır. insanlar konuşur.", memory, "claude")
        self.assertIsNotNone(memory.direct("insan", IS_A, "canlı"))
        self.assertIsNotNone(memory.direct("insan", CAN, "konuşmak"))


if __name__ == "__main__":
    unittest.main()
