"""The pipeline end to end: sources in, a model file out.

Same stages an LLM goes through — collect, clean, audit, learn, generalise,
evaluate, ship. Different mechanism in the middle, and no gradient anywhere.
"""
import os
import shutil
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, INFERRED
from lmm.pack import export_pack
from lmm.train import collect, train, evaluate
from lmm.cli import Session

PACKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "packs")


class TestCollecting(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        for name in ("belge.txt", "tr-cekirdek-deneme.txt", "paket.json",
                     "okunmaz.md"):
            open(os.path.join(self.directory, name), "w").close()

    def test_sources_are_classified_and_ordered(self):
        found = collect(self.directory)
        kinds = dict((os.path.basename(path), kind) for kind, path in found)
        self.assertEqual(kinds["belge.txt"], "doküman")
        self.assertEqual(kinds["tr-cekirdek-deneme.txt"], "damıtma")
        self.assertEqual(kinds["paket.json"], "paket")
        self.assertNotIn("okunmaz.md", kinds)

    def test_the_order_is_stable_so_a_run_repeats(self):
        self.assertEqual(collect(self.directory), collect(self.directory))


class TestARun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.mkdtemp()
        for name in ("hayvanlar.txt", "tr-cekirdek.txt"):
            shutil.copy(os.path.join(PACKS, name), cls.directory)
        source = Memory()
        source.learn_word("kazmak", "kazar", "kazamaz")
        source.write(Edge("köstebek", IS_A, "hayvan", source="uzman"))
        source.write(Edge("köstebek", CAN, "kazmak", source="uzman"))
        export_pack(source, os.path.join(cls.directory, "ek.json"), name="ek")

        cls.model = os.path.join(cls.directory, "model.lmm")
        cls.memory, cls.report = train(cls.directory, cls.model)

    def test_every_kind_of_source_was_absorbed(self):
        self.assertGreater(self.report.learned, 100)
        self.assertGreater(self.report.words, 0)
        self.assertIsNotNone(self.memory.direct("köstebek", CAN, "kazmak"))

    def test_contradictions_were_refused_and_attributed(self):
        self.assertTrue(self.report.refused)
        for source, concept, explanation in self.report.refused:
            self.assertTrue(source.endswith((".txt", ".json")))
            self.assertTrue(explanation)

    def test_unparsable_sentences_were_skipped_not_guessed(self):
        self.assertTrue(any("kalıpta" in sentence
                            for _, sentence in self.report.skipped))

    def test_it_formed_rules_of_its_own_and_marked_them(self):
        self.assertTrue(self.report.rules)
        inferred = [e for e in self.memory.edges if e.source == INFERRED]
        self.assertEqual(len(inferred), len(self.report.rules))

    def test_most_of_what_it_can_answer_was_never_written(self):
        stated, derived, guessed = evaluate(self.memory)
        self.assertGreater(derived, stated)

    def test_the_model_file_is_a_working_system(self):
        session = Session(self.model)
        self.assertTrue(session.respond("kartal uçar mı").startswith("evet"))
        self.assertIn("bilmiyorum", session.respond("zürafa nedir"))

    def test_a_second_run_resumes_rather_than_starting_over(self):
        before = len(self.memory.edges)
        memory, _ = train(self.directory, self.model)
        self.assertGreaterEqual(len(memory.edges), before)


if __name__ == "__main__":
    unittest.main()
