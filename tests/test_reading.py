"""Reading a document: the same learning loop with a different source."""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT
from lmm.reading import (read_text, read_file, sentences, NOT_UNDERSTOOD,
                         A_QUESTION)
from lmm.cli import Session


class TestSentenceSplitting(unittest.TestCase):
    def test_splits_on_endings_and_line_breaks(self):
        self.assertEqual(sentences("Kuşlar uçar. Penguen bir kuştur!\nKedi koşar"),
                         ["Kuşlar uçar", "Penguen bir kuştur", "Kedi koşar"])

    def test_ignores_empty_stretches(self):
        self.assertEqual(sentences("\n\n  \n Kuşlar uçar.  \n\n"), ["Kuşlar uçar"])


class TestReading(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()

    def test_learns_and_records_the_document_as_the_source(self):
        report = read_text("Kuşlar uçar. Penguen bir kuştur.", self.memory,
                           "kuslar.txt")
        self.assertEqual(len(report.learned), 2)
        self.assertEqual(self.memory.direct("penguen", IS_A, "kuş").source,
                         "kuslar.txt")

    def test_skips_what_it_cannot_parse_instead_of_guessing(self):
        report = read_text("Kuşlar uçar. Bu cümle hiçbir kalıba uymaz elbette.",
                           self.memory, "karisik.txt")
        self.assertEqual(len(report.learned), 1)
        self.assertEqual([reason for _, reason in report.skipped], [NOT_UNDERSTOOD])

    def test_questions_in_a_document_are_not_lessons(self):
        report = read_text("Penguen uçar mı? Kuşlar uçar.", self.memory, "s.txt")
        self.assertEqual(len(report.learned), 1)
        self.assertIn(A_QUESTION, [reason for _, reason in report.skipped])

    def test_repeated_facts_reinforce(self):
        report = read_text("Kuşlar uçar. Kuşlar uçar.", self.memory, "t.txt")
        self.assertEqual(len(report.learned), 1)
        self.assertEqual(len(report.reinforced), 1)

    def test_an_exception_a_document_states_is_taken(self):
        """"penguen uçamaz" is not a contradiction — it is the exception.

        A document cannot be asked "are you sure?", and this used to mean every
        pack could say "penguen uçamaz" while the trained model still answered
        that penguins fly. Nothing is guessed by taking it: "kuşlar uçar" stays
        exactly as true, and only the penguin steps out from under it.
        """
        report = read_text("Kuşlar uçar. Penguen bir kuştur. Penguen uçamaz.",
                           self.memory, "hayvanlar.txt")
        self.assertEqual(len(report.conflicts), 0)
        edge = self.memory.direct("penguen", CANNOT, "uçmak")
        self.assertIsNotNone(edge)
        self.assertTrue(edge.is_exception)
        self.assertIsNotNone(self.memory.direct("kuş", CAN, "uçmak"))

    def test_a_contradiction_is_reported_and_refused(self):
        """A clash with what was said about this very concept is a real one."""
        report = read_text("Penguen uçar. Penguen uçamaz.",
                           self.memory, "celiskili.txt")
        self.assertEqual(len(report.conflicts), 1)
        self.assertIsNone(self.memory.direct("penguen", CANNOT, "uçmak"))
        edge, explanation = report.conflicts[0]
        self.assertEqual(edge.concept, "penguen")
        self.assertIn("penguen", explanation)

    def test_reading_a_file(self):
        path = os.path.join(tempfile.mkdtemp(), "belge.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("Kediler koşar. Kedi bir hayvandır.")
        report = read_file(path, self.memory)
        self.assertEqual(report.source, "belge.txt")
        self.assertEqual(len(report.learned), 2)


class TestReadingThenTalking(unittest.TestCase):
    def test_after_reading_it_answers_instead_of_asking(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "memory.json")
        document = os.path.join(directory, "hayvanlar.txt")
        with open(document, "w", encoding="utf-8") as f:
            f.write("Kuşlar uçar. Kuşlar tüylüdür. Penguen bir kuştur. "
                    "Serçe bir kuştur.")

        memory = Memory()
        read_file(document, memory)
        memory.save(path)

        session = Session(path)
        answer = session.respond("penguen tüylü mü")
        self.assertTrue(answer.startswith("evet"))
        self.assertNotIn("bilmiyorum", answer)
        self.assertIn("serçe", session.respond("kimler uçar"))
        self.assertIn("hayvanlar.txt", session.respond("penguen nedir"))

    def test_a_conflict_the_document_raised_can_be_settled_in_conversation(self):
        """Read, report, then let a person decide — the loop closes.

        A person speaking directly outranks a document they handed over, so the
        correction is taken without an argument and the superseded source is
        named aloud.
        """
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "memory.json")
        memory = Memory()
        read_text("Kuşlar uçar. Penguen bir kuştur. Penguen uçamaz.", memory,
                  "hayvanlar.txt")
        memory.save(path)

        session = Session(path)
        self.assertTrue(session.respond("penguen uçar mı").startswith("hayır"))
        correction = session.respond("penguen uçar")
        self.assertIn("hayvanlar.txt", correction)
        self.assertIn("senin sözünü üstün tutuyorum", correction)
        self.assertTrue(session.respond("penguen uçar mı").startswith("evet"))


if __name__ == "__main__":
    unittest.main()
