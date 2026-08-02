"""Reading prose written for people, by restating it into shapes we can hold.

The model is faked here — these check the loop, the audit and the corpus, not
the network.
"""
import json
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT
from lmm.trust import TEACHER
from lmm.comprehend import comprehend, chunks, restate

PROSE = ("Bankacılıkta kullanılan büyük dil modelleri, kendi ağırlıklarında "
         "tuttukları bilgiyi denetlenebilir biçimde gösteremez. Bu nedenle "
         "kurumsal mimaride ayrı bir bellek katmanı önerilir.")


def steady(text):
    return lambda passages, verbs: text


def wavering(*texts):
    answers = list(texts)

    def asker(passages, verbs):
        return answers.pop(0) if answers else texts[-1]
    return asker


class TestChunking(unittest.TestCase):
    def test_prose_is_split_into_passages(self):
        found = list(chunks("Bir. İki. Üç. Dört.", size=2))
        self.assertEqual(len(found), 2)
        self.assertIn("Bir", found[0])
        self.assertIn("Üç", found[1])


class TestRestating(unittest.TestCase):
    def test_agreement_is_required(self):
        self.assertEqual(restate(PROSE, asker=steady("model bir yazılımdır.")),
                         "model bir yazılımdır.")

    def test_a_model_that_wavers_teaches_nothing(self):
        self.assertEqual(restate(PROSE, asker=wavering("model bir yazılımdır.",
                                                       "model bir donanımdır.")),
                         "")


class TestComprehending(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()

    def test_the_document_stays_the_source_not_the_model(self):
        comprehend(self.memory, PROSE, "bankacilik.txt",
                   asker=steady("model bir yazılımdır."))
        edge = self.memory.direct("model", IS_A, "yazılım")
        self.assertIsNotNone(edge)
        self.assertEqual(edge.source, "bankacilik.txt")

    def test_restated_sentences_face_the_same_audit(self):
        """A restatement that fights what is known is refused like any source.

        The verbs here are core ones on purpose: a test that depends on a word
        being unknown is at the mercy of whatever another test taught the shared
        lexicon.
        """
        # Stated about the model itself, so the restatement is a head-on
        # contradiction rather than an exception to what software can do.
        self.memory.write(Edge("model", IS_A, "yazılım", source=TEACHER))
        self.memory.write(Edge("model", CAN, "okumak", source=TEACHER))
        _, learned, refused, _ = comprehend(
            self.memory, PROSE, "belge.txt", asker=steady("model okuyamaz."))
        self.assertEqual(refused, 1)
        self.assertIsNone(self.memory.direct("model", CANNOT, "okumak"))

    def test_vocabulary_the_restatement_needs_is_taken_too(self):
        comprehend(self.memory, PROSE, "belge.txt",
                   asker=steady("kelime: denetlemek = denetler / denetleyemez\n"
                                "kurum denetler."))
        self.assertIsNotNone(self.memory.direct("kurum", CAN, "denetlemek"))
        self.assertIn({"infinitive": "denetlemek", "positive": "denetler",
                       "negative": "denetleyemez"}, self.memory.vocabulary)

    def test_every_restatement_is_recorded_for_later(self):
        """The pairs are the training data for doing this without a model."""
        path = os.path.join(tempfile.mkdtemp(), "cift.json")
        comprehend(self.memory, PROSE, "belge.txt",
                   asker=steady("model bir yazılımdır."), corpus=path)
        with open(path, encoding="utf-8") as f:
            pairs = json.load(f)
        self.assertEqual(len(pairs), 1)
        self.assertIn("Bankacılıkta", pairs[0]["düzyazı"])
        self.assertIn("model bir yazılımdır", pairs[0]["sade"])

    def test_the_corpus_accumulates_across_documents(self):
        path = os.path.join(tempfile.mkdtemp(), "cift.json")
        for _ in range(2):
            comprehend(self.memory, PROSE, "belge.txt",
                       asker=steady("model bir yazılımdır."), corpus=path)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)), 2)


if __name__ == "__main__":
    unittest.main()
