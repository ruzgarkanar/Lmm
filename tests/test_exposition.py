"""Exposition: paragraphs composed from the graph, not sampled from a model."""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT, HAS_PROPERTY, INFERRED
from lmm.reasoning import Reasoning
from lmm.exposition import Exposition
from lmm.phrasing import predicate, capitalize, clitic_da
from lmm.cli import Session


class TestPhrasingPieces(unittest.TestCase):
    def test_predicates_drop_the_subject(self):
        self.assertEqual(predicate(CAN, "uçmak"), "uçar")
        self.assertEqual(predicate(CANNOT, "uçmak"), "uçamaz")
        self.assertEqual(predicate(HAS_PROPERTY, "tüylü"), "tüylüdür")
        self.assertEqual(predicate(IS_A, "kuş"), "bir kuştur")

    def test_turkish_capitalisation(self):
        self.assertEqual(capitalize("içer"), "İçer")     # not "Icer"
        self.assertEqual(capitalize("kuş"), "Kuş")

    def test_the_da_clitic_harmonises_two_ways(self):
        self.assertEqual(clitic_da("kuş"), "da")
        self.assertEqual(clitic_da("memeli"), "de")


class TestComposition(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        for edge in [Edge("kuş", IS_A, "hayvan", source="sen"),
                     Edge("kuş", CAN, "uçmak", source="sen"),
                     Edge("kuş", HAS_PROPERTY, "tüylü", source="sen"),
                     Edge("penguen", IS_A, "kuş", source="sen"),
                     Edge("penguen", CANNOT, "uçmak", source="sen"),
                     Edge("penguen", CAN, "yüzmek", source="sen")]:
            self.memory.write(edge)
        self.exposition = Exposition(self.memory, Reasoning(self.memory))

    def test_it_opens_with_what_the_thing_is(self):
        paragraph = self.exposition.describe("penguen")
        self.assertTrue(paragraph.startswith("Penguen bir kuştur"))
        self.assertIn("kuş da bir hayvandır", paragraph)

    def test_an_exception_gets_the_contrast_it_deserves(self):
        paragraph = self.exposition.describe("penguen")
        self.assertIn("kuş uçar ama penguen uçamaz", paragraph.lower())

    def test_an_exception_is_not_contradicted_a_sentence_later(self):
        """It once said 'penguen uçamaz' and then 'kuş olduğu için uçar'."""
        paragraph = self.exposition.describe("penguen")
        after = paragraph.split("ama penguen uçamaz.")[1]
        self.assertNotIn("uçar", after)

    def test_inherited_traits_are_named_as_inherited(self):
        self.assertIn("Kuş olduğu için tüylüdür",
                      self.exposition.describe("penguen"))

    def test_its_own_facts_come_last(self):
        self.assertIn("Ayrıca yüzer", self.exposition.describe("penguen"))

    def test_a_guess_is_flagged_inside_the_paragraph(self):
        self.memory.write(Edge("penguen", HAS_PROPERTY, "hızlı",
                               source=INFERRED, confidence=0.45))
        self.assertIn("(sanırım)", self.exposition.describe("penguen"))

    def test_an_unknown_concept_is_admitted(self):
        self.assertIn("bilmiyorum", self.exposition.describe("zürafa"))

    def test_nothing_in_the_paragraph_is_unsupported(self):
        """Every clause traces to a fact — that is the whole difference."""
        paragraph = self.exposition.describe("penguen").lower()
        for invented in ("koşar", "yürür", "büyüktür", "siyahtır"):
            self.assertNotIn(invented, paragraph)


class TestInConversation(unittest.TestCase):
    def test_anlat_composes_a_paragraph(self):
        path = os.path.join(tempfile.mkdtemp(), "memory.json")
        session = Session(path)
        session.respond("kuşlar uçar")
        session.respond("penguen bir kuştur")
        session.respond("penguen yüzer")
        paragraph = session.respond("penguen anlat")
        self.assertIn("Penguen bir kuştur", paragraph)
        self.assertIn("yüzer", paragraph)


if __name__ == "__main__":
    unittest.main()
