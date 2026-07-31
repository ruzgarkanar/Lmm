"""Properties: "kar beyazdır" knowledge, inherited and excepted like abilities.

Turkish draws the line for us: "penguen bir kuştur" says what something is,
"penguen siyahtır" says what it is like. The word "bir" is the whole difference,
and the reasoning machinery is shared.
"""
import os
import tempfile
import unittest

from lmm.memory import Memory, Edge, IS_A, HAS_PROPERTY, LACKS_PROPERTY
from lmm.reasoning import Reasoning
from lmm.intuition import Intuition, TEACH, ASK, ASK_PROPERTIES
from lmm.phrasing import property_clause, property_question, property_summary
from lmm.cli import Session


class TestPropertyPhrasing(unittest.TestCase):
    def test_clauses_obey_vowel_harmony(self):
        self.assertEqual(property_clause("kar", "beyaz"), "kar beyazdır")
        self.assertEqual(property_clause("veri", "gizli"), "veri gizlidir")
        self.assertEqual(property_clause("kar", "sıcak", False), "kar sıcak değildir")

    def test_questions_obey_vowel_harmony(self):
        self.assertEqual(property_question("kar", "beyaz"), "kar beyaz mı?")
        self.assertEqual(property_question("kuş", "tüylü"), "kuş tüylü mü?")

    def test_summary_separates_what_is_and_what_is_not(self):
        summary = property_summary("kar", [("beyaz", True), ("sıcak", False)])
        self.assertEqual(summary, "kar beyazdır ama sıcak değildir")


class TestPropertyParsing(unittest.TestCase):
    def setUp(self):
        self.intuition = Intuition()

    def test_bir_makes_it_a_type_not_a_property(self):
        typed = self.intuition.understand("penguen bir kuştur")
        self.assertEqual(typed.relation, IS_A)
        described = self.intuition.understand("penguen siyahtır")
        self.assertEqual((described.kind, described.concept, described.relation,
                          described.target), (TEACH, "penguen", HAS_PROPERTY,
                                              "siyah"))

    def test_negative_property(self):
        intent = self.intuition.understand("kar sıcak değildir")
        self.assertEqual((intent.relation, intent.target),
                         (LACKS_PROPERTY, "sıcak"))

    def test_property_question(self):
        intent = self.intuition.understand("kar beyaz mı?")
        self.assertEqual((intent.kind, intent.concept, intent.relation,
                          intent.target), (ASK, "kar", HAS_PROPERTY, "beyaz"))

    def test_all_properties_question(self):
        intent = self.intuition.understand("kar nasıldır?")
        self.assertEqual((intent.kind, intent.concept), (ASK_PROPERTIES, "kar"))

    def test_verbs_still_win_over_the_property_reading(self):
        intent = self.intuition.understand("kuşlar uçar")
        self.assertEqual(intent.target, "uçmak")


class TestPropertyReasoning(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.reasoning = Reasoning(self.memory)
        self.memory.write(Edge("kuş", HAS_PROPERTY, "tüylü", source="sen"))
        self.memory.write(Edge("penguen", IS_A, "kuş", source="sen"))

    def test_properties_are_inherited(self):
        known, chain = self.reasoning.has_property("penguen", "tüylü")
        self.assertTrue(known)
        self.assertIn("penguen bir kuş", " ".join(chain))

    def test_direct_denial_overrides_inheritance(self):
        self.memory.write(Edge("penguen", LACKS_PROPERTY, "tüylü", source="sen"))
        self.assertFalse(self.reasoning.has_property("penguen", "tüylü")[0])

    def test_unknown_property_returns_none(self):
        self.assertIsNone(self.reasoning.has_property("penguen", "hızlı")[0])


class TestPropertiesInConversation(unittest.TestCase):
    def setUp(self):
        self.session = Session(os.path.join(tempfile.mkdtemp(), "memory.json"))

    def test_teaching_and_asking(self):
        self.session.respond("kuşlar tüylüdür")
        self.session.respond("penguen bir kuştur")
        self.assertTrue(self.session.respond("penguen tüylü mü").startswith("evet"))

    def test_property_conflict_is_noticed(self):
        self.session.respond("kuşlar tüylüdür")
        self.session.respond("penguen bir kuştur")
        answer = self.session.respond("penguen tüylü değildir")
        self.assertIn("çelişki", answer)
        self.session.respond("evet")
        self.assertTrue(self.session.respond("penguen tüylü mü").startswith("hayır"))

    def test_listing_properties(self):
        self.session.respond("kar beyazdır")
        self.session.respond("kar sıcak değildir")
        answer = self.session.respond("kar nasıldır")
        self.assertIn("beyazdır", answer)
        self.assertIn("sıcak değildir", answer)

    def test_unknown_property_is_admitted(self):
        self.assertIn("bilmiyorum", self.session.respond("kar beyaz mı"))

    def test_curiosity_asks_about_properties(self):
        self.session.respond("kar beyazdır")
        reply = self.session.respond("buz bir sudur")
        self.assertIn("?", reply)   # it wondered about something


if __name__ == "__main__":
    unittest.main()
