"""Suffix rules worked out from words, standing in for the lists we wrote.

Discovery already finds a language's families from a word list. What it cannot
find is what a suffix is *for* — nothing in the shape of "-lar" says it marks
number — so it is told once, with an example, and derives the rest.
"""
import os
import tempfile
import unittest

from lmm.turkish import TurkishMorphology, turkish, ANCHORS
from lmm.discovered import DiscoveredMorphology, words_of, PLURAL, COPULA
from lmm.memory import Memory, Edge, IS_A, CAN
from lmm.intuition import Intuition, TEACH
from lmm.cli import Session

WORDS = """kuş kuşlar kuştur kedi kediler kedidir kar karlar kardır
balık balıklar balıktır at atlar attır ev evler evdir göz gözler gözdür
su sular sudur yol yollar yoldur el eller eldir dal dallar daldır
gül güller güldür kol kollar koldur diş dişler diştir dağ dağlar dağdır
kuzu kuzular kuzudur köy köyler köydür süt sütler süttür""".split()

BATTERY = ["kuş", "kedi", "ev", "göz", "at", "balık", "kuzu", "köy", "süt",
           "zürafa", "köpek", "çiçek", "orman", "deniz", "gemi"]


def discovered():
    return DiscoveredMorphology(TurkishMorphology(), WORDS, ANCHORS)


class TestItLearnsWhatWeUsedToWrite(unittest.TestCase):
    def setUp(self):
        self.discovered = discovered()
        self.declared = TurkishMorphology()

    def test_both_roles_are_worked_out(self):
        self.assertEqual(self.discovered.learned, [COPULA, PLURAL])

    def test_it_agrees_with_the_hand_written_lists(self):
        """Every word in the battery must strip the same either way."""
        for word in BATTERY:
            plural = self.discovered.families[PLURAL].attach(word)
            with self.subTest(word):
                self.assertEqual(self.discovered.strip_plural(plural), word)
                self.assertEqual(self.declared.strip_plural(plural), word)

    def test_the_copula_carries_its_consonant(self):
        attach = self.discovered.attach_copula
        self.assertEqual(attach("kuş"), "kuştur")
        self.assertEqual(attach("at"), "attır")
        self.assertEqual(attach("kedi"), "kedidir")
        self.assertEqual(attach("köpek"), "köpektir")

    def test_it_recognises_a_copula_and_its_absence(self):
        self.assertTrue(self.discovered.has_copula("kuştur"))
        self.assertFalse(self.discovered.has_copula("kuş"))

    def test_closed_class_words_are_still_declared(self):
        """Nothing in a word list reveals that "bazı" means some."""
        self.assertEqual(self.discovered.question_particles,
                         self.declared.question_particles)
        self.assertIn("bazı", self.discovered.quantifiers)

    def test_too_few_words_falls_back_rather_than_breaking(self):
        thin = DiscoveredMorphology(TurkishMorphology(), ["kuş", "kedi"], ANCHORS)
        self.assertEqual(thin.learned, [])
        self.assertEqual(thin.strip_plural("kuşlar"), "kuş")


class TestTheGrammarWithDiscoveredRules(unittest.TestCase):
    def test_sentences_read_the_same(self):
        by_rule = Intuition(grammar=turkish(WORDS))
        by_list = Intuition()
        for sentence in ("penguen bir kuştur", "kuşlar uçar", "kar beyazdır",
                         "penguen nedir", "kartal uçar mı"):
            with self.subTest(sentence):
                one, two = by_rule.understand(sentence), by_list.understand(sentence)
                self.assertEqual((one.kind, one.concept, one.relation, one.target),
                                 (two.kind, two.concept, two.relation, two.target))


class TestASessionUsesWhatItHasSeen(unittest.TestCase):
    def test_the_words_a_memory_holds_feed_the_discovery(self):
        memory = Memory()
        memory.write(Edge("kuş", IS_A, "hayvan", source="sen"))
        memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        memory.learn_word("koşmak", "koşar", "koşamaz")
        found = words_of(memory)
        for word in ("kuş", "hayvan", "uçmak", "koşar", "koşamaz"):
            self.assertIn(word, found)

    def test_a_session_still_works_from_an_empty_memory(self):
        session = Session(os.path.join(tempfile.mkdtemp(), "yeni.lmm"))
        self.assertIn("öğrendim", session.respond("penguen bir kuştur"))
        self.assertIn("kuş", session.respond("penguen nedir"))


if __name__ == "__main__":
    unittest.main()
