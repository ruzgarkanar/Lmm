import unittest

from lmm.intuition import tokenize, Intuition, TEACH, ASK, UNKNOWN
from lmm.memory import IS_A, CAN, CANNOT


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_drops_punctuation(self):
        self.assertEqual(tokenize("Penguen bir KUŞTUR!"), ["penguen", "bir", "kuştur"])

    def test_keeps_turkish_characters(self):
        self.assertEqual(tokenize("Kuşlar uçar."), ["kuşlar", "uçar"])


class TestPatternParser(unittest.TestCase):
    def setUp(self):
        self.intuition = Intuition()

    def test_teach_type(self):
        intent = self.intuition.understand("Penguen bir kuştur.")
        self.assertEqual((intent.kind, intent.concept, intent.relation, intent.target),
                         (TEACH, "penguen", IS_A, "kuş"))

    def test_teach_positive_ability(self):
        intent = self.intuition.understand("Kuşlar uçar.")
        self.assertEqual((intent.kind, intent.concept, intent.relation, intent.target),
                         (TEACH, "kuş", CAN, "uçmak"))

    def test_teach_negative_ability(self):
        intent = self.intuition.understand("Penguen uçamaz.")
        self.assertEqual((intent.kind, intent.concept, intent.relation, intent.target),
                         (TEACH, "penguen", CANNOT, "uçmak"))

    def test_ask_ability(self):
        intent = self.intuition.understand("Penguen uçar mı?")
        self.assertEqual((intent.kind, intent.concept, intent.relation, intent.target),
                         (ASK, "penguen", CAN, "uçmak"))

    def test_ask_definition(self):
        intent = self.intuition.understand("Penguen nedir?")
        self.assertEqual((intent.kind, intent.concept, intent.relation),
                         (ASK, "penguen", IS_A))

    def test_unparsable_sentence(self):
        intent = self.intuition.understand("florp glorp zzz qqq")
        self.assertEqual(intent.kind, UNKNOWN)


if __name__ == "__main__":
    unittest.main()
