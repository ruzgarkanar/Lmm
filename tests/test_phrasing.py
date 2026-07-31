import unittest

from lmm.memory import IS_A, CAN, CANNOT
from lmm.phrasing import is_a_clause, ability_clause, verb_form, describe


class TestCopulaHarmony(unittest.TestCase):
    def test_vowel_harmony_and_assimilation(self):
        self.assertEqual(is_a_clause("penguen", "kuş"), "penguen bir kuştur")
        self.assertEqual(is_a_clause("kedi", "hayvan"), "kedi bir hayvandır")
        self.assertEqual(is_a_clause("çiçek", "canlı"), "çiçek bir canlıdır")
        self.assertEqual(is_a_clause("masa", "eşya"), "masa bir eşyadır")
        self.assertEqual(is_a_clause("kedi", "süt"), "kedi bir süttür")


class TestVerbForms(unittest.TestCase):
    def test_positive_and_negative(self):
        self.assertEqual(verb_form("uçmak", True), "uçar")
        self.assertEqual(verb_form("uçmak", False), "uçamaz")
        self.assertEqual(verb_form("okumak", False), "okuyamaz")

    def test_ability_clause(self):
        self.assertEqual(ability_clause("penguen", "uçmak", False), "penguen uçamaz")
        self.assertEqual(ability_clause("kuş", "uçmak", True), "kuş uçar")


class TestDescribe(unittest.TestCase):
    def test_relations_render_as_sentences(self):
        self.assertEqual(describe("penguen", IS_A, "kuş"), "penguen bir kuştur")
        self.assertEqual(describe("kuş", CAN, "uçmak"), "kuş uçar")
        self.assertEqual(describe("penguen", CANNOT, "uçmak"), "penguen uçamaz")

    def test_internal_labels_never_leak(self):
        for text in (describe("penguen", IS_A, "kuş"),
                     describe("kuş", CAN, "uçmak"),
                     describe("penguen", CANNOT, "uçmak")):
            self.assertNotIn("type", text)
            self.assertNotIn("can", text)


if __name__ == "__main__":
    unittest.main()
