"""Ansiklopedi tanımlarından olgu çıkarma."""
import unittest

from lmm.definitions import extract, head_noun, strip_possessive, subject
from lmm.turkish import TurkishMorphology

MORPHOLOGY = TurkishMorphology()


class TestPossessive(unittest.TestCase):
    """İki hata da ölçümde yakalandı ve ikisi de sessizce yanlış bilgi üretti."""

    def test_the_s_of_cins_belongs_to_the_word_not_to_the_suffix(self):
        """3621 sahte "cin" üretmişti. "-sı" yalnızca ünlüden sonra gelir."""
        self.assertEqual(strip_possessive("cinsi"), "cins")

    def test_softening_is_undone_so_guvec_becomes_guvec_with_a_tail(self):
        self.assertEqual(strip_possessive("güveci"), "güveç")
        self.assertEqual(strip_possessive("kitabı"), "kitap")

    def test_a_genuine_possessive_after_a_vowel_is_stripped(self):
        self.assertEqual(strip_possessive("arabası"), "araba")
        self.assertEqual(strip_possessive("sporu"), "spor")

    def test_a_word_with_no_possessive_is_left_alone(self):
        for word in ("kuş", "köy", "mahalle", "familya"):
            self.assertEqual(strip_possessive(word), word)


class TestReadingADefinition(unittest.TestCase):
    def test_it_takes_the_kind_from_the_predicate(self):
        self.assertEqual(
            extract("Futbol, on birer oyuncudan oluşan bir takım sporudur.",
                    MORPHOLOGY),
            ("futbol", "spor"))

    def test_the_subject_is_what_stands_before_the_first_comma(self):
        self.assertEqual(subject("Kartal, yırtıcı bir kuştur."), "kartal")

    def test_a_parenthetical_after_the_name_is_dropped(self):
        self.assertEqual(subject("Biber (Capsicum), bir cinstir."), "biber")

    def test_a_multi_word_name_is_refused_rather_than_guessed_at(self):
        """Hangi kelimenin baş olduğuna tahminle karar vermek grafı bozar."""
        self.assertIsNone(subject("Çayır papatyası, bir cinstir."))

    def test_a_sentence_that_is_not_a_definition_yields_nothing(self):
        for line in ("Kartal uçar.", "Bugün hava çok güzel.",
                     "Penguen uçamaz ve yüzer."):
            self.assertIsNone(extract(line, MORPHOLOGY))

    def test_a_definition_that_defines_a_thing_as_itself_is_refused(self):
        self.assertIsNone(extract("Köy, bir köydür.", MORPHOLOGY))

    def test_the_head_survives_the_copula(self):
        self.assertEqual(head_noun("... bir kuştur.", MORPHOLOGY), "kuş")
        self.assertEqual(head_noun("... bir ilçedir.", MORPHOLOGY), "ilçe")


class TestOnRealLines(unittest.TestCase):
    REAL = [
        ("Yoda, George Lucas tarafından yaratılan bir karakterdir.",
         ("yoda", "karakter")),
        ("Biber ( ), patlıcangillerden bir cinstir.", ("biber", "cins")),
        ("Futbol, kendine özgü bir topla oynanan takım sporudur.",
         ("futbol", "spor")),
    ]

    def test_lines_taken_straight_from_the_dump(self):
        for line, expected in self.REAL:
            self.assertEqual(extract(line, MORPHOLOGY), expected, line)

    def test_two_names_for_one_thing_are_refused_rather_than_halved(self):
        """"Basketbol ya da sepettopu" iki addır; birini seçmek uydurmaktır.

        Bunu çözmek mümkün — ikisini de aynı kavramın adı saymak — ama tahminle
        değil, eş adlılık ilişkisi eklenerek. O gelene kadar susmak doğru olan.
        """
        self.assertIsNone(extract(
            "Basketbol ya da sepettopu, elle oynanan bir spor dalıdır.",
            MORPHOLOGY))


if __name__ == "__main__":
    unittest.main()
