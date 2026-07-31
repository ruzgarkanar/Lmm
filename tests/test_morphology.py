"""Discovering a language's suffixes from its words, with nobody describing them.

The suffix lists in turkish.py were the last thing here a person wrote about a
particular language — and the thing that would have to be written eighty more
times. These check it can be found instead of stated.

The word list below is the only input. Nothing in it says which words are
plurals, nothing says Turkish has vowel harmony, and nothing says a suffix may
begin with d or t depending on what it follows.
"""
import unittest

from lmm.morphology import discover, last_vowel

WORDS = """kuş kuşlar kuştur kedi kediler kedidir kar karlar kardır
balık balıklar balıktır at atlar attır ev evler evdir göz gözler gözdür
kalp kalpler kalptir su sular sudur yol yollar yoldur el eller eldir
dal dallar daldır gül güller güldür kol kollar koldur diş dişler diştir
dağ dağlar dağdır kuzu kuzular kuzudur köy köyler köydür süt sütler süttür""".split()


def family_with(found, variant):
    for family in found:
        if variant in family.variants:
            return family
    return None


class TestDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.found = discover(WORDS, minimum=4)

    def test_the_plural_is_found(self):
        family = family_with(self.found, "lar")
        self.assertIsNotNone(family)
        self.assertIn("ler", family.variants)

    def test_the_copula_is_found_with_all_its_faces(self):
        family = family_with(self.found, "dır")
        self.assertIsNotNone(family)
        for variant in ("dir", "dur", "dür", "tır", "tir", "tur", "tür"):
            self.assertIn(variant, family.variants)

    def test_vowel_harmony_falls_out_of_the_counts(self):
        family = family_with(self.found, "lar")
        self.assertEqual(family.attach("kuş"), "kuşlar")
        self.assertEqual(family.attach("ev"), "evler")
        self.assertEqual(family.attach("göz"), "gözler")
        self.assertEqual(family.attach("balık"), "balıklar")

    def test_consonant_assimilation_falls_out_too(self):
        """d after a voiced ending, t after a voiceless one — never stated."""
        family = family_with(self.found, "dır")
        self.assertEqual(family.attach("kuş"), "kuştur")
        self.assertEqual(family.attach("at"), "attır")
        self.assertEqual(family.attach("kedi"), "kedidir")
        self.assertEqual(family.attach("yol"), "yoldur")

    def test_it_generalises_to_a_word_it_never_saw(self):
        family = family_with(self.found, "lar")
        self.assertEqual(family.attach("zürafa"), "zürafalar")
        self.assertEqual(family.attach("köpek"), "köpekler")

    def test_two_different_suffixes_are_not_merged(self):
        """"kuşlar" and "kuştur" both exist, so -lar and -dır are not one thing."""
        plural = family_with(self.found, "lar")
        copula = family_with(self.found, "dır")
        self.assertIsNot(plural, copula)

    def test_stripping_undoes_attaching(self):
        family = family_with(self.found, "lar")
        for stem in ("kuş", "ev", "kuzu", "göz"):
            self.assertEqual(family.strip(family.attach(stem)), stem)


class TestExceptionsDoNotBecomeRules(unittest.TestCase):
    def test_a_loanword_cannot_outvote_the_language(self):
        """"kalp" takes the front variant; one such word once taught the wrong
        rule for every word with an "a" in it."""
        family = family_with(discover(WORDS, minimum=4), "lar")
        self.assertEqual(family.attach("kar"), "karlar")
        self.assertEqual(family.attach("dal"), "dallar")


class TestLastVowel(unittest.TestCase):
    def test_it_reads_from_the_end(self):
        self.assertEqual(last_vowel("kitap"), "a")
        self.assertEqual(last_vowel("gözlük"), "ü")
        self.assertEqual(last_vowel(""), "")


if __name__ == "__main__":
    unittest.main()
