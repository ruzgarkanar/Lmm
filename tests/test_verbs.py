"""Fiilleri metinden bulmak: olumlu ve olumsuz biçim birlikte geçiyorsa fiildir.

Bu modülün varlık sebebi ölçümdür. Bir belgeyi okurken kazancın tamamının
sözlükten geldiği, bilgi grafının hiç katkı vermediği ölçüldü: boş bellek 5,6
olgu/100 cümle, 486 kavramlık graf tek başına yine 5,6, ama 98 fiillik sözlük
tek başına 7,1. Fiil sayısı 724'e çıkınca 19,4.
"""
import unittest

from lmm.verbs import (discover, from_text, infinitive_of, negative_of,
                       stems_of)


class TestShapes(unittest.TestCase):
    def test_harmony_decides_the_negative(self):
        self.assertEqual(negative_of("uygulan"), "uygulanmaz")
        self.assertEqual(negative_of("gel"), "gelmez")
        self.assertEqual(negative_of("uyu"), "uyumaz")
        self.assertEqual(negative_of("gör"), "görmez")

    def test_harmony_decides_the_infinitive(self):
        self.assertEqual(infinitive_of("bak"), "bakmak")
        self.assertEqual(infinitive_of("gel"), "gelmek")
        self.assertEqual(infinitive_of("uyu"), "uyumak")

    def test_a_plural_is_not_a_verb_even_though_it_ends_in_r(self):
        """"kuşlar" -r ile bitiyor; bu tuzağa düşmemek gerekiyor."""
        self.assertEqual(stems_of("kuşlar"), [])
        self.assertEqual(stems_of("evler"), [])

    def test_several_stems_are_offered_and_the_pair_test_decides(self):
        """"gelir" hem "gel"+ir hem "geli"+r okunabilir; kararı olumsuz verir."""
        self.assertIn("gel", stems_of("gelir"))


class TestThePairTest(unittest.TestCase):
    """Tek testin tamamı: iki biçim de geçiyorsa fiil, geçmiyorsa değil."""

    def test_a_word_with_both_forms_is_a_verb(self):
        words = ["uygulanır"] * 3 + ["uygulanmaz"] * 3 + ["banka"] * 9
        found = dict((i, (p, n)) for i, p, n in discover(words))
        self.assertIn("uygulanmak", found)
        self.assertEqual(found["uygulanmak"], ("uygulanır", "uygulanmaz"))

    def test_a_noun_that_merely_ends_in_r_is_not_taken(self):
        """"broker" fiil değildir ve "brokermaz" hiçbir metinde geçmez."""
        found = [i for i, _, _ in discover(["broker"] * 20 + ["şeker"] * 20)]
        self.assertEqual(found, [])

    def test_one_sighting_of_each_is_not_enough(self):
        """Bir kez geçen çift yazım hatası olabilir; iki kez düzenliliktir."""
        self.assertEqual(discover(["koşar", "koşmaz"], minimum=2), [])
        self.assertEqual(len(discover(["koşar", "koşmaz"], minimum=1)), 1)

    def test_it_reads_plain_text(self):
        # "kaydeder" ile "kaydedilmez" AYRI fiillerdir (etken ve edilgen);
        # ilk yazdığımda onları çift sanmıştım ve test haklı olarak düştü.
        text = ("Her işlem kaydedilir, hatalı işlem kaydedilmez. "
                "Doğru veri kaydedilir ama bozuk veri kaydedilmez.")
        found = [i for i, _, _ in from_text(text, minimum=2)]
        self.assertIn("kaydedilmek", found)

    def test_nothing_in_nothing_out(self):
        self.assertEqual(discover([]), [])
        self.assertEqual(from_text(""), [])


class TestAgainstWhatWeWroteByHand(unittest.TestCase):
    def test_it_recovers_verbs_nobody_told_it_about(self):
        """Ölçüldü: 400M karakterde elle yazdığımız fiillerin %67'sini buldu."""
        words = []
        for positive, negative in (("uçar", "uçmaz"), ("koşar", "koşmaz"),
                                   ("yüzer", "yüzmez"), ("uyur", "uyumaz")):
            words += [positive] * 2 + [negative] * 2
        found = {i for i, _, _ in discover(words, minimum=2)}
        self.assertEqual(found,
                         {"uçmak", "koşmak", "yüzmek", "uyumak"})


if __name__ == "__main__":
    unittest.main()
