"""Uzun cümleyi cümleciklere bölmek."""
import unittest

from lmm.clauses import split, readable


class TestSplitting(unittest.TestCase):
    def test_a_conjunction_marks_a_boundary(self):
        self.assertEqual(
            split("kuşlar uçar ve balıklar yüzer"),
            ["kuşlar uçar", "kuşlar balıklar yüzer"])

    def test_the_subject_is_carried_to_the_second_clause(self):
        """Türkçe'de bağlaçtan sonra özne düşer; okunabilmesi için geri konur."""
        found = split("Sistem, her işlemi kaydeder ve şüpheli hareketleri raporlar")
        self.assertEqual(found, ["Sistem her işlemi kaydeder",
                                 "Sistem şüpheli hareketleri raporlar"])

    def test_a_single_word_before_a_comma_is_the_subject_not_a_clause(self):
        """Atılırsa özne yanlış seçiliyordu: "Penguen" gidince "soğuk" özne oluyordu."""
        found = split("Penguen, soğuk yerlerde yaşar ve balıkla beslenir")
        self.assertEqual(found, ["Penguen soğuk yerlerde yaşar",
                                 "Penguen balıkla beslenir"])

    def test_a_clause_that_already_names_the_subject_is_left_alone(self):
        found = split("kredi bir üründür ve kredi faiz taşır")
        self.assertEqual(found[1], "kredi faiz taşır")

    def test_a_dangling_conjunction_is_trimmed(self):
        """Sonda boşta kalan bağlaç, olmayan bir cümleciği varmış gibi gösterir."""
        self.assertEqual(split("kuşlar uçar ve"), ["kuşlar uçar"])

    def test_a_sentence_with_no_boundary_comes_back_whole(self):
        self.assertEqual(split("kuşlar uçar"), ["kuşlar uçar"])

    def test_nothing_in_produces_nothing_out(self):
        self.assertEqual(split(""), [])
        self.assertEqual(split(",,,"), [])


class TestWhenToSplit(unittest.TestCase):
    def test_a_short_sentence_is_never_split(self):
        """Kalıpların okuyabildiği cümleyi bölmek, kazançsız risk almaktır."""
        self.assertEqual(readable("penguen bir kuştur"), ["penguen bir kuştur"])

    def test_a_long_sentence_is_split(self):
        found = readable("Sistem her işlemi kaydeder ve şüpheli olanı raporlar")
        self.assertEqual(len(found), 2)

    def test_a_long_sentence_with_no_boundary_comes_back_whole(self):
        # "ya da" bir bağlaçtır; ilk yazdığım örnekte kendi bağlacım vardı.
        sentence = "bu uzun cümlede hiçbir sınır işareti bulunmuyor gerçekten"
        self.assertEqual(readable(sentence), [sentence])


if __name__ == "__main__":
    unittest.main()
