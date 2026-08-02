"""Geri okuma kapısı: üretilen cümle graftaki kaydın tersini söylüyor mu.

Bu kapı gerçek bir açıktan doğdu ve açığı ben açtım. Kapılı ağzı gevşetirken
sorunun kelimelerini izinli listeye ekledim — "kullanıcının söylediğini tekrar
etmek uydurmak değildir" diye. Sonuç:

    soru : penguen uçar mı
    graf : hayır, penguen uçamaz
    ağız : "hayır, çünkü penguen UÇAR."      <- kapı geçirdi

Gecenin en tehlikeli deliği, başka biçimde geri açılmıştı.

**Ders: kelime listesi kutbu kodlayamaz.** `uçar` ile `uçamaz` aynı kökten
geliyor ve bir liste "bu kelimeyi anabilirsin ama iddia edemezsin" diyemez.

Doğrusu cümleyi kelime kelime denetlemek değil, **geri okumak**: üretilen
cümleyi kendi ayrıştırıcımızdan geçir, çıkan olguyu grafa sor. Okuyucu kutbu
görüyor, liste görmüyor.

Okunamayan cümle çelişki sayılmıyor — bilmemek, yanlış bilmek değildir.
"""
import unittest

from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT, HAS_PROPERTY


def _contradicts(said, memory):
    from core.bridge import _contradicts as check
    return check(said, memory)


class TestReadBack(unittest.TestCase):
    def setUp(self):
        self.memory = Memory()
        self.memory.learn_word("uçmak", "uçar", "uçamaz")
        self.memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        self.memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        self.memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        self.memory.write(Edge("penguen", CANNOT, "uçmak", source="sen",
                               is_exception=True))
        self.memory.write(Edge("penguen", HAS_PROPERTY, "tüylü", source="sen"))

    def test_the_opposite_of_a_record_is_caught(self):
        """Grafta "penguen uçamaz" yazıyorken "penguen uçar" geçemez."""
        self.assertTrue(_contradicts("penguen uçar", self.memory))

    def test_it_is_caught_inside_a_longer_sentence(self):
        """Çelişki cümlenin ortasında da olsa yakalanmalı."""
        self.assertTrue(_contradicts("hayır, çünkü penguen uçar", self.memory))

    def test_what_the_graph_says_passes(self):
        self.assertFalse(_contradicts("penguen uçamaz", self.memory))
        self.assertFalse(_contradicts("penguen tüylüdür", self.memory))

    def test_another_concept_is_not_confused(self):
        """Penguen uçamaz diye kartal da uçamaz sayılmamalı."""
        self.assertFalse(_contradicts("kartal uçar", self.memory))

    def test_an_unreadable_sentence_is_not_a_contradiction(self):
        """Bilmemek, yanlış bilmek değildir."""
        self.assertFalse(_contradicts("bugün hava çok güzel", self.memory))
        self.assertFalse(_contradicts("", self.memory))


class TestFirstSentence(unittest.TestCase):
    """Model istemin devamını da yazıyor; o tekrarlar cevabın parçası değil."""

    def check(self, raw):
        from core.bridge import _first_sentence
        return _first_sentence(raw)

    def test_the_scaffolding_is_stripped(self):
        self.assertEqual(self.check("Cevap: penguen bir kuştur."),
                         "penguen bir kuştur.")

    def test_it_stops_at_the_prompt_echo(self):
        said = self.check("penguen bir kuştur. Soru: kartal nedir")
        self.assertNotIn("Soru:", said)

    def test_it_stops_at_a_line_break(self):
        self.assertEqual(self.check("kuş uçar.\nBaşka bir şey"), "kuş uçar.")


if __name__ == "__main__":
    unittest.main()
