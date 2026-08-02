"""Bilmediğini fark edip gidip okumak.

Üçüncü şartın etkin hâli. Bu testlerin hiçbiri ağa çıkmıyor: getirici de
okuyucu da dışarıdan veriliyor, çünkü sınanan şey ağ değil **döngü** — neyin
soruşturulduğu, neyin yazıldığı, neyin reddedildiği.
"""
import unittest

from lmm import inquiry
from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT


PAGE = ("Penguen, uçamayan deniz kuşlarının ortak adıdır. "
        "Penguenler güney yarım kürede yaşar. Penguenler çok iyi yüzer.")


def page_of(text, title="Penguen"):
    """Sabit bir sayfa döndüren getirici."""
    return lambda concept: (title, text)


def nothing(concept):
    """Sayfası olmayan kavram."""
    return None


class TestFetching(unittest.TestCase):
    def test_a_missing_page_is_not_an_error(self):
        """Bulamamak bir arıza değil bir sonuçtur; çağıran çökmemeli."""
        report = inquiry.investigate(Memory(), "olmayanşey", fetcher=nothing)
        self.assertIsNone(report["kaynak"])
        self.assertEqual(report["yazılan"], 0)

    def test_the_source_is_citable(self):
        """Künye olmadan olgu, kaynağı gösterilemeyen bir iddiadır."""
        memory = Memory()
        inquiry.investigate(memory, "penguen", fetcher=page_of(PAGE))
        for edge in memory.edges:
            self.assertTrue(edge.source.startswith(inquiry.SOURCE_PREFIX),
                            edge.source)


class TestReading(unittest.TestCase):
    def test_the_reader_is_replaceable(self):
        """Okuyucu verilirse derleyici çalışır ve iddiaları metne karşı denetler.

        Buradaki okuyucu metinde geçen bir iddia ile geçmeyen bir iddia
        döndürüyor; ikincisi yazılmamalı.
        """
        def reader(chunk):
            return ("penguen | property | uçamayan | "
                    "Penguen, uçamayan deniz kuşlarının ortak adıdır.\n"
                    "penguen | can | uçmak | Penguen uçabilir.")

        memory = Memory()
        report = inquiry.investigate(memory, "penguen",
                                     fetcher=page_of(PAGE), reader=reader)
        self.assertEqual(report["uydurma"], 1)
        self.assertNotIn("uçmak", [e.target for e in memory.edges])

    def test_without_a_reader_it_still_works(self):
        """Ağ ya da model yoksa kendi okuyucumuza düşülür — körleşir, durmaz."""
        report = inquiry.investigate(Memory(), "penguen",
                                     fetcher=page_of(PAGE))
        self.assertIsNotNone(report["kaynak"])

    def test_brackets_are_dropped_before_reading(self):
        """"(Latince: Aquila)" cümlenin durum yapısını bozuyor."""
        cleaned = inquiry._cleaned("Kartal (Latince: Aquila) bir kuştur.")
        self.assertNotIn("Latince", cleaned)
        self.assertIn("Kartal", cleaned)
        self.assertIn("kuştur", cleaned)


class TestWriting(unittest.TestCase):
    def test_a_clash_with_what_is_known_is_refused_and_counted(self):
        """Kaynak yanılırsa graf onu almaz — ve bu sessizce olmaz."""
        memory = Memory()
        memory.learn_word("uçmak", "uçar", "uçamaz")
        memory.write(Edge("penguen", CANNOT, "uçmak", source="sen"))

        def reader(chunk):
            return "penguen | can | uçmak | Penguenler çok iyi uçar."

        report = inquiry.investigate(memory, "penguen",
                                     fetcher=page_of("Penguenler çok iyi uçar."),
                                     reader=reader)
        self.assertEqual(report["çelişen"] + report["uydurma"], 1)
        self.assertEqual(
            [e.relation for e in memory.query("penguen")], [CANNOT])

    def test_a_long_page_is_cut_and_says_so(self):
        """Sınır sessizce uygulanmaz; kesildiği söylenir."""
        traits = ("siyah", "beyaz", "küçük", "hızlı", "sessiz")

        def many(chunk):
            return "\n".join(f"penguen | property | {t} | Penguen {t}dır."
                              for t in traits)

        memory = Memory()
        text = " ".join(f"Penguen {t}dır." for t in traits)
        report = inquiry.investigate(memory, "penguen", fetcher=page_of(text),
                                     reader=many, most=2)
        self.assertTrue(report["kesildi"])
        self.assertEqual(report["yazılan"], 2)


class TestChoosingWhatToInvestigate(unittest.TestCase):
    def test_only_what_is_not_known_is_investigated(self):
        """Bilinen kavram için ağa gidilmez; graf zaten büyüyor."""
        memory = Memory()
        memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        self.assertEqual(inquiry.unknown_in(memory, ["kartal", "penguen"]),
                         ["penguen"])

    def test_nothing_unknown_means_no_investigation(self):
        memory = Memory()
        memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        self.assertEqual(inquiry.unknown_in(memory, ["kartal"]), [])


if __name__ == "__main__":
    unittest.main()


class TestWhatIsWrittenIsUsable(unittest.TestCase):
    """Okunan olgu grafta işe yarar hâlde olmalı.

    İki kusur canlı bir okumada yakalandı — Vikipedi'nin ahtapot sayfası:

        ahtapot --can--> rastlanma      "rastlanma" bir eylem değil, eylem ADI
        ahtapotlar --property--> zehirli   çoğul ayrı bir kavram sanıldı

    Birincisi cevabı bozuyordu ("ahtapot rastlanma"), ikincisi bilgiyi ikiye
    bölüyordu: birinde öğrenilen diğerinde bulunamıyor.
    """

    def test_a_verbal_noun_is_not_an_action(self):
        """Türkçe'de "-mak/-mek" mastardır, "-ma/-me" addır."""
        from lmm.compiler import _an_action
        self.assertTrue(_an_action("uçmak"))
        self.assertTrue(_an_action("yumurtaları bırakmak"))
        self.assertFalse(_an_action("rastlanma"))
        self.assertFalse(_an_action("yaşlanma"))
        self.assertFalse(_an_action(""))

    def test_a_can_claim_with_a_verbal_noun_is_refused(self):
        from lmm.compiler import Candidate, NOT_AN_ACTION, normalise, verify
        anchor = "Ahtapota her denizde rastlanma durumu vardır."
        claim = Candidate("ahtapot", "can", "rastlanma", anchor)
        self.assertFalse(verify(claim, normalise(anchor)))
        self.assertEqual(claim.refusal, NOT_AN_ACTION)

    def test_the_plural_joins_the_singular(self):
        """"ahtapotlar" ile "ahtapot" iki kavram değil."""
        memory = Memory()
        memory.write(Edge("ahtapot", IS_A, "hayvan", source="sen"))

        def reader(chunk):
            return ("ahtapotlar | property | zehirli | "
                    "Bazı ahtapotlar zehirlidir.")

        inquiry.investigate(memory, "ahtapot", reader=reader,
                            fetcher=page_of("Bazı ahtapotlar zehirlidir.",
                                            "Ahtapot"))
        self.assertTrue(memory.query("ahtapot", "property"))
        self.assertFalse(memory.query("ahtapotlar"))
