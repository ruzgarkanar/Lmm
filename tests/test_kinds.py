"""Bir ilişkinin kutbu: hangi ucu olumluyor, hangi ucu yadsıyor.

Kutup dizgi sezgisiyle okunuyordu — ad "not_" ile başlıyorsa yadsıma sayılıyordu.
Envanterdeki `has_not` ve `must_not` bu kalıba uymuyor, dolayısıyla `pair()` bu
iki ilişkide sırayı ters döndürüyordu. Ölçülen sonucu bir sıralama hatası değil,
projenin merkez iddiasının ihlaliydi: bellekte yalnızca ('kuş','has','kanadı')
varken sistem "kuşun kanadı YOK" cümlesini üretiyor ve onu "(kaynak: sen)" diye
öğretmene atfediyordu. Yani bellekte olmayan bir şeyi söyleyip kaynak gösteriyordu.

Buradaki testler kutbun VERİDEN okunduğunu ve iki uçtan da aynı cevabı verdiğini
tutuyor: ilişkiler veri olduğuna göre kutup da veridir, ada bakmak dilin içine
kaçmaktır ve dil bu projede yalnız phrasing/turkish dosyalarında yaşar.
"""
import os
import tempfile
import unittest

from lmm.cli import Session
from lmm.kinds import DEFAULT, Kind, Kinds
from lmm.memory import Memory
from lmm.relations import (CAN, CANNOT, HAS_PART, HAS_PROPERTY, IS_A,
                           LACKS_PART, LACKS_PROPERTY, LACKS_REQUIREMENT,
                           MUST, MUST_NOT, NOT_A, REQUIRES)


PAIRS = [(IS_A, NOT_A), (CAN, CANNOT), (HAS_PROPERTY, LACKS_PROPERTY),
         (HAS_PART, LACKS_PART), (MUST, MUST_NOT),
         (REQUIRES, LACKS_REQUIREMENT)]


class TestPolarityIsData(unittest.TestCase):
    def test_every_core_pair_keeps_its_order_from_either_end(self):
        """Asıl kırılma buydu: `has` ucundan sorulunca ('has','has_not'),
        `has_not` ucundan sorulunca ('has_not','has') geliyordu."""
        for affirms, denies in PAIRS:
            self.assertEqual((affirms, denies), DEFAULT.pair(affirms))
            self.assertEqual((affirms, denies), DEFAULT.pair(denies))

    def test_no_core_relation_leaves_its_polarity_unsaid(self):
        """Künye eksikse motor sıraya düşer; çekirdek bunu hiç yapmamalı."""
        for kind in Kinds().by_name.values():
            if kind.negation_of:
                self.assertIsNotNone(kind.denies, kind.name)

    def test_a_new_relation_arrives_as_data_with_its_polarity(self):
        """Kod değişmeden: adında hiçbir ipucu olmayan bir çift, yalnız
        künyesiyle doğru kuruluyor."""
        kinds = Kinds()
        kinds.load([{"name": "acik", "negation_of": "gizli", "denies": False},
                    {"name": "gizli", "negation_of": "acik", "denies": True}])
        self.assertEqual(("acik", "gizli"), kinds.pair("acik"))
        self.assertEqual(("acik", "gizli"), kinds.pair("gizli"))

    def test_one_end_is_enough(self):
        """Kutbu tek uç bildirse de çift belirlenir: eşin künyesine bakılır,
        adına değil."""
        kinds = Kinds()
        kinds.add(Kind("temiz", negation_of="kirli"))
        kinds.add(Kind("kirli", negation_of="temiz", denies=True))
        self.assertEqual(("temiz", "kirli"), kinds.pair("kirli"))
        self.assertEqual(("temiz", "kirli"), kinds.pair("temiz"))

    def test_an_unsaid_pair_is_still_the_same_from_both_ends(self):
        """Kutbunu kimse bildirmezse yön tahmindir — ama iki yönde birden
        tahmin edilemez, çelişki uydurmasının kaynağı tam olarak buydu."""
        kinds = Kinds()
        kinds.load([{"name": "zzz", "negation_of": "zzz_yok"},
                    {"name": "zzz_yok", "negation_of": "zzz"}])
        self.assertEqual(kinds.pair("zzz"), kinds.pair("zzz_yok"))

    def test_a_record_written_before_this_field_keeps_its_polarity(self):
        """Kayıtlı modellerde bu alan yok. Veri künyeyi olduğu gibi ezseydi,
        dosyadan yüklenen her oturumda hata geri gelirdi."""
        old = [dict(kind.to_dict()) for kind in Kinds().by_name.values()]
        for entry in old:
            entry.pop("denies")
        kinds = Kinds()
        kinds.load(old)
        for affirms, denies in PAIRS:
            self.assertEqual((affirms, denies), kinds.pair(denies))

    def test_polarity_survives_a_round_trip(self):
        kinds = Kinds()
        kinds.load(Kinds().to_list())
        self.assertEqual((HAS_PART, LACKS_PART), kinds.pair(LACKS_PART))


class TestItDoesNotInventWhatMemoryDoesNotHold(unittest.TestCase):
    """Ölçülen kanıt: `Memory` bir tek olumlu olgu tutarken, çelişki uyarısı
    olumsuz cümleyi yazıyor ve kaynağını öğretmen gösteriyordu."""

    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".lmm")
        os.close(handle)
        os.remove(self.path)
        self.session = Session(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_the_clash_it_reports_is_the_fact_it_holds(self):
        self.session.respond("kuşun kanadı var")
        answer = self.session.respond("kuşun kanadı yok")
        held = [(edge.concept, edge.relation, edge.target)
                for edge in self.session.memory.edges]
        self.assertIn(("kuş", HAS_PART, "kanadı"), held)
        self.assertIn("var", answer)

    def test_and_the_same_the_other_way_round(self):
        self.session.respond("kuşun kanadı yok")
        answer = self.session.respond("kuşun kanadı var")
        held = [(edge.concept, edge.relation, edge.target)
                for edge in self.session.memory.edges]
        self.assertIn(("kuş", LACKS_PART, "kanadı"), held)
        self.assertIn("yok", answer)


class TestWhatTheReasonerReadsFromAPair(unittest.TestCase):
    def test_a_denial_is_not_read_as_an_affirmation(self):
        memory = Memory()
        self.assertEqual((HAS_PART, LACKS_PART), memory.kinds.pair(LACKS_PART))
        self.assertEqual((MUST, MUST_NOT), memory.kinds.pair(MUST_NOT))


if __name__ == "__main__":
    unittest.main()
