"""Söyleyiş öğrenmek: anlaşılmayan cümleden kalıcı kalıp.

Projenin kendi tezinin dile uygulanmış hâli. Bilgi için zaten böyle çalışıyor —
bilmiyorsa gider okur, denetler, künyesiyle yazar, bir daha sormaz. Dil için
yapılmıyordu: ayrıştırıcı 48 elle yazılmış kalıptı ve hiç büyümüyordu.

Buradaki testlerin hiçbiri ağa çıkmaz. Sahte okuyucu bilerek yanlış da söyler,
çünkü sınanan şey okuyucunun iyi niyeti değil DENETİM: işe yaramayan bir okuma
kalıba dönüşmemeli.
"""
import os
import tempfile
import unittest

from lmm import wording
from lmm.cli import Session
from lmm.intuition import ASK_DESCRIBE, tokenize
from lmm.memory import Memory, Edge, IS_A, CAN


class TestReadingTheReply(unittest.TestCase):
    def test_it_reads_the_three_column_form(self):
        found = wording.parse_reply("ANLAT | 1 | -1", 4)
        self.assertEqual(found.kind, ASK_DESCRIBE)
        self.assertEqual(found.concept_at, 1)
        self.assertIsNone(found.target_at)

    def test_an_invented_kind_is_refused(self):
        """Model niyet icat edemez; kapalı bir kümeden seçer."""
        self.assertIsNone(wording.parse_reply("HAVA_DURUMU | 0 | -1", 3))

    def test_an_index_outside_the_sentence_is_refused(self):
        self.assertIsNone(wording.parse_reply("ANLAT | 9 | -1", 3))

    def test_a_malformed_reply_is_not_guessed_at(self):
        self.assertIsNone(wording.parse_reply("bence bu bir anlatma isteği", 4))
        self.assertIsNone(wording.parse_reply("", 4))


class TestPeelingTheConcept(unittest.TestCase):
    """"penguenlerden" grafta durmaz, "penguen" durur."""

    def setUp(self):
        self.memory = Memory()
        self.memory.write(Edge("penguen", IS_A, "kuş", source="test"))
        self.morphology = Session(
            os.path.join(tempfile.mkdtemp(), "m.lmm")).language.grammar.morphology

    def test_a_case_marked_concept_is_found(self):
        found = wording._concept_at(["bana", "penguenlerden"], 1, self.memory,
                                    self.morphology)
        self.assertEqual(found, "penguen")

    def test_a_word_that_is_no_concept_is_not_invented(self):
        found = wording._concept_at(["bana", "ejderhadan"], 1, self.memory,
                                    self.morphology)
        self.assertIsNone(found)

    def test_peeling_reaches_through_two_suffixes(self):
        """Tek geçiş yetmiyor: hem çoğul hem durum eki var."""
        self.assertIn("penguen", wording._forms("penguenlerden",
                                                self.morphology))


class TestTheGateOnPatterns(unittest.TestCase):
    """Bir kalıp, işe yaradığı görülmeden yazılmaz."""

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "s.lmm")
        memory = Memory()
        memory.learn_word("uçmak", "uçar", "uçamaz")
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        memory.save(self.path)

    def session(self, reply):
        return Session(self.path, wording=lambda chunk: reply)

    def test_a_working_reading_becomes_a_permanent_pattern(self):
        session = self.session("ANLAT | 1 | -1")
        said = session.respond("bana penguenlerden bahseder misin")
        self.assertIn("kuş", said.lower())
        self.assertEqual(len(session.memory.patterns), 1)

    def test_a_reading_that_answers_nothing_is_not_written(self):
        """Kavram grafta yoksa okuma cevap üretemez, kalıp da yazılmaz."""
        session = self.session("ANLAT | 1 | -1")
        session.respond("bana ejderhadan bahseder misin")
        self.assertEqual(session.memory.patterns, [])

    def test_an_unreadable_reply_writes_nothing(self):
        session = self.session("bilmiyorum ki")
        session.respond("bana penguenlerden bahseder misin")
        self.assertEqual(session.memory.patterns, [])

    def test_the_same_sentence_is_asked_about_only_once(self):
        """Model bir kez sorulur; aynı cümle için ikinci kez ağa çıkılmaz."""
        asked = []

        def counting(chunk):
            asked.append(chunk)
            return "saçmalık"

        session = Session(self.path, wording=counting)
        line = "bana penguenlerden bahseder misin"
        session.respond(line)
        session.respond(line)
        self.assertEqual(len(asked), 1)
        self.assertIn(line, session.asked_about)

    def test_without_a_reader_nothing_changes(self):
        session = Session(self.path)
        session.respond("bana penguenlerden bahseder misin")
        self.assertEqual(session.memory.patterns, [])


class TestThePatternOutlivesTheModel(unittest.TestCase):
    """Asıl iddia: bir kez sorulur, sonra model olmadan çalışır."""

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "k.lmm")
        memory = Memory()
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        memory.save(self.path)
        session = Session(self.path, wording=lambda chunk: "ANLAT | 1 | -1")
        session.respond("bana penguenlerden bahseder misin")
        session.save()

    def test_it_survives_a_restart_without_the_model(self):
        session = Session(self.path)
        said = session.respond("bana penguenlerden bahseder misin")
        self.assertIn("kuş", said.lower())

    def test_it_reaches_a_concept_it_never_saw_in_that_sentence(self):
        """Kalıp öğrenildi, ezber değil: başka kavramla da çalışmalı."""
        session = Session(self.path)
        said = session.respond("bana kartaldan bahseder misin")
        self.assertIn("kuş", said.lower())

    def test_the_pattern_says_where_it_came_from(self):
        session = Session(self.path)
        self.assertTrue(any("öğrenildi" in entry.get("name", "")
                            for entry in session.memory.patterns))


if __name__ == "__main__":
    unittest.main()


class TestTheReadingMustFitTheSentence(unittest.TestCase):
    """Cevap üretebilmek yetmez — okuma cümleye de uymalı.

    Bu kapı gerçek bir kusurdan doğdu. Denetim "graf bu okumayı cevaplayabiliyor
    mu" diye soruyordu ve "bir kuşun uçabilmesi için ne gerekir" cümlesi "kuş ne
    yapabilir" diye okunmuştu: graf cevapladı, denetim geçti, **yanlış kalıp
    kalıcı yazıldı.** Kalıcı hafızaya yanlış kural yazmak, bu mimaride
    yapılabilecek en pahalı hata.

    Ayıran işaret cümlenin fiillerinde ve eş anlamlılık grafıyla çözülüyor.
    """

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "u.lmm")
        memory = Memory()
        memory.write(Edge("kuş", IS_A, "hayvan", source="sen"))
        memory.write(Edge("anlat", "same_as", "bahset",
                          source="vikisözlük:çeviri"))
        memory.save(self.path)
        self.session = Session(self.path)

    def fits(self, kind, sentence):
        return wording.accounted_for(
            kind, tokenize(sentence), self.session.memory.lexicon,
            self.session.language.grammar.patterns, self.session._meanings)

    def test_an_unexplained_verb_blocks_the_reading(self):
        """"gerekir" hiçbir niyetin sözcüğüne bağlı değil."""
        self.assertFalse(
            self.fits("ASK_ABILITIES", "bir kuşun uçabilmesi için ne gerekir"))

    def test_a_synonym_of_a_request_word_is_explained(self):
        """"bahseder" ile "anlat" grafta eş; okuma cümleye uyuyor."""
        self.assertTrue(
            self.fits("ASK_DESCRIBE", "bana kuşlardan bahseder misin"))

    def test_a_sentence_with_no_verb_is_never_blocked(self):
        self.assertTrue(self.fits("ASK_PROPERTIES", "kuşların özellikleri neler"))

    def test_consonant_softening_does_not_break_the_match(self):
        """"bahset" + ünlü -> "bahsed"er. Düz önek karşılaştırması kaçırıyordu."""
        self.assertTrue(wording._starts_with("bahseder", "bahset"))
        self.assertTrue(wording._starts_with("ekmeği", "ekmek"))     # k -> ğ
        self.assertTrue(wording._starts_with("kitabı", "kitap"))     # p -> b
        self.assertFalse(wording._starts_with("gerekir", "bahset"))


class TestAWrongReadingIsNotWritten(unittest.TestCase):
    """Uçtan uca: yanlış okuma kalıcı kalıba dönüşmemeli."""

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "y.lmm")
        memory = Memory()
        memory.learn_word("uçmak", "uçar", "uçamaz")
        memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        memory.write(Edge("kuş", IS_A, "hayvan", source="sen"))
        memory.save(self.path)

    def test_a_reading_that_ignores_the_verb_writes_nothing(self):
        """Ağ "ne gerekir"i "ne yapabilir" sansa bile kalıp yazılmaz."""
        session = Session(self.path,
                          intent=lambda line: ("ASK_ABILITIES", 0.9))
        session.respond("bir kuşun uçabilmesi için ne gerekir")
        self.assertEqual(session.memory.patterns, [])

    def test_a_fitting_reading_still_writes_its_pattern(self):
        session = Session(self.path,
                          intent=lambda line: ("ASK_ABILITIES", 0.9))
        said = session.respond("kuş neler yapar")
        self.assertIn("uçar", said.lower())
