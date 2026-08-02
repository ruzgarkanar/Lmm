"""Eş anlamlılık: aynı şeyin başka söylenişi, graftan okunarak.

Ayrıştırıcının tavanı ölçüldü ve iki çözüm denendi. Dağılımsal kümeleme
çürütüldü — işlevsel kelimelerde aynı gruptaki çift, farklı gruptakinin altında
kalıyordu. Buradaki yol başka: eş anlamlılık bir olgudur, o hâlde yeri graftır —
künyeli, denetlenebilir, düzeltilebilir.

Mekanizma çalışıyor ve burada sınanıyor. Ama ölçüldü ki **tek başına yetmiyor**:
gerçek sorularda skor değişmedi, çünkü darboğaz kelime değil YAPI.
"""
import unittest

from lmm import synonyms
from lmm.cli import Session
from lmm.memory import Memory, Edge, IS_A
from lmm.relations import SAME_AS

import os
import tempfile


ENTRY = """==Türkçe==
===Eylem===
# Bilgi vermek.
# [[nakletmek]]
====Çeviriler====
*İngilizce: {{ç|en|explain}}, {{ç|en|tell}}
*Fransızca: {{ç|fr|expliquer}}
*Fince: {{ç|fi|selittää}}
"""

OTHER = """==Türkçe==
===Eylem===
# Söz etmek.
====Çeviriler====
*İngilizce: {{ç|en|explain}}
*Fransızca: {{ç|fr|parler}}
*Fince: {{ç|fi|selittää}}
"""

UNRELATED = """==Türkçe==
===Eylem===
# Hızlı gitmek.
====Çeviriler====
*İngilizce: {{ç|en|run}}
*Fransızca: {{ç|fr|courir}}
*Fince: {{ç|fi|juosta}}
"""


class TestReadingTheDictionary(unittest.TestCase):
    def test_a_definition_that_is_only_a_link_is_a_synonym(self):
        self.assertIn("nakletmek", synonyms.stated(ENTRY))

    def test_prose_definitions_are_not_synonyms(self):
        self.assertNotIn("bilgi vermek", synonyms.stated(ENTRY))

    def test_translations_are_read_per_language(self):
        found = synonyms.translations(ENTRY)
        self.assertEqual(found["en"], {"explain", "tell"})
        self.assertEqual(found["fr"], {"expliquer"})

    def test_an_empty_entry_yields_nothing(self):
        self.assertEqual(synonyms.stated(None), [])
        self.assertEqual(synonyms.translations(None), {})


class TestTheTranslationPivot(unittest.TestCase):
    """Aynı yabancı kelimeye giden iki Türkçe kelime eş anlamlıdır.

    Ölçüldü: gerçek çiftler %12,5-62,5, alakasız çiftler tam %0. Ayıran şey
    oranın büyüklüğü değil varlığı.
    """

    def test_words_sharing_translations_are_alike(self):
        yes, why = synonyms.alike(ENTRY, OTHER, "anlatmak", "bahsetmek")
        self.assertTrue(yes)
        self.assertIn("çeviri", why)

    def test_words_sharing_nothing_are_not(self):
        yes, _ = synonyms.alike(ENTRY, UNRELATED, "anlatmak", "koşmak")
        self.assertFalse(yes)

    def test_the_reason_is_recorded_not_just_the_verdict(self):
        """Künye kadar önemli olan şey, hangi kanıtla yazıldığı."""
        _, why = synonyms.alike(ENTRY, OTHER, "anlatmak", "bahsetmek")
        self.assertTrue(why)

    def test_too_few_languages_is_not_evidence(self):
        thin = "*İngilizce: {{ç|en|explain}}"
        share, _ = synonyms.shared_translations(thin, ENTRY)
        self.assertEqual(share, 0.0)

    def test_a_stated_synonym_needs_no_translations(self):
        yes, why = synonyms.alike(ENTRY, UNRELATED, "anlatmak", "nakletmek")
        self.assertTrue(yes)
        self.assertIn("tanım", why)


class TestStems(unittest.TestCase):
    def test_an_infinitive_becomes_a_stem(self):
        self.assertEqual(synonyms.stem("anlatmak"), "anlat")
        self.assertEqual(synonyms.stem("bahsetmek"), "bahset")

    def test_a_word_that_is_not_an_infinitive_is_left_alone(self):
        self.assertEqual(synonyms.stem("özellik"), "özellik")


class TestInConversation(unittest.TestCase):
    """Grafta "anlat = bahset" yazıyorsa, kalıp eklemeden anlaşılmalı."""

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "es.lmm")
        memory = Memory()
        memory.learn_word("uçmak", "uçar", "uçamaz")
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        memory.write(Edge("kuş", IS_A, "hayvan", source="sen"))
        memory.write(Edge("anlat", SAME_AS, "bahset",
                          source="vikisözlük:çeviri"))
        memory.save(self.path)

    def test_a_synonym_reaches_the_same_pattern(self):
        session = Session(self.path)
        said = session.respond("penguen bahset")
        self.assertIn("kuş", said.lower())
        self.assertNotIn("anlamadım", said)

    def test_the_original_word_still_works(self):
        session = Session(self.path)
        self.assertIn("kuş", session.respond("penguen anlat").lower())

    def test_an_unrelated_word_is_not_accepted(self):
        """Eş anlamlılık kapı açmıyor, yalnızca kayıtlı olanı geçiriyor."""
        session = Session(self.path)
        self.assertNotIn("hayvan", session.respond("penguen koş").lower())


if __name__ == "__main__":
    unittest.main()
