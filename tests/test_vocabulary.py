"""Vocabulary as learned knowledge: packs teach words, not just facts.

Before this, LMM could only discuss what the source code had hardcoded. A domain
now extends the language by shipping its verbs — with no change to the engine.
"""
import os
import tempfile
import unittest

from lmm.lexicon import Lexicon, ACTIVE
from lmm.memory import Memory, Edge, CAN, CANNOT
from lmm.intuition import Intuition, TEACH, UNKNOWN_WORD
from lmm.pack import export_pack, read_pack, merge_pack
from lmm.cli import Session


class TestLexicon(unittest.TestCase):
    def test_core_words_are_known(self):
        lexicon = Lexicon()
        self.assertEqual(lexicon.reading("uçamaz"), ("uçmak", False))
        self.assertEqual(lexicon.surface("uçmak", True), "uçar")

    def test_learning_a_verb_works_both_ways(self):
        lexicon = Lexicon()
        self.assertFalse(lexicon.knows("öğrenir"))
        lexicon.learn_verb("öğrenmek", "öğrenir", "öğrenemez")
        self.assertEqual(lexicon.reading("öğrenemez"), ("öğrenmek", False))
        self.assertEqual(lexicon.surface("öğrenmek", True), "öğrenir")

    def test_signature_changes_when_words_are_learned(self):
        lexicon = Lexicon()
        before = lexicon.signature()
        lexicon.learn_verb("öğrenmek", "öğrenir", "öğrenemez")
        self.assertNotEqual(before, lexicon.signature())

    def test_an_unknown_verb_is_named_rather_than_shrugged_at(self):
        """The 'before' half of the story, on a lexicon of its own."""
        intent = Intuition(lexicon=Lexicon()).understand("robotlar çalışır")
        self.assertEqual(intent.kind, UNKNOWN_WORD)
        self.assertEqual(intent.target, "çalışır")   # it asks for this word

    def test_a_learned_verb_can_be_parsed(self):
        lexicon = Lexicon()
        lexicon.learn_verb("çalışmak", "çalışır", "çalışamaz")
        intent = Intuition(lexicon=lexicon).understand("robotlar çalışır")
        self.assertEqual((intent.kind, intent.concept, intent.relation,
                          intent.target), (TEACH, "robot", CAN, "çalışmak"))


class TestVocabularyInPacks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.pack_path = os.path.join(self.directory, "domain.json")
        source = Memory()
        source.learn_word("hesaplamak", "hesaplar", "hesaplayamaz")
        source.write(Edge("bilgisayar", CAN, "hesaplamak", source="uzman"))
        source.write(Edge("taş", CANNOT, "hesaplamak", source="uzman"))
        export_pack(source, self.pack_path, name="hesap", version="1.0")

    def test_pack_carries_its_words(self):
        pack = read_pack(self.pack_path)
        self.assertIn({"infinitive": "hesaplamak", "positive": "hesaplar",
                       "negative": "hesaplayamaz"}, pack.vocabulary)

    def test_merging_teaches_the_words_then_the_facts(self):
        memory = Memory()
        report = merge_pack(memory, read_pack(self.pack_path))
        self.assertEqual(len(report.words), 1)
        self.assertEqual(len(report.added), 2)
        self.assertTrue(ACTIVE.knows("hesaplar"))

    def test_installed_words_make_new_sentences_understandable(self):
        session = Session(os.path.join(self.directory, "memory.json"))
        merge_pack(session.memory, read_pack(self.pack_path))
        session.save()

        talking = Session(os.path.join(self.directory, "memory.json"))
        self.assertTrue(talking.respond("bilgisayar hesaplar mı").startswith("evet"))
        self.assertIn("bilgisayar", talking.respond("kimler hesaplar"))

    def test_words_survive_a_restart(self):
        """A word learned once is a word known forever, like any other fact."""
        path = os.path.join(self.directory, "memory.json")
        first = Session(path)
        merge_pack(first.memory, read_pack(self.pack_path))
        first.save()

        with open(path, encoding="utf-8") as f:
            self.assertIn("hesaplar", f.read())    # stored, not just in RAM

        second = Session(path)
        self.assertEqual(len(second.memory.vocabulary), 1)
        reply = second.respond("robotlar hesaplar")
        self.assertIn("öğrendim", reply)


if __name__ == "__main__":
    unittest.main()


class TestInflectionsBeyondTheAorist(unittest.TestCase):
    """Fiil keşfi geniş zaman çiftine dayanıyor ama insanlar öyle konuşmuyor.

    "penguen neden UÇAMIYOR" sorusu sözlükte karşılığı olmadığı için nitelik
    sorusu sanılıyor ve cevapsız kalıyordu. Her fiil için sekiz biçim saklamak
    yerine ek soyuluyor — sözlüğün `-ebilir` için zaten yaptığı şeyin aynısı.
    """

    def setUp(self):
        self.lexicon = Lexicon()
        self.lexicon.learn_verb("uçmak", "uçar", "uçamaz")
        self.lexicon.learn_verb("yüzmek", "yüzer", "yüzemez")

    def test_the_aorist_still_reads(self):
        self.assertEqual(self.lexicon.reading("uçar"), ("uçmak", True))
        self.assertEqual(self.lexicon.reading("uçamaz"), ("uçmak", False))

    def test_the_present_continuous_reads_in_both_polarities(self):
        self.assertEqual(self.lexicon.reading("uçuyor"), ("uçmak", True))
        self.assertEqual(self.lexicon.reading("uçmuyor"), ("uçmak", False))
        self.assertEqual(self.lexicon.reading("uçamıyor"), ("uçmak", False))

    def test_the_past_and_the_future_read(self):
        self.assertEqual(self.lexicon.reading("uçtu"), ("uçmak", True))
        self.assertEqual(self.lexicon.reading("uçmadı"), ("uçmak", False))
        self.assertEqual(self.lexicon.reading("uçacak"), ("uçmak", True))

    def test_a_word_that_is_not_a_verb_is_still_refused(self):
        """Ek soymak her kelimeyi fiil yapmamalı."""
        for word in ("masa", "penguen", "kırmızı"):
            self.assertIsNone(self.lexicon.reading(word), word)

    def test_it_reaches_the_right_verb_among_several(self):
        self.assertEqual(self.lexicon.reading("yüzemiyor"), ("yüzmek", False))
