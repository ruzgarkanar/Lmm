"""Grammar as data: patterns that can be listed, matched, and learned.

Patterns used to be a chain of ifs in the parser, which meant the system could
learn any fact but not one new way of saying one. These check that the data-driven
grammar behaves exactly as the hand-written chain did, and that a pattern can now
be worked out from a single example.
"""
import os
import unittest

from lmm.grammar import (Grammar, Pattern, learn_pattern, KAVRAM, TUR, NITELIK,
                         SOZ, FIIL, SORU, FROM_VERB)
from lmm.turkish import turkish, TurkishMorphology, TEACH, ASK, ASK_DESCRIBE
from lmm.relations import IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY
from lmm.lexicon import Lexicon
from lmm.intuition import Intuition


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.grammar = turkish()
        self.lexicon = Lexicon()

    def _read(self, sentence):
        """(kind, relation, concept, target) — the object is checked elsewhere."""
        tokens = sentence.split()
        pattern, captured = self.grammar.match(tokens, self.lexicon)
        self.assertIsNotNone(pattern, sentence)
        return self.grammar.read(pattern, captured)[:4]

    def test_a_type_lesson(self):
        self.assertEqual(self._read("penguen bir kuştur"),
                         (TEACH, IS_A, "penguen", "kuş"))

    def test_a_property_lesson(self):
        self.assertEqual(self._read("kuşlar tüylüdür"),
                         (TEACH, HAS_PROPERTY, "kuş", "tüylü"))

    def test_polarity_comes_from_the_verb(self):
        self.assertEqual(self._read("penguen uçamaz"),
                         (TEACH, CANNOT, "penguen", "uçmak"))
        self.assertEqual(self._read("kuşlar uçar"),
                         (TEACH, CAN, "kuş", "uçmak"))

    def test_a_denial(self):
        self.assertEqual(self._read("penguen bir memeli değildir"),
                         (TEACH, NOT_A, "penguen", "memeli"))

    def test_order_decides_between_two_shapes(self):
        """"kim uçar" and "kuşlar uçar" have the same shape."""
        kind = self._read("kim uçar")[0]
        self.assertEqual(kind, "ASK_WHO")

    def test_a_sentence_with_no_pattern_does_not_match(self):
        """Hiçbir yüklem taşımayan bir dizi kalıba girmez.

        Bu test önce "bu cümle hiçbir kalıba uymaz" kullanıyordu ve `uymaz`
        derlemde tanıklanan bir fiil olduğu için artık KALIP EŞLEŞİYOR:
        `bu --cannot--> uymak`. Kalıbın gevşemesi kusur değil — alttaki kavram
        kapısı `bu`yu reddediyor ve grafa hiçbir şey yazılmıyor; katmanlı
        savunmanın çalıştığı yer tam burası. Test onu ayrıca ölçüyor.

        Buradaki asıl soru başka: yüklemsiz bir dizi kalıba girmemeli.
        """
        pattern, _ = self.grammar.match("masa sandalye pencere".split(),
                                        self.lexicon)
        self.assertIsNone(pattern)

    def test_a_loose_match_still_writes_nothing(self):
        """Kalıp eşleşse bile kavram olmayan bir kavram grafa geçemez."""
        import shutil
        import tempfile
        from lmm.cli import Session
        from lmm.memory import Memory
        path = os.path.join(tempfile.mkdtemp(), "gevsek.lmm")
        Memory().save(path)
        session = Session(path)
        session.respond("bu cümle hiçbir kalıba uymaz")
        self.assertEqual(list(session.memory.edges), [])


class TestSlots(unittest.TestCase):
    def setUp(self):
        self.grammar = Grammar([], TurkishMorphology())
        self.lexicon = Lexicon()

    def _capture(self, slot, token):
        return self.grammar._capture(slot, token, self.lexicon)

    def test_a_property_slot_demands_a_copula(self):
        self.assertEqual(self._capture(NITELIK, "beyazdır"), "beyaz")
        self.assertIsNone(self._capture(NITELIK, "beyaz"))

    def test_a_type_slot_accepts_either(self):
        self.assertEqual(self._capture(TUR, "kuştur"), "kuş")
        self.assertEqual(self._capture(TUR, "kuş"), "kuş")

    def test_a_concept_slot_drops_the_plural(self):
        self.assertEqual(self._capture(KAVRAM, "kuşlar"), "kuş")

    def test_a_verb_slot_reads_the_polarity(self):
        self.assertEqual(self._capture(FIIL, "uçamaz"), ("uçmak", False))
        self.assertIsNone(self._capture(FIIL, "zıplar"))

    def test_a_literal_must_match_exactly(self):
        self.assertEqual(self._capture("bir", "bir"), "bir")
        self.assertIsNone(self._capture("bir", "iki"))


class TestLearningAPattern(unittest.TestCase):
    """The point of the refactor: a new way of saying something can be taught."""

    def setUp(self):
        self.lexicon = Lexicon()
        self.morphology = TurkishMorphology()

    def test_a_pattern_is_worked_out_from_one_example(self):
        pattern = learn_pattern("penguen hakkında konuş".split(), ASK_DESCRIBE,
                                None, 0, None, self.lexicon, self.morphology)
        self.assertEqual(pattern.tokens, [KAVRAM, "hakkında", "konuş"])

    def test_the_learned_pattern_then_matches_new_sentences(self):
        grammar = turkish()
        grammar.add(learn_pattern("penguen hakkında konuş".split(), ASK_DESCRIBE,
                                  None, 0, None, self.lexicon, self.morphology),
                    first=True)
        intuition = Intuition(lexicon=self.lexicon, grammar=grammar)
        intent = intuition.understand("Kartal hakkında konuş")
        self.assertEqual(intent.kind, ASK_DESCRIBE)
        self.assertEqual(intent.concept, "kartal")

    def test_a_verb_in_the_example_becomes_a_verb_slot(self):
        pattern = learn_pattern("acaba kuşlar uçar mı".split(), ASK, CAN, 1, 2,
                                self.lexicon, self.morphology)
        self.assertEqual(pattern.tokens, ["acaba", KAVRAM, FIIL, SORU])


class TestTheModelFileCarriesHowToSpeak(unittest.TestCase):
    """The artifact is everything it learned — facts, words, and ways of saying.

    An LLM ships weights. This ships what it knows and how it can be spoken to,
    in one file, all of it readable.
    """

    def test_a_learned_pattern_survives_a_restart(self):
        import os
        import tempfile
        from lmm.cli import Session

        path = os.path.join(tempfile.mkdtemp(), "model.lmm")
        session = Session(path)
        session.respond("penguen bir kuştur")
        pattern = learn_pattern("penguen hakkında konuş".split(), ASK_DESCRIBE,
                                None, 0, None, session.language.lexicon,
                                session.language.grammar.morphology)
        session.memory.learn_pattern(pattern)
        session.save()

        later = Session(path)                      # a fresh set of organs
        answer = later.respond("penguen hakkında konuş")
        self.assertIn("Penguen bir kuştur", answer)

    def test_a_pack_carries_patterns_too(self):
        import os
        import tempfile
        from lmm.memory import Memory
        from lmm.pack import export_pack, read_pack, merge_pack

        source = Memory()
        source.learn_pattern(Pattern([KAVRAM, "hakkında", "konuş"], ASK_DESCRIBE,
                                     None, 0, None))
        path = os.path.join(tempfile.mkdtemp(), "dil.json")
        export_pack(source, path, name="konusma")

        target = Memory()
        merge_pack(target, read_pack(path))
        self.assertEqual(len(target.patterns), 1)


class TestInducingPatternsWithNobodyLabelling(unittest.TestCase):
    """Where the patterns come from once we stop writing them.

    We already know what a plain sentence means, because our own parser reads
    it. So a passage paired with its restatement is a labelled example that no
    human labelled — and the shape of the prose can be read off it.
    """

    def setUp(self):
        from lmm.grammar import induce
        from lmm.lexicon import ACTIVE
        self.induce = induce
        self.lexicon = ACTIVE
        self.morphology = TurkishMorphology()
        self.parser = Intuition().understand

    def _induce(self, pairs, **kwargs):
        return self.induce(pairs, self.parser, self.morphology, self.lexicon,
                           **kwargs)

    def test_a_shape_seen_twice_becomes_a_pattern(self):
        found = self._induce([
            {"düzyazı": "Penguen aslında bir kuştur.",
             "sade": "penguen bir kuştur"},
            {"düzyazı": "Serçe aslında bir kuştur.",
             "sade": "serçe bir kuştur"},
        ])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].tokens, [KAVRAM, "aslında", "bir", TUR])
        self.assertEqual(found[0].kind, TEACH)
        self.assertEqual(found[0].relation, IS_A)

    def test_the_induced_pattern_reads_a_sentence_it_never_saw(self):
        found = self._induce([
            {"düzyazı": "Penguen aslında bir kuştur.",
             "sade": "penguen bir kuştur"},
            {"düzyazı": "Serçe aslında bir kuştur.",
             "sade": "serçe bir kuştur"},
        ])
        grammar = turkish()
        for pattern in found:
            grammar.add(pattern, first=True)
        intent = Intuition(grammar=grammar).understand("Devekuşu aslında bir kuştur")
        self.assertEqual((intent.kind, intent.concept, intent.target),
                         (TEACH, "devekuşu", "kuş"))

    def test_one_sighting_is_a_coincidence_not_a_rule(self):
        found = self._induce([{"düzyazı": "Penguen aslında bir kuştur.",
                               "sade": "penguen bir kuştur"}])
        self.assertEqual(found, [])

    def test_long_sentences_teach_nothing(self):
        """A pattern taken from a twenty-word sentence matches only itself."""
        prose = ("Penguen bilindiği üzere kanatları olmasına rağmen uçamayan "
                 "ama çok iyi yüzebilen bir kuştur")
        found = self._induce([{"düzyazı": prose, "sade": "penguen bir kuştur"},
                              {"düzyazı": prose, "sade": "penguen bir kuştur"}])
        self.assertEqual(found, [])


class TestConceptsMadeOfSeveralWords(unittest.TestCase):
    """Real domain knowledge is terms, not single words.

    "müşteri bakiyesi", "kredi tahsisi", "tool broker" — every concept in this
    system used to be one token, which is why a real banking paper taught it
    almost nothing.
    """

    def setUp(self):
        self.intuition = Intuition()

    def _read(self, sentence):
        intent = self.intuition.understand(sentence)
        return intent.kind, intent.relation, intent.concept, intent.target

    def test_a_term_is_held_together(self):
        self.assertEqual(self._read("müşteri bakiyesi bir bilgi türüdür"),
                         (TEACH, IS_A, "müşteri bakiyesi", "bilgi türü"))

    def test_the_subject_takes_the_modifiers_and_the_predicate_stays_short(self):
        """It once read this as 'müşteri' being 'bakiyesi gizli'."""
        self.assertEqual(self._read("müşteri bakiyesi gizlidir"),
                         (TEACH, HAS_PROPERTY, "müşteri bakiyesi", "gizli"))

    def test_terms_work_in_questions_too(self):
        self.assertEqual(self._read("müşteri bakiyesi gizli mi"),
                         (ASK, HAS_PROPERTY, "müşteri bakiyesi", "gizli"))

    def test_single_word_sentences_read_exactly_as_before(self):
        self.assertEqual(self._read("kuşlar uçar"), (TEACH, CAN, "kuş", "uçmak"))
        self.assertEqual(self._read("kar beyazdır"),
                         (TEACH, HAS_PROPERTY, "kar", "beyaz"))
        self.assertEqual(self._read("penguen bir kuştur"),
                         (TEACH, IS_A, "penguen", "kuş"))

    def test_a_verb_is_never_swallowed_by_the_subject(self):
        """Descending widths once broke out of the search on the first try."""
        self.assertEqual(self._read("kuşlar uçar")[3], "uçmak")


if __name__ == "__main__":
    unittest.main()


class TestAQuestionIsNeverAFact(unittest.TestCase):
    """Soru eki koşaç alınca soru olmaktan çıkmaz.

    "mudur" koşaç taşıdığı için NITELIK yuvasına giriyordu ve "penguen mutlu
    mudur" cümlesi [KAVRAM=penguen mutlu, NITELIK=mu] diye okunup TEACH
    sayılıyordu. Sonuç grafa yazılan bir saçmalıktı:

        sence penguenler mutlu —property→ mu   [sen]

    Cevap uydurulmuyordu ama hafıza kirleniyordu — ki hafıza bu projenin tek
    varlığı. Ayrıştırıcı soru ile bildirmeyi ayıramadığında varsayılanı "bu bir
    bilgidir" olamaz.
    """

    def setUp(self):
        from lmm.intuition import Intuition
        from lmm.network import MiniNetwork
        self.language = Intuition(network=MiniNetwork.default())

    def kind(self, sentence):
        return self.language.understand(sentence).kind

    def test_a_question_with_a_fused_copula_is_not_teaching(self):
        for sentence in ("penguen mutlu mudur", "penguen bir kuş mudur",
                         "sence penguenler mutlu mudur"):
            self.assertNotEqual(self.kind(sentence), TEACH, sentence)

    def test_the_bare_particle_still_asks(self):
        self.assertEqual(self.kind("penguen mutlu mu"), ASK)

    def test_a_statement_is_still_teaching(self):
        for sentence in ("penguen mutludur", "kartal bir kuştur"):
            self.assertEqual(self.kind(sentence), TEACH, sentence)

    def test_not_understanding_beats_writing_nonsense(self):
        """Anlaşılmayan bir soru, bilgi diye yazılmaktansa anlaşılmasın."""
        self.assertNotEqual(self.kind("kuşların özellikleri nelerdir"), TEACH)


class TestDenialIsNotSwallowed(unittest.TestCase):
    """Olumsuzluk kelimesi bir öbeğin içine giremez.

    "penguen bir kuş değil mi" cümlesinin hedefi `kuş değil` diye okunuyordu;
    graf öyle bir kavram arıyor, bulamıyor ve soru sessizce cevapsız kalıyordu.
    Oysa Türkçe'de bu kalıp olumsuzluk sormaz, TEYİT ister — cevabı "evet, bir
    kuştur" olmalı.
    """

    def setUp(self):
        import os
        import tempfile
        from lmm.memory import Memory, Edge, IS_A, HAS_PROPERTY
        from lmm.cli import Session
        path = os.path.join(tempfile.mkdtemp(), "d.lmm")
        memory = Memory()
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        memory.write(Edge("kartal", HAS_PROPERTY, "hızlı", source="sen"))
        memory.save(path)
        self.session = Session(path)

    def test_a_negative_confirmation_question_is_answered_yes(self):
        said = self.session.respond("penguen bir kuş değil mi")
        self.assertTrue(said.startswith("evet"), said)

    def test_it_works_for_properties_too(self):
        said = self.session.respond("kartal hızlı değil mi")
        self.assertTrue(said.startswith("evet"), said)

    def test_a_trailing_filler_does_not_break_it(self):
        """"yani" hiçbir şey eklemiyor; cümleyi düşürmemeli."""
        said = self.session.respond("penguen bir kuş değil mi yani")
        self.assertTrue(said.startswith("evet"), said)

    def test_a_real_denial_is_still_a_denial(self):
        said = self.session.respond("penguen bir balık değildir")
        self.assertNotIn("evet, penguen bir balık", said)


class TestAnInterrogativeIsNeverATeaching(unittest.TestCase):
    """Soru sözcüğü taşıyan cümle bildirme olamaz.

    Ayrıştırıcı yanılabilir; yanıldığında bedeli ağır ve kalıcı. Ölçüldü:

        "kartal kaç yaşında yaşar"
          -> TEACH sayıldı
          -> grafa `kartal kaç --can--> yaşamak [rol=yer, nesne=yaşın]` yazıldı

    Hafızayı kirletmek bu mimaride yapılabilecek en pahalı hata: cevap
    uydurulmuyor ama yanlış bilgi kalıcı oluyor ve bir daha çıkmıyor. Kapı,
    yanılmayı cevapsızlığa çeviriyor — kirletmektense anlamamak.
    """

    def setUp(self):
        import os
        import tempfile
        from lmm.memory import Memory, Edge, IS_A
        from lmm.cli import Session
        path = os.path.join(tempfile.mkdtemp(), "s.lmm")
        memory = Memory()
        memory.learn_word("yaşamak", "yaşar", "yaşamaz")
        memory.learn_word("uçmak", "uçar", "uçamaz")
        memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        memory.save(path)
        self.session = Session(path)
        self.before = len(self.session.memory.edges)

    def test_a_question_word_stops_the_lesson(self):
        for line in ("kartal kaç yaşında yaşar", "kuş nasıl uçar",
                     "kartal niye uçar", "penguen ne kadar yaşar"):
            self.session.respond(line)
        self.assertEqual(len(self.session.memory.edges), self.before)

    def test_a_real_statement_still_gets_through(self):
        self.session.respond("kartal güçlüdür")
        self.assertGreater(len(self.session.memory.edges), self.before)

    def test_the_closed_class_covers_the_common_ones(self):
        from lmm.intuition import _MORPHOLOGY
        for word in ("kaç", "nasıl", "neden", "nerede", "hangi", "kim"):
            self.assertIn(word, _MORPHOLOGY.interrogatives, word)


class TestNothingWrongEntersAConcept(unittest.TestCase):
    """Bir kavram öbeğine neyin giremeyeceği — hepsi aynı sınıftan hata.

    Gece boyunca aynı kusur beş kelime sınıfında ayrı ayrı yakalandı ve her
    biri grafa saçma bir kayıt yazıyordu:

        değil       "penguen bir kuş değil mi"      -> hedef `kuş değil`
        ile         "penguen ile kartal aynı mı"    -> kavram `penguen ile kartal`
        çok         "kartal çok çok çok hızlıdır"   -> kavram `kartal çok çok`
        hem         "kartal hem hızlı hem güçlüdür" -> kavram `kartal hem hızlı`
        FİİL        "penguen uçar ve uçamaz"        -> kavram `penguen uçar`

    Sonuncusu en tehlikelisi: cümlenin yüklemi kavramın parçası sanılıyor, hem
    kavram hem olgu bozuluyor.

    Ve bir ders: kural kelimenin kendisinde değil YERİNDE. Pekiştireci her
    yuvada yasaklamak "çok hızlı" gibi geçerli nitelikleri düşürüyordu.
    """

    def setUp(self):
        import os
        import tempfile
        from lmm.memory import Memory, Edge, IS_A
        from lmm.cli import Session
        path = os.path.join(tempfile.mkdtemp(), "n.lmm")
        memory = Memory()
        memory.learn_word("uçmak", "uçar", "uçamaz")
        memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        memory.save(path)
        self.session = Session(path)
        self.before = len(self.session.memory.edges)

    def written(self, line):
        self.session.respond(line)
        return [(e.concept, e.relation, e.target)
                for e in self.session.memory.edges[self.before:]]

    def test_an_intensifier_is_not_part_of_a_concept(self):
        self.assertEqual(self.written("kartal çok çok çok hızlıdır"), [])

    def test_a_correlative_is_not_part_of_a_concept(self):
        self.assertEqual(self.written("kartal hem hızlı hem güçlüdür"), [])

    def test_a_verb_is_never_inside_a_concept(self):
        self.assertEqual(self.written("penguen uçar ve uçamaz"), [])

    def test_an_intensifier_is_still_fine_in_a_property(self):
        """Kural yerinde: "çok hızlı" geçerli bir niteliktir."""
        found = self.written("kartal çok hızlıdır")
        self.assertEqual(found, [("kartal", "property", "çok hızlı")])

    def test_a_real_compound_concept_survives(self):
        found = self.written("müşteri bakiyesi gizlidir")
        self.assertEqual(found, [("müşteri bakiyesi", "property", "gizli")])


class TestCliticsAreNotContent(unittest.TestCase):
    """Ayrı yazılan "da"/"de" her zaman pekiştirmedir, hiçbir zaman kavram.

    Türkçe'nin yazımı ayrımı zaten yapıyor: ek olduğunda bitişik yazılır
    ("kartalda" bulunma), pekiştirme olduğunda ayrı ("kartal da"). Biz yalnızca
    okuyoruz — liste değil, imla kuralı.

    Üç ayrı yerde sızıyordu: çok kelimeli öbeğe, tek kelimelik kavram yuvasına,
    ve soru dizilişine.
    """

    def setUp(self):
        import os
        import tempfile
        from lmm.memory import Memory, Edge, IS_A, CAN
        from lmm.cli import Session
        path = os.path.join(tempfile.mkdtemp(), "c.lmm")
        memory = Memory()
        memory.learn_word("uçmak", "uçar", "uçamaz")
        memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        memory.save(path)
        self.session = Session(path)

    def test_a_clitic_before_the_verb_is_ignored(self):
        self.assertTrue(self.session.respond("kartal da uçar mı").startswith("evet"))

    def test_a_clitic_before_the_particle_is_ignored(self):
        self.assertTrue(self.session.respond("kartal da mı uçuyor").startswith("evet"))

    def test_other_clitics_work_the_same(self):
        self.assertTrue(self.session.respond("kartal bile uçar mı").startswith("evet"))

    def test_the_focus_order_is_read(self):
        """"kartal mı uçuyor" — ek fiilden önce; Türkçe'nin gerçek dizilişi."""
        self.assertTrue(self.session.respond("kartal mı uçuyor").startswith("evet"))

    def test_the_plain_question_still_works(self):
        self.assertTrue(self.session.respond("kartal uçar mı").startswith("evet"))
