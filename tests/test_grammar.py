"""Grammar as data: patterns that can be listed, matched, and learned.

Patterns used to be a chain of ifs in the parser, which meant the system could
learn any fact but not one new way of saying one. These check that the data-driven
grammar behaves exactly as the hand-written chain did, and that a pattern can now
be worked out from a single example.
"""
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
        pattern, _ = self.grammar.match("bu cümle hiçbir kalıba uymaz".split(),
                                        self.lexicon)
        self.assertIsNone(pattern)


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
