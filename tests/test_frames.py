"""Cümleyi durum ekleriyle okumak.

Bu modülün varlık sebebi ölçüm: aynı belgede sözcük sıralı kalıplar 23,8
olgu/100 cümle veriyordu, ek tabanlı çerçeveler 49,3 — ve tür testleriyle
temizlenince 21,3, ama bu sefer çıkanların çoğu gerçek.
"""
import collections
import unittest

from lmm.frames import (ABLATIVE, ACCUSATIVE, AORIST, COPULA, DATIVE,
                        LOCATIVE, NOMINATIVE, case_of, predicate_of, read,
                        to_fact)
from lmm.lexicon import Lexicon


class TestCases(unittest.TestCase):
    def test_it_reads_the_role_off_the_ending(self):
        self.assertEqual(case_of("müşteriye"), ("müşteri", DATIVE))
        self.assertEqual(case_of("kutupta"), ("kutup", LOCATIVE))
        self.assertEqual(case_of("evden"), ("ev", ABLATIVE))
        self.assertEqual(case_of("veriyi"), ("veri", ACCUSATIVE))

    def test_a_bare_word_is_nominative(self):
        self.assertEqual(case_of("sistem"), ("sistem", NOMINATIVE))

    def test_the_corpus_settles_an_ambiguous_ending(self):
        """"banka" bir kelime mi, "bank"a yönelme mi? Kural karar veremez.

        Sıklık karar verir: kök kelimeden nadirse ek gerçek değildir.
        """
        seen = collections.Counter({"banka": 500, "bank": 3,
                                    "müşteriye": 4, "müşteri": 900})
        self.assertEqual(case_of("banka", seen), ("banka", NOMINATIVE))
        self.assertEqual(case_of("müşteriye", seen), ("müşteri", DATIVE))

    def test_without_a_corpus_the_longest_ending_wins(self):
        """Sayaç yoksa eski davranış sürsün; sistem sayaçsız da çalışmalı."""
        self.assertEqual(case_of("banka"), ("bank", DATIVE))


class TestPredicates(unittest.TestCase):
    def test_it_names_the_suffix_class(self):
        self.assertEqual(predicate_of("kuştur"), ("kuş", COPULA))
        self.assertEqual(predicate_of("beyazdır"), ("beyaz", COPULA))

    def test_a_plural_is_not_a_predicate(self):
        self.assertEqual(predicate_of("kuşlar"), (None, None))

    def test_the_lexicon_settles_a_verb_that_looks_like_a_copula(self):
        """"üretir" = "üret"+ir geniş zamandır, "üre"+tir koşaç değil.

        Aynı harfler iki ayrı yapı; kural ayıramaz, TANIK ayırır.

        Bu test önce "sözlüksüz yanılır" diye kuralın kusurunu belgeliyordu.
        Artık derlem ikinci tanık: `üretir` orada da geçiyor ve sözlük
        öğretilmemişken bile doğru okunuyor. Kusur kalktı, test onu takip
        ediyor — kuralın tek başına yanıldığı, hiçbir tanığın olmadığı
        uydurma bir kelimeyle gösteriliyor.
        """
        lexicon = Lexicon()
        lexicon.learn_verb("üretmek", "üretir", "üretmez")
        self.assertEqual(predicate_of("üretir", lexicon), ("üretmek", AORIST))
        self.assertEqual(predicate_of("üretir")[0], "üretmek")  # derlem de bilir
        self.assertEqual(predicate_of("zördetir")[1], COPULA)  # tanıksız yanılır


class TestFrames(unittest.TestCase):
    def setUp(self):
        self.lexicon = Lexicon()
        for infinitive, positive, negative in (
                ("kaydetmek", "kaydeder", "kaydetmez"),
                ("yaşamak", "yaşar", "yaşamaz"),
                ("vermek", "verir", "vermez")):
            self.lexicon.learn_verb(infinitive, positive, negative)

    def frame(self, sentence, seen=None):
        return read(sentence.split(), seen, self.lexicon)

    def test_every_role_is_placed(self):
        # Sayaç veriliyor: "banka" sık, "bank" nadir -> yalın kalır ve yönelme
        # yuvasını kapatmaz. Sayaçsız hâli ayrı testte.
        seen = collections.Counter({"banka": 500, "bank": 2,
                                    "müşteriye": 5, "müşteri": 800,
                                    "krediyi": 5, "kredi": 400})
        found = self.frame("banka müşteriye krediyi verir", seen)
        self.assertEqual(found.roles[DATIVE], "müşteri")
        self.assertEqual(found.object, "kredi")
        self.assertEqual(found.tense, AORIST)

    def test_the_head_of_a_phrase_is_the_last_word_not_the_first(self):
        """Türkçe tamlamada baş sondadır: "basit sistem" öznesi "sistem"dir.

        İlkini almak sıfatları özne yapıyordu.
        """
        self.assertEqual(self.frame("basit sistem kaydeder").subject, "sistem")

    def test_a_determiner_cannot_be_the_head(self):
        """"kartal bir kuştur" — son yalın kelime "bir"dir ama baş "kartal"."""
        self.assertEqual(self.frame("kartal bir kuştur").subject, "kartal")

    def test_a_sentence_with_no_predicate_yields_nothing(self):
        self.assertIsNone(self.frame("kırmızı büyük ev"))
        self.assertIsNone(self.frame("kuş"))


class TestFacts(unittest.TestCase):
    def setUp(self):
        self.lexicon = Lexicon()
        self.lexicon.learn_verb("yaşamak", "yaşar", "yaşamaz")
        self.seen = collections.Counter()
        for word in ("kartal", "kar", "penguen"):
            self.seen[word] = 100
            for suffix in ("i", "e", "de", "den", "in", "ler"):
                self.seen[word + suffix] = 20

    def fact(self, sentence):
        found = read(sentence.split(), self.seen, self.lexicon)
        return to_fact(found, self.lexicon, self.seen) if found else None

    def test_the_article_marks_a_type_claim(self):
        self.assertEqual(self.fact("kartal bir kuştur"),
                         ("kartal", "type", "kuş"))

    def test_without_the_article_it_is_a_property(self):
        self.assertEqual(self.fact("kar beyazdır"), ("kar", "property", "beyaz"))

    def test_a_known_verb_gives_an_ability(self):
        self.assertEqual(self.fact("penguen kutupta yaşar"),
                         ("penguen", "can", "yaşamak"))

    def test_an_adverb_cannot_be_a_subject(self):
        """"henüz" hiçbir durum eki almaz, dolayısıyla isim değildir.

        Bu denetim olmadan "henüz | can | yapılmak" gibi olgular çıkıyordu.
        """
        self.seen["henüz"] = 100        # sık, ama çekimsiz
        self.assertIsNone(self.fact("henüz kutupta yaşar"))

    def test_a_predicate_the_lexicon_does_not_know_is_refused(self):
        """Kuraldan üretilen kök güvenilmez: "riskti" -> "ti" çıkıyordu."""
        self.seen["risk"] = 100
        for suffix in ("i", "e", "de", "den"):
            self.seen["risk" + suffix] = 20
        self.assertIsNone(self.fact("risk büyükti"))


if __name__ == "__main__":
    unittest.main()


class TestPolarityAndNumber(unittest.TestCase):
    """İki kapalı sınıf: olumsuzluk eki ve çoğul eki.

    İkisi de atlanınca sessizce yanlış olgu üretiliyordu — "penguen uçamaz"
    cümlesinden "penguen can uçmak", ve "kuşlar" ile "kuş" iki ayrı kavram.
    """

    def setUp(self):
        self.lexicon = Lexicon()
        self.lexicon.learn_verb("uçmak", "uçar", "uçamaz")
        self.seen = collections.Counter()
        for word in ("penguen", "kuş"):
            self.seen[word] = 100
            for suffix in ("i", "e", "de", "den", "in", "ler"):
                self.seen[word + suffix] = 20

    def fact(self, sentence):
        found = read(sentence.split(), self.seen, self.lexicon)
        return to_fact(found, self.lexicon, self.seen) if found else None

    def test_a_negative_verb_gives_a_negative_fact(self):
        self.assertEqual(self.fact("penguen uçamaz"),
                         ("penguen", "cannot", "uçmak"))

    def test_a_positive_verb_gives_a_positive_fact(self):
        self.assertEqual(self.fact("penguen uçar"), ("penguen", "can", "uçmak"))

    def test_the_plural_is_stripped_from_the_subject(self):
        """"kuşlar uçar" kuşlar hakkında değil, kuş hakkındadır."""
        self.assertEqual(self.fact("kuşlar uçar"), ("kuş", "can", "uçmak"))


class TestMood(unittest.TestCase):
    """Kip de kapalı bir sınıf: gereklilik ve koşul.

    İkisi de atlanınca yanlış okunuyordu: "tanımlanmalı" hiç tanınmıyordu,
    "yapılmalıdır" koşaç sanılıyordu, "yağarsa" görülmüyordu.
    """

    def setUp(self):
        self.lexicon = Lexicon()
        self.seen = collections.Counter()
        for word in ("sistem", "yağmur"):
            self.seen[word] = 100
            for suffix in ("i", "e", "de", "den", "in", "ler"):
                self.seen[word + suffix] = 20

    def fact(self, sentence):
        found = read(sentence.split(), self.seen, self.lexicon)
        return to_fact(found, self.lexicon, self.seen) if found else None

    def test_an_obligation_gets_its_own_relation(self):
        self.assertEqual(self.fact("sistem kaydetmelidir"),
                         ("sistem", "must", "kaydetmek"))

    def test_the_stem_is_returned_as_an_infinitive(self):
        """"kısıtlan" bir kelime değil; graf kavramları tam tutmalı."""
        self.seen["erişim"] = 100
        for suffix in ("i", "e", "de", "den"):
            self.seen["erişim" + suffix] = 20
        self.assertEqual(self.fact("erişim kısıtlanmalı"),
                         ("erişim", "must", "kısıtlanmak"))

    def test_a_conditional_clause_is_not_a_fact(self):
        """"yağmur yağarsa" doğru ya da yanlış olamaz — bir kuralın önkoşulu.

        Olgu saymak, henüz temsil edemediğimiz bir şeyi varmış gibi göstermek.
        """
        self.assertIsNone(self.fact("yağmur yağarsa"))
