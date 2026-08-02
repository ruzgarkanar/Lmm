"""Derleyici: dil modeli okur, çapa denetler, graf kabul eder.

Buradaki testlerin hepsi sahte bir okuyucuyla çalışır ve hiçbiri ağa çıkmaz. Bu
bir kolaylık değil, testin konusu: denetimin doğruluğu okuyucunun iyi niyetine
bağlı olmamalı. Sahte okuyucu bilerek yalan söyler ve yalanın tutulup
tutulmadığına bakılır.
"""
import unittest

from lmm.compiler import (_appears, Candidate, Report, compile_text,
                          normalise, parse, states_a_type,
                          verify, NOT_IN_DOCUMENT, NOT_IN_ANCHOR, NO_ANCHOR,
                          BAD_RELATION, EMPTY)

DOCUMENT = (
    "Kredi başvurusu müşteri tarafından yapılır. "
    "Vadeli mevduat bir bankacılık ürünüdür. "
    "Sistem, her işlemi kayıt altına alır. "
    "Müşteri verileri gizlidir ve dışarı çıkarılamaz."
)


def reader(reply):
    """Ne söylenirse onu döndüren sahte okuyucu."""
    return lambda chunk: reply


class TestTheAnchorHolds(unittest.TestCase):
    def setUp(self):
        self.document = normalise(DOCUMENT)

    def test_a_fact_the_document_really_states_is_accepted(self):
        candidate = Candidate("vadeli mevduat", "type", "ürün",
                              "Vadeli mevduat bir bankacılık ürünüdür.")
        self.assertTrue(verify(candidate, self.document))

    def test_a_sentence_the_document_never_contained_is_refused(self):
        """Okuyucu cümleyi uydurduysa, olgusu da uydurmadır."""
        candidate = Candidate("kredi", "property", "faizsiz",
                              "Tüm krediler faizsizdir.")
        self.assertFalse(verify(candidate, self.document))
        self.assertEqual(candidate.refusal, NOT_IN_DOCUMENT)

    def test_a_real_sentence_that_does_not_say_the_claim_is_refused(self):
        """En sinsi hata: cümle gerçek, iddia ona ait değil."""
        candidate = Candidate("kredi", "property", "gizli",
                              "Vadeli mevduat bir bankacılık ürünüdür.")
        self.assertFalse(verify(candidate, self.document))
        self.assertEqual(candidate.refusal, NOT_IN_ANCHOR)

    def test_a_fact_with_no_sentence_at_all_is_refused(self):
        candidate = Candidate("kredi", "type", "ürün", "")
        self.assertFalse(verify(candidate, self.document))
        self.assertEqual(candidate.refusal, NO_ANCHOR)

    def test_an_invented_relation_is_refused(self):
        candidate = Candidate("sistem", "sever", "kayıt",
                              "Sistem, her işlemi kayıt altına alır.")
        self.assertFalse(verify(candidate, self.document))
        self.assertEqual(candidate.refusal, BAD_RELATION)

    def test_an_empty_field_is_refused(self):
        candidate = Candidate("", "type", "ürün", "Vadeli mevduat bir ürünüdür.")
        self.assertFalse(verify(candidate, self.document))
        self.assertEqual(candidate.refusal, EMPTY)

    def test_turkish_suffixes_do_not_break_the_check(self):
        """"müşteri" cümlede "müşteri tarafından" diye geçiyor; bu sayılmalı."""
        candidate = Candidate("kredi başvurusu", "can", "yapılmak",
                              "Kredi başvurusu müşteri tarafından yapılır.")
        self.assertTrue(verify(candidate, self.document))


class TestReadingTheReply(unittest.TestCase):
    def test_it_reads_the_four_column_form(self):
        found = parse("kredi | type | ürün | Kredi bir üründür.")
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].concept, found[0].relation, found[0].target),
                         ("kredi", "type", "ürün"))

    def test_a_malformed_line_is_skipped_not_guessed_at(self):
        found = parse("kredi bir üründür\nbaşlık:\nkredi | type | ürün | Kredi bir üründür.")
        self.assertEqual(len(found), 1)

    def test_turkish_capital_i_survives(self):
        found = parse("İstanbul | type | şehir | İstanbul bir şehirdir.")
        self.assertEqual(found[0].concept, "istanbul")


class TestCompilingADocument(unittest.TestCase):
    def test_truth_passes_and_invention_does_not(self):
        """Aynı cevapta biri gerçek biri uydurma; yalnızca gerçek geçmeli."""
        reply = ("vadeli mevduat | type | ürün | Vadeli mevduat bir bankacılık "
                 "ürünüdür.\n"
                 "kredi | property | faizsiz | Tüm krediler faizsizdir.")
        report = compile_text(["cümle"], DOCUMENT, reader(reply))
        self.assertEqual(len(report.accepted), 1)
        self.assertEqual(len(report.refused), 1)
        self.assertEqual(report.accepted[0].concept, "vadeli mevduat")
        self.assertEqual(report.reasons(), {NOT_IN_DOCUMENT: 1})

    def test_a_reader_that_invents_everything_yields_nothing(self):
        """Asıl iddia: okuyucu tamamen yalan söylese bile graf temiz kalır."""
        reply = "\n".join(
            f"uydurma{i} | type | şey | Bu cümle belgede hiç geçmiyor {i}."
            for i in range(20))
        report = compile_text(["cümle"], DOCUMENT, reader(reply))
        self.assertEqual(report.accepted, [])
        self.assertEqual(len(report.refused), 20)

    def test_a_silent_reader_produces_nothing_rather_than_failing(self):
        report = compile_text(["cümle"], DOCUMENT, reader(""))
        self.assertEqual(report.total, 0)

    def test_every_sentence_is_offered_to_the_reader(self):
        seen = []

        def watching(chunk):
            seen.extend(chunk)
            return ""
        compile_text([f"c{i}" for i in range(25)], DOCUMENT, watching, batch=10)
        self.assertEqual(len(seen), 25)


class TestNormalising(unittest.TestCase):
    def test_punctuation_and_spacing_do_not_decide_the_outcome(self):
        self.assertEqual(normalise("Kredi,  bir   ürün!"), "kredi bir ürün")

    def test_turkish_letters_survive(self):
        self.assertEqual(normalise("İŞÇİ ÇĞÖŞÜ"), "işçi çğöşü")



class TestTypeDirection(unittest.TestCase):
    """Tür ilişkisi grafın omurgası; ters yazılan bir tanesi altını bozar."""

    def check(self, sentence, concept, target):
        candidate = Candidate(concept, "type", target, sentence)
        return verify(candidate, normalise(sentence))

    def test_a_real_type_claim_passes(self):
        self.assertTrue(self.check("Linux bir işletim sistemidir.",
                                   "linux", "işletim sistemi"))
        self.assertTrue(self.check("Penguen bir kuştur.", "penguen", "kuş"))

    def test_a_compound_noun_is_not_a_type_claim(self):
        """"Linux dağıtımları" bir tamlamadır; Linux bir dağıtım türü değildir."""
        self.assertFalse(self.check("Linux dağıtımları çok popülerdir.",
                                    "linux", "dağıtım"))

    def test_the_target_coming_first_is_not_a_type_claim(self):
        self.assertFalse(self.check("Kuş türleri arasında penguen de vardır.",
                                    "penguen", "kuş"))

    def test_other_relations_are_not_subject_to_this_check(self):
        candidate = Candidate("linux", "property", "özgür",
                              "Linux özgür bir yazılımdır.")
        self.assertTrue(verify(candidate, normalise("Linux özgür bir yazılımdır.")))

class TestRootMatching(unittest.TestCase):
    """Çapa denetimi ekli bir dilde nasıl yapılır.

    Yüzey biçimi aramak sistematik kayıp veriyordu: "Vaşak ... yabani hayvan
    türlerinin ortak adı" cümlesinden çıkan doğru bir iddia, `türü` ile
    `türlerinin` aynı görünmediği için reddediliyordu. Kök eşleşmesine geçildi.

    Gevşetmenin bir riski var ve sistemin en tehlikeli hata biçimi tam olarak
    o: doğru kavram, ters iddia. "uçar" ile "uçamaz" aynı kökten. O yüzden
    burada asıl sınanan şey uzunluk değil **kutup**.
    """

    def test_an_inflected_form_is_found(self):
        self.assertTrue(_appears("hayvan türü",
                                 "yabani hayvan türlerinin ortak adı"))
        self.assertTrue(_appears("kredi", "kredinin faizi"))

    def test_the_opposite_polarity_is_never_a_match(self):
        self.assertFalse(_appears("uçmak", "penguen uçamaz"))
        self.assertFalse(_appears("uçamamak", "kartal uçar"))
        self.assertFalse(_appears("yapmak", "işçi yapmaz"))
        self.assertFalse(_appears("beslenmek", "bu kuşlar balıkla beslenmez"))

    def test_the_same_polarity_is_a_match(self):
        self.assertTrue(_appears("uçmak", "kartal uçar"))
        self.assertTrue(_appears("yapmak", "işçi yapar"))
        self.assertTrue(_appears("yüzmek", "penguenler çok iyi yüzer"))

    def test_a_root_starting_with_a_negative_letter_is_not_negative(self):
        """"memeli" ile "yapmaz" ikisi de "me" taşır; fark sondadır."""
        self.assertTrue(_appears("memeli", "balinalar memelilerdir"))

    def test_an_unrelated_word_is_not_found(self):
        self.assertFalse(_appears("balina", "kartal bir kuştur"))


class TestTypeClaims(unittest.TestCase):
    """Tür ilişkisi grafın omurgası: ters yazılmış bir tanesi altındaki her
    şeyi bozar. O yüzden ayrı bir kapısı var — ve o kapı iki kez yanlış
    ayarlanmıştı, ikisinde de sessizce.

    Birincisi: bitişiklik izin sayılmış, yani "linux dağıtımları" bir tür
    iddiası olarak geçiyordu. İkincisi: aranan virgül işareti normalleştirilmiş
    metinde hiç bulunmuyordu, yani kural hep "hayır" diyor ve Türkçe
    ansiklopedi tanımlarının olağan biçimini tümden eliyordu.
    """

    def claim(self, concept, target, anchor):
        return Candidate(concept, "type", target, anchor)

    def test_an_indefinite_definition_is_a_type_claim(self):
        for concept, target, anchor in (
                ("kartal", "kuş", "Kartal bir kuştur."),
                ("kredi", "ürün", "Kredi bir bankacılık ürünüdür.")):
            claim = self.claim(concept, target, anchor)
            self.assertTrue(states_a_type(claim, normalise(anchor)), anchor)

    def test_an_appositive_definition_is_a_type_claim(self):
        """Türkçe ansiklopedi tanımının olağan biçimi: "X, ... Y'dir"."""
        for concept, target, anchor in (
                ("volkanoloji", "bilim dalı",
                 "Volkanoloji, yanardağları inceleyen bilim dalıdır."),
                ("penguen", "kuş",
                 "Penguenler, güney yarım kürede yaşayan kuşlardır."),
                ("vaşak", "hayvan türü",
                 "Vaşak, kedigiller familyasından Lynx cinsini oluşturan "
                 "yabani hayvan türlerinin ortak adı.")):
            claim = self.claim(concept, target, anchor)
            self.assertTrue(states_a_type(claim, normalise(anchor)), anchor)

    def test_a_noun_compound_is_not_a_type_claim(self):
        """"Linux dağıtımları" bir tamlamadır; linux bir dağıtım DEĞİLdir."""
        anchor = "Linux dağıtımları arasında Ubuntu vardır."
        claim = self.claim("linux", "dağıtım", anchor)
        self.assertFalse(states_a_type(claim, normalise(anchor)))

    def test_the_other_direction_is_not_a_type_claim(self):
        anchor = "Kuş türleri arasında penguen de vardır."
        claim = self.claim("penguen", "kuş", anchor)
        self.assertFalse(states_a_type(claim, normalise(anchor)))


if __name__ == "__main__":
    unittest.main()
