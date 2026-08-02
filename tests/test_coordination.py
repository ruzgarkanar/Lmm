"""İki özneli sorular: "penguen ve kartal kuş mu".

Ölçümde başarısız olan sorulardan biriydi ve sebebi bilgi eksikliği değildi —
graf ikisini de biliyor. Eksik olan, bir cümlede İKİ ÖZNE olabileceğiydi;
ayrıştırıcı "penguen ve kartal" öbeğini tek bir kavram sanıyordu.

Çözüm kalıp eklemek değil, cümleyi bölmek: her özne için aynı soru normal
yoldan, kapıdan geçerek soruluyor. Birleştirme yalnızca söyleyişte — yani bu
yol hiçbir denetimi atlamıyor.
"""
import os
import tempfile
import unittest

from lmm import coordination
from lmm.cli import Session
from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT, HAS_PROPERTY


class TestSplitting(unittest.TestCase):
    JOINERS = {"ve", "veya"}
    KNOWN = {"penguen", "kartal", "kuş"}

    def test_it_finds_two_subjects(self):
        found = coordination.split_subjects(
            ["penguen", "ve", "kartal", "uçar", "mı"], self.JOINERS, self.KNOWN)
        self.assertEqual(found, (["penguen", "kartal"], ["uçar", "mı"]))

    def test_echo_words_are_dropped(self):
        """"ikisi de" bilgi taşımıyor; iki özne olduğunu bağlaç zaten söylüyor."""
        found = coordination.split_subjects(
            ["penguen", "ve", "kartal", "ikisi", "de", "kuş", "mu"],
            self.JOINERS, self.KNOWN)
        self.assertEqual(found[1], ["kuş", "mu"])

    def test_an_unknown_second_subject_is_refused(self):
        """Bilinmeyen bir şeyi özne diye ayırmak uydurma olurdu."""
        found = coordination.split_subjects(
            ["penguen", "ve", "ejderha", "uçar", "mı"], self.JOINERS, self.KNOWN)
        self.assertEqual(found, (None, None))

    def test_a_sentence_without_a_joiner_is_left_alone(self):
        found = coordination.split_subjects(
            ["penguen", "uçar", "mı"], self.JOINERS, self.KNOWN)
        self.assertEqual(found, (None, None))

    def test_a_joiner_at_the_edge_is_not_a_split(self):
        for tokens in (["ve", "penguen", "uçar"], ["penguen", "uçar", "ve"]):
            self.assertEqual(
                coordination.split_subjects(tokens, self.JOINERS, self.KNOWN),
                (None, None))


class TestCombining(unittest.TestCase):
    def test_two_yeses_become_one(self):
        said = coordination.combine(["evet, penguen bir kuştur.",
                                     "evet, kartal bir kuştur."])
        self.assertTrue(said.startswith("evet, ikisi de"))

    def test_a_disagreement_keeps_both(self):
        """Farkın kendisi cevabın parçası."""
        said = coordination.combine(["hayır, penguen uçamaz.",
                                     "evet, kartal uçar."])
        self.assertIn("hayır", said)
        self.assertIn("evet", said)

    def test_nothing_is_not_invented(self):
        self.assertIsNone(coordination.combine([]))


class TestInConversation(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "b.lmm")
        memory = Memory()
        memory.learn_word("uçmak", "uçar", "uçamaz")
        memory.write(Edge("kuş", CAN, "uçmak", source="sen"))
        memory.write(Edge("kuş", HAS_PROPERTY, "tüylü", source="sen"))
        for concept in ("penguen", "kartal"):
            memory.write(Edge(concept, IS_A, "kuş", source="sen"))
        memory.write(Edge("penguen", CANNOT, "uçmak", source="sen",
                          is_exception=True))
        memory.save(self.path)

    def test_both_are_answered_as_one(self):
        session = Session(self.path)
        said = session.respond("penguen ve kartal kuş mu")
        self.assertTrue(said.startswith("evet, ikisi de"), said)

    def test_it_works_with_the_echo_words(self):
        session = Session(self.path)
        said = session.respond("penguen ve kartal ikisi de kuş mu")
        self.assertTrue(said.startswith("evet, ikisi de"), said)

    def test_a_real_difference_is_reported_as_one(self):
        """Penguen uçamaz, kartal uçar — cevap ikisini de söylemeli."""
        session = Session(self.path)
        said = session.respond("penguen ve kartal uçar mı").lower()
        self.assertIn("hayır", said)
        self.assertIn("evet", said)

    def test_an_unknown_concept_falls_through(self):
        session = Session(self.path)
        said = session.respond("penguen ve ejderha kuş mu")
        self.assertNotIn("ikisi de", said)

    def test_a_single_subject_is_untouched(self):
        session = Session(self.path)
        self.assertTrue(session.respond("penguen uçar mı").startswith("hayır"))


if __name__ == "__main__":
    unittest.main()
