"""Sohbetin kendisi ve iki kavramın sıralanması.

İkisi de on iki soruluk ölçümde başarısız olan sorulardı ve ikisi de bilgi
eksikliği değildi — sistemin böyle bir sorusu hiç yoktu.
"""
import os
import tempfile
import unittest

from lmm.cli import Session
from lmm.memory import Memory, Edge, IS_A, HAS_PROPERTY
from lmm.thread import Thread


class TestTheThread(unittest.TestCase):
    """Sohbet geçmişi grafa yazılmaz: bir sohbetin neden bahsettiği dünya
    hakkında bir olgu değildir."""

    def test_it_remembers_in_order_newest_first(self):
        thread = Thread()
        for topic in ("kuş", "penguen", "kartal"):
            thread.note(topic)
        self.assertEqual(thread.recent(3), ["kartal", "penguen", "kuş"])

    def test_the_same_topic_twice_running_counts_once(self):
        thread = Thread()
        thread.note("kuş")
        thread.note("kuş")
        self.assertEqual(len(thread), 1)

    def test_returning_to_a_topic_moves_it_to_the_front(self):
        thread = Thread()
        for topic in ("kuş", "penguen", "kuş"):
            thread.note(topic)
        self.assertEqual(thread.recent(1), ["kuş"])

    def test_it_forgets_the_oldest(self):
        thread = Thread(kept=2)
        for topic in ("a", "b", "c"):
            thread.note(topic)
        self.assertEqual(len(thread), 2)

    def test_nothing_is_not_invented(self):
        self.assertEqual(Thread().recent(), [])


class TestTheThreadInConversation(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "t.lmm")
        memory = Memory()
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        memory.save(self.path)

    def test_an_empty_conversation_says_so(self):
        session = Session(self.path)
        self.assertIn("henüz", session.respond("az önce ne konuşuyorduk"))

    def test_it_names_what_was_asked_about(self):
        session = Session(self.path)
        session.respond("penguen nedir")
        session.respond("kartal nedir")
        said = session.respond("az önce ne konuşuyorduk").lower()
        self.assertIn("kartal", said)
        self.assertIn("penguen", said)

    def test_the_thread_never_reaches_the_graph(self):
        """Geçici bağlam kalıcı hafızayı kirletmemeli."""
        session = Session(self.path)
        before = len(session.memory.edges)
        session.respond("penguen nedir")
        session.respond("az önce ne konuşuyorduk")
        self.assertEqual(len(session.memory.edges), before)


class TestWhichIsMore(unittest.TestCase):
    """Sıralama kayıttan gelir, tahminden değil."""

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "k.lmm")
        memory = Memory()
        for concept in ("kartal", "serçe", "penguen"):
            memory.write(Edge(concept, IS_A, "kuş", source="sen"))
        memory.write(Edge("kartal", HAS_PROPERTY, "hızlı", object="serçe",
                          role="çıkış", source="sen"))
        memory.save(self.path)
        self.session = Session(self.path)

    def test_a_recorded_comparison_is_answered(self):
        said = self.session.respond("hangisi daha hızlı kartal mı serçe mi")
        self.assertTrue(said.startswith("kartal"), said)

    def test_the_order_of_the_question_does_not_change_the_answer(self):
        said = self.session.respond("hangisi daha hızlı serçe mi kartal mı")
        self.assertTrue(said.startswith("kartal"), said)

    def test_without_a_comparison_it_does_not_rank(self):
        """Sıralamayı tahmin etmek, cevabı uydurmakla aynı şey."""
        said = self.session.respond("hangisi daha hızlı kartal mı penguen mi")
        self.assertIn("bilmiyorum", said)

    def test_an_unknown_concept_is_refused(self):
        said = self.session.respond("hangisi daha hızlı kartal mı ejderha mı")
        self.assertIn("bilmiyorum", said)


if __name__ == "__main__":
    unittest.main()


class TestSayingMore(unittest.TestCase):
    """"başka ne biliyorsun" — söylenmemiş olanı söylemek, tekrar etmemek.

    Bir insan sohbeti en çok bununla ilerliyor ve sistemin böyle bir sorusu
    hiç yoktu. Neyin söylendiği oturumda tutuluyor, grafa yazılmıyor: bir
    şeyin anlatılmış olması dünya hakkında bir olgu değil.
    """

    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "m.lmm")
        memory = Memory()
        memory.write(Edge("penguen", IS_A, "kuş", source="sen"))
        memory.write(Edge("penguen", HAS_PROPERTY, "siyah", source="sen"))
        memory.write(Edge("penguen", HAS_PROPERTY, "tüylü", source="sen"))
        memory.save(self.path)
        self.session = Session(self.path)

    def test_it_says_something_new_each_time(self):
        self.session.respond("penguen nedir")
        first = self.session.respond("başka ne biliyorsun")
        second = self.session.respond("başka ne biliyorsun")
        self.assertNotEqual(first, second)

    def test_it_stops_instead_of_repeating(self):
        self.session.respond("penguen nedir")
        for _ in range(5):
            said = self.session.respond("başka ne biliyorsun")
        self.assertIn("başka bir şey yok", said)

    def test_without_a_topic_it_says_so(self):
        session = Session(self.path)
        self.assertIn("henüz", session.respond("başka ne biliyorsun"))

    def test_it_never_reaches_the_graph(self):
        before = len(self.session.memory.edges)
        self.session.respond("penguen nedir")
        self.session.respond("başka ne biliyorsun")
        self.assertEqual(len(self.session.memory.edges), before)
