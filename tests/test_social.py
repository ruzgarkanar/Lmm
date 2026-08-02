"""Sohbetin bilgi taşımayan kısmı ve eksiltili sorular.

Sekiz turluk gerçek bir sohbet denemesinde beş tur "anlamadım" ile geçti:
"selam", "nasılsın", "teşekkürler" bilgi sorusu sanılıp grafta arandı, ve
"peki ya kartal" hiç tanınmadı. İkisi de kapalı sınıf ve ikisi de eksikti.
"""
import os
import tempfile
import unittest

from lmm import social
from lmm.cli import Session
from lmm.memory import Memory, Edge, IS_A, CAN, CANNOT


class TestRecognising(unittest.TestCase):
    def test_it_hears_a_greeting(self):
        for word in ("selam", "merhaba", "Günaydın", "iyi günler"):
            self.assertEqual(social.recognise(word), social.GREETING, word)

    def test_it_hears_the_other_exchanges(self):
        self.assertEqual(social.recognise("nasılsın"), social.WELLBEING)
        self.assertEqual(social.recognise("teşekkürler"), social.THANKS)
        self.assertEqual(social.recognise("kimsin"), social.IDENTITY)
        self.assertEqual(social.recognise("görüşürüz"), social.FAREWELL)

    def test_the_question_wins_over_the_greeting(self):
        """"selam nasılsın" iki şey taşır; sorulan şey hatırdır."""
        self.assertEqual(social.recognise("selam nasılsın"), social.WELLBEING)

    def test_a_knowledge_question_is_not_a_social_exchange(self):
        for line in ("penguen nedir", "kuşlar uçar", "kartal nasıldır"):
            self.assertIsNone(social.recognise(line), line)

    def test_a_long_sentence_is_never_a_social_exchange(self):
        self.assertIsNone(social.recognise(
            "merhaba bugün sana penguenler hakkında bir şey sormak istiyorum"))


class TestReplying(unittest.TestCase):
    def test_the_numbers_come_from_memory_not_from_thin_air(self):
        memory = Memory()
        memory.write(Edge("kuş", IS_A, "hayvan", source="test"))
        said = social.reply(social.WELLBEING, memory)
        self.assertIn("1 bilgi", said)

    def test_it_works_without_a_memory(self):
        self.assertIn("birçok", social.reply(social.IDENTITY))

    def test_an_unknown_exchange_has_no_reply(self):
        self.assertIsNone(social.reply("olmayan"))


class TestInConversation(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "sohbet.lmm")
        session = Session(self.path)
        for line in ("kelime: uçmak = uçar / uçamaz", "kuşlar uçar",
                     "penguen bir kuştur", "kartal bir kuştur",
                     "penguen uçamaz", "evet"):
            session.respond(line)
        session.save()

    def test_a_greeting_is_answered_not_refused(self):
        session = Session(self.path)
        self.assertIn("merhaba", session.respond("selam"))

    def test_an_elliptical_question_repeats_the_last_one(self):
        """"penguen uçar mı" sonrası "peki ya kartal" kartal için aynı soru."""
        session = Session(self.path)
        self.assertTrue(session.respond("penguen uçar mı").startswith("hayır"))
        self.assertTrue(session.respond("peki ya kartal").startswith("evet"))

    def test_ellipsis_needs_a_previous_question(self):
        """İlk cümle eksiltili olamaz — tekrarlanacak bir şey yok."""
        session = Session(self.path)
        self.assertNotIn("evet", session.respond("peki ya kartal"))

    def test_an_unknown_concept_is_not_carried(self):
        """Tanımadığı bir şey için önceki soruyu tekrarlamak uydurma olurdu."""
        session = Session(self.path)
        session.respond("penguen uçar mı")
        answer = session.respond("peki ya ejderha")
        self.assertNotIn("evet, çünkü ejderha", answer)


if __name__ == "__main__":
    unittest.main()
