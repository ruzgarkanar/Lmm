"""Curiosity: noticing its own gaps and asking about them.

Saying "bilmiyorum" when asked is honesty. Noticing an absence nobody pointed at,
and asking about it unprompted, is curiosity — and it is what closes the loop
from ignorance to permanent knowledge without anyone retraining anything.

Every question is grounded in structure that actually exists in memory: a concept
with no place in the hierarchy, or an action the world is known to contain that
this concept has never been connected to. The system does not invent curiosity
any more than it invents answers.
"""
from lmm.relations import IS_A, CAN, HAS_PROPERTY
from lmm import serialize


class Question:
    def __init__(self, key, text):
        self.key = key      # stable identity, so it is asked at most once
        self.text = text


class Curiosity:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning

    def next_question(self):
        """The first unasked gap, or None when nothing is genuinely missing."""
        for question in self._gaps():
            if not self.memory.has_asked(question.key):
                self.memory.mark_asked(question.key)
                return question
        return None

    def _gaps(self):
        # HAKKINDA BİR ŞEY BİLİNENLER ÖNCE. Süzmek değil sıralamak: `concepts()`
        # özneleri de HEDEF olarak geçen adları da veriyor, ve ikinciler çoğu
        # zaman bir tamlamanın parçası — `dalı`, `sistemi`, `resm`. Onları
        # elemek küçük bir grafta her şeyi elerdi (yeni öğrenilen bir hedef
        # hakkında soru sormak doğru), ama büyük grafta öne almak yanlış.
        #
        # Ölçüldü: bu sıralama olmadan merak döngüsü bir dil modeline "dalı
        # nedir?" diye soruyor ve model olmayan bir şeye tanım uyduruyordu.
        # Kötü soru, kötü cevabın kaynağıdır.
        concepts = sorted(self.memory.concepts(),
                          key=lambda name: -len(self.memory.query(name)))
        for concept in concepts:            # what is this thing, anyway?
            if not self.memory.query(concept, IS_A):
                yield Question(f"type:{concept}",
                               f"{concept} → {serialize.label(IS_A)} → ?")
        for concept in concepts:            # the world does this — can it?
            for action in self.memory.actions():
                if self.reasoning.can_do(concept, action)[0] is None:
                    yield Question(f"can:{concept}:{action}",
                                   f"{serialize.fact(concept, CAN, action)} ?")
        for concept in concepts:            # the world is like this — is it?
            for prop in self.memory.properties():
                if self.reasoning.has_property(concept, prop)[0] is None:
                    yield Question(f"property:{concept}:{prop}",
                                   f"{serialize.fact(concept, HAS_PROPERTY, prop)} ?")
