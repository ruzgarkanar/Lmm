"""Önkoşul: "uçmak için ne gerekir".

Bu ilişki envanterde yoktu ve sonucu ölçüldü — soru yaklaşık bir ilişkiye
çevriliyor, graf onu cevaplıyor ve YANLIŞ kalıp kalıcı yazılıyordu. İlişkiler
veri olduğu için eklemek motoru değiştirmedi: `lmm/kinds.py` içinde iki satır.
"""
import os
import tempfile
import unittest

from lmm.cli import Session
from lmm.memory import Memory, Edge, IS_A
from lmm.phrasing import describe
from lmm.relations import REQUIRES


class TestTheRelationIsData(unittest.TestCase):
    def test_the_registry_knows_it(self):
        memory = Memory()
        self.assertIn(REQUIRES, memory.kinds.known())

    def test_it_is_inherited_but_not_transitive(self):
        """Kuş için geçerli olan kartal için de geçerli. Ama A->B, B->C ise
        A->C denemez: aradaki koşul kopabilir."""
        kind = Memory().kinds.by_name[REQUIRES]
        self.assertTrue(kind.inherits)
        self.assertFalse(kind.transitive)

    def test_an_unknown_relation_is_said_with_its_own_label(self):
        """describe tanımadığı ilişkiyi YETENEK sanıyor ve bozuk cümle
        üretiyordu: "kuş kanat gerektirir" -> "kuş kanat"."""
        memory = Memory()
        said = describe("kuş", REQUIRES, "kanat", kinds=memory.kinds)
        self.assertIn("gerektirir", said)


class TestInConversation(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "g.lmm")
        memory = Memory()
        memory.write(Edge("kuş", IS_A, "hayvan", source="sen"))
        memory.write(Edge("kartal", IS_A, "kuş", source="sen"))
        memory.save(self.path)
        self.session = Session(self.path)

    def test_it_is_taught_and_said_back_properly(self):
        said = self.session.respond("kuş kanat gerektirir")
        self.assertIn("gerektirir", said)

    def test_it_is_answered(self):
        self.session.respond("kuş kanat gerektirir")
        self.assertIn("kanat", self.session.respond("kuş için ne gerekir"))

    def test_a_child_inherits_the_requirement(self):
        self.session.respond("kuş kanat gerektirir")
        self.assertIn("kanat", self.session.respond("kartal için ne gerekir"))

    def test_what_is_not_recorded_is_not_invented(self):
        """Bir önkoşulu tahmin etmek, cevabı uydurmakla aynı şey."""
        said = self.session.respond("kartal için ne gerekir")
        self.assertIn("bilmiyorum", said)

    def test_an_unknown_concept_is_refused(self):
        self.assertIn("bilmiyorum",
                      self.session.respond("ejderha için ne gerekir"))


if __name__ == "__main__":
    unittest.main()
