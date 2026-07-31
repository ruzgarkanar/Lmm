"""Curiosity: noticing its own gaps and asking about them.

Saying "bilmiyorum" when asked is honesty. Noticing an absence nobody pointed at,
and asking about it unprompted, is curiosity — and it is what closes the loop
from ignorance to permanent knowledge without anyone retraining anything.

Every question is grounded in structure that actually exists in memory: a concept
with no place in the hierarchy, or an action the world is known to contain that
this concept has never been connected to. The system does not invent curiosity
any more than it invents answers.
"""
from lmm.memory import IS_A, CAN, CANNOT
from lmm import phrasing


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
        concepts = self._concepts()
        for concept in concepts:            # what is this thing, anyway?
            if not self.memory.query(concept, IS_A):
                yield Question(f"type:{concept}",
                               phrasing.definition_question(concept))
        for concept in concepts:            # the world does this — can it?
            for action in self._actions():
                if self.reasoning.can_do(concept, action)[0] is None:
                    yield Question(f"can:{concept}:{action}",
                                   phrasing.ability_question(concept, action))

    def _concepts(self):
        """Everything that behaves like a thing, in the order it was learned."""
        ordered = []
        for edge in self.memory.edges:
            for candidate in (edge.concept,
                              edge.target if edge.relation == IS_A else None):
                if candidate is not None and candidate not in ordered:
                    ordered.append(candidate)
        return ordered

    def _actions(self):
        ordered = []
        for edge in self.memory.edges:
            if edge.relation in (CAN, CANNOT) and edge.target not in ordered:
                ordered.append(edge.target)
        return ordered
