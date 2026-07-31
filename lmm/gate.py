"""Epistemic Gate: answers are produced from memory alone.

There is no path from this class to a sentence that memory does not support, so
hallucination is not filtered out — it is unreachable.
"""
from lmm.memory import IS_A, CAN
from lmm.phrasing import is_a_clause

HEDGE_THRESHOLD = 0.5


class EpistemicGate:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning

    def answer(self, intent):
        if intent.relation == IS_A:
            return self._definition(intent.concept)
        if intent.relation == CAN:
            return self._ability(intent.concept, intent.target)
        return self._dont_know(intent.concept)

    def _definition(self, concept):
        edges = self.memory.query(concept, IS_A)
        if not edges:
            return self._dont_know(concept)
        best = max(edges, key=lambda e: e.confidence)
        answer = f"{is_a_clause(concept, best.target)} (kaynak: {best.source})."
        if best.confidence < HEDGE_THRESHOLD:
            return "emin değilim ama " + answer
        return answer

    def _ability(self, concept, action):
        known, chain = self.reasoning.can_do(concept, action)
        if known is None:
            return self._dont_know(concept)
        prefix = "evet" if known else "hayır"
        return f"{prefix}, çünkü {' ve '.join(chain)}."

    def _dont_know(self, concept):
        return f"bilmiyorum. {concept} hakkında bunu bana öğretir misin?"
