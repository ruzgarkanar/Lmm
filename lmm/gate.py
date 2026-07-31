"""Epistemic Gate: answers are produced from memory alone.

There is no path from this class to a sentence that memory does not support, so
hallucination is not filtered out — it is unreachable.
"""
from lmm.relations import IS_A, NOT_A, CAN, HAS_PROPERTY
from lmm.phrasing import (is_a_clause, is_not_a_clause, ability_clause,
                          property_clause,
                          who_clause, ability_summary, property_summary,
                          verb_form, dont_know)
from lmm.similarity import nearest
from lmm.intuition import (ASK_WHO, ASK_ABILITIES, ASK_WHY, ASK_PROPERTIES,
                           ASK_DESCRIBE)
from lmm.exposition import Exposition

HEDGE_THRESHOLD = 0.5


class EpistemicGate:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning
        self.exposition = Exposition(memory, reasoning)

    def answer(self, intent):
        if intent.kind == ASK_WHO:
            return self._who(intent.target, intent.relation == CAN)
        if intent.kind == ASK_ABILITIES:
            return self._abilities(intent.concept)
        if intent.kind == ASK_WHY:
            return self._why(intent)
        if intent.kind == ASK_DESCRIBE:
            return self.exposition.describe(intent.concept)
        if intent.kind == ASK_PROPERTIES:
            return self._all_properties(intent.concept)
        if intent.relation == IS_A:
            if intent.target:
                return self._is_a(intent.concept, intent.target)
            return self._definition(intent.concept)
        if intent.relation == CAN:
            return self._ability(intent.concept, intent.target, intent.object,
                                 intent.role)
        if intent.relation == HAS_PROPERTY:
            return self._property(intent.concept, intent.target, intent.object,
                                  intent.role)
        return self._dont_know(intent.concept)

    def _is_a(self, concept, target):
        """"kalp bir organ mı" — a yes/no about a place in the hierarchy."""
        if self.memory.direct(concept, NOT_A, target) is not None:
            return f"hayır, {is_not_a_clause(concept, target)}."
        ancestors = self.reasoning.ancestors(concept)
        if target in ancestors:
            chain = [is_a_clause(concept, step) for step in ancestors
                     if step == target or ancestors.index(step) == 0]
            return f"evet, {chain[-1]}."
        if not self.memory.query(concept, IS_A):
            return self._dont_know(concept)
        return f"bildiğim kadarıyla {is_not_a_clause(concept, target)}."

    def _property(self, concept, prop, object=None, role=None):
        known, chain = self.reasoning.has_property(concept, prop, object, role)
        if known is None:
            return self._dont_know(concept)
        prefix = "evet" if known else "hayır"
        return f"{prefix}, çünkü {' ve '.join(chain)}."

    def _all_properties(self, concept):
        found = self.reasoning.properties(concept)
        if not found:
            return f"{concept} nasıldır, bunu hiç öğrenmedim."
        return property_summary(concept, found) + "."

    def _who(self, action, positive):
        if action not in self.memory.actions():
            return f"{verb_form(action, True)} diye bir şeyi hiç duymadım."
        found = self.reasoning.who_can(action, positive)
        if not found:
            return (f"bildiğim hiçbir şeyin {verb_form(action, positive)}ini "
                    f"öğrenmedim.")
        return who_clause(found, action, positive) + "."

    def _abilities(self, concept):
        found = self.reasoning.abilities(concept)
        if not found:
            return f"{concept} ne yapabilir, bunu hiç öğrenmedim."
        return ability_summary(concept, found) + "."

    def _why(self, intent):
        """Why questions work the same for what a thing does and how it is."""
        about_property = intent.relation == HAS_PROPERTY
        lookup = (self.reasoning.has_property if about_property
                  else self.reasoning.can_do)
        clause = property_clause if about_property else ability_clause
        known, chain = lookup(intent.concept, intent.target)
        if known is None:
            return self._dont_know(intent.concept)
        if known is not True:
            return f"aslında {clause(intent.concept, intent.target, known)}."
        return "çünkü " + " ve ".join(chain) + "."

    def _definition(self, concept):
        edges = self.memory.query(concept, IS_A)
        if not edges:
            return self._dont_know(concept)
        best = max(edges, key=lambda e: e.confidence)
        answer = f"{is_a_clause(concept, best.target)} (kaynak: {best.source})."
        if best.confidence < HEDGE_THRESHOLD:
            return "emin değilim ama " + answer
        return answer

    def _ability(self, concept, action, object=None, role=None):
        known, chain = self.reasoning.can_do(concept, action, object, role)
        if known is None:
            return self._dont_know(concept)
        prefix = "evet" if known else "hayır"
        return f"{prefix}, çünkü {' ve '.join(chain)}."

    def _dont_know(self, concept):
        known = self.memory.concepts()
        return dont_know(concept, nearest(concept, known) if concept else ())
