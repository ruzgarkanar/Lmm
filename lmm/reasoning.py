"""Reasoning Engine: inheritance, exceptions and conflict detection over memory.

Every answer it produces can name the chain it came from, so nothing the system
says is unexplainable.
"""
from lmm.memory import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY, LACKS_PROPERTY,
                        TYPE_RELATIONS, ABILITY_RELATIONS, PROPERTY_RELATIONS)
from lmm.phrasing import (ability_clause, property_clause, is_a_clause,
                          is_not_a_clause, attribution)
from lmm.trust import INFERENCE


class Reasoning:
    def __init__(self, memory):
        self.memory = memory

    def ancestors(self, concept):
        """Type ancestors, nearest first."""
        result, queue, seen = [], [concept], {concept}
        while queue:
            current = queue.pop(0)
            for edge in self.memory.query(current, IS_A):
                if edge.target not in seen:
                    seen.add(edge.target)
                    result.append(edge.target)
                    queue.append(edge.target)
        return result

    def can_do(self, concept, action):
        """Returns (True | False | None, explanation chain).

        None means memory holds nothing on this — the caller must not guess.
        """
        answer, chain, _ = self._resolve(concept, action, CAN, CANNOT,
                                         ability_clause)
        return answer, chain

    def has_property(self, concept, prop):
        """Same shape as can_do, for "kar beyazdır" style knowledge."""
        answer, chain, _ = self._resolve(concept, prop, HAS_PROPERTY,
                                         LACKS_PROPERTY, property_clause)
        return answer, chain

    def basis(self, candidate):
        """The edge behind the current belief about a candidate's claim.

        Lets a caller ask *why* it believes something — in particular whether a
        belief came from a teacher or from the system's own generalisation.
        """
        if candidate.relation in ABILITY_RELATIONS:
            return self._resolve(candidate.concept, candidate.target, CAN,
                                 CANNOT, ability_clause)[2]
        if candidate.relation in PROPERTY_RELATIONS:
            return self._resolve(candidate.concept, candidate.target,
                                 HAS_PROPERTY, LACKS_PROPERTY,
                                 property_clause)[2]
        return self.memory.direct(candidate.concept,
                                  IS_A if candidate.relation == NOT_A else NOT_A,
                                  candidate.target)

    def _resolve(self, concept, target, affirms, denies, clause):
        """Direct knowledge first, then the type hierarchy, nearest ancestor first.

        A direct fact always beats an inherited one — that is exactly what makes
        an exception an exception.
        """
        for polarity, relation in ((False, denies), (True, affirms)):
            edge = self.memory.direct(concept, relation, target)
            if edge:
                return polarity, [f"{clause(concept, target, polarity)} "
                                  f"({attribution(edge.source)})"], edge
        for ancestor in self.ancestors(concept):
            for polarity, relation in ((False, denies), (True, affirms)):
                edge = self.memory.direct(ancestor, relation, target)
                if edge:
                    inherited = clause(ancestor, target, polarity)
                    if edge.source == INFERENCE:
                        # An inherited guess is still a guess, and must say so.
                        inherited += " (kendi çıkarımım)"
                    return polarity, [f"{concept} bir {ancestor}", inherited], edge
        return None, [], None

    def abilities(self, concept):
        """[(action, True|False)] for every action this memory knows about."""
        return self._known_of(self.memory.actions(), self.can_do, concept)

    def properties(self, concept):
        """[(property, True|False)] for every property this memory knows about."""
        return self._known_of(self.memory.properties(), self.has_property, concept)

    def _known_of(self, targets, lookup, concept):
        found = []
        for target in targets:
            known, _ = lookup(concept, target)
            if known is not None:
                found.append((target, known))
        return found

    def who_can(self, action, positive=True):
        """Every concept known to do (or known not to do) an action."""
        return [c for c in self.memory.concepts()
                if self.can_do(c, action)[0] is positive]

    def find_conflict(self, candidate):
        """Explanation string if the candidate edge conflicts with what we know."""
        if candidate.relation in TYPE_RELATIONS:
            return self._type_conflict(candidate)
        if candidate.relation in ABILITY_RELATIONS:
            known, chain = self.can_do(candidate.concept, candidate.target)
            claimed = candidate.relation == CAN
        elif candidate.relation in PROPERTY_RELATIONS:
            known, chain = self.has_property(candidate.concept, candidate.target)
            claimed = candidate.relation == HAS_PROPERTY
        else:
            return None
        if known is not None and known != claimed:
            return "şu an bildiğim: " + " çünkü ".join(chain)
        return None

    def _type_conflict(self, candidate):
        """"penguen bir kuş değildir" against a hierarchy that says it is."""
        opposite = NOT_A if candidate.relation == IS_A else IS_A
        edge = self.memory.direct(candidate.concept, opposite, candidate.target)
        if edge is not None:
            clause = (is_a_clause if opposite == IS_A else is_not_a_clause)
            return (f"şu an bildiğim: {clause(candidate.concept, candidate.target)} "
                    f"(kaynak: {edge.source})")
        if (candidate.relation == NOT_A
                and candidate.target in self.ancestors(candidate.concept)):
            return ("şu an bildiğim: "
                    + is_a_clause(candidate.concept, candidate.target))
        return None
