"""Reasoning Engine: inheritance, exceptions and conflict detection over memory.

Every answer it produces can name the chain it came from, so nothing the system
says is unexplainable.
"""
from lmm.memory import IS_A, NOT_A, CAN, CANNOT, TYPE_RELATIONS
from lmm.phrasing import ability_clause, is_a_clause, is_not_a_clause


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
        for polarity, relation in ((False, CANNOT), (True, CAN)):
            edge = self.memory.direct(concept, relation, action)
            if edge:
                return polarity, [f"{ability_clause(concept, action, polarity)} "
                                  f"(doğrudan bilgi, kaynak: {edge.source})"]
        for ancestor in self.ancestors(concept):  # nearer ancestor wins
            for polarity, relation in ((False, CANNOT), (True, CAN)):
                edge = self.memory.direct(ancestor, relation, action)
                if edge:
                    return polarity, [f"{concept} bir {ancestor}",
                                      ability_clause(ancestor, action, polarity)]
        return None, []

    def abilities(self, concept):
        """[(action, True|False)] for every action this memory knows about."""
        found = []
        for action in self.memory.actions():
            known, _ = self.can_do(concept, action)
            if known is not None:
                found.append((action, known))
        return found

    def who_can(self, action, positive=True):
        """Every concept known to do (or known not to do) an action."""
        return [c for c in self.memory.concepts()
                if self.can_do(c, action)[0] is positive]

    def find_conflict(self, candidate):
        """Explanation string if the candidate edge conflicts with what we know."""
        if candidate.relation in TYPE_RELATIONS:
            return self._type_conflict(candidate)
        if candidate.relation not in (CAN, CANNOT):
            return None
        known, chain = self.can_do(candidate.concept, candidate.target)
        if known is None:
            return None
        if known != (candidate.relation == CAN):
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
