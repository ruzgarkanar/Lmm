"""Reasoning Engine: inheritance, exceptions and conflict detection over memory.

Every answer it produces can name the chain it came from, so nothing the system
says is unexplainable.
"""
from lmm.memory import IS_A, CAN, CANNOT


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
        edge = self.memory.direct(concept, CANNOT, action)
        if edge:
            return False, [f"{concept} {action} yapamaz "
                           f"(doğrudan bilgi, kaynak: {edge.source})"]
        edge = self.memory.direct(concept, CAN, action)
        if edge:
            return True, [f"{concept} {action} yapabilir "
                          f"(doğrudan bilgi, kaynak: {edge.source})"]
        for ancestor in self.ancestors(concept):  # nearer ancestor wins
            edge = self.memory.direct(ancestor, CANNOT, action)
            if edge:
                return False, [f"{concept} bir {ancestor}", f"{ancestor} {action} yapamaz"]
            edge = self.memory.direct(ancestor, CAN, action)
            if edge:
                return True, [f"{concept} bir {ancestor}", f"{ancestor} {action} yapabilir"]
        return None, []

    def find_conflict(self, candidate):
        """Explanation string if the candidate edge conflicts with what we know."""
        if candidate.relation not in (CAN, CANNOT):
            return None
        known, chain = self.can_do(candidate.concept, candidate.target)
        if known is None:
            return None
        if known != (candidate.relation == CAN):
            return "şu an bildiğim: " + " çünkü ".join(chain)
        return None
