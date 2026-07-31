"""Learning Loop: new information -> conflict check -> permanent write.

Nothing is written blindly. When a fact clashes with what is already known the
system says so and asks, and only a confirmed clash becomes an exception.
"""
from lmm.memory import Edge, CycleError, CAN, INFERRED
from lmm.phrasing import describe, corrected

LEARNED = "learned"
REINFORCED = "reinforced"
CORRECTED = "corrected"
CONFLICT = "conflict"
REJECTED = "rejected"


class LearningLoop:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning

    def teach(self, intent, source="sen"):
        """Returns (status, message, edge). Status is one of the module constants."""
        candidate = Edge(intent.concept, intent.relation, intent.target, source=source)
        conflict = self.reasoning.find_conflict(candidate)
        if conflict is not None and self._only_my_own_guess(candidate):
            # A teacher outranks a generalisation the system made up itself.
            self.memory.write(candidate)
            return CORRECTED, corrected(describe(candidate.concept,
                                                 candidate.relation,
                                                 candidate.target)), candidate
        if conflict is not None:
            statement = describe(candidate.concept, candidate.relation, candidate.target)
            question = (f"bir çelişki fark ettim: {conflict}. yine de "
                        f"'{statement}' olarak öğreneyim mi? (evet/hayır)")
            return CONFLICT, question, candidate
        existing = self.memory.direct(candidate.concept, candidate.relation,
                                      candidate.target)
        if existing is not None:
            self.memory.write(candidate)
            return REINFORCED, "bunu zaten biliyordum, güvenim arttı.", existing
        try:
            self.memory.write(candidate)
        except CycleError:
            return REJECTED, "bu tür ilişkisi döngü oluşturur, kabul edemem.", None
        statement = describe(candidate.concept, candidate.relation, candidate.target)
        return LEARNED, f"öğrendim: {statement}.", candidate

    def _only_my_own_guess(self, candidate):
        basis = self.reasoning.basis(candidate)
        return basis is not None and basis.source == INFERRED

    def confirm_exception(self, edge):
        """A conflict the teacher stands behind becomes a permanent exception."""
        edge.is_exception = True
        return self.memory.write(edge)
