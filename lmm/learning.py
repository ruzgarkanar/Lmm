"""Learning Loop: new information -> conflict check -> permanent write.

Nothing is written blindly. When a fact clashes with what is already known the
system says so and asks, and only a confirmed clash becomes an exception.
"""
from lmm.memory import Edge, CycleError, CAN
from lmm.phrasing import describe, corrected
from lmm.trust import TEACHER, confidence_for, outranks

LEARNED = "learned"
REINFORCED = "reinforced"
CORRECTED = "corrected"
CONFLICT = "conflict"
REJECTED = "rejected"


class LearningLoop:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning

    def teach(self, intent, source=TEACHER):
        """Returns (status, message, edge). Status is one of the module constants."""
        candidate = Edge(intent.concept, intent.relation, intent.target,
                         object=getattr(intent, "object", None),
                         source=source, confidence=confidence_for(source))
        conflict = self.reasoning.find_conflict(candidate)
        basis = self.reasoning.basis(candidate)
        if conflict is not None and basis is not None and \
                outranks(candidate.source, basis.source):
            # A better source simply replaces a belief held on a weaker one.
            self.memory.write(candidate)
            statement = describe(candidate.concept, candidate.relation,
                                 candidate.target)
            return CORRECTED, corrected(statement, basis.source), candidate
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
        statement = describe(candidate.concept, candidate.relation,
                             candidate.target, candidate.object)
        return LEARNED, f"öğrendim: {statement}.", candidate

    def confirm_exception(self, edge):
        """A conflict the teacher stands behind becomes a permanent exception."""
        edge.is_exception = True
        return self.memory.write(edge)
