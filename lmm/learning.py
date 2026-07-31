"""Learning Loop: new information -> conflict check -> permanent write.

Nothing is written blindly. When a fact clashes with what is already known the
system says so and asks, and only a confirmed clash becomes an exception.
"""
from lmm.memory import Edge, CycleError, CAN, ALL
from lmm.phrasing import describe, corrected, disagreement
from lmm.trust import (TEACHER, HUMAN, confidence_for, outranks, level,
                       arbitrate, note, CANDIDATE, DISPUTED)

LEARNED = "learned"
DISPUTE = "dispute"
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
                         role=getattr(intent, "role", None),
                         quantifier=getattr(intent, "quantifier", ALL),
                         source=source, confidence=confidence_for(source))
        conflict = self.reasoning.find_conflict(candidate)
        basis = self.reasoning.basis(candidate)
        statement = describe(candidate.concept, candidate.relation,
                             candidate.target, candidate.object, candidate.role)
        if conflict is not None and basis is not None:
            verdict = arbitrate(basis, candidate, self.memory.reputation)
            if verdict == CANDIDATE:
                # A better-backed claim replaces the one held on weaker ground.
                note(self.memory.reputation, candidate.source, True)
                for voice in basis.sources:
                    note(self.memory.reputation, voice, False)
                written = self.memory.write(candidate)
                written.disputed = False    # arbitration settled it
                return CORRECTED, corrected(statement, basis.source), candidate
            if verdict == DISPUTED and level(candidate.source) < HUMAN:
                # Wikidata's answer: keep both and record that they disagree.
                # Quietly picking a winner between equals is the guess we refuse.
                basis.disputed = candidate.disputed = True
                self.memory.write(candidate)
                for voice in list(basis.sources) + list(candidate.sources):
                    note(self.memory.reputation, voice, False)
                return DISPUTE, disagreement(statement, basis.sources), candidate
        if conflict is not None:
            question = (f"bir çelişki fark ettim: {conflict}. yine de "
                        f"'{statement}' olarak öğreneyim mi? (evet/hayır)")
            return CONFLICT, question, candidate
        existing = self.memory.direct(candidate.concept, candidate.relation,
                                      candidate.target, candidate.object,
                                      candidate.role)
        if existing is not None:
            self.memory.write(candidate)
            return REINFORCED, "bunu zaten biliyordum, güvenim arttı.", existing
        try:
            self.memory.write(candidate)
        except CycleError:
            return REJECTED, "bu tür ilişkisi döngü oluşturur, kabul edemem.", None
        return LEARNED, f"öğrendim: {statement}.", candidate

    def confirm_exception(self, edge):
        """A conflict the teacher stands behind becomes a permanent exception."""
        edge.is_exception = True
        return self.memory.write(edge)
