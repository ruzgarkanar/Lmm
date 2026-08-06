"""Learning Loop: new information -> conflict check -> permanent write.

Nothing is written blindly. When a fact clashes with what is already known the
system says so and asks, and only a confirmed clash becomes an exception.
"""
from lmm.memory import Edge, CycleError, CAN, ALL
from lmm import serialize


def describe(concept, relation, target, object=None, role=None, kinds=None):
    return serialize.fact(concept, relation, target, object=object)
from lmm.drift import may_write, branch_of, record
from lmm.trust import (TEACHER, HUMAN, confidence_for, outranks, level,
                       arbitrate, note, CANDIDATE, DISPUTED)

LEARNED = "learned"
FROZEN = "frozen"
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
        if not may_write(self.memory, self.reasoning, candidate.concept, source):
            branch = branch_of(self.reasoning, candidate.concept)
            return FROZEN, f"❄ {serialize.cite(branch)}", candidate
        conflict = self.reasoning.find_conflict(candidate)
        basis = self.reasoning.basis(candidate)
        statement = describe(candidate.concept, candidate.relation,
                             candidate.target, candidate.object, candidate.role,
                             self.memory.kinds)
        if conflict is not None and basis is not None:
            verdict = arbitrate(basis, candidate, self.memory.reputation)
            if verdict == CANDIDATE:
                # A better-backed claim replaces the one held on weaker ground.
                note(self.memory.reputation, candidate.source, True)
                for voice in basis.sources:
                    note(self.memory.reputation, voice, False)
                written = self.memory.write(candidate)
                written.disputed = False    # arbitration settled it
                return CORRECTED, f"{serialize.cite(basis.source)} ⇐ {statement}", candidate
            if verdict == DISPUTED and level(candidate.source) < HUMAN:
                record(self.memory, branch_of(self.reasoning, candidate.concept),
                       False)
                # Wikidata's answer: keep both and record that they disagree.
                # Quietly picking a winner between equals is the guess we refuse.
                basis.disputed = candidate.disputed = True
                self.memory.write(candidate)
                for voice in list(basis.sources) + list(candidate.sources):
                    note(self.memory.reputation, voice, False)
                return DISPUTE, f"⚡ {serialize.cite(', '.join(list(basis.sources)))} ⊥ {statement}", candidate
        if conflict is not None:
            record(self.memory, branch_of(self.reasoning, candidate.concept),
                   False)
            return CONFLICT, f"⚡ {conflict} ⊥ {statement} ? [+/−]", candidate
        # Nicelik de sorulmalı. Sorulmayınca `direct()` yuvadaki EN GENİŞ
        # iddiayı döndürüyor ve yeni bir kayıt yazıldığı hâlde "zaten
        # biliyordum" deniyordu: "bazı kuşlar uçmaz" üzerine "hiçbir kuş
        # uçmaz" gelince bellek iki ayrı kayıt tutuyor ama cümle tek kayıt
        # varmış gibi konuşuyordu. Söylenen ile yapılanın ayrışması, bu
        # mimaride en pahalı hata türü.
        existing = self.memory.direct(candidate.concept, candidate.relation,
                                      candidate.target, candidate.object,
                                      candidate.role,
                                      quantifier=candidate.quantifier)
        if existing is not None:
            self.memory.write(candidate)
            return REINFORCED, f"= {statement}", existing
        try:
            self.memory.write(candidate)
        except CycleError:
            return REJECTED, "✗ ⟳", None
        record(self.memory, branch_of(self.reasoning, candidate.concept), True)
        return LEARNED, f"+ {statement}", candidate

    def confirm_exception(self, edge):
        """A conflict the teacher stands behind becomes a permanent exception."""
        edge.is_exception = True
        return self.memory.write(edge)
