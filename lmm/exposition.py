"""Exposition: composing a paragraph out of what is known.

A language model writes fluently by sampling likely words, which is the same
machine that makes it invent. This composes instead: it decides what is worth
saying about a concept, puts it in an order a person would use, and joins it
with the connectives the meaning calls for.

Nobody wrote the paragraph that comes out, so it is generative in the sense that
matters — but every clause in it can be traced to a fact, and the exceptions get
the "ama" they deserve because the reasoning knows they are exceptions.

What it will not do is write a poem. Composition from knowledge is a different
machine from composition from a probability distribution, and this is the one
that cannot make things up.
"""
from lmm.relations import IS_A, CAN, CANNOT, HAS_PROPERTY, LACKS_PROPERTY
from lmm.trust import INFERENCE
from lmm import phrasing

OPPOSITES = {CAN: CANNOT, CANNOT: CAN,
             HAS_PROPERTY: LACKS_PROPERTY, LACKS_PROPERTY: HAS_PROPERTY}


class Exposition:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning

    def describe(self, concept):
        """Everything worth saying about a concept, as connected prose."""
        parts = [self._identity(concept), self._exceptions(concept),
                 self._inherited(concept), self._own(concept)]
        said = [part for part in parts if part]
        if not said:
            return phrasing.dont_know(concept)
        return " ".join(said)

    def _identity(self, concept):
        edges = self.memory.query(concept, IS_A)
        if not edges:
            return ""
        best = max(edges, key=lambda e: e.confidence)
        sentence = phrasing.capitalize(phrasing.is_a_clause(concept, best.target))
        ancestors = self.reasoning.ancestors(concept)
        if len(ancestors) > 1:
            sentence += (f", {best.target} {phrasing.clitic_da(best.target)} "
                         f"{phrasing.predicate(IS_A, ancestors[-1])}")
        return sentence + "."

    def _exceptions(self, concept):
        """Where the concept breaks its own family's rule — the interesting part."""
        said = []
        for edge in self.memory.edges:
            if edge.concept != concept or not self._contradicts_family(edge):
                continue
            family = self._family_clause(edge)
            own = phrasing.predicate(edge.relation, edge.target, edge.object, edge.role)
            said.append(f"{phrasing.capitalize(family)} ama {concept} {own}.")
        return " ".join(said)

    def _inherited(self, concept):
        """What it gets from what it is, named as coming from there."""
        parent = self._nearest_parent(concept)
        if parent is None:
            return ""
        clauses = []
        for target, relation, obj, role in self._traits_of(parent):
            if self._speaks_for_itself(concept, relation, target):
                continue        # already told, either as its own or as an exception
            clauses.append(phrasing.predicate(relation, target, obj, role))
        if not clauses:
            return ""
        return (f"{phrasing.capitalize(parent)} olduğu için "
                f"{phrasing.listing(clauses)}.")

    def _own(self, concept):
        """Facts stated about it directly, minus the exceptions already told."""
        clauses = []
        for edge in self.memory.edges:
            if edge.concept != concept or edge.relation == IS_A:
                continue
            if self._contradicts_family(edge):
                continue
            clause = phrasing.predicate(edge.relation, edge.target, edge.object, edge.role)
            if edge.source == INFERENCE:
                clause += " (sanırım)"
            clauses.append(clause)
        if not clauses:
            return ""
        return f"Ayrıca {phrasing.listing(clauses)}."

    def _speaks_for_itself(self, concept, relation, target):
        """Does the concept hold its own view on this — agreeing or not?"""
        opposite = OPPOSITES.get(relation, relation)
        return (self.memory.direct(concept, relation, target) is not None
                or self.memory.direct(concept, opposite, target) is not None)

    def _contradicts_family(self, edge):
        opposite = OPPOSITES.get(edge.relation)
        if opposite is None:
            return False
        return any(self.memory.direct(ancestor, opposite, edge.target)
                   for ancestor in self.reasoning.ancestors(edge.concept))

    def _family_clause(self, edge):
        opposite = OPPOSITES[edge.relation]
        for ancestor in self.reasoning.ancestors(edge.concept):
            if self.memory.direct(ancestor, opposite, edge.target):
                return phrasing.describe(ancestor, opposite, edge.target)
        return ""

    def _nearest_parent(self, concept):
        ancestors = self.reasoning.ancestors(concept)
        return ancestors[0] if ancestors else None

    def _traits_of(self, parent):
        for edge in self.memory.edges:
            if edge.concept == parent and edge.relation != IS_A:
                yield edge.target, edge.relation, edge.object, edge.role
