"""Induction: forming knowledge nobody stated.

Until now the system only knew what it had been told, and chaining given facts
is not the same as learning. An LLM's real advantage is that it generalises —
it answers about things nobody wrote down, because it absorbed patterns.

This does the same thing, in the open. When several children of a type share a
trait and none contradict it, the trait is proposed for the type itself. The
resulting fact is written like any other but sourced to the system rather than a
teacher, kept at a confidence low enough that answers hedge, and beaten by any
human correction without argument.

That is the part an LLM cannot do: it generalises too, but it cannot tell you
that it did, cannot show you the examples that convinced it, and cannot be
corrected on one point without retraining.
"""
from lmm.memory import Edge, IS_A, CAN, CANNOT, HAS_PROPERTY, LACKS_PROPERTY, \
    INFERRED, INFERRED_CONFIDENCE

MINIMUM_EXAMPLES = 2


class Hypothesis:
    def __init__(self, concept, relation, target, examples):
        self.concept = concept
        self.relation = relation
        self.target = target
        self.examples = examples    # the children that suggested it

    def as_edge(self):
        return Edge(self.concept, self.relation, self.target,
                    source=INFERRED, confidence=INFERRED_CONFIDENCE)


class Induction:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning

    def propose(self):
        """The first generalisation memory supports but nobody has stated."""
        for hypothesis in self._candidates():
            return hypothesis
        return None

    def learn(self, hypothesis):
        return self.memory.write(hypothesis.as_edge())

    def _candidates(self):
        for parent in self.memory.concepts():
            children = self._children(parent)
            if len(children) < MINIMUM_EXAMPLES:
                continue
            for target, affirms, denies, lookup in self._traits():
                if lookup(parent, target)[0] is not None:
                    continue                    # the type is already settled
                agree = [c for c in children if lookup(c, target)[0] is True]
                disagree = [c for c in children if lookup(c, target)[0] is False]
                if len(agree) >= MINIMUM_EXAMPLES and not disagree:
                    yield Hypothesis(parent, affirms, target, agree)
                elif len(disagree) >= MINIMUM_EXAMPLES and not agree:
                    yield Hypothesis(parent, denies, target, disagree)

    def _children(self, parent):
        return [edge.concept for edge in self.memory.edges
                if edge.relation == IS_A and edge.target == parent]

    def _traits(self):
        for action in self.memory.actions():
            yield action, CAN, CANNOT, self.reasoning.can_do
        for prop in self.memory.properties():
            yield prop, HAS_PROPERTY, LACKS_PROPERTY, self.reasoning.has_property
