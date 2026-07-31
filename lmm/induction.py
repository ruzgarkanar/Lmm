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
from lmm.memory import (Edge, IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                        LACKS_PROPERTY, INFERRED, INFERRED_CONFIDENCE)

MINIMUM_EXAMPLES = 2
MINIMUM_TRAITS = 2      # one shared habit is a coincidence


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
        """The first generalisation memory supports but nobody has stated.

        Placing a stranger comes before generalising over a family: knowing what
        something *is* unlocks everything its kind already knows, so it is worth
        more than one more rule about a type we have already placed.
        """
        for hypothesis in self.placements():
            return hypothesis
        for hypothesis in self.candidates():
            return hypothesis
        return None

    def placements(self):
        """Concepts with no place, put where their behaviour says they belong.

        A stranger that flies and has feathers is probably a bird. Nobody said
        so, and it may be wrong — a bat behaves like that too — so the guess is
        written as the system's own and yields to the first person who corrects
        it. What it buys is everything else birds know.
        """
        placed = []
        for concept in self.memory.concepts():
            if self.memory.query(concept, IS_A):
                continue                    # already knows what it is
            traits = self._behaviour(concept)
            if len(traits) < MINIMUM_TRAITS:
                continue                    # too little behaviour to go on
            for category in self._categories():
                if category == concept:
                    continue
                shared = traits & self._behaviour(category)
                if len(shared) >= MINIMUM_TRAITS and shared == traits:
                    placed.append(Hypothesis(concept, IS_A, category,
                                             sorted(t[1] for t in shared)))
                    break
        return placed

    def _categories(self):
        """Shelves: concepts something is already said to be, tightest first.

        Any concept with two habits in common was too loose a bar. Fire and the
        sun are both hot and both warm things, so a stranger that behaved like
        either was filed under the other — "sanırım ateş bir güneştir". A shelf
        has to be a category, and what makes a concept a category is that
        something already belongs to it.
        """
        kinds = self.memory.kinds.hierarchical()
        shelves = {edge.target for edge in self.memory.edges
                   if edge.relation == kinds}
        ranked = [(len(self._behaviour(shelf)), shelf) for shelf in shelves
                  if len(self._behaviour(shelf)) >= MINIMUM_TRAITS]
        ranked.sort()           # the tightest fit is the most informative one
        return [shelf for _, shelf in ranked]

    def _behaviour(self, concept):
        """What a concept is known to do — evidence only, never guesses.

        A guess must not become the ground for the next guess. If the system's
        own inferences counted as evidence, one wrong placement would breed a
        rule, the rule would breed another placement, and the chain would look
        exactly as confident as anything it was actually told. Inference reads
        from what it was given; it never reads from itself.
        """
        return {(edge.relation, edge.target, edge.object, edge.role)
                for edge in self.memory.query(concept)
                if edge.relation not in (IS_A, NOT_A)
                and edge.source != INFERRED}

    def still_open(self, hypothesis):
        """Whether a proposal is still unsettled — an earlier one may have closed it."""
        if hypothesis.relation == IS_A:
            return not self.memory.query(hypothesis.concept, IS_A)
        lookup = (self.reasoning.has_property
                  if hypothesis.relation in (HAS_PROPERTY, LACKS_PROPERTY)
                  else self.reasoning.can_do)
        return lookup(hypothesis.concept, hypothesis.target)[0] is None

    def learn(self, hypothesis):
        return self.memory.write(hypothesis.as_edge())

    def candidates(self):
        """Every generalisation the memory supports right now, in one pass."""
        return list(self._candidates())

    def _candidates(self):
        for parent in self.memory.concepts():
            children = self._children(parent)
            if len(children) < MINIMUM_EXAMPLES:
                continue
            for target, affirms, denies, lookup in self._traits():
                if lookup(parent, target)[0] is not None:
                    continue                    # the type is already settled
                # Only what the system was told counts as evidence, never what
                # it worked out itself.
                agree = [c for c in children
                         if self._stated(c, target, affirms)]
                disagree = [c for c in children
                            if self._stated(c, target, denies)]
                # Two out of twenty is not a rule about the twenty, and two out
                # of eight is not either: "kuş siyahtır" came from a penguin and
                # a crow, "hayvan küçüktür" from a cat and a beetle. A third of
                # the family has to agree before it counts as a rule about it.
                enough = max(MINIMUM_EXAMPLES, -(-len(children) // 3))
                if len(agree) >= enough and not disagree:
                    yield Hypothesis(parent, affirms, target, agree)
                elif len(disagree) >= enough and not agree:
                    yield Hypothesis(parent, denies, target, disagree)

    def _stated(self, concept, target, relation):
        edge = self.memory.direct(concept, relation, target)
        return edge is not None and edge.source != INFERRED

    def _children(self, parent):
        return [edge.concept for edge in self.memory.edges
                if edge.relation == IS_A and edge.target == parent]

    def _traits(self):
        for action in self.memory.actions():
            yield action, CAN, CANNOT, self.reasoning.can_do
        for prop in self.memory.properties():
            yield prop, HAS_PROPERTY, LACKS_PROPERTY, self.reasoning.has_property
