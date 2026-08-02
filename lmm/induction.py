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
                        LACKS_PROPERTY, INFERRED, INFERRED_CONFIDENCE,
                        INHERITING)

MINIMUM_EXAMPLES = 2
MINIMUM_TRAITS = 2      # one shared habit is a coincidence


def succession(agreeing, consulted):
    """Laplace's rule of succession: the odds the rest of the family agrees too.

    After k members are seen to share a trait and none seen to lack it, the
    chance that a further member shares it is (k+1)/(k+2), and the chance that
    *all* of the remaining n-k do is (k+1)/(n+1). Laplace derived it in 1774 for
    exactly this question — how far a run of agreeing observations licenses a
    claim about the ones you have not looked at.

    This replaces a threshold we tuned by hand, and it earns its place because
    it separates the cases we spent a night arguing with: three cold-blooded
    animals out of eight scores 0.44, three running mammals out of three scores
    1.00, and two black birds out of twenty — the very first over-generalisation
    this system ever made — scores 0.14. One formula, no dial.
    """
    return (agreeing + 1) / (consulted + 1)


# A rule has to be likelier than not to hold of the members nobody described,
# with room to spare. Below this the honest move is to keep the examples and
# decline the generalisation.
SUPPORT = 0.75


class Hypothesis:
    def __init__(self, concept, relation, target, examples, support=None):
        self.concept = concept
        self.relation = relation
        self.target = target
        self.examples = examples    # the children that suggested it
        self.support = support      # how far the evidence reaches, 0..1

    def as_edge(self):
        # A guess whose evidence reaches further is held more firmly, but never
        # as firmly as something it was actually told.
        confidence = INFERRED_CONFIDENCE
        if self.support is not None:
            confidence = min(INFERRED_CONFIDENCE * self.support * 2,
                             INFERRED_CONFIDENCE)
        return Edge(self.concept, self.relation, self.target,
                    source=INFERRED, confidence=confidence)


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
                # "Most of the family" has to mean most of the family that
                # has an opinion. Counting silent children as dissent kills good
                # rules — three mammals run and the other twelve were never
                # described. The opinionated ones are the ones that were
                # actually consulted, so they are the denominator, and how far
                # their agreement reaches is arithmetic rather than a dial.
                #
                # Note this cannot be done by spotting opposites instead:
                # measured over 1288 facts, only 5% of trait pairs ever co-occur
                # and "sıcak" and "soğuk" are among the pairs that do.
                # Complementary distribution finds suffix families; it does not
                # find antonyms.
                consulted = [c for c in children if self._opinionated(c, affirms)]
                for holds, relation in ((agree, affirms), (disagree, denies)):
                    other = disagree if holds is agree else agree
                    if other or len(holds) < MINIMUM_EXAMPLES:
                        continue
                    support = succession(len(holds), len(consulted))
                    if support >= SUPPORT:
                        yield Hypothesis(parent, relation, target, holds, support)

    def _opinionated(self, concept, relation):
        """Whether this child was ever described in this respect at all."""
        return any(edge.source != INFERRED and edge.quantifier in INHERITING
                   for edge in self.memory.query(concept)
                   if edge.relation in self.memory.kinds.pair(relation))

    def _stated(self, concept, target, relation):
        """Evidence for a rule about everyone must itself be about everyone.

        "bazı kuşlar yüzer" was being counted towards "hayvan yüzer", which is
        the existence claim leaking into a universal by the back door — the very
        thing the quantifier was added to stop.
        """
        edge = self.memory.direct(concept, relation, target)
        return (edge is not None and edge.source != INFERRED
                and edge.quantifier in INHERITING)

    def _children(self, parent):
        return [edge.concept for edge in self.memory.edges
                if edge.relation == IS_A and edge.target == parent]

    def _traits(self):
        for action in self.memory.actions():
            yield action, CAN, CANNOT, self.reasoning.can_do
        for prop in self.memory.properties():
            yield prop, HAS_PROPERTY, LACKS_PROPERTY, self.reasoning.has_property
