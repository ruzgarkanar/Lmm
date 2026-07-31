"""Pursuit: holding a question it cannot answer, and working out what it needs.

Curiosity wanders — it notices any gap. Pursuit is aimed: given one question the
system cannot answer, it derives from memory the single fact that would unlock
it, asks for that, and comes back to the original question the moment it can.

This is the agent loop without an agent guessing at tools. The plan is not
invented; it is read off the hierarchy. "I don't know whether a penguin flies,
but I know a penguin is a bird — do birds fly?" is a step the memory itself
dictates.
"""
from lmm.relations import IS_A, CAN, HAS_PROPERTY
from lmm import phrasing
from lmm.similarity import nearest


class Step:
    def __init__(self, key, text):
        self.key = key      # stable identity, so the same step is not repeated
        self.text = text


class Pursuit:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning

    def resolved(self, goal):
        """Can the gate answer this question from what memory holds now?"""
        if goal.relation == IS_A:
            return bool(self.memory.query(goal.concept, IS_A))
        return self._lookup(goal) is not None

    def next_step(self, goal):
        """The one thing worth asking next, or None when nothing would help."""
        if self.resolved(goal):
            return None
        if goal.relation == IS_A or not self.memory.query(goal.concept, IS_A):
            # Without a place in the hierarchy there is nothing to reason from.
            return Step(f"type:{goal.concept}",
                        phrasing.definition_question(goal.concept))
        for ancestor in self.reasoning.ancestors(goal.concept):
            if self._lookup(goal, ancestor) is None:
                return Step(f"{goal.relation}:{ancestor}:{goal.target}",
                            self._question(ancestor, goal))
        return None

    def opening(self, goal):
        """What to say when a question arrives that cannot be answered yet."""
        step = self.next_step(goal)
        if step is None:
            return phrasing.dont_know(goal.concept)
        if step.key.startswith("type:"):
            # A concept with nothing at all behind it might be a word the person
            # spelled differently from the one we know.
            suggestions = ()
            if not self.memory.query(goal.concept):
                suggestions = nearest(goal.concept, self.memory.concepts())
            return phrasing.need_first(step.text, suggestions)
        ancestors = self.reasoning.ancestors(goal.concept)
        return phrasing.climbing(goal.concept, ancestors[0], step.text)

    def _lookup(self, goal, concept=None):
        concept = concept if concept is not None else goal.concept
        obj = getattr(goal, "object", None)
        role = getattr(goal, "role", None)
        if goal.relation == HAS_PROPERTY:
            return self.reasoning.has_property(concept, goal.target, obj, role)[0]
        return self.reasoning.can_do(concept, goal.target, obj, role)[0]

    def _question(self, concept, goal):
        if goal.relation == HAS_PROPERTY:
            return phrasing.property_question(concept, goal.target)
        return phrasing.ability_question(concept, goal.target,
                                         getattr(goal, "object", None),
                                         getattr(goal, "role", None))
