"""Living Memory: LMM's knowledge organ.

Knowledge does not live in weights here — it lives in this graph, writable at any
moment, persisted to disk, and always carrying its source and confidence.
"""
import gzip
import json
import os
import time

from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,  # noqa: F401
                           LACKS_PROPERTY, TYPE_RELATIONS, ABILITY_RELATIONS,
                           PROPERTY_RELATIONS)

from lmm.lexicon import ACTIVE
from lmm.trust import INFERENCE, confidence_for

FORMAT_VERSION = 3

# The source on facts nobody stated — the system worked them out itself.
INFERRED = INFERENCE
INFERRED_CONFIDENCE = confidence_for(INFERENCE)


class CycleError(Exception):
    """Raised when a write would create a cycle in the type hierarchy."""


class Edge:
    def __init__(self, concept, relation, target, source="unknown",
                 confidence=0.6, is_exception=False, timestamp=None):
        self.concept = concept
        self.relation = relation        # IS_A | CAN | CANNOT
        self.target = target
        self.source = source
        self.confidence = confidence
        self.is_exception = is_exception
        self.timestamp = timestamp if timestamp is not None else time.time()

    def to_dict(self):
        return {"concept": self.concept, "relation": self.relation,
                "target": self.target, "source": self.source,
                "confidence": self.confidence, "is_exception": self.is_exception,
                "timestamp": self.timestamp}

    @staticmethod
    def from_dict(data):
        return Edge(**data)


class Memory:
    """Facts the system knows, plus the questions it has already put to a teacher.

    Both are memory: one is what it believes, the other is what it has already
    wondered aloud. Without the second, curiosity would ask the same question
    forever — which is the opposite of learning.
    """

    def __init__(self):
        self.edges = []
        self.asked = set()
        self.vocabulary = []    # words learned beyond the core lexicon
        self._rebuild()

    def _rebuild(self):
        """Indexes over the edges. Everything here is derivable from self.edges.

        Without them every question scanned every fact, which is fine at three
        hundred and hopeless at a million — and curiosity and induction sweep the
        whole memory, so the cost was quadratic where it hurt most.
        """
        self._by_concept = {}
        self._exact = {}
        self._concepts = []
        self._concept_set = set()
        self._actions, self._action_set = [], set()
        self._properties, self._property_set = [], set()
        for edge in self.edges:
            self._index(edge)

    def _index(self, edge):
        self._by_concept.setdefault(edge.concept, []).append(edge)
        self._exact[(edge.concept, edge.relation, edge.target)] = edge
        self._note_concept(edge.concept)
        if edge.relation in TYPE_RELATIONS:
            self._note_concept(edge.target)
        elif edge.relation in ABILITY_RELATIONS:
            self._note(edge.target, self._actions, self._action_set)
        elif edge.relation in PROPERTY_RELATIONS:
            self._note(edge.target, self._properties, self._property_set)

    def _note_concept(self, name):
        self._note(name, self._concepts, self._concept_set)

    @staticmethod
    def _note(name, ordered, seen):
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    def learn_word(self, infinitive, positive, negative, lexicon=None):
        """Add a verb to what this memory knows how to say, and can hear."""
        entry = {"infinitive": infinitive, "positive": positive,
                 "negative": negative}
        if entry not in self.vocabulary:
            self.vocabulary.append(entry)
        (lexicon or ACTIVE).learn_verb(infinitive, positive, negative)
        return entry

    def mark_asked(self, key):
        self.asked.add(key)

    def has_asked(self, key):
        return key in self.asked

    def query(self, concept, relation=None):
        found = self._by_concept.get(concept, ())
        if relation is None:
            return list(found)
        return [e for e in found if e.relation == relation]

    def direct(self, concept, relation, target):
        return self._exact.get((concept, relation, target))

    def write(self, edge):
        if edge.relation == IS_A and self._creates_cycle(edge):
            raise CycleError(f"{edge.concept} -> {edge.target} creates a cycle")
        existing = self.direct(edge.concept, edge.relation, edge.target)
        if existing is not None:
            existing.confidence = min(1.0, existing.confidence + 0.2)
            return existing
        self.edges.append(edge)
        self._index(edge)
        return edge

    def concepts(self):
        """Everything that behaves like a thing, in the order it was learned."""
        return list(self._concepts)

    def actions(self):
        """Every action this memory has ever heard of, in learning order."""
        return list(self._actions)

    def properties(self):
        """Every property this memory has ever heard of, in learning order."""
        return list(self._properties)

    def forget(self, concept):
        """Erase everything known about a concept, in or out. Returns the count.

        Selective deletion from a trained model's weights is famously impractical;
        here it is a filter over a list, and afterwards the system genuinely does
        not know — it goes back to saying "bilmiyorum".
        """
        remaining = [e for e in self.edges
                     if e.concept != concept and e.target != concept]
        removed = len(self.edges) - len(remaining)
        self.edges = remaining
        self._rebuild()
        return removed

    def _creates_cycle(self, edge):
        # Walk type edges up from the target: can we reach the concept again?
        queue, seen = [edge.target], set()
        while queue:
            current = queue.pop(0)
            if current == edge.concept:
                return True
            if current in seen:
                continue
            seen.add(current)
            queue.extend(e.target for e in self.query(current, IS_A))
        return False

    @staticmethod
    def _opener(path):
        """A .lmm file is the same JSON, gzipped — around twenty times smaller.

        Compression is a storage detail, not a change of kind: gunzip it and you
        are looking at the same readable facts. Nothing becomes a black box.
        """
        return (gzip.open, "wt", "rt") if path.endswith(".lmm") else (open, "w", "r")

    def save(self, path):
        payload = {"format": FORMAT_VERSION,
                   "edges": [e.to_dict() for e in self.edges],
                   "asked": sorted(self.asked),
                   "vocabulary": self.vocabulary}
        opener, write_mode, _ = self._opener(path)
        temp = path + ".tmp"
        with opener(temp, write_mode, encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(temp, path)  # atomic: a half-written file never appears

    @staticmethod
    def load(path):
        memory = Memory()
        try:
            opener, _, read_mode = Memory._opener(path)
            with opener(path, read_mode, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):        # format 1: a bare list of edges
                memory.edges = [Edge.from_dict(d) for d in data]
                memory._rebuild()
            else:
                memory.edges = [Edge.from_dict(d) for d in data["edges"]]
                memory._rebuild()
                memory.asked = set(data.get("asked", []))
                for word in data.get("vocabulary", []):
                    memory.learn_word(**word)   # words come back with the facts
        except (OSError, ValueError, KeyError, TypeError):
            pass  # missing or corrupt file: start with an empty memory
        return memory
