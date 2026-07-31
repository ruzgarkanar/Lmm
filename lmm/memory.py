"""Living Memory: LMM's knowledge organ.

Knowledge does not live in weights here — it lives in this graph, writable at any
moment, persisted to disk, and always carrying its source and confidence.
"""
import json
import os
import time

FORMAT_VERSION = 2

# Relation kinds stored on edges.
IS_A = "type"
CAN = "can"
CANNOT = "cannot"


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

    def mark_asked(self, key):
        self.asked.add(key)

    def has_asked(self, key):
        return key in self.asked

    def query(self, concept, relation=None):
        return [e for e in self.edges
                if e.concept == concept and (relation is None or e.relation == relation)]

    def direct(self, concept, relation, target):
        for e in self.edges:
            if e.concept == concept and e.relation == relation and e.target == target:
                return e
        return None

    def write(self, edge):
        if edge.relation == IS_A and self._creates_cycle(edge):
            raise CycleError(f"{edge.concept} -> {edge.target} creates a cycle")
        existing = self.direct(edge.concept, edge.relation, edge.target)
        if existing is not None:
            existing.confidence = min(1.0, existing.confidence + 0.2)
            return existing
        self.edges.append(edge)
        return edge

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

    def save(self, path):
        payload = {"format": FORMAT_VERSION,
                   "edges": [e.to_dict() for e in self.edges],
                   "asked": sorted(self.asked)}
        temp = path + ".tmp"
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(temp, path)  # atomic: a half-written file never appears

    @staticmethod
    def load(path):
        memory = Memory()
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):        # format 1: a bare list of edges
                memory.edges = [Edge.from_dict(d) for d in data]
            else:
                memory.edges = [Edge.from_dict(d) for d in data["edges"]]
                memory.asked = set(data.get("asked", []))
        except (OSError, ValueError, KeyError, TypeError):
            pass  # missing or corrupt file: start with an empty memory
        return memory
