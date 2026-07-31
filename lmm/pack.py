"""Memory Packs: how LMM knowledge travels between installations.

An LLM ships as weights nobody can read, and merging two of them averages
numbers until the disagreement disappears. A pack ships as facts anybody can
read, diff, delete or refuse — each carrying the source it came from. Merging
surfaces contradictions instead of absorbing them: the receiving memory asks
rather than quietly picking a side.
"""
import json
import sys
import time

from lmm.memory import Memory, Edge, CAN, CANNOT
from lmm.reasoning import Reasoning

FORMAT_VERSION = 1


class Pack:
    def __init__(self, name, facts, vocabulary=None, version="1.0",
                 author="unknown", created=None):
        self.name = name
        self.facts = facts                      # list of Edge
        self.vocabulary = vocabulary or []      # words the facts need
        self.version = version
        self.author = author
        self.created = created if created is not None else time.time()

    @property
    def provenance(self):
        """What answers will cite once these facts are learned."""
        return f"{self.name}@{self.version}"

    def to_dict(self):
        return {"format": FORMAT_VERSION, "name": self.name, "version": self.version,
                "author": self.author, "created": self.created,
                "vocabulary": self.vocabulary,
                "facts": [e.to_dict() for e in self.facts]}

    @staticmethod
    def from_dict(data):
        return Pack(name=data["name"], version=data.get("version", "1.0"),
                    author=data.get("author", "unknown"), created=data.get("created"),
                    vocabulary=data.get("vocabulary", []),
                    facts=[Edge.from_dict(f) for f in data["facts"]])


class MergeReport:
    def __init__(self):
        self.added = []
        self.reinforced = []
        self.words = []          # vocabulary the pack brought with it
        self.conflicts = []      # (edge, explanation) pairs, never written

    @property
    def clean(self):
        return not self.conflicts


def export_pack(memory, path, name, version="1.0", author="unknown"):
    pack = Pack(name=name, version=version, author=author,
                facts=list(memory.edges), vocabulary=list(memory.vocabulary))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pack.to_dict(), f, ensure_ascii=False, indent=1)
    return pack


def read_pack(path):
    with open(path, encoding="utf-8") as f:
        return Pack.from_dict(json.load(f))


def _is_declared_exception(memory, candidate):
    """An exception its author already adjudicated, not a fresh disagreement.

    "penguen uçamaz" clashes with the inherited "kuşlar uçar" by definition —
    that is what makes it an exception, and it must survive the journey. But if
    the receiving memory holds the opposite as a fact of its own, two sources
    genuinely disagree and that has to be reported instead.
    """
    if not candidate.is_exception:
        return False
    opposite = CAN if candidate.relation == CANNOT else CANNOT
    return memory.direct(candidate.concept, opposite, candidate.target) is None


def merge_pack(memory, pack, reasoning=None):
    """Fold a pack into a memory, refusing anything that contradicts what is known.

    Conflicts are reported, not resolved: a machine that silently picks a winner
    between two sources is guessing, and guessing is what LMM exists to avoid.
    """
    reasoning = reasoning if reasoning is not None else Reasoning(memory)
    report = MergeReport()
    for word in pack.vocabulary:        # words first: the facts are said with them
        memory.learn_word(**word)
        report.words.append(word)
    for fact in pack.facts:
        candidate = Edge(fact.concept, fact.relation, fact.target,
                         source=pack.provenance, confidence=fact.confidence,
                         is_exception=fact.is_exception)
        conflict = reasoning.find_conflict(candidate)
        if conflict is not None and not _is_declared_exception(memory, candidate):
            report.conflicts.append((candidate, conflict))
            continue
        existing = memory.direct(candidate.concept, candidate.relation, candidate.target)
        memory.write(candidate)
        (report.reinforced if existing is not None else report.added).append(candidate)
    return report


def main(argv):
    """Usage: python3 -m lmm.pack export|merge <memory.json> <pack.json> [name]"""
    if len(argv) < 4:
        print(main.__doc__)
        return 1
    command, memory_path, pack_path = argv[1], argv[2], argv[3]
    memory = Memory.load(memory_path)
    if command == "export":
        name = argv[4] if len(argv) > 4 else "unnamed"
        pack = export_pack(memory, pack_path, name)
        print(f"{len(pack.facts)} bilgi paketlendi: {pack_path} ({pack.provenance})")
        return 0
    if command == "merge":
        report = merge_pack(memory, read_pack(pack_path))
        memory.save(memory_path)
        print(f"{len(report.words)} yeni kelime, {len(report.added)} yeni bilgi, "
              f"{len(report.reinforced)} pekişen, {len(report.conflicts)} çelişkili.")
        for edge, explanation in report.conflicts:
            print(f"  çelişki: {edge.concept} — {explanation}")
        return 0
    print(main.__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
