"""Measuring what a memory is worth.

The claims LMM makes are the kind that can be counted, so they should be. This
reports how much a memory holds, how fast it answers, and — the number that
matters most — how much of what it can answer nobody ever wrote down.

Usage: python3 scripts/olcum.py memory.json
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.memory import Memory, INFERRED  # noqa: E402
from lmm.reasoning import Reasoning  # noqa: E402
from lmm.trust import level, DISTILLED, HUMAN, DOCUMENT  # noqa: E402


def answerable(memory, reasoning):
    """Every question the memory can answer, split by where the answer came from.

    Directly stated, derived through the hierarchy, or generalised by the system
    itself. The derived share is what a lookup table could never give you.
    """
    stated = derived = generalised = 0
    concepts = memory.concepts()
    for concept in concepts:
        for target, lookup, relations in (
                (memory.actions(), reasoning.can_do, ("can", "cannot")),
                (memory.properties(), reasoning.has_property,
                 ("property", "not_property"))):
            for item in target:
                answer, _ = lookup(concept, item)
                if answer is None:
                    continue
                edge = (memory.direct(concept, relations[0], item)
                        or memory.direct(concept, relations[1], item))
                if edge is None:
                    derived += 1
                elif edge.source == INFERRED:
                    generalised += 1
                else:
                    stated += 1
    return stated, derived, generalised


def report(path):
    memory = Memory.load(path)
    reasoning = Reasoning(memory)
    size = os.path.getsize(path)

    sources = {}
    for edge in memory.edges:
        sources[level(edge.source)] = sources.get(level(edge.source), 0) + 1

    started = time.time()
    for concept in memory.concepts():
        reasoning.can_do(concept, "uçmak")
    elapsed = (time.time() - started) / max(1, len(memory.concepts())) * 1000

    stated, derived, generalised = answerable(memory, reasoning)
    total = stated + derived + generalised

    print(f"BELLEK: {path}")
    print(f"  bilgi          : {len(memory.edges)}")
    print(f"  kavram         : {len(memory.concepts())}")
    print(f"  eylem/nitelik  : {len(memory.actions())} / {len(memory.properties())}")
    print(f"  kelime         : {len(memory.vocabulary)}")
    print(f"  dosya          : {size / 1024:.1f} KB  ({size / max(1, len(memory.edges)):.0f} bayt/bilgi)")
    print()
    print("KAYNAK DAĞILIMI")
    for name, key in (("insan", HUMAN), ("doküman", DOCUMENT),
                      ("dil modeli", DISTILLED)):
        if sources.get(key):
            print(f"  {name:12}: {sources[key]}")
    print()
    print("CEVAPLANABİLİR SORU")
    print(f"  toplam         : {total}")
    print(f"  doğrudan yazılı: {stated}")
    print(f"  çıkarımla      : {derived}  (%{derived / max(1, total) * 100:.0f})")
    print(f"  genellemeyle   : {generalised}")
    print()
    print(f"HIZ: soru başına {elapsed:.3f} ms, GPU yok")
    print(f"KAZANÇ: yazılan her bilgi {total / max(1, len(memory.edges)):.1f} soru cevaplıyor")


if __name__ == "__main__":
    report(sys.argv[1] if len(sys.argv) > 1 else "memory.json")
