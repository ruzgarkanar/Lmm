"""Builds a pack that teaches LMM about itself — in words it does not yet know.

Run: python3 packs/build_lmm_pack.py

Nothing here touches the engine. The pack brings its own verbs, so installing it
widens what the system can hear and say. That is the point: a domain extends the
language without a code change.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.memory import Memory, Edge, IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY, \
    LACKS_PROPERTY  # noqa: E402
from lmm.pack import export_pack  # noqa: E402

AUTHOR = "rüzgar"
NAME = "lmm-tanitim"
VERSION = "1.0"

# Verbs the core lexicon has never heard of.
WORDS = [
    ("öğrenmek", "öğrenir", "öğrenemez"),
    ("hatırlamak", "hatırlar", "hatırlayamaz"),
    ("unutmak", "unutur", "unutamaz"),
    ("uydurmak", "uydurur", "uyduramaz"),
]

FACTS = [
    ("llm", IS_A, "modeldir"),
    ("lmm", IS_A, "modeldir"),
    ("model", IS_A, "yazılımdır"),
    ("lmm", NOT_A, "llm"),

    ("llm", CANNOT, "öğrenmek"),        # weights freeze when training ends
    ("lmm", CAN, "öğrenmek"),
    ("lmm", CAN, "hatırlamak"),
    ("lmm", CAN, "unutmak"),            # deliberately, fact by fact
    ("llm", CANNOT, "unutmak"),
    ("llm", CAN, "uydurmak"),
    ("lmm", CANNOT, "uydurmak"),

    ("llm", HAS_PROPERTY, "kapalı"),
    ("lmm", HAS_PROPERTY, "okunabilir"),
    ("llm", LACKS_PROPERTY, "okunabilir"),
]


def build(path):
    memory = Memory()
    for infinitive, positive, negative in WORDS:
        memory.learn_word(infinitive, positive, negative)
    for concept, relation, target in FACTS:
        target = target[:-3] if relation in (IS_A, NOT_A) and \
            target.endswith(("dır", "dir", "dur", "dür", "tır", "tir")) else target
        memory.write(Edge(concept, relation, target, source=AUTHOR, confidence=0.9))
    pack = export_pack(memory, path, name=NAME, version=VERSION, author=AUTHOR)
    print(f"{len(pack.facts)} bilgi, {len(pack.vocabulary)} kelime -> {path}")
    return pack


if __name__ == "__main__":
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"{NAME}.json"))
