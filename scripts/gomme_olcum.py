"""Does counting alone find the word classes nobody taught it?

The claim in vectors.py is that distribution carries grammar: words that keep
the same company behave the same way. If that is true, a verb's nearest
neighbours should be verbs, without the lexicon ever being consulted — and the
score should beat what picking at random would give.

This is the falsifiable form of "the system learns the language itself". A
result near the baseline means the geometry is decorative and should be dropped.

Usage: python3 scripts/gomme_olcum.py
"""
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmm.vectors import Vectors                        # noqa: E402
from lmm.intuition import tokenize                     # noqa: E402
from lmm.reading import sentences                      # noqa: E402
from lmm.memory import Memory                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from lmm import registry                                    # noqa: E402
NEIGHBOURS = 5


def corpus():
    """Every Turkish sentence this project has, whatever it was written for."""
    found = []
    packs = os.path.join(ROOT, "packs")
    for name in sorted(os.listdir(packs)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(packs, name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and not line.startswith("kelime:"):
                    found.append(tokenize(line))
    document = os.path.join(ROOT, "docs",
                            "Bankacilikta_LLM_Teknik_Mimari_Ruzgar_Kanar (1).docx")
    if os.path.exists(document):
        with zipfile.ZipFile(document) as archive:
            xml = archive.read("word/document.xml").decode("utf-8")
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", xml))
        found.extend(tokenize(s) for s in sentences(text) if s.strip())
    return [tokens for tokens in found if len(tokens) > 1]


def classes(memory):
    """What we happen to know, used only to mark the answers — never as input."""
    verbs = set()
    for entry in memory.vocabulary:
        verbs.update((entry["positive"], entry["negative"]))
    properties = {p for p in memory.properties()}
    concepts = {c for c in memory.concepts()}
    return {"fiil": verbs, "nitelik": properties, "kavram": concepts}


def measure(vectors, known):
    print(f"{'sınıf':<10} {'kelime':>7} {'komşusu aynı sınıf':>20} {'rastgele olsa':>15}")
    total_hits = total_slots = 0
    for label, members in known.items():
        present = [w for w in members if vectors.vector(w) is not None]
        if len(present) < 3:
            continue
        hits = slots = 0
        for word in present:
            for neighbour, _ in vectors.similar(word, NEIGHBOURS):
                slots += 1
                if neighbour in members:
                    hits += 1
        baseline = len(present) / len(vectors.vectors)
        total_hits += hits
        total_slots += slots
        print(f"{label:<10} {len(present):>7} {hits / slots * 100:>19.0f}%"
              f" {baseline * 100:>14.0f}%")
    if total_slots:
        print(f"\nTOPLAM  komşuların %{total_hits / total_slots * 100:.0f}'i "
              f"aynı sınıftan")
    return total_hits / total_slots if total_slots else 0.0


def main():
    text = corpus()
    print(f"derlem: {len(text)} cümle, {sum(len(s) for s in text)} kelime")
    vectors = Vectors().learn(text)
    print(f"sözcük dağarcığı: {len(vectors.vectors)}, "
          f"boyut: {vectors.dimensions}  (GPU yok, gradyan yok, rastgelelik yok)")
    memory = Memory.load(registry.where("graph"))
    measure(vectors, classes(memory))
    print("\nörnek komşuluklar:")
    for word in ("uçar", "kuş", "beyaz", "banka", "model"):
        found = vectors.similar(word, 6)
        if found:
            print(f"  {word:<8} {', '.join(w for w, _ in found)}")


if __name__ == "__main__":
    main()
