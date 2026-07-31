"""The pipeline: sources in, a model file out.

A language model is trained by collecting text, cleaning it, and running gradient
descent over it for weeks on a rented cluster. The stages here are the same
stages — collect, clean, audit, learn, generalise, evaluate, ship — and the
mechanism in the middle is the whole difference. Nothing here computes a
gradient, so nothing here needs a GPU.

    1  TOPLAMA      documents, packs, and a model's answers to our own questions
    2  TEMİZLEME    every sentence parsed or skipped; nothing half-understood
    3  DENETİM      anything contradicting what is known is refused, and named
    4  ÖĞRENME      what survives is written with the source it came from
    5  GENELLEME    rules the memory supports are formed and marked as guesses
    6  DEĞERLENDİRME what it can answer, and how much of it nobody wrote
    7  PAKETLEME    one file: facts, words, patterns, questions already asked

Usage: python3 -m lmm.train <kaynak-klasörü> [model.lmm] [--hasat N]
"""
import os
import sys

from lmm.memory import Memory, INFERRED
from lmm.reasoning import Reasoning
from lmm.reading import read_file
from lmm.distill import distill_file, generalise
from lmm.pack import read_pack, merge_pack
from lmm.trust import level, DISTILLED, DOCUMENT, HUMAN
from lmm.drift import frozen_branches, refusal_rate

DOCUMENT_SUFFIX = ".txt"
PACK_SUFFIX = ".json"
DISTILLED_MARK = "-cekirdek"    # a document written by a model, not a person


class Run:
    """What each stage did, kept so the report can be honest about all of it."""

    def __init__(self):
        self.sources = []
        self.learned = 0
        self.reinforced = 0
        self.refused = []       # (source, concept, explanation)
        self.skipped = []       # (source, sentence)
        self.words = 0
        self.rules = []


def collect(directory):
    """Stage 1. Every source, in a stable order so a run is reproducible."""
    found = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if name.endswith(DOCUMENT_SUFFIX):
            kind = "damıtma" if DISTILLED_MARK in name else "doküman"
            found.append((kind, path))
        elif name.endswith(PACK_SUFFIX):
            found.append(("paket", path))
    return found


def absorb(memory, kind, path, run, model_name="claude"):
    """Stages 2-4: parse, audit, and write what survives."""
    name = os.path.basename(path)
    if kind == "paket":
        report = merge_pack(memory, read_pack(path))
        run.words += len(report.words)
        run.learned += len(report.added)
        run.reinforced += len(report.reinforced)
        for edge, explanation in report.conflicts:
            run.refused.append((name, edge.concept, explanation))
        return
    reader = distill_file if kind == "damıtma" else read_file
    report = (reader(path, memory, model_name) if kind == "damıtma"
              else reader(path, memory))
    run.words += len(getattr(report, "words", []))
    run.learned += len(report.learned)
    run.reinforced += len(report.reinforced)
    for edge, explanation in report.conflicts:
        run.refused.append((name, edge.concept, explanation))
    for sentence, reason in report.skipped:
        run.skipped.append((name, sentence))


def evaluate(memory):
    """Stage 6. The numbers that say whether the run was worth anything."""
    reasoning = Reasoning(memory)
    stated = derived = guessed = 0
    for concept in memory.concepts():
        for targets, lookup, pair in (
                (memory.actions(), reasoning.can_do, ("can", "cannot")),
                (memory.properties(), reasoning.has_property,
                 ("property", "not_property"))):
            for target in targets:
                if lookup(concept, target)[0] is None:
                    continue
                edge = (memory.direct(concept, pair[0], target)
                        or memory.direct(concept, pair[1], target))
                if edge is None:
                    derived += 1
                elif edge.source == INFERRED:
                    guessed += 1
                else:
                    stated += 1
    return stated, derived, guessed


def train(directory, model_path, harvest_rounds=0):
    memory = Memory.load(model_path) if os.path.exists(model_path) else Memory()
    run = Run()

    print(f"1  TOPLAMA      {directory}")
    run.sources = collect(directory)
    for kind, path in run.sources:
        print(f"     {kind:9} {os.path.basename(path)}")

    print("2  TEMİZLEME / 3  DENETİM / 4  ÖĞRENME")
    for kind, path in run.sources:
        absorb(memory, kind, path, run)
    print(f"     {run.learned} bilgi, {run.words} kelime öğrenildi; "
          f"{run.reinforced} pekişti")
    print(f"     {len(run.refused)} çelişki reddedildi, "
          f"{len(run.skipped)} cümle anlaşılmadı")
    for source, concept, explanation in run.refused[:5]:
        print(f"     reddedildi [{source}] {concept} — {explanation}")

    if harvest_rounds:
        print(f"   + HASAT       merakın soruları, {harvest_rounds} tur")
        _harvest(memory, harvest_rounds, run)

    print("5  GENELLEME")
    run.rules = generalise(memory)
    print(f"     {len(run.rules)} kural kendi çıkarıldı")
    for rule in run.rules[:3]:
        print(f"     {rule.concept} — {rule.relation} → {rule.target} "
              f"({', '.join(rule.examples)})")

    frozen = frozen_branches(memory)
    if frozen:
        print("   ! DONMUŞ DALLAR — çok fazla çelişki, insan bakmalı")
        for branch in frozen:
            print(f"     {branch}: ret oranı %{refusal_rate(memory, branch)*100:.0f}")
    print("6  DEĞERLENDİRME")
    stated, derived, guessed = evaluate(memory)
    total = stated + derived + guessed
    share = derived / total * 100 if total else 0
    sources = {}
    for edge in memory.edges:
        sources[level(edge.source)] = sources.get(level(edge.source), 0) + 1
    print(f"     bilgi {len(memory.edges)}, kavram {len(memory.concepts())}, "
          f"kelime {len(memory.vocabulary)}, kalıp {len(memory.patterns)}")
    print(f"     kaynak: insan {sources.get(HUMAN, 0)}, "
          f"doküman {sources.get(DOCUMENT, 0)}, "
          f"dil modeli {sources.get(DISTILLED, 0)}")
    print(f"     cevaplanabilir {total} — yazılı {stated}, "
          f"çıkarımla {derived} (%{share:.0f}), genellemeyle {guessed}")

    print("7  PAKETLEME")
    memory.save(model_path)
    size = os.path.getsize(model_path) / 1024
    print(f"     {model_path}  {size:.1f} KB  (GPU kullanılmadı)")
    return memory, run


def _harvest(memory, rounds, run):
    from lmm.harvest import harvest, HarvestError
    for _ in range(rounds):
        try:
            questions, report = harvest(memory, count=8)
        except HarvestError as error:
            print(f"     hasat atlandı: {error}")
            return
        if report is None:
            print("     merak edecek bir şey kalmadı")
            return
        run.learned += len(report.learned)
        for edge, explanation in report.conflicts:
            run.refused.append(("hasat", edge.concept, explanation))
        print(f"     {len(questions)} soruldu, {len(report.learned)} bilgi alındı")


def main(argv):
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[-1])
        return 1
    directory = argv[1]
    model_path = argv[2] if len(argv) > 2 and not argv[2].startswith("--") \
        else "model.lmm"
    rounds = 0
    if "--hasat" in argv:
        rounds = int(argv[argv.index("--hasat") + 1])
    train(directory, model_path, rounds)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
