"""Distillation: taking knowledge out of a language model and into a memory.

A language model has read far more than any person could type, and it can state
what it knows in plain sentences. That makes it a usable source of raw material
— once, at the start. What it cannot do is be checked, corrected or trusted at
a specific point, which is exactly what happens to its output here.

Distilled facts enter marked as machine-sourced and at lower confidence than a
person's word. They are checked against each other on the way in: a model that
says birds fly and that a penguin is a bird and that a penguin cannot fly has
contradicted itself, and that contradiction is reported rather than absorbed.
Anything a human later says overrides them without argument.

The point is what happens afterwards. The model is not consulted again. The
memory it seeded runs on its own, with no GPU and no dependency on it.

Input format — plain Turkish sentences, plus optional word declarations:

    kelime: koşmak = koşar / koşamaz
    Kedi bir hayvandır. Kediler koşar. Kedi tüylüdür.

Usage: python3 -m lmm.distill memory.json distilled.txt [model-adı]
"""
import os
import sys

from lmm.memory import Memory
from lmm.reasoning import Reasoning
from lmm.reading import read_text
from lmm.induction import Induction
from lmm.phrasing import describe, listing
from lmm.trust import distilled_source
from lmm.intuition import lower

WORD_PREFIX = "kelime:"


def split_words(text):
    """Separate word declarations from the sentences. Returns (words, rest)."""
    words, lines = [], []
    for line in text.splitlines():
        stripped = line.strip()
        if not lower(stripped).startswith(WORD_PREFIX):
            lines.append(line)
            continue
        body = stripped[len(WORD_PREFIX):]
        if "=" not in body or "/" not in body:
            continue                    # malformed declarations are ignored
        infinitive, forms = body.split("=", 1)
        positive, negative = forms.split("/", 1)
        words.append((infinitive.strip(), positive.strip(), negative.strip()))
    return words, "\n".join(lines)


def distill_text(text, memory, model_name, reasoning=None, language=None):
    """Fold a model's output into a memory, marked and audited."""
    words, sentences = split_words(text)
    for infinitive, positive, negative in words:
        memory.learn_word(infinitive, positive, negative)
    report = read_text(sentences, memory, distilled_source(model_name),
                       reasoning or Reasoning(memory), language)
    report.words = words
    return report


def generalise(memory, reasoning=None, limit=1000):
    """Let the system draw its own rules from what it just took in.

    A bulk ingest is exactly when patterns become visible, and a rule the system
    forms itself is marked as such — it is beaten by anything a person says.
    """
    induction = Induction(memory, reasoning or Reasoning(memory))
    formed = []
    while len(formed) < limit:
        # One sweep collects every rule the memory currently supports; learning
        # one can settle another, so each is re-checked before it is written.
        batch = [h for h in induction.candidates() if induction.still_open(h)]
        if not batch:
            break
        for hypothesis in batch:
            if induction.still_open(hypothesis):
                induction.learn(hypothesis)
                formed.append(hypothesis)
    return formed


def distill_file(path, memory, model_name, reasoning=None, language=None):
    with open(path, encoding="utf-8") as f:
        return distill_text(f.read(), memory, model_name, reasoning, language)


def main(argv):
    if len(argv) < 3:
        print("Kullanım: python3 -m lmm.distill memory.json distilled.txt [model]")
        return 1
    memory_path, document_path = argv[1], argv[2]
    model_name = argv[3] if len(argv) > 3 else os.path.basename(document_path)
    memory = Memory.load(memory_path)
    report = distill_file(document_path, memory, model_name)
    formed = generalise(memory)
    memory.save(memory_path)

    print(f"{distilled_source(model_name)}: {report.total} cümle işlendi — "
          f"{len(report.words)} kelime, {len(report.learned)} bilgi alındı, "
          f"{len(report.reinforced)} pekişen, "
          f"{len(report.conflicts)} kendi kendiyle çelişti, "
          f"{len(report.skipped)} atlandı.")
    for edge, explanation in report.conflicts:
        print(f"  modelin çelişkisi: {edge.concept} — {explanation}")
    if formed:
        print(f"  ardından {len(formed)} kuralı kendi çıkardı, ilk üçü:")
        for hypothesis in formed[:3]:
            statement = describe(hypothesis.concept, hypothesis.relation,
                                 hypothesis.target)
            print(f"    {statement}  ({listing(hypothesis.examples)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
