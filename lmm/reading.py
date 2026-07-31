"""Reading: learning from a document instead of from a teacher.

Conversation is one way in, not the definition of the system. A document is the
same learning loop with a different source: sentences it understands become
permanent knowledge, sentences it does not are skipped and reported rather than
half-guessed, and anything contradicting what is already known is refused.

Refusing is deliberate. There is nobody at the keyboard to adjudicate, and a
machine that silently picks a winner between two sources is guessing. The
contradictions come back in the report for a person to settle.

Usage: python3 -m lmm.reading memory.json document.txt
"""
import os
import sys

from lmm.memory import Memory
from lmm.reasoning import Reasoning
from lmm.learning import LearningLoop, LEARNED, REINFORCED, CONFLICT, REJECTED
from lmm.intuition import Intuition, TEACH, UNKNOWN
from lmm.network import MiniNetwork

SENTENCE_ENDINGS = ".!?\n;"

NOT_UNDERSTOOD = "anlaşılmadı"
A_QUESTION = "soru"
REFUSED = "kabul edilmedi"


class ReadingReport:
    def __init__(self, source):
        self.source = source
        self.learned = []
        self.reinforced = []
        self.conflicts = []      # (edge, explanation) — never written
        self.skipped = []        # (sentence, reason)

    @property
    def total(self):
        return (len(self.learned) + len(self.reinforced) + len(self.conflicts)
                + len(self.skipped))


def sentences(text):
    """Split on sentence endings and line breaks, dropping the empties."""
    found, current = [], ""
    for character in text:
        if character in SENTENCE_ENDINGS:
            if current.strip():
                found.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        found.append(current.strip())
    return found


def read_text(text, memory, source, reasoning=None, language=None):
    """Fold a document into a memory. Returns a ReadingReport."""
    reasoning = reasoning or Reasoning(memory)
    language = language or Intuition(network=MiniNetwork.default())
    learning = LearningLoop(memory, reasoning)
    report = ReadingReport(source)
    for sentence in sentences(text):
        intent = language.understand(sentence)
        if intent.kind == UNKNOWN:
            report.skipped.append((sentence, NOT_UNDERSTOOD))
            continue
        if intent.kind != TEACH:
            report.skipped.append((sentence, A_QUESTION))
            continue
        status, _, edge = learning.teach(intent, source=source)
        if status == LEARNED:
            report.learned.append(edge)
        elif status == REINFORCED:
            report.reinforced.append(edge)
        elif status == CONFLICT:
            report.conflicts.append((edge, _explain(reasoning, edge)))
        elif status == REJECTED:
            report.skipped.append((sentence, REFUSED))
    return report


def _explain(reasoning, edge):
    return reasoning.find_conflict(edge) or ""


def read_file(path, memory, reasoning=None, language=None):
    with open(path, encoding="utf-8") as f:
        return read_text(f.read(), memory, os.path.basename(path),
                         reasoning, language)


def main(argv):
    if len(argv) < 3:
        print(__doc__.strip().splitlines()[-1])
        return 1
    memory_path, document_path = argv[1], argv[2]
    memory = Memory.load(memory_path)
    report = read_file(document_path, memory)
    memory.save(memory_path)

    print(f"{report.source}: {report.total} cümle okundu — "
          f"{len(report.learned)} yeni bilgi, {len(report.reinforced)} pekişen, "
          f"{len(report.conflicts)} çelişkili, {len(report.skipped)} atlandı.")
    for edge, explanation in report.conflicts:
        print(f"  çelişki: {edge.concept} — {explanation}")
    for sentence, reason in report.skipped:
        if reason == NOT_UNDERSTOOD:
            print(f"  atlandı ({reason}): {sentence}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
