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
from lmm import lexicon
from lmm.clauses import readable
from lmm.compiler import usable


def words_of(memory):
    """Keşfedilen biçimbilim silindi: okuyucuya kelime tablosu verilmiyor."""
    return set()
from lmm.trust import level, DOCUMENT
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
        self.words = []          # verbs the document declared

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
    lexicon.use(memory.lexicon)     # read in this memory's own words
    reasoning = reasoning or Reasoning(memory)
    # The grammar needs this memory's own words and concepts: without them a
    # two-way possessive split cannot be resolved and every "kuşun kanadı var"
    # falls out as ambiguous. The conversation path already passed them.
    language = language or Intuition(network=MiniNetwork.default(),
                                     lexicon=memory.lexicon,
                                     words=words_of(memory),
                                     known=memory.concepts,
                                     memory=memory)
    learning = LearningLoop(memory, reasoning)
    report = ReadingReport(source)
    # A document may declare the words it is about to use. This only worked for
    # text written by a model, so a person could not hand the system a verb it
    # had just asked for — the one thing it says when it does not understand.
    from lmm.distill import split_words   # the two read each other
    declared, text = split_words(text)
    for infinitive, positive, negative in declared:
        memory.learn_word(infinitive, positive, negative)
        report.words.append(infinitive)
    if declared:
        language = Intuition(network=MiniNetwork.default(),
                             lexicon=memory.lexicon,
                             words=words_of(memory), known=memory.concepts)
    # Uzun cümle olduğu gibi okunamıyor: en uzun kalıbımız dört kelime, gerçek
    # cümlelerin %78'i beş ve üzeri. Bölünce ölçülen kazanç, bankacılık
    # dokümanında 3,1 -> 8,8 olgu/100 cümle. Kısa cümleye dokunulmuyor.
    for sentence in [piece for whole in sentences(text)
                     for piece in readable(whole, lexicon=memory.lexicon)]:
        intent = language.understand(sentence)
        if intent.kind == UNKNOWN:
            report.skipped.append((sentence, NOT_UNDERSTOOD))
            continue
        if intent.kind != TEACH:
            report.skipped.append((sentence, A_QUESTION))
            continue
        # Cümlecik bölme okunabilir parça sayısını artırdı ama ayrıştırıcı
        # bazılarını yanlış okuyor: "tool broker /" bir kavram değil, bir
        # ayrıştırma kazası. Derleyicide aynı sorun için yazılan hijyen denetimi
        # burada da geçerli — bir düğüm, başka olgularda da geçebilecek bir ad
        # olmalı.
        if not (usable(intent.concept) and usable(intent.target or "x")):
            report.skipped.append((sentence, NOT_UNDERSTOOD))
            continue
        status, _, edge = learning.teach(intent, source=source)
        if status == LEARNED:
            report.learned.append(edge)
        elif status == REINFORCED:
            report.reinforced.append(edge)
        elif status == CONFLICT:
            # A document cannot be asked "are you sure?", so an exception it
            # states was being thrown away: every pack said "penguen uçamaz" and
            # the trained model still answered that penguins fly. But a clash
            # with an *inherited* fact is not a contradiction — it is the
            # exception the inheritance was always allowed to have, and a direct
            # statement about a concept beats what its type says. Only a clash
            # with something stated about this very concept is a real dispute,
            # and that still goes to the report rather than being decided here.
            # Only a source we would trust to state a rule may state an
            # exception to one. A language model's answer that fights what is
            # known is exactly the thing the gate exists to stop, and it stays
            # refused; a document written by a person is deliberate.
            basis = reasoning.basis(edge)
            if (level(source) >= DOCUMENT and basis is not None
                    and basis.concept != edge.concept):
                report.learned.append(learning.confirm_exception(edge))
            else:
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
