"""LMM v0 chat interface. Usage: python3 -m lmm.cli [memory_file]"""
import sys

from lmm.memory import Memory
from lmm.reasoning import Reasoning
from lmm.gate import EpistemicGate
from lmm.intuition import Intuition, ASK, UNKNOWN
from lmm.learning import LearningLoop, CONFLICT
from lmm.network import MiniNetwork

CONFIDENCE_THRESHOLD = 0.35
AFFIRMATIVE = ("evet", "e")
EXIT_WORDS = ("çık", "cik", "exit")


class Session:
    """One conversation: the five organs wired together over a memory file."""

    def __init__(self, path):
        self.path = path
        self.memory = Memory.load(path)
        reasoning = Reasoning(self.memory)
        self.gate = EpistemicGate(self.memory, reasoning)
        self.intuition = Intuition(network=MiniNetwork.default())
        self.learning = LearningLoop(self.memory, reasoning)
        self.pending = None   # an edge awaiting the teacher's confirmation

    def respond(self, line):
        if self.pending is not None:
            return self._resolve_pending(line)
        intent = self.intuition.understand(line)
        if intent.kind == UNKNOWN or intent.confidence < CONFIDENCE_THRESHOLD:
            return ("bunu anlamadım. 'x bir y', 'x uçar', 'x uçar mı', 'x nedir' "
                    "kalıplarıyla konuşabiliyorum şimdilik.")
        if intent.kind == ASK:
            return self.gate.answer(intent)
        status, message, edge = self.learning.teach(intent)
        if status == CONFLICT:
            self.pending = edge
        return message

    def _resolve_pending(self, line):
        edge, self.pending = self.pending, None
        if line.lower() in AFFIRMATIVE:
            self.learning.confirm_exception(edge)
            return "öğrendim (istisna olarak işledim)."
        return "tamam, öğrenmedim."

    def save(self):
        self.memory.save(self.path)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "memory.json"
    session = Session(path)
    print(f"LMM v0 — yaşayan bellek: {path} ({len(session.memory.edges)} bilgi)")
    print("öğret: 'penguen bir kuştur' | sor: 'penguen uçar mı' | çık: 'çık'")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line.lower() in EXIT_WORDS:
            break
        print(session.respond(line))
        session.save()
    session.save()
    print("bellek kaydedildi. hoşça kal.")


if __name__ == "__main__":
    main()
