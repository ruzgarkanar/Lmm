"""LMM v0 chat interface. Usage: python3 -m lmm.cli [memory_file]"""
import sys

from lmm.memory import Memory
from lmm.reasoning import Reasoning
from lmm.gate import EpistemicGate
from lmm.intuition import (Intuition, Intent, TEACH, ASK, ASK_WHO, UNKNOWN,
                           UNKNOWN_WORD, PRONOUNS, lower)
from lmm.distill import split_words
from lmm.grammar import Pattern
from lmm.learning import LearningLoop, CONFLICT, LEARNED, CORRECTED
from lmm.induction import Induction
from lmm.network import MiniNetwork
from lmm.curiosity import Curiosity
from lmm.pursuit import Pursuit
from lmm.phrasing import (wondering, not_understood, now_i_can, generalising,
                          describe, teach_me_the_word, learned_word)

# A custom LanguageOrgan may report graded confidence; ours parses or does not.
CONFIDENCE_THRESHOLD = 0.35
RESEMBLANCE = 0.5       # below this, guessing at what you meant is noise
AFFIRMATIVE = ("evet", "e")
EXIT_WORDS = ("çık", "cik", "exit")
SUBJECTLESS = (ASK_WHO, UNKNOWN, UNKNOWN_WORD)   # kinds needing no subject


class Session:
    """One conversation: the five organs wired together over a memory file."""

    def __init__(self, path, language=None):
        self.path = path
        self.memory = Memory.load(path)
        reasoning = Reasoning(self.memory)
        self.gate = EpistemicGate(self.memory, reasoning)
        # any LanguageOrgan fits here; the other organs never see a sentence
        self.language = language or Intuition(network=MiniNetwork.default())
        for entry in self.memory.patterns:      # ways of speaking it was taught
            self.language.grammar.add(Pattern.from_dict(entry), first=True)
        self.learning = LearningLoop(self.memory, reasoning)
        self.curiosity = Curiosity(self.memory, reasoning)
        self.pursuit = Pursuit(self.memory, reasoning)
        self.induction = Induction(self.memory, reasoning)
        self.pending = None   # an edge awaiting the teacher's confirmation
        self.goal = None      # a question it is still trying to earn the answer to
        self.goal_steps = set()
        self.focus = None     # what "o" and a bare question refer to

    def respond(self, line):
        if self.pending is not None:
            return self._resolve_pending(line)
        words, rest = split_words(line)
        if words:                       # "kelime: uçmak = uçar / uçamaz"
            for infinitive, positive, negative in words:
                self.memory.learn_word(infinitive, positive, negative)
            infinitive, positive, negative = words[-1]
            return learned_word(infinitive, positive, negative)
        intent = self._in_context(self.language.understand(line))
        if intent.kind == UNKNOWN_WORD:
            return teach_me_the_word(intent.target)
        if intent.kind == UNKNOWN or intent.confidence < CONFIDENCE_THRESHOLD:
            resembles = intent.resembles if intent.resemblance >= RESEMBLANCE else None
            return not_understood(resembles)
        if intent.kind != TEACH:
            return self._question(intent)
        status, message, edge = self.learning.teach(intent)
        if status == CONFLICT:
            self.pending = edge
            return message
        if status in (LEARNED, CORRECTED):
            return f"{message} {self._after_learning()}".strip()
        return message

    def _in_context(self, intent):
        """Fill in what the sentence left out, and remember what it was about.

        Nothing here is a context window: knowledge is already permanent. This
        only tracks the thread of the conversation, so "peki yüzer mi" means
        what a person would take it to mean.
        """
        if intent.concept is None or intent.concept in PRONOUNS:
            intent.concept = self.focus
        elif intent.concept:
            self.focus = intent.concept
        if intent.concept is None and intent.kind not in SUBJECTLESS:
            # Nothing said, nothing to carry on from: better to admit it than
            # to answer about None.
            return Intent(UNKNOWN, confidence=0.0)
        return intent

    def _question(self, intent):
        if intent.kind == ASK and not self.pursuit.resolved(intent):
            self.goal = intent          # hold it; a plan beats a shrug
            self.goal_steps = set()
            opening = self.pursuit.opening(intent)
            step = self.pursuit.next_step(intent)
            if step is not None:
                self.goal_steps.add(step.key)
            return opening
        self.goal = None
        return self.gate.answer(intent)

    def _after_learning(self):
        """A goal in hand outranks idle wondering."""
        if self.goal is not None:
            if self.pursuit.resolved(self.goal):
                answer = self.gate.answer(self.goal)
                self.goal = None
                return now_i_can(answer)
            step = self.pursuit.next_step(self.goal)
            if step is not None and step.key not in self.goal_steps:
                self.goal_steps.add(step.key)
                return step.text
            return ""
        hypothesis = self.induction.propose()
        if hypothesis is not None:
            self.induction.learn(hypothesis)
            return generalising(hypothesis.examples,
                                describe(hypothesis.concept, hypothesis.relation,
                                         hypothesis.target))
        question = self.curiosity.next_question()
        return wondering(question.text) if question is not None else ""

    def _resolve_pending(self, line):
        edge, self.pending = self.pending, None
        if lower(line) in AFFIRMATIVE:
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
        if lower(line) in EXIT_WORDS:
            break
        print(session.respond(line))
        session.save()
    session.save()
    print("bellek kaydedildi. hoşça kal.")


if __name__ == "__main__":
    main()
