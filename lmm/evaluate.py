"""A scorecard for the things LMM claims, measured rather than asserted.

The eight headings are borrowed from Eywa's error taxonomy for long-term agent
memory (coverage, grounding, revision, scope, temporal, retrieval, synthesis,
measurement), because a system that keeps adding mechanisms needs a fixed place
to see whether each one still earns its keep. Abstention is a category of its
own here, as it has become in the 2026 agent-memory benchmarks — the difference
is that we are not scoring how often the system chooses to decline, but checking
that declining is the only thing it *can* do when memory is silent.

Everything runs offline, from one fixed script, and is deterministic. A
benchmark that needs a network or a random seed cannot be used to catch a
regression.

Usage: python3 -m lmm.evaluate
"""
import os
import sys
import tempfile

from lmm.cli import Session

LESSON = [
    # The word has to be taught before the lesson that uses it. The first
    # version of this script skipped that and the system said so — twice —
    # while the benchmark counted it as a failure of inheritance.
    "kelime: büyümek = büyür / büyümez",
    "kuşlar uçar",
    "kuşlar tüylüdür",
    "kuş bir hayvandır",
    "hayvanlar büyür",
    "serçe bir kuştur",
    "kartal bir kuştur",
    "penguen bir kuştur",
    "penguen uçamaz",
    "evet",                     # the exception is confirmed
    "balık bir hayvandır",
    "balıklar yüzer",
]

# What it was told outright, and must therefore be able to say back.
COVERAGE = [("serçe nedir", "kuş"), ("balık yüzer mi", "evet"),
            ("penguen uçar mı", "hayır"), ("kuş nedir", "hayvan")]

# What nobody stated, and only the hierarchy can reach.
DERIVED = [("serçe uçar mı", "evet"), ("kartal tüylü mü", "evet"),
           ("serçe büyür mü", "evet"), ("balık büyür mü", "evet")]

# What memory is silent about, where declining is the only honest answer.
UNKNOWN = ["zürafa nedir", "zürafa uçar mı", "ejderha nedir",
           "kaplan yüzer mi", "bulut nedir"]

DECLINED = ("bilmiyorum", "anlamadım", "öğrenmedim", "duymadım")
GROUNDED = ("kaynak:", "çıkarımım", "dil modelinden", "anlaşmıyor", "çünkü")


class Result:
    def __init__(self, name, passed, total, note=""):
        self.name = name
        self.passed = passed
        self.total = total
        self.note = note

    @property
    def share(self):
        return self.passed / self.total if self.total else 0.0


def _taught(path):
    session = Session(path)
    for line in LESSON:
        session.respond(line)
    session.save()
    return session


def coverage(session):
    """Can it say back what it was told?"""
    passed = sum(1 for question, expected in COVERAGE
                 if expected in session.respond(question))
    return Result("kapsam", passed, len(COVERAGE), "doğrudan öğretilen bilgi")


def retrieval(session):
    """Does the hierarchy reach what nobody stated?"""
    passed = sum(1 for question, expected in DERIVED
                 if session.respond(question).startswith(expected))
    return Result("çıkarım", passed, len(DERIVED), "yalnızca kalıtımla ulaşılan")


def scope(session):
    """Silence in memory must produce a refusal, never a guess."""
    passed = sum(1 for question in UNKNOWN
                 if any(mark in session.respond(question) for mark in DECLINED))
    return Result("sınır", passed, len(UNKNOWN), "bilinmeyende çekimserlik")


def grounding(session):
    """Every answer about a fact must carry where it came from."""
    questions = [q for q, _ in COVERAGE + DERIVED]
    passed = sum(1 for question in questions
                 if any(mark in session.respond(question) for mark in GROUNDED))
    return Result("temellendirme", passed, len(questions), "kaynak veya gerekçe")


def revision(path):
    """A correction must take effect, and survive a restart."""
    session = _taught(path)
    checks = []
    session.respond("serçe uçamaz")
    session.respond("evet")
    checks.append(session.respond("serçe uçar mı").startswith("hayır"))
    checks.append(session.respond("kartal uçar mı").startswith("evet"))
    session.save()
    later = Session(path)
    checks.append(later.respond("serçe uçar mı").startswith("hayır"))
    return Result("revizyon", sum(checks), len(checks), "düzeltme kalıcı mı")


def temporal(directory):
    """The order lessons arrive in must not change what is believed."""
    forward = _taught(os.path.join(directory, "ileri.lmm"))
    backward = Session(os.path.join(directory, "geri.lmm"))
    for line in reversed(LESSON):
        backward.respond(line)
    checks = [forward.respond(question) == backward.respond(question)
              for question, _ in COVERAGE]
    return Result("sıra", sum(checks), len(checks), "öğretme sırasından bağımsız")


def synthesis(session):
    """A composed paragraph must contain nothing memory does not hold."""
    invented = ("koşar", "yüzemez", "siyah", "büyüktür")
    checks = []
    for concept in ("penguen", "serçe", "balık"):
        paragraph = session.respond(f"{concept} anlat").lower()
        checks.append(not any(word in paragraph for word in invented))
    return Result("sentez", sum(checks), len(checks), "paragrafta uydurma yok")


def measurement(session):
    """The same question must give the same answer, every time."""
    checks = []
    for question, _ in COVERAGE + DERIVED:
        answers = {session.respond(question) for _ in range(5)}
        checks.append(len(answers) == 1)
    return Result("ölçüm", sum(checks), len(checks), "aynı soru, aynı cevap")


def run():
    directory = tempfile.mkdtemp()
    session = _taught(os.path.join(directory, "olcum.lmm"))
    return [coverage(session), retrieval(session), scope(session),
            grounding(session), revision(os.path.join(directory, "revizyon.lmm")),
            temporal(directory), synthesis(session), measurement(session)]


def main():
    results = run()
    width = max(len(r.name) for r in results)
    print("LMM DEĞERLENDİRME")
    for result in results:
        mark = "✓" if result.passed == result.total else "✗"
        print(f"  {mark} {result.name:<{width}}  {result.passed}/{result.total}"
              f"  %{result.share * 100:3.0f}   {result.note}")
    total = sum(r.passed for r in results)
    possible = sum(r.total for r in results)
    print(f"\n  TOPLAM {total}/{possible} — %{total / possible * 100:.0f}"
          f"   (ağ yok, rastgelelik yok)")
    return 0 if total == possible else 1


if __name__ == "__main__":
    sys.exit(main())
