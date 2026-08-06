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

Fixed is the word to be careful with. This exam is twelve hand-written lesson
sentences and thirteen hand-written questions, all about four animals, and it
never changes. That makes it a good regression detector and a *bad* measure of
ability: anything tuned while watching this number is studying the exam paper.
The measure of ability is `scripts/olcut.py`, which builds its questions from
the graph itself, so nobody picks what gets asked.

So the exam lives in `packs/olcum/ders.txt` rather than here. Two reasons, and
the second is the one that mattered. First, a fixed exam sitting in the engine's
own source quietly becomes a target while you are editing next to it. Second,
this file is otherwise language-free: an exam is written in a language, an
engine is not, and the twenty-five Turkish sentences that used to be here were
the only thing tying the scorecard to Turkish.

Usage: python3 -m lmm.evaluate
"""
import os
import sys
import tempfile

from lmm.cli import Session

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAM = os.path.join(HERE, "packs", "olcum", "ders.txt")
ARROW = "->"        # soru ile beklenen cevabı ayıran işaret


def _sections(path):
    """`[başlık]` ile bölünmüş düz metin -> {başlık: [satır]}.

    Yorum satırı `#` ile başlar. Ders satırlarının içinde `#` geçebiliyor
    (`kelime: ...` bildirimleri geçmiyor ama bir gün geçebilir), o yüzden
    yalnızca satırın BAŞINDAKİ `#` yorum sayılıyor.
    """
    found, current = {}, None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1]
                found[current] = []
            elif current is not None:
                found[current].append(line)
    return found


def _pairs(lines):
    return [tuple(part.strip() for part in line.split(ARROW, 1))
            for line in lines]


_EXAM = _sections(EXAM)

# The lesson every category is scored against, and the questions themselves.
# `_pairs` gives (soru, beklenen); the plain lists are questions with no
# expected answer, because declining has no single wording to match.
LESSON = _EXAM["ders"]
COVERAGE = _pairs(_EXAM["kapsam"])          # söylenen, geri söylenebilmeli
DERIVED = _pairs(_EXAM["çıkarım"])          # söylenmeyen, kalıtımla ulaşılan
UNKNOWN = _EXAM["bilinmeyen"]               # bellek sessiz, tek cevap çekimserlik
CORRECTION = _EXAM["düzeltme-ders"]
CORRECTED = _pairs(_EXAM["düzeltme-sınav"])
DESCRIBE = _EXAM["sentez-soru"][0]          # "{kavram} anlat"
DESCRIBED = _EXAM["sentez-kavram"]
INVENTED = tuple(_EXAM["sentez-uydurma"])
GROUNDED = tuple(_EXAM["temellendirme"])

# Tek kaynak: aynı imza iki yerde kopyalanmıştı ve biri eksik kalmıştı.
# Söyleyiş öğrenen organ bu imzaya güveniyor — eksik bir madde, yanlış bir
# kalıbın kalıcı olarak yazılması demek.
from lmm.serialize import is_refusal


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
                 if is_refusal(session.respond(question)))
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
    for line in CORRECTION:
        session.respond(line)
    checks = [session.respond(question).startswith(expected)
              for question, expected in CORRECTED]
    session.save()
    later = Session(path)
    # Yeniden açılışta ilk soru bir kez daha: düzeltmenin yaşayıp yaşamadığını
    # ölçen tek şey bu. İkincisi zaten dokunulmamış olanı bekliyor.
    question, expected = CORRECTED[0]
    checks.append(later.respond(question).startswith(expected))
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
    checks = []
    for concept in DESCRIBED:
        paragraph = session.respond(DESCRIBE.format(kavram=concept)).lower()
        checks.append(not any(word in paragraph for word in INVENTED))
    return Result("sentez", sum(checks), len(checks), "paragrafta uydurma yok")


# Aynı soru kaç kez sorulacak. Cevap yolunda rastgelelik yok, dolayısıyla tek
# tekrar farkı görmeye yeter; beş olmasının sebebi rastgelelik değil **çağrıyla
# değişen durum**: bir önbellek, dönüşümlü bir söyleyiş ya da bir sayaç ilk iki
# çağrıda aynı görünüp üçüncüde değişebilir. Ölçüldü: bugünkü sistemde 5'te de
# 50'de de sonuç aynı — yani sayı bir güvenlik payı, bir bulgu değil.
REPEATS = 5


def measurement(session):
    """The same question must give the same answer, every time."""
    checks = []
    for question, _ in COVERAGE + DERIVED:
        answers = {session.respond(question) for _ in range(REPEATS)}
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
