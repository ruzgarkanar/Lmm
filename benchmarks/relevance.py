"""Is the relation check any good? — the first number this project has
for answer relevance.

`generate.answers_asked` is this library's relevance gate: does the
evidence state the relation THE QUESTION ASKS, rather than merely
everything the answer happens to assert. It was written against a
measured failure — asked who SIGNED a document, the answer gave the
PREPARER field, with coverage 1.0 and the read-back saying yes — and in
the field it has let one through that mattered: a requirement asking for
a simulation feature in a product, answered with the document's account
of unit testing during the project.

The literature calls this axis ANSWER RELEVANCE and treats it as a
metric beside faithfulness, on the grounds that a perfectly faithful
answer to the wrong question is still useless. This library has used it
as a gate and never scored it.

THE NEGATIVES ARE BUILT, NOT WRITTEN. A hard negative here is not a
fabrication — the gate above this one already stops those. It is a
sentence that is TRUE, drawn from this very document, properly sourced,
and about something else. So each question is paired with the answer to
a DIFFERENT question of the same set: faithful by construction,
irrelevant by construction, and labelled without anybody judging
anything.

    export LMM_BACKEND=azure
    python3 benchmarks/relevance.py

Two calls a question, about 26 for the NIST set. Reports the two error
rates separately, because they are not interchangeable: letting an
irrelevant answer through is a wrong answer shipped, and rejecting a
relevant one is an answer lost.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from lmm import Memory, evidence, generate              # noqa: E402

DOC = os.path.join(HERE, "field", "documents", "nist_sp800-63b.pdf")
QS = os.path.join(HERE, "field", "questions_nist.json")


def main():
    if not os.environ.get("LMM_BACKEND"):
        print("LMM_BACKEND is not set — the relation check needs an engine.")
        return 1
    m = Memory(None)
    m.learn(DOC, deep=False)
    store = m.session.evidence
    rows = [q for q in json.load(open(QS, encoding="utf-8")) if q.get("altin")]

    # each question's own answering line, read off the store by its gold
    answer = {}
    for q in rows:
        gold = [str(g).lower() for g in q["altin"]]
        for line in store.find(q["soru"], most=30):
            if all(g in line.lower() for g in gold):
                answer[q["soru"]] = line[:300]
                break
    asked = [q for q in rows if q["soru"] in answer]
    print("%d questions · %d with an answering line in the store\n"
          % (len(rows), len(asked)))

    kept = dropped = let_in = refused = 0
    out = []
    for at, q in enumerate(asked):
        own = answer[q["soru"]]
        # THE NEGATIVE IS THE FURTHEST ANSWER, NOT THE NEXT ONE. Taking
        # the neighbour paired "how many characters shall a memorized
        # secret be" with the answer to "what length of memorized secret
        # should verifiers permit" — two questions about one thing, so
        # the "irrelevant" answer was relevant and the gate was marked
        # wrong for being right. The negative is now the answering line
        # that shares the fewest words with this question.
        mine = set(evidence._words(q["soru"]))
        other = min(
            (answer[o["soru"]] for o in asked if o["soru"] != q["soru"]),
            key=lambda line: len(mine & set(evidence._words(line))))
        block = "\n".join(store.find(q["soru"], most=6) + [own, other])
        try:
            yes_own = generate.answers_asked(q["soru"], own, block)
            yes_other = generate.answers_asked(q["soru"], other, block)
        except Exception as broke:                      # noqa: BLE001
            print("engine: %r" % (broke,))
            return 1
        kept += yes_own
        refused += not yes_own
        let_in += yes_other
        dropped += not yes_other
        out.append({"soru": q["soru"], "own": own, "other": other,
                    "kept_own": bool(yes_own), "let_other_in": bool(yes_other)})
        print("%2d %-46s own=%-3s other=%-3s"
              % (at + 1, q["soru"][:46], "yes" if yes_own else "NO",
                 "YES" if yes_other else "no"))
    n = len(out)
    print("\nrelevant answers kept      %2d / %d   (%.0f%%)"
          % (kept, n, 100.0 * kept / max(1, n)))
    print("irrelevant answers stopped %2d / %d   (%.0f%%)"
          % (dropped, n, 100.0 * dropped / max(1, n)))
    print("\n  an answer LOST       %d" % refused)
    print("  a wrong answer SHIPPED %d" % let_in)
    json.dump(out, open(os.path.join(HERE, "relevance_result.json"), "w"),
              indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
