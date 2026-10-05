"""Does widening seat the answering line? — one call a question, then free.

The arm measurement says WIDEN rescued the paraphrase ("what is the
shortest password a user may choose", which the document answers under
"memorized secret") and broke one question that worked. Both are single
observations inside measured noise, and three runs of two arms would
cost about 500 calls to settle.

Most of that is unnecessary. Whether widening HELPS is a retrieval
question — does the answering line reach the seats — and retrieval needs
no engine. The only engine call is the proposal itself,
`generate.phrasings`, one per question, and it is deterministic at
temperature 0. So: ask once, keep the proposal, and measure seating
before and after for nothing.

    export LMM_BACKEND=azure
    python3 benchmarks/widen_recall.py

Thirteen calls, one a question.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from lmm import Memory, evidence, generate            # noqa: E402

DOC = os.path.join(HERE, "field", "documents", "nist_sp800-63b.pdf")
QS = os.path.join(HERE, "field", "questions_nist.json")
SEATS = 6


def seated(store, question, gold):
    """Does a seated line carry every gold token?"""
    return any(all(g in line.lower() for g in gold)
               for line in store.find(question, most=SEATS))


def main():
    if not os.environ.get("LMM_BACKEND"):
        print("LMM_BACKEND is not set — the proposals need an engine.")
        return 1
    m = Memory(None)
    m.learn(DOC, deep=False)
    store = m.session.evidence
    print("lines: %d · one call a question\n" % len(store.sentences))

    rows, calls = [], 0
    for q in json.load(open(QS, encoding="utf-8")):
        gold = [str(g).lower() for g in (q.get("altin") or [])]
        if not gold:
            continue                        # a trap has no line to seat
        question = q["soru"]
        before = seated(store, question, gold)
        # THE SAME CALL `find_again` MAKES, lines and all — an earlier
        # version of this file asked `phrasings(question)` with nothing
        # else and so measured a reading the library no longer does.
        proposed = generate.phrasings(question, lines=store.find(question,
                                                                most=SEATS))
        calls += 1
        # the same rule `find_again` keeps: a word nobody wrote here
        # cannot enter the search
        known = []
        for phrase in proposed:
            for w in evidence._words(phrase, known=store.units):
                if w in store.index and w not in known:
                    known.append(w)
        after = (seated(store, "%s %s" % (question, " ".join(known)), gold)
                 if known else before)
        rows.append((question, before, after, known[:6]))
        mark = {(False, True): "RESCUED", (True, False): "BROKEN ",
                (True, True): "kept   ", (False, False): "missed "}[(before, after)]
        print("%s %-46s +%s" % (mark, question[:46], ", ".join(known[:5]) or "-"))

    kept = sum(1 for _q, b, a, _k in rows if b and a)
    rescued = sum(1 for _q, b, a, _k in rows if not b and a)
    broken = sum(1 for _q, b, a, _k in rows if b and not a)
    print("\nbefore %d/%d · after %d/%d · rescued %d · broken %d · %d calls"
          % (sum(1 for _q, b, _a, _k in rows if b), len(rows),
             kept + rescued, len(rows), rescued, broken, calls))
    json.dump([{"soru": q, "before": b, "after": a, "added": k}
               for q, b, a, k in rows],
              open(os.path.join(HERE, "widen_recall_result.json"), "w"),
              indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
