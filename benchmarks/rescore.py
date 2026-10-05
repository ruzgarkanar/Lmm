"""Re-score a saved run the way the project scores — strict OR judge.

MY OWN SCORER WAS THE ONE THE PROJECT ALREADY FIXED ONCE. LongMemEval's
reference answers are prose — "30 days. 31 days (including the last day)
is also acceptable" — and strict containment asks for all of it. Nothing
answers that, so strict alone marks almost every derivation question
wrong. The published figures use strict OR an engine judge for exactly
this reason (README, "Honest limits"), and the harnesses written today
used strict alone.

What it cost: `plan_trace.json` records the turn that answered "How much
more money did I raise than my initial goal" with

    50 — charity cycling event (250), initially aimed to raise $200 ...

against a gold of "$50". The arithmetic is right, the answer is right,
and it was counted as a failure because the reply does not contain a
dollar sign.

This re-reads a saved file and asks the judge, one call a row. Nothing
is re-run.

    export LMM_BACKEND=azure
    python3 benchmarks/rescore.py benchmarks/gate_blame.json
"""
import json
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import lme_score                                        # noqa: E402
import longmemeval as lme                               # noqa: E402


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(HERE, "gate_blame.json")
    if not os.environ.get("LMM_BACKEND"):
        print("LMM_BACKEND is not set — the judge needs an engine.")
        return 1
    rows = json.load(open(path, encoding="utf-8"))
    was = sum(1 for r in rows if r.get("right"))
    now = 0
    for at, r in enumerate(rows, 1):
        gold = str(r.get("gold") or "")
        # the stamp is not part of the answer (the harness's own rule)
        claimed = str(r.get("said") or "").split("(~", 1)[0]
        absent = str(r.get("question_id") or "").endswith("_abs")
        verdict, strict = lme_score.accepted(
            r["question"], gold, r.get("said"), absent=absent,
            abstained=bool(r.get("abstained")))
        r["right_strict"] = bool(strict)
        r["right"] = verdict
        now += verdict
        if verdict and not strict:
            print("%2d JUDGE  %-44s" % (at, r["question"][:44]))
            print("        gold: %s" % gold[:70])
            print("        said: %s" % claimed[:70])
    json.dump(rows, open(path, "w"), indent=1, ensure_ascii=False)
    print("\n%s · strict only: %d/%d · strict OR judge: %d/%d"
          % (os.path.basename(path), was, len(rows), now, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
