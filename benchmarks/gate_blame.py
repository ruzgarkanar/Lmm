"""Did the GATE kill the answer, or was it never found? — one run.

Over the runs kept in `benchmarks/longmemeval_runs`, 233 of 360 turns
are wrong or silent, 185 of them silent, and 178 end on the route
`chain · refuse`. That route has two readings and the saved files cannot
tell them apart:

  (a) candidates were written and the gate rejected every one — in which
      case the checking is too strict and the answer existed;
  (b) no candidate ever carried the answer, because the evidence did not
      — in which case the gate is innocent and the work is retrieval.

Loosening the gate on a guess would put the one claim this project can
defend — no wrong fact asserted, on any document run — at risk to fix a
problem it may not have. So the rejected candidates are read instead.
`Session._select` already keeps them (`tried`); this wraps it and asks,
of every refused turn, whether the gold answer was in something the gate
threw away.

NOTHING IN THE PACKAGE CHANGES. The measurement lives here.

    export LMM_BACKEND=azure
    python3 benchmarks/gate_blame.py longmemeval_s.json --most 30

The data is not in this repository; it comes from the benchmark authors'
own release. About 250 calls for thirty questions.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import longmemeval as lme                               # noqa: E402
from lmm.session import Session                         # noqa: E402


import lme_score as score                                         # noqa: E402


def _carries(text, gold):
    """Did this candidate hold the answer? Used for CANDIDATES, where no
    judge is available — a rejected candidate is scored strictly on
    purpose, since the question is whether the gate threw away something
    that contained the answer."""
    low = (text or "").lower()
    return all(part.lower() in low for part in gold) if gold else False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--most", type=int, default=30)
    ap.add_argument("--save", default=os.path.join(HERE, "gate_blame.json"))
    args = ap.parse_args()
    if not os.environ.get("LMM_BACKEND"):
        print("LMM_BACKEND is not set — this needs an engine.")
        return 1

    # THE WRAPPER. `_select` returns (chosen, tried); the second is every
    # candidate the gate scored and did not speak.
    real_select = Session._select
    seen = {"tried": []}

    def watched(self, question, records, proof, fact_block, block):
        chosen, tried = real_select(self, question, records, proof,
                                    fact_block, block)
        seen["tried"] = list(tried or ())
        return chosen, tried
    Session._select = watched

    rows = json.load(open(args.data, encoding="utf-8"))
    rows = lme.slice_of(rows, args.most)
    out, killed, starved = [], 0, 0
    try:
        for at, row in enumerate(rows, 1):
            m = lme.learn_instance(row)
            gold = [str(row.get("answer") or "")]
            seen["tried"] = []
            said = m.ask(row["question"], explain=True, quoted=True)
            absent = score.is_absent(row)
            right, _strict = score.accepted(
                row["question"], row.get("answer"), said,
                absent=absent, abstained=bool(said.abstained))
            refused = bool(said.abstained) or not right
            rejected = [t for t in seen["tried"] if not _carries(str(said), [t])]
            had = [t for t in rejected if _carries(t, gold)]
            if refused and had:
                killed += 1
            elif refused:
                starved += 1
            out.append({
                "question_id": row.get("question_id"),
                "absent": absent,
                "question": row["question"],
                "gold": row.get("answer"),
                "said": str(said),
                "right": right,
                "abstained": bool(said.abstained),
                "route": list(said.route or ()),
                "rejected": rejected,
                "gold_was_in_a_rejected_candidate": bool(had),
            })
            print("%3d/%d %-7s %s" % (at, len(rows),
                                      "OK" if right else
                                      ("GATE" if had else "starved"),
                                      row["question"][:54]))
    finally:
        Session._select = real_select

    bad = killed + starved
    print("\n%d questions · %d not answered" % (len(out), bad))
    if bad:
        print("  the gate threw away the answer : %d  (%.0f%%)"
              % (killed, 100.0 * killed / bad))
        print("  no candidate ever had it       : %d  (%.0f%%)"
              % (starved, 100.0 * starved / bad))
    json.dump(out, open(args.save, "w"), indent=1, ensure_ascii=False)
    print("written: %s" % args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
