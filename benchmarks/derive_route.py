"""Is it the organ that fails, or the router that never calls it?

Nine derivation questions failed in `gate_blame.json`. Six of them never
reached the plan organ at all — route `chain · refuse` — although the
operations they need have shipped since 0.13.0: "how much did I spend on
EACH mug" is `per`, "what PERCENTAGE of positions" is `ratio`, "how much
MORE than my goal" is `minus`. The vocabulary exists and the question
does not arrive.

So the shape is declared instead. `ask(shape=...)` skips the classifier
and routes the turn directly (it has been a public argument since
0.13.0, and `gate_arms.py` measured what declaring it saves). If the
answers appear, the organ was always able and the router was the defect
— a far smaller thing to repair than the empty graph this was first
blamed on.

    export LMM_BACKEND=azure
    python3 benchmarks/derive_route.py data/longmemeval_s_cleaned.json

Nine questions, one arm: about 70 calls. The baseline is not re-run —
every one of these is already recorded as failed.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import meter                                            # noqa: E402
import longmemeval as lme                               # noqa: E402
import lme_score as score                                            # noqa: E402

# the shapes these questions are, read off what they ask for — a span
# between two dates, or a figure derived from others
SPAN = ("how many days", "how long", "how many months")


def shape_for(question):
    low = question.lower()
    return "span" if any(low.startswith(s) for s in SPAN) else "derive"


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "data", "longmemeval_s_cleaned.json")
    if not os.environ.get("LMM_BACKEND"):
        print("LMM_BACKEND is not set — this needs an engine.")
        return 1
    blame = {r["question"]: r for r in
             json.load(open(os.path.join(HERE, "gate_blame.json")))}
    rows = lme.slice_of(json.load(open(data, encoding="utf-8")), 30)
    wanted = [r for r in rows
              if not blame.get(r["question"], {}).get("right", True)
              and any(w in r["question"].lower() for w in
                      ("how many days", "how much", "percentage",
                       "how long", "how many months", "since"))]
    print("%d derivation questions, all recorded as failed\n" % len(wanted))
    meter.install()
    out, won = [], 0
    for at, row in enumerate(wanted, 1):
        shape = shape_for(row["question"])
        before = meter.METER.totals()["calls"]
        m = lme.learn_instance(row)
        said = m.ask(row["question"], explain=True, shape=shape)
        right, _strict = score.accepted(
            row["question"], row.get("answer"), said,
            absent=score.is_absent(row), abstained=bool(said.abstained))
        won += right
        calls = meter.METER.totals()["calls"] - before
        out.append({"question": row["question"], "shape": shape,
                    "gold": row.get("answer"), "said": str(said),
                    "right": right, "route": list(said.route or ()),
                    "calls": calls,
                    "was": blame[row["question"]]["route"]})
        print("%2d %-7s %-7s %-44s %s"
              % (at, shape, "OK" if right else "--",
                 row["question"][:44], " · ".join(said.route or ())[:22]))
    print("\n%d of %d answered once the shape was declared · %d calls"
          % (won, len(out), sum(r["calls"] for r in out)))
    json.dump(out, open(os.path.join(HERE, "derive_route.json"), "w"),
              indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
