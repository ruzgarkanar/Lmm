"""Where does the plan die — in the proposal, or in the anchoring?

Three suspects were eliminated by measurement before this one. The gates
kill nothing (`gate_blame.py`: 0 of 22). Retrieval seats the answering
session in 86% of the failures (`seat_recall.py`). And declaring the
shape changes nothing (`derive_route.py`: 0 of 10, and seven of them
still ended on `chain · refuse`, because the `derive` seat falls through
rather than refusing).

That leaves the plan organ itself, and it can fail in two different
places. The engine proposes a plan in a fixed vocabulary; the library
then executes it, and every step must ANCHOR — the phrase it names has
to be found in the store. So either:

  (a) no plan is proposed, or one outside the vocabulary is — the
      reading is the defect;
  (b) a sensible plan is proposed and a step cannot be anchored — the
      store cannot find what the plan asks for, and the repair is in
      retrieval or in the amount/date readers, not in the plan.

This records the proposal and the step that killed it. Ten questions,
about 80 calls.

    export LMM_BACKEND=azure
    python3 benchmarks/plan_trace.py data/longmemeval_s_cleaned.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import longmemeval as lme                               # noqa: E402
import lme_score as score                                            # noqa: E402
from lmm import generate                                # noqa: E402
from lmm.session import Session                         # noqa: E402

DERIV = ("how many days", "how much", "percentage", "how long",
         "how many months", "since")


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "data", "longmemeval_s_cleaned.json")
    if not os.environ.get("LMM_BACKEND"):
        print("LMM_BACKEND is not set — this needs an engine.")
        return 1
    blame = {r["question"]: r for r in
             json.load(open(os.path.join(HERE, "gate_blame.json")))}
    rows = [r for r in lme.slice_of(json.load(open(data, encoding="utf-8")), 30)
            if not blame.get(r["question"], {}).get("right", True)
            and any(w in r["question"].lower() for w in DERIV)]

    seen = {}
    real_plan = generate.plan_of
    real_amounts = generate.amounts_of
    real_events = generate.events_of

    def plan_of(question):
        out = real_plan(question)
        seen.setdefault("plan", []).append(out)
        return out

    def amounts_of(question, block):
        out = real_amounts(question, block)
        seen.setdefault("amounts", []).append(out)
        return out

    def events_of(*a, **k):
        out = real_events(*a, **k)
        seen.setdefault("events", []).append(out)
        return out

    generate.plan_of = plan_of
    generate.amounts_of = amounts_of
    generate.events_of = events_of

    real_answer = Session._plan_answer

    def traced(self, question, want=()):
        said = real_answer(self, question, want=want)
        seen.setdefault("said", []).append(said)
        return said
    Session._plan_answer = traced

    out = []
    try:
        for at, row in enumerate(rows, 1):
            seen.clear()
            m = lme.learn_instance(row)
            said = m.ask(row["question"], explain=True, shape="derive")
            plans = seen.get("plan") or []
            right, _strict = score.accepted(
                row["question"], row.get("answer"), said,
                absent=score.is_absent(row),
                abstained=bool(said.abstained))
            row_out = {
                "right": right,
                "question": row["question"],
                "gold": row.get("answer"),
                "said": str(said),
                "route": list(said.route or ()),
                "plan_proposed": plans,
                "amounts_read": seen.get("amounts"),
                "events_read": seen.get("events"),
                "plan_answer": seen.get("said"),
            }
            out.append(row_out)
            first = plans[0] if plans else None
            print("%2d %-40s plan=%s" % (at, row["question"][:40],
                                         ("none" if not first else
                                          " | ".join("%s=%s:%s" % (s[0], s[1],
                                                     ",".join(s[2:]))[:38]
                                                     for s in first[:3]))))
    finally:
        generate.plan_of = real_plan
        generate.amounts_of = real_amounts
        generate.events_of = real_events
        Session._plan_answer = real_answer

    no_plan = sum(1 for r in out if not r["plan_proposed"]
                  or not r["plan_proposed"][0])
    print("\n%d questions · no plan proposed in %d · a plan proposed and "
          "unanswered in %d" % (len(out), no_plan, len(out) - no_plan))
    json.dump(out, open(os.path.join(HERE, "plan_trace.json"), "w"),
              indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
