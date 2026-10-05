"""Does the planner propose a plan when it can SEE the store?

`generate.plan_of` is given the QUESTION and nothing else. It must decide
how to compute an answer without knowing whether the store holds the
figures the plan would need, and measured on 5 October 2026 it declined
on five of ten derivation questions — including "what percentage of
leadership positions do women hold", which is almost word for word one of
its own prompt's examples. The engine was asked and returned nothing.

It is the second instance of one pattern found the same day: the other
is `generate.phrasings`, which is asked how else a question might be
WRITTEN and is never shown the document, so it proposes only words the
question already used.

So this asks the same planner twice — once as the library asks it, once
with the lines retrieval already seated in front of it — and reports
whether a plan appears. Nothing in the package changes; the prompt is
rebuilt here from the one `plan_of` uses.

    export LMM_BACKEND=azure
    python3 benchmarks/plan_sees_store.py data/longmemeval_s_cleaned.json

Ten questions, two calls each: about 20 calls.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import longmemeval as lme                               # noqa: E402
from lmm import generate, runtime                       # noqa: E402

DERIV = ("how many days", "how much", "percentage", "how long",
         "how many months", "since")


def system_of():
    """The planner's own instructions, lifted from `generate.plan_of`."""
    src = open(os.path.join(ROOT, "src", "lmm", "generate.py"),
               encoding="utf-8").read()
    body = src[src.index("def plan_of("):]
    body = body[:body.index("raw = runtime.generate(")]
    scope = {"PLAN_OPS": getattr(generate, "PLAN_OPS", "")}
    exec(body[body.index("system = ("):], scope)          # noqa: S102
    return scope["system"]


def parse(raw):
    steps = []
    for line in (raw or "").splitlines():
        if "=" not in line:
            continue
        name, _eq, rest = line.partition("=")
        op, _colon, args = rest.partition(":")
        parts = [a.strip() for a in args.split(",") if a.strip()]
        steps.append(tuple([name.strip(), op.strip()] + parts))
    return steps[:6]


def ask(system, prompt):
    raw = (runtime.generate(prompt, system=system, max_tokens=120,
                            temperature=0.0) or "").strip()
    if not raw or raw.upper().startswith("NONE"):
        return []
    return parse(raw)


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
    system = system_of()
    blind = seeing = 0
    out = []
    for at, row in enumerate(rows, 1):
        m = lme.learn_instance(row)
        lines = m.session.evidence.find(row["question"], most=6)
        block = "\n".join(lines)
        a = ask(system, "Q: %s" % row["question"])
        b = ask(system + "\nThe store holds these lines; plan only over "
                         "what they could supply.",
                "EVIDENCE:\n%s\n\nQ: %s" % (block, row["question"]))
        blind += bool(a)
        seeing += bool(b)
        out.append({"question": row["question"], "blind": a, "seeing": b})
        print("%2d blind=%-5s seeing=%-5s %s"
              % (at, "plan" if a else "none", "plan" if b else "none",
                 row["question"][:46]))
    print("\n%d questions · a plan proposed blind in %d · seeing the store "
          "in %d" % (len(out), blind, seeing))
    json.dump(out, open(os.path.join(HERE, "plan_sees_store.json"), "w"),
              indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
