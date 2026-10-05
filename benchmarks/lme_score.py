"""One LongMemEval scorer for the harnesses here, because three got it wrong.

NAMED `lme_score`, NOT `score`: `benchmarks/score.py` already exists and is
the field benchmark's own, imported by `evaluate.py` and by two invariants.
Writing this one as `score.py` overwrote it — recovered from git, and the
lesson is the obvious one: a new file in a directory is a name taken, so
read the directory first.

LongMemEval's reference answers are prose — "30 days. 31 days (including
the last day) is also acceptable" — so strict containment asks for all of
it and nothing answers that. The published figures have always used
strict OR an engine judge for that reason (README, "Honest limits"), and
the harnesses written on 5 October 2026 used strict alone. Measured
cost: `gate_blame` read 8 of 30 where the rule gives 11, so a quarter of
its "failures" were answers, and the chain of reasoning built on that set
had to be recomputed.

Three rules, and all three were learned the hard way:

  1. THE STAMP IS NOT THE ANSWER. A reply ends with "(~ chat 2023/…)",
     and strict containment once found the gold "2" inside "2023".
  2. STRICT OR JUDGE. Either accepts the reply.
  3. AN ABSENT QUESTION IS ANSWERED BY SILENCE. Its id ends `_abs`, its
     reference answer says the user never mentioned it, and the correct
     reply is an abstention — scoring its text is meaningless.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import longmemeval as lme                               # noqa: E402


def is_absent(row):
    """Is the right answer silence? (the benchmark's own marking)"""
    return str(row.get("question_id") or "").endswith("_abs")


def claimed(said):
    """The reply without its provenance mark."""
    return str(said or "").split("(~", 1)[0]


def strict_ok(gold, said, absent=False, abstained=False):
    if absent:
        return bool(abstained)
    gold = str(gold or "")
    return bool(gold) and lme._plain(gold) in lme._plain(claimed(said))


def accepted(question, gold, said, absent=False, abstained=False,
             judge=True):
    """Correct by the rule this project publishes. One call when the
    strict reading fails and a judge is allowed; none otherwise."""
    if absent:
        return bool(abstained), bool(abstained)
    strict = strict_ok(gold, said)
    if strict or not judge:
        return strict, strict
    return bool(lme.judged(question, gold, claimed(said))), strict
