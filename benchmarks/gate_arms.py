"""Three gates, measured against the same questions — calls and answers.

Two switches in `session.py` ship OFF, each for a reason recorded beside
it, and one of those reasons does not survive reading:

  WIDEN   the widened second retrieval (`evidence.find_again`), which
          asks the engine how else the thing might be WRITTEN and keeps
          only words the store actually holds. Its default moved on
          three corpora, one of them an integrator's 44-cell requirement
          catalogue whose own RFP says the catalogue was DERIVED from the
          proposals it is matched against — so requirement and proposal
          share vocabulary and a tool for the vocabulary gap was measured
          where there is no vocabulary gap.

  ENOUGH  the sufficient-context verdict (`generate.enough`), asked once
          before any candidate is written. Over the recorded runs in
          `benchmarks/longmemeval_runs`, turns whose route contains
          `refuse` take 80% of every engine call at about ten each; this
          asks one question instead.

NIST SP 800-63B asks the same fact twice, once in the document's words
and once in a reader's — "memorized secret" against "password", "acting
NIST director" against "who currently heads NIST". Measured engine-free
on 4 October 2026, retrieval seats the answering line for 8 of 11
answerable questions, and the three it misses are those paraphrases and
the metadata question.

    export LMM_BACKEND=azure        # plus endpoint / key / deployment
    python3 benchmarks/gate_arms.py              # all three arms
    python3 benchmarks/gate_arms.py base enough  # or name them

EACH ARM RUNS IN A FRESH PROCESS, and the first version of this file did
not — which is why its first result has to be thrown away. The engine
pools identical deterministic calls for the life of a process
(`runtime._asked_before`), so a second arm in the same process reads the
first arm's answers back out of the pool: measured, it reported 12 calls
against 81 and an identical 9 of 13, because the only calls it actually
made were the new verdicts. Appendix C has said "each time in a fresh
process" for as long as it has existed. Results already written for an
arm are kept, so a contaminated arm can be re-run on its own.

Each arm is 13 questions at roughly 6.5 calls a question: about 85 calls
per arm, 255 for all three. Nothing is written to the repository but the
result file named at the end.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import meter                                            # noqa: E402
from lmm import Memory                                  # noqa: E402

DOC = os.path.join(HERE, "field", "documents", "nist_sp800-63b.pdf")
QS = os.path.join(HERE, "field", "questions_nist.json")
# (switches set on the session, keywords passed to `ask`)
ARMS = {
    "base": ({}, {}),
    "widen": ({"WIDEN": True}, {}),
    "enough": ({"ENOUGH": True}, {}),
    # THE BUCKET THE MEASUREMENT POINTED AT. On NIST, `other` plus
    # `extract` is 60% of a turn's calls — the readings that route the
    # question, bought before a word of the answer exists — while the
    # read-back, the thing everyone optimises, was 4 calls of 84 because
    # full coverage skips it. `standalone` tells the turn it has no
    # conversation on either side of it, which is true of every question
    # an API is sent, and the classification is then not bought at all.
    "standalone": ({}, {"standalone": True}),
}


def arm(spec):
    """One configuration, in its own store, over all thirteen questions."""
    switches, asking = spec
    m = Memory(None)
    for name, value in switches.items():
        setattr(m.session, name, value)
    m.learn(DOC, deep=False)
    rows = []
    for q in json.load(open(QS, encoding="utf-8")):
        gold = [str(g).lower() for g in (q.get("altin") or [])]
        before = meter.METER.totals()["calls"]
        marks = {k: v.get("calls", 0)
                 for k, v in (meter.METER.by_bucket() or {}).items()}
        said = m.ask(q["soru"], explain=True, **asking)
        # WHERE THE TURN SPENT IT, not just how much. The first clean run
        # showed the verdict working perfectly — four refusals, all four
        # right, none of the nine answerable questions wrongly stopped —
        # and saving nothing, because a refusing turn pays most of its
        # calls BEFORE the evidence is ever judged. Totals cannot say
        # which reading those were; buckets can.
        spent = {k: v.get("calls", 0) - marks.get(k, 0)
                 for k, v in (meter.METER.by_bucket() or {}).items()
                 if v.get("calls", 0) - marks.get(k, 0) > 0}
        rows.append({
            "buckets": spent,
            "soru": q["soru"],
            "altin": q.get("altin"),
            "cevap": str(said),
            "abstained": bool(said.abstained),
            "route": list(said.route or ()),
            "calls": meter.METER.totals()["calls"] - before,
            # a trap is right when the turn declines; everything else is
            # right when the gold token survives into the sentence
            "right": (bool(said.abstained) if not gold
                      else (not said.abstained
                            and all(g in str(said).lower() for g in gold))),
        })
    return rows


def main():
    if not os.environ.get("LMM_BACKEND"):
        print("LMM_BACKEND is not set — these arms need an engine.")
        return 1
    # THE CHILD: one arm, its own process, its own empty call pool.
    if sys.argv[1:2] == ["--arm"]:
        meter.install()                 # the meter belongs to THIS process
        json.dump(arm(ARMS[sys.argv[2]]), sys.stdout)
        return 0
    wanted = [a for a in (sys.argv[1:] or list(ARMS)) if a in ARMS]
    if not wanted:
        print("arms: %s" % ", ".join(ARMS))
        return 2
    print("engine: %s · arms: %s · about %d calls"
          % (os.environ["LMM_BACKEND"], ", ".join(wanted), 85 * len(wanted)))
    import subprocess                                   # noqa: PLC0415
    path = os.path.join(HERE, "gate_arms_result.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    for name in wanted:
        done = subprocess.run([sys.executable, __file__, "--arm", name],
                              capture_output=True, text=True)
        if done.returncode:
            print("%-8s FAILED\n%s" % (name, done.stderr[-600:]))
            return 1
        rows = out[name] = json.loads(done.stdout)
        print("%-8s dogru %2d/%d · cagri %4d · soru basina %.1f"
              % (name, sum(r["right"] for r in rows), len(rows),
                 sum(r["calls"] for r in rows),
                 sum(r["calls"] for r in rows) / len(rows)))
        pot = {}
        for r in rows:
            for bucket, calls in (r.get("buckets") or {}).items():
                pot[bucket] = pot.get(bucket, 0) + calls
        for bucket, calls in sorted(pot.items(), key=lambda kv: -kv[1]):
            print("           %-18s %4d" % (bucket, calls))
    if "base" in out:
        for name in wanted:
            if name == "base":
                continue
            print("\n--- %s vs base, only what moved ---" % name)
            for a, b in zip(out["base"], out[name]):
                if a["right"] != b["right"] or a["calls"] != b["calls"]:
                    print("%-52s %-10s %-10s"
                          % (a["soru"][:52],
                             "%s/%dc" % ("ok" if a["right"] else "--", a["calls"]),
                             "%s/%dc" % ("ok" if b["right"] else "--", b["calls"])))
            print("verdict: %+d correct for %+d calls"
                  % (sum(r["right"] for r in out[name])
                     - sum(r["right"] for r in out["base"]),
                     sum(r["calls"] for r in out[name])
                     - sum(r["calls"] for r in out["base"])))
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print("\nwritten: %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
