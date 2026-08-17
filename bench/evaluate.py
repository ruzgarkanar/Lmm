"""STABILITY-AWARE SCORING — aggregate N samples of the SAME configuration.

Why this exists: the benchmarks are noisy. Identical code scored hospital
9-11/16, manual 21-28/30 and corpus 13-16/17 across repeated runs, and a
single lucky sample (hospital 15/16) was once mistaken for a plateau, which
sent a whole session chasing a regression that never happened.

So: never judge a change by one run. This tool takes several result files
produced by the same configuration and reports
  - the median / min / max total (the median is the honest headline),
  - per-question STABILITY: how many samples got it right,
which separates three very different things a single run cannot:
  stable pass (n/n)    — really solved
  flaky (between)      — the answer-selection tail, the real work item
  stable fail (0/n)    — genuinely missing, needs a mechanism, not tuning

Usage:
    python3.11 bench/evaluate.py questions.json result_a.json result_b.json ...
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from score import abstains, fold                                # noqa: E402


def correct(question, answer):
    """Same criterion as score.py: gold keyword present, or — for absence
    questions — a refusal."""
    if question["altin"] is None:
        return abstains(answer or "")
    ans = fold(answer or "")
    return any(fold(gold) in ans for gold in question["altin"])


def main():
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[-1])
        return
    questions = json.load(open(sys.argv[1], encoding="utf-8"))
    samples = []
    for path in sys.argv[2:]:
        data = json.load(open(path, encoding="utf-8"))
        samples.append({row["soru"]: row.get("cevap", "")
                        for row in data["cevaplar"]})

    totals = []
    hits = {q["soru"]: 0 for q in questions}
    for sample in samples:
        total = 0
        for q in questions:
            if q["soru"] in sample and correct(q, sample[q["soru"]]):
                total += 1
                hits[q["soru"]] += 1
        totals.append(total)

    n, q_count = len(samples), len(questions)
    ordered = sorted(totals)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1]
                                            + ordered[n // 2]) / 2
    print(f"samples: {n}   per-sample totals: {totals}")
    print(f"MEDIAN {median}/{q_count}   min {min(totals)}   max {max(totals)}"
          f"   spread {max(totals) - min(totals)}")

    stable_pass = [q for q in questions if hits[q["soru"]] == n]
    stable_fail = [q for q in questions if hits[q["soru"]] == 0]
    flaky = [q for q in questions if 0 < hits[q["soru"]] < n]
    print(f"\nstable pass {len(stable_pass)}   flaky {len(flaky)}"
          f"   stable fail {len(stable_fail)}   (of {q_count})")
    for label, group in (("FLAKY", flaky), ("STABLE FAIL", stable_fail)):
        for q in group:
            print(f"  [{label}] {hits[q['soru']]}/{n}  {q['soru']}")


if __name__ == "__main__":
    main()
