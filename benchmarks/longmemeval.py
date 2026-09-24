"""LongMemEval, run against this memory — and kept in the repository.

This project measured itself on LongMemEval twice (0.6.4, 0.7) and could
not do it a third time: the harness and the data both lived outside the
repo and are gone, so the numbers in those release notes cannot be
reproduced by anyone, including us. That is the defect this file exists
to fix. The DATA stays out (it is 278 MB and not ours to vendor); the
reading of it, the slice rule and the scoring live here.

    python benchmarks/longmemeval.py longmemeval_s --most 30 --judge

WHAT A SESSION BECOMES. One instance is one memory. Each haystack session
enters as its own source, stamped with the session's own date in the
shape `stamp_day` reads — `chat 2023/04/09 (Sun) 09:00` is that
function's own example — so a store built here is wholly dated and the
recency vote (W188) is live, which is what this benchmark is about.
A turn's speaker is written in front of its text, because "I bought a
Samsung" and "you bought a Samsung" are different claims and the store
holds sentences, not roles.

THE SLICE IS DETERMINISTIC AND STATED. Questions are sorted by id and
taken round-robin across the six types, so `--most 30` is the same
thirty questions on every machine and in every release. The full set is
`--most 0`.

TWO SCORES, BECAUSE THEY MEASURE DIFFERENT THINGS. `strict` asks whether
the gold string is in the answer — free, side-blind, and harsh on a
correct answer worded differently. `judge` asks an engine whether the
answer says what the gold says — one extra call per question, and it is
what published LongMemEval figures use, so it is the comparable one. A
run reports both and says which is which; neither is called accuracy on
its own.

ABSTENTION QUESTIONS (`_abs`) ARE SCORED AS ABSTENTIONS, not by their
gold sentence: the benchmark's own gold for them is a paragraph
explaining that the information is not there, and what this memory owes
is silence, which it stamps structurally (`Answer.abstained`).
"""
import argparse
import json
import os
import sys
import time
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

TYPES = ("knowledge-update", "multi-session", "single-session-assistant",
         "single-session-preference", "single-session-user",
         "temporal-reasoning")


def _plain(text):
    text = unicodedata.normalize("NFD", (text or "").lower())
    return " ".join("".join(c for c in text
                            if not unicodedata.combining(c)).split())


def slice_of(data, most):
    """The same questions on every machine: sorted by id, round-robin
    across the six types, so a slice is a statement and not a sample."""
    by_type = defaultdict(list)
    for row in sorted(data, key=lambda r: str(r["question_id"])):
        by_type[row["question_type"]].append(row)
    if not most:
        return [row for kind in TYPES for row in by_type[kind]]
    out, at = [], 0
    while len(out) < most and any(len(by_type[k]) > at for k in TYPES):
        for kind in TYPES:
            if len(by_type[kind]) > at and len(out) < most:
                out.append(by_type[kind][at])
        at += 1
    return out


def learn_instance(row):
    """One instance, as a memory: a session per source, stamped by day."""
    from lmm.api import Memory
    m = Memory(None)
    dates = row.get("haystack_dates") or []
    for nth, session in enumerate(row["haystack_sessions"]):
        said = []
        for turn in session:
            who = "The user" if turn.get("role") == "user" else "The assistant"
            body = (turn.get("content") or "").strip()
            if body:
                said.append("%s: %s" % (who, body))
        if not said:
            continue
        when = dates[nth] if nth < len(dates) else ""
        m.learn("\n".join(said), deep=False,
                source="chat %s" % (when or "session %d" % nth))
    return m


def judged(question, gold, said):
    """Does the answer say what the gold says — the engine, asked once."""
    from lmm import runtime
    out = runtime.generate(
        "QUESTION: %s\n\nREFERENCE ANSWER: %s\n\nANSWER: %s"
        % (question, gold, said),
        system=('Does the ANSWER give the same information as the '
                'REFERENCE ANSWER for this QUESTION? Wording, length and '
                'extra detail do not matter; the facts asked for do. '
                'Answer ONLY "yes" or "no".'),
        max_tokens=4, temperature=0.0, small=True)
    return (out or "").strip().lower().startswith("yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="longmemeval_s / _m / _oracle (JSON)")
    ap.add_argument("--most", type=int, default=30, help="0 = all 500")
    ap.add_argument("--judge", action="store_true",
                    help="also ask the engine whether the answer matches")
    ap.add_argument("--save")
    ap.add_argument("--shape", default=None,
                    help='declare the turn kind, e.g. "ask" — a declared '
                         'shape also skips the count/span door (W93)')
    ap.add_argument("--quoted", action="store_true",
                    help="answer from one line the store holds")
    ap.add_argument("--conversational", action="store_true",
                    help="do NOT pass standalone, so the turn is classified "
                         "and carries a subject")
    args = ap.parse_args()

    from lmm import runtime
    data = json.load(open(args.data, encoding="utf-8"))
    chosen = slice_of(data, args.most)
    print("# %d questions · shape=%s quoted=%s standalone=%s"
          % (len(chosen), args.shape, args.quoted,
             not args.conversational), file=sys.stderr)
    print("# %d questions · %s" % (len(chosen),
                                   dict(Counter(r["question_type"]
                                                for r in chosen))),
          file=sys.stderr)

    rows, t0 = [], time.time()
    for nth, row in enumerate(chosen, 1):
        absent = str(row["question_id"]).endswith("_abs")
        before = runtime.CALLS
        built = time.time()
        m = learn_instance(row)
        ingest = time.time() - built
        said = m.ask(row["question"], explain=True,
                     standalone=not args.conversational,
                     shape=args.shape, quoted=args.quoted)
        gold = row["answer"]
        strict = (said.abstained if absent
                  else _plain(str(gold)) in _plain(str(said)))
        verdict = None
        if args.judge:
            verdict = (bool(said.abstained) if absent
                       else judged(row["question"], gold, str(said)))
        rows.append({"id": row["question_id"], "type": row["question_type"],
                     "abs": absent, "strict": bool(strict),
                     "judge": verdict, "abstained": bool(said.abstained),
                     "calls": runtime.CALLS - before, "ingest_s": ingest,
                     "question": row["question"], "gold": str(gold),
                     "said": str(said), "route": list(said.route)})
        print("%3d/%d %-28s %-6s %s %s" % (
            nth, len(chosen), row["question_type"][:28],
            "strict" if strict else ("JUDGE" if verdict else "-"),
            "abs" if absent else "   ", str(said)[:60].replace("\n", " ")),
            file=sys.stderr, flush=True)

    took = time.time() - t0
    print("\n%-30s %s" % ("", "strict   judge   n"))
    for kind in TYPES:
        mine = [r for r in rows if r["type"] == kind]
        if not mine:
            continue
        s = sum(r["strict"] for r in mine)
        j = sum(bool(r["judge"]) for r in mine)
        print("%-30s %5d %7s %4d"
              % (kind, s, (j if args.judge else "-"), len(mine)))
    s = sum(r["strict"] for r in rows)
    j = sum(bool(r["judge"]) for r in rows)
    print("-" * 52)
    print("%-30s %5d %7s %4d   (%.0f%% / %s)"
          % ("TOTAL", s, (j if args.judge else "-"), len(rows),
             100.0 * s / max(1, len(rows)),
             ("%.0f%%" % (100.0 * j / max(1, len(rows))))
             if args.judge else "-"))
    print("%d engine calls · %.0f s · %.1f s a question"
          % (sum(r["calls"] for r in rows), took, took / max(1, len(rows))))
    if args.save:
        json.dump(rows, open(args.save, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
