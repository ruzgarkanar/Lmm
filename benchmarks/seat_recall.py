"""Does the answering SESSION reach the seats? — engine-free.

`gate_blame.py` measured that the gates kill nothing: of 22 unanswered
turns, none had the answer in a candidate the gate rejected. The failure
is upstream, and LongMemEval labels exactly where the answer lives —
`answer_session_ids`. So the question can be asked without an engine:
of the sessions this instance holds, does retrieval seat a line from the
one that carries the answer?

If it does, the evidence arrives and the reading fails, and building
records from the retrieved sessions (extraction at question time) is
worth measuring. If it does not, no amount of extraction helps, because
extraction would run over the wrong sessions.

    python3 benchmarks/seat_recall.py data/longmemeval_s_cleaned.json

No model calls at all: ingestion is lines-only and `find` is the store's
own.
"""
import builtins
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
os.environ.pop("LMM_BACKEND", None)
_real = builtins.__import__


def _no_models(name, *a, **k):
    if name.split(".")[0] in ("transformers", "torch", "llama_cpp"):
        raise ImportError("no model (simulated)")
    return _real(name, *a, **k)


builtins.__import__ = _no_models

import longmemeval as lme                               # noqa: E402
from lmm import Memory                                  # noqa: E402

SEATS = 6


def one(row):
    """Build the instance exactly as the benchmark harness does, and
    remember which SOURCE each session was stored under."""
    m = Memory(None)
    m.session.asked_at = lme._day(row.get("question_date"))
    dates = row.get("haystack_dates") or []
    ids = row.get("haystack_session_ids") or []
    answer_ids = set(row.get("answer_session_ids") or [])
    answering_sources = set()
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
        source = "chat %s" % (when or "session %d" % nth)
        m.learn("\n".join(said), deep=False, source=source)
        if nth < len(ids) and ids[nth] in answer_ids:
            answering_sources.add(source)
    return m, answering_sources


def main():
    data = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "data", "longmemeval_s_cleaned.json")
    blame = os.path.join(HERE, "gate_blame.json")
    failed = set()
    if os.path.exists(blame):
        failed = {r["question"] for r in json.load(open(blame))
                  if not r["right"]}
    rows = lme.slice_of(json.load(open(data, encoding="utf-8")), 30)
    seated = missed = 0
    out = []
    for at, row in enumerate(rows, 1):
        only_failed = bool(failed)
        if only_failed and row["question"] not in failed:
            continue
        m, answering = one(row)
        store = m.session.evidence
        lines = store.find(row["question"], most=SEATS)
        sources = list(store.last_sources[:len(lines)])
        hit = any(src in answering for src in sources)
        seated += hit
        missed += not hit
        out.append({"question": row["question"], "seated": hit,
                    "answer_sources": sorted(answering),
                    "seated_sources": sources})
        print("%3d %-8s %s" % (at, "SEATED" if hit else "not seen",
                               row["question"][:56]))
    total = seated + missed
    print("\n%d failed questions · the answering session reached the seats "
          "in %d (%.0f%%) · never reached in %d (%.0f%%)"
          % (total, seated, 100.0 * seated / max(1, total),
             missed, 100.0 * missed / max(1, total)))
    json.dump(out, open(os.path.join(HERE, "seat_recall.json"), "w"),
              indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
