"""ENGINE INDEPENDENCE — the same questions, on a 3B int4 model on this machine.

    python3.11 benchmarks/gguf_run.py <corpus> <questions> <out.json>

The claim the graph-first path makes is not only "cheaper". It is that a
question the GRAPH can settle is answered the same way whatever engine is
installed, because no engine is consulted — and the field trial
(`benchmarks/field/REPORT.md`) is where that stopped being an abstraction: on a
weak local engine two fabrications walked through the semantic gates, because
the gates ask the same weak engine whether the evidence supports the claim. A
judge is only as good as the model judging.

So this script runs BOTH paths against ONE session, one ingestion and one
engine, and the difference between the two columns is the whole measurement:

    fluent=False   the graph answers what the graph can settle (the new path)
    fluent=True    every question goes to the engine (exactly the old path —
                   `Session.respond` skips `lookup` on this flag, so this is a
                   real before-column produced by the same binary, not a
                   remembered number from another run)

Model calls are counted at `runtime.generate`, which is the single point every
call in this system passes through whichever backend is selected — `meter.py`
counts the SERVER's own token report and only a hosted API has one, so on a
local engine the honest unit is the call and the second.

Engine: whatever LMM_BACKEND says. Intended:

    LMM_BACKEND=gguf LMM_N_GPU_LAYERS=99 python3.11 benchmarks/gguf_run.py ...
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)


def counted():
    """Wrap the engine chokepoint in a per-question counter. Returns the list
    the counts land in; nothing about the call itself changes."""
    from lmm import runtime
    original = runtime.generate
    tally = {"calls": 0, "seconds": 0.0}

    def generate(*a, **kw):
        t0 = time.time()
        try:
            return original(*a, **kw)
        finally:
            tally["calls"] += 1
            tally["seconds"] += time.time() - t0

    runtime.generate = generate
    return tally


def ask_all(session, questions, tally, fluent):
    rows = []
    for q in questions:
        session.history = []            # each question stands alone
        before, seconds = tally["calls"], tally["seconds"]
        t0 = time.time()
        said = session.respond(q["soru"], fluent=fluent)
        rows.append({"soru": q["soru"], "cevap": said,
                     "ms": round((time.time() - t0) * 1000),
                     "abstained": bool(session.last_abstained),
                     "from_graph": bool(session.last_from_graph),
                     "calls": tally["calls"] - before,
                     "engine_seconds": round(tally["seconds"] - seconds, 3)})
        print(f"> [{'graph' if rows[-1]['from_graph'] else 'engine'}"
              f" {rows[-1]['calls']} calls] {q['soru']}\n  {said}", flush=True)
    return rows


def main():
    corpus_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "corpus.txt")
    q_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "questions.json")
    out_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "cost", "gguf.json")

    from lmm.session import Session
    from lmm.mind import Mind

    tally = counted()
    questions = json.load(open(q_path, encoding="utf-8"))
    corpus = open(corpus_path, encoding="utf-8").read()

    s = Session(None)
    m = Mind(s)
    t0 = time.time()
    wrote, skipped = s.learn_text(corpus, source="#doc:corpus", deep=True)
    m.run()
    ingest = {"wall_seconds": round(time.time() - t0, 1), "facts": wrote,
              "skipped": skipped, "calls": tally["calls"],
              "derived": sum(1 for r in s.memory.records.values()
                             if r.source == "#inference")}
    print(f"ingest: {ingest}", flush=True)

    out = {"backend": os.environ.get("LMM_BACKEND", ""),
           "gpu_layers": os.environ.get("LMM_N_GPU_LAYERS", "0"),
           "corpus": os.path.basename(corpus_path), "ingest": ingest}
    for name, fluent in (("engine_only", True), ("graph_first", False)):
        # ENGINE-ONLY FIRST, so the graph-first column cannot be credited with
        # a model that has warmed up on these questions — the local engine
        # loads on its first call and the first column pays for that.
        print(f"\n===== {name} (fluent={fluent}) =====", flush=True)
        start = tally["calls"]
        rows = ask_all(s, questions, tally, fluent)
        out[name] = {
            "questions": rows,
            "calls": tally["calls"] - start,
            "zero_call_questions": sum(1 for r in rows if r["calls"] == 0),
            "question_count": len(rows),
        }
        print(f"{name}: {out[name]['calls']} calls over {len(rows)} q, "
              f"zero-call {out[name]['zero_call_questions']}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
