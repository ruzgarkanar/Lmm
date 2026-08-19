#!/usr/bin/env python3
"""FIVE DOCUMENTS, TEN QUESTIONS EACH — the short protocol.

What it measures, per document and per configuration: how many questions were
answered CORRECTLY, how many were ABSTAINED on, how many came back WRONG, how
many cost NO MODEL CALL at all, the calls per question and the wall time. Three
samples of each configuration; the median is what is reported.

THE INGESTION IS PAID ONCE AND REUSED. Every configuration and every sample is
loaded from the same saved graph, so ingestion variance — which is the largest
source of noise in this system and has nothing to do with the answer path being
compared — cannot move the comparison. What is re-run per sample is the
answering, which is what the configurations differ in.

    python3.11 benchmarks/field/measure.py ingest          # once, pays for it
    python3.11 benchmarks/field/measure.py run  base       # LMM_DIRECT_LOOKUP=0
    python3.11 benchmarks/field/measure.py run  graph      # the shipped path
    python3.11 benchmarks/field/measure.py repeat        # the answer cache
    python3.11 benchmarks/field/measure.py report

The two field documents are NOT in this repository — fetch them first with
`fetch_documents.py`, which carries every URL and its licence.
"""
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
os.environ.setdefault("LMM_BACKEND", "azure")

DOCS = os.path.join(HERE, "documents")
# WHERE THE INGESTED GRAPHS AND THE ANSWERS LIVE. Overridable because some
# variables are READ-TIME ones (`LMM_XLSX_HEADER_BLOCK` changes what the
# document becomes, not how it is asked), and those cannot share one ingestion:
# the arm is a different graph, so it needs a different directory to keep it in
# and a different directory to write its answers to. Nothing else about the
# protocol changes — same documents, same questions, same three samples.
GRAPHS = os.environ.get("LMM_FIELD_GRAPHS") or os.path.join(HERE, "graphs")
RESULTS = os.environ.get("LMM_FIELD_RESULTS") or os.path.join(HERE, "results")

# name, what to learn, deep?, question file
SETS = [
    ("tr", os.path.join(ROOT, "benchmarks", "corpus.txt"), True,
     "questions_tr10.json"),
    ("en", os.path.join(ROOT, "benchmarks", "corpus_en.txt"), True,
     "questions_en10.json"),
    ("es", os.path.join(ROOT, "benchmarks", "corpus_es.txt"), True,
     "questions_es10.json"),
    ("nist", os.path.join(DOCS, "nist_sp800-63b.pdf"), False,
     "questions_nist.json"),
    ("census", os.path.join(DOCS, "us_state_population.xlsx"), False,
     "questions_census.json"),
]

CONFIGS = {
    # the answer path as it was before the graph-first route existed at all
    "base": {"LMM_DIRECT_LOOKUP": "0", "LMM_LOOKUP_RUNS": "1"},
    # graph-first, naming nodes one word at a time — what shipped before this
    # task, and the arm the multi-word rule has to beat
    "words": {"LMM_DIRECT_LOOKUP": "1", "LMM_LOOKUP_RUNS": "0"},
    # graph-first, naming a node by the question's contiguous word runs
    "graph": {"LMM_DIRECT_LOOKUP": "1", "LMM_LOOKUP_RUNS": "1",
              "LMM_LOOKUP_CANDIDATES": "0"},
    # ... and answering with EVERY record the question's reading names, rather
    # than declining and letting the semantic path pick one of them
    "candidates": {"LMM_DIRECT_LOOKUP": "1", "LMM_LOOKUP_RUNS": "1",
                   "LMM_LOOKUP_CANDIDATES": "1"},
}


def questions(name):
    for tag, _, _, path in SETS:
        if tag == name:
            return json.load(open(os.path.join(HERE, path), encoding="utf-8"))
    raise KeyError(name)


def _counted():
    """Count every model call. `runtime.generate` is the one chokepoint the
    whole system reaches the engine through — the same fact `tests/test_core.py`
    rests its zero-call assertions on."""
    from lmm import runtime
    was, seen = runtime.generate, []

    def counting(*args, **kwargs):
        seen.append(1)
        return was(*args, **kwargs)

    runtime.generate = counting
    return seen, (lambda: setattr(runtime, "generate", was))


def ingest():
    """Read every document once and save the graph. Paid once, not per sample."""
    from lmm import Memory
    os.makedirs(GRAPHS, exist_ok=True)
    for name, what, deep, _ in SETS:
        path = os.path.join(GRAPHS, name + ".lmm")
        if os.path.exists(path):
            print(f"{name}: already ingested")
            continue
        if not os.path.exists(what):
            print(f"{name}: MISSING {what} — run fetch_documents.py")
            continue
        memory = Memory(path)
        seen, restore = _counted()
        t0 = time.time()
        try:
            if deep:                      # a corpus is text, read into the graph
                learned = memory.learn(open(what, encoding="utf-8").read(),
                                       source=f"#doc:{name}", deep=True)
                from lmm.mind import Mind
                Mind(memory.session).run()
            else:                         # a document is read as a document
                learned = memory.learn(what)
        finally:
            restore()
        memory.save()
        print(f"{name}: {learned!r} — {len(seen)} calls, "
              f"{round(time.time() - t0, 1)}s")


def expand():
    """PAY FOR THE OFFLINE EXPANSION on graphs that are already ingested.

    It runs over a COPY of the baseline graphs (`LMM_FIELD_GRAPHS` points at
    it), so the two arms of the A/B differ in exactly one thing: the expansion
    index. Re-ingesting for the expanded arm would have let ingestion variance
    — the largest noise source in this system — into the comparison, which is
    the same reason the protocol pays for ingestion once in the first place.

        cp -r graphs graphs_expand
        LMM_FIELD_GRAPHS=.../graphs_expand python3.11 measure.py expand
    """
    from lmm.api import Memory
    for name, _, _, _ in SETS:
        path = os.path.join(GRAPHS, name + ".lmm")
        if not os.path.exists(path):
            print(f"{name}: no graph — run `ingest` first")
            continue
        memory = Memory(path)
        if memory.session.evidence.expansions:
            print(f"{name}: already expanded")
            continue
        seen, restore = _counted()
        t0 = time.time()
        try:
            asked, kept = memory.session.expand()
        finally:
            restore()
        memory.save()
        print(f"{name}: {asked} units asked, {kept} queries kept — "
              f"{len(seen)} calls, {round(time.time() - t0, 1)}s", flush=True)


def _scored(question, said, abstained):
    """correct / abstain / WRONG — the three outcomes, decided structurally.

    A question with no answer in the document is answered correctly by
    ABSTAINING; anything asserted for it is wrong. A question with an answer is
    correct when the answer carries one of the gold strings, and WRONG when it
    asserts something else. Abstaining on an answerable question is neither —
    it is a miss, not a fabrication, and it is counted separately because those
    two failures are not the same failure.
    """
    gold = question["altin"]
    if gold is None:
        return "correct" if abstained else "wrong"
    if abstained:
        return "abstain"
    low = said.lower()
    return "correct" if any(g.lower() in low for g in gold) else "wrong"


def run(config, only=None, samples=3):
    """`only` restricts the arm to one document. A read-time arm (the header
    block) changes ONE of the five documents and leaves the other four
    byte-identical, so re-asking those four would measure the same graph with
    the same config twice — the arm it is being compared against already has
    them."""
    from lmm.api import Memory
    os.makedirs(RESULTS, exist_ok=True)
    for key, value in CONFIGS[config].items():
        os.environ[key] = value
    for name, _, _, _ in SETS:
        if only and name != only:
            continue
        graph = os.path.join(GRAPHS, name + ".lmm")
        if not os.path.exists(graph):
            print(f"{name}: no graph — run `ingest` first")
            continue
        for sample in range(1, samples + 1):
            out = os.path.join(RESULTS, f"{config}_{name}_{sample}.json")
            if os.path.exists(out):
                continue
            # A FRESH MEMORY PER SAMPLE. The answer cache is per Memory, so a
            # sample must not inherit the previous one's answers — that would
            # measure the cache instead of the path under test.
            memory = Memory(graph)
            rows = []
            for q in questions(name):
                seen, restore = _counted()
                t0 = time.time()
                try:
                    said = memory.ask(q["soru"], explain=True)
                finally:
                    restore()
                rows.append({
                    "soru": q["soru"], "cevap": str(said),
                    "calls": len(seen), "ms": round((time.time() - t0) * 1000),
                    "abstained": said.abstained, "from_graph": said.from_graph,
                    "verdict": _scored(q, str(said), said.abstained),
                })
                print(f"[{config}/{name}/{sample}] {q['soru'][:44]:46} "
                      f"{rows[-1]['verdict']:8} {len(seen)} calls", flush=True)
            json.dump(rows, open(out, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=1)


def repeat():
    """THE ANSWER CACHE, measured: the same ten questions twice, one memory.

    The second pass is the whole measurement. Its calls and its seconds are
    what a repeated question costs once the first one has been paid for, and
    the answers are compared word for word — a cheaper answer that is not the
    SAME answer would be a defect, not a saving.
    """
    from lmm.api import Memory
    print("| document | pass | calls/q | s/q | answers identical |")
    print("|---|---|---|---|---|")
    for name, _, _, _ in SETS:
        graph = os.path.join(GRAPHS, name + ".lmm")
        if not os.path.exists(graph):
            continue
        memory = Memory(graph)
        passes = []
        for _ in (1, 2):
            said, calls, t0 = [], 0, time.time()
            for q in questions(name):
                seen, restore = _counted()
                try:
                    said.append(str(memory.ask(q["soru"])))
                finally:
                    restore()
                calls += len(seen)
            passes.append((said, calls, time.time() - t0))
        n = len(passes[0][0])
        same = sum(a == b for a, b in zip(passes[0][0], passes[1][0]))
        for i, (_, calls, wall) in enumerate(passes, start=1):
            print(f"| {name} ({n}) | {i} | {round(calls / n, 2)} | "
                  f"{round(wall / n, 2)} | {same}/{n} |")


def _summary(config, name):
    rows = []
    for sample in (1, 2, 3):
        path = os.path.join(RESULTS, f"{config}_{name}_{sample}.json")
        if os.path.exists(path):
            rows.append(json.load(open(path, encoding="utf-8")))
    if not rows:
        return None
    def med(fn):
        return statistics.median(fn(one) for one in rows)
    return {
        "correct": med(lambda r: sum(x["verdict"] == "correct" for x in r)),
        "abstain": med(lambda r: sum(x["verdict"] == "abstain" for x in r)),
        "wrong": med(lambda r: sum(x["verdict"] == "wrong" for x in r)),
        "zero_call": med(lambda r: sum(x["calls"] == 0 for x in r)),
        "calls_per_q": round(med(lambda r: sum(x["calls"] for x in r)
                                 / len(r)), 2),
        "s_per_q": round(med(lambda r: sum(x["ms"] for x in r)
                             / len(r)) / 1000, 2),
        "n": len(rows[0]),
    }


def report():
    print("| document | config | correct | abstain | WRONG | zero-call | "
          "calls/q | s/q |")
    print("|---|---|---|---|---|---|---|---|")
    for name, _, _, _ in SETS:
        for config in CONFIGS:
            got = _summary(config, name)
            if got is None:
                continue
            print(f"| {name} ({got['n']}) | {config} | {got['correct']:.0f} | "
                  f"{got['abstain']:.0f} | {got['wrong']:.0f} | "
                  f"{got['zero_call']:.0f} | {got['calls_per_q']} | "
                  f"{got['s_per_q']} |")


def types(config):
    """The same samples, split by what KIND of question they are.

    The whole-document score cannot judge a retrieval change aimed at one class
    of question: three differently-worded questions inside a set of thirteen
    move the total by at most three, and every other row can hide them. The
    class the expansion targets — `esanlam`, a question that asks for something
    the document states in OTHER words — is therefore reported on its own line.
    """
    print(f"| document | kind | n | correct | abstain | WRONG | zero-call | "
          f"({config}) |")
    print("|---|---|---|---|---|---|---|---|")
    for name, _, _, _ in SETS:
        rows = []
        for sample in (1, 2, 3):
            path = os.path.join(RESULTS, f"{config}_{name}_{sample}.json")
            if os.path.exists(path):
                rows.append(json.load(open(path, encoding="utf-8")))
        if not rows:
            continue
        kinds = {q["soru"]: q["tip"] for q in questions(name)}
        for kind in sorted({kinds[r["soru"]] for r in rows[0]}):
            def med(verdict, kind=kind, rows=rows):
                return statistics.median(
                    sum(x["verdict"] == verdict for x in one
                        if kinds[x["soru"]] == kind) for one in rows)
            n = sum(kinds[x["soru"]] == kind for x in rows[0])
            zero = statistics.median(
                sum(x["calls"] == 0 for x in one if kinds[x["soru"]] == kind)
                for one in rows)
            print(f"| {name} | {kind} | {n} | {med('correct'):.0f} | "
                  f"{med('abstain'):.0f} | {med('wrong'):.0f} | {zero:.0f} | |")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "report"
    if what == "ingest":
        ingest()
    elif what == "expand":
        expand()
    elif what == "types":
        types(sys.argv[2] if len(sys.argv) > 2 else "candidates")
    elif what == "run":
        run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    elif what == "repeat":
        repeat()
    else:
        report()
