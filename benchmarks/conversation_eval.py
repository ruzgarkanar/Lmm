"""Multi-run conversation evaluator — the referee this week earned twice.

Measured lesson, twice in one day: byte-identical code and store gave
opposite verdicts on the same question forty minutes apart. Engine wobble
makes any single run meaningless for judging a conversational-architecture
change, so no such change is judged without this harness again.

Usage:
    python benchmarks/conversation_eval.py QUESTIONS.json CORPUS_DIR \
        [--runs 3] [--baseline BASELINE.json] [--save BASELINE.json]

QUESTIONS.json schema (all keys optional):
    {"olgu":  [[question, [gold, ...]], ...],     # factual, gold substrings
     "tuzak": [[question, null], ...],            # unanswerable traps
     "kiyas": [[question, [gold, ...]], ...],     # cross-document
     "takip": [[[q, [gold,...]], [q, [gold,...]]], ...]}  # dialogues

CORPUS_DIR holds .txt files; each becomes one source. Every run rebuilds a
fresh store (ingestion is seconds) and a fresh session, so runs are
independent. The report is the per-category MEDIAN across runs plus a
FLIP count — questions whose verdict changed between runs — because the
flip set, not the score, is where a boundary case lives. With --baseline
it prints the deltas; a change that moves medians down or flips up has to
argue for itself."""
import argparse
import glob
import json
import os
import statistics
import sys
import time
import unicodedata

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))


def _plain(t):
    t = unicodedata.normalize("NFD", (t or "").lower()
                              .replace("İ", "i").replace("I", "ı"))
    return "".join(c for c in t if not unicodedata.combining(c))


# refusal is detected by ABSENCE of asserted substance rather than by a
# phrase list where possible; the phrase net below is the fallback and is
# deliberately generous — a trap is failed only by a confident assertion
REFUSAL_HINTS = ["bilmiyorum", "bilgi yok", "sahip degil", "yer almiyor",
                 "bulunmuyor", "belirtilmemis", "mevcut degil", "veremem",
                 "bulamadim", "verilmemis", "yoktur", "bahsedilmiyor",
                 "icermiyor", "paylasamam", "i don't know", "no information",
                 "belirtirsen", "hangi egitim", "not stated", "cannot"]


def one_run(questions, corpus_dir):
    from lmm.api import Memory
    m = Memory(None)
    for f in sorted(glob.glob(os.path.join(corpus_dir, "*.txt"))):
        m.learn(open(f, encoding="utf-8").read(), deep=False,
                source="#docx:" + os.path.basename(f)[:-4])
    verdicts = {}

    def hit(answer, golds):
        # a gold entry may itself be a list: EVERY inner group must land
        # (any-of within, all-of across) — how a multi-part question
        # ("goals AND outcomes") demands both of its parts
        #
        # ...and a gold entry may be {"all": [...]}, which lands when ALL
        # of its parts do, as one alternative among the flat ones. That
        # shape exists because the referee was marking correct answers
        # wrong: asked whether two courses run the same length, the
        # memory answered "both are 2 full days" — which answers the
        # question completely — and scored zero for not containing the
        # word "same". Two of fifteen comparisons were being lost to
        # vocabulary, not to substance. Stating BOTH VALUES is answering
        # a comparison; the words for it are the engine's business.
        pa = _plain(answer)
        def one(g):
            if isinstance(g, dict):
                return all(_plain(str(x)) in pa for x in g.get("all", ()))
            if isinstance(g, list):
                return any(_plain(x) in pa for x in g)
            return _plain(g) in pa
        groups = [g for g in golds if isinstance(g, list)]
        others = [g for g in golds if not isinstance(g, list)]
        if groups:
            # the AND-groups are the requirement; a plain string or an
            # {"all": [...]} alternative satisfies the gold on its own
            return (any(one(g) for g in others)
                    or all(one(g) for g in groups))
        return any(one(g) for g in others)

    for q, golds in questions.get("olgu", []):
        verdicts[("olgu", q)] = hit(str(m.ask(q)), golds)
    for q, _none in questions.get("tuzak", []):
        # a trap is passed by NOT ASSERTING, and the system already owns
        # the organ that reads assertion: the abstention stamp. Judging
        # refusals by a phrase net measured the phrasing's mood, not the
        # substance — the referee's own trap category flipped on wording
        # while nothing factual changed. The phrase net remains only as a
        # backstop for answers the stamp cannot see.
        ans = m.ask(q, explain=True)
        a = _plain(str(ans))
        verdicts[("tuzak", q)] = bool(getattr(ans, "abstained", False)) or             any(_plain(x) in a for x in REFUSAL_HINTS)
    for q, golds in questions.get("kiyas", []):
        verdicts[("kiyas", q)] = hit(str(m.ask(q)), golds)
    # the frontier category: questions whose discriminating word the
    # document never writes — scored exactly like the factual ones, kept
    # apart because a change may move it in the opposite direction
    for q, golds in questions.get("sinir", []):
        verdicts[("sinir", q)] = hit(str(m.ask(q)), golds)
    for dialog in questions.get("takip", []):
        for q, golds in dialog:
            verdicts[("takip", q)] = hit(
                m.session.respond(q, teach=False), golds)
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions")
    ap.add_argument("corpus_dir")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--baseline")
    ap.add_argument("--save")
    args = ap.parse_args()
    questions = json.load(open(args.questions, encoding="utf-8"))

    runs = []
    for i in range(args.runs):
        t0 = time.time()
        runs.append(one_run(questions, args.corpus_dir))
        print(f"# run {i + 1}/{args.runs}: {time.time() - t0:.0f}s",
              file=sys.stderr)

    cats = sorted({c for v in runs for c, _q in v})
    report = {"runs": args.runs, "medians": {}, "flips": {}}
    for cat in cats:
        keys = sorted({q for v in runs for c, q in v if c == cat})
        scores = [sum(v.get((cat, q), False) for q in keys) for v in runs]
        flips = [q for q in keys
                 if len({v.get((cat, q), False) for v in runs}) > 1]
        report["medians"][cat] = (statistics.median(scores), len(keys))
        report["flips"][cat] = flips
        print(f"{cat:6} median {statistics.median(scores)}/{len(keys)}"
              f"  (runs: {scores})  flips: {len(flips)}")
        for q in flips:
            print(f"       ~ {q[:70]}")

    if args.baseline and os.path.exists(args.baseline):
        base = json.load(open(args.baseline))
        print("\n# vs baseline:")
        for cat, (med, n) in report["medians"].items():
            b = base.get("medians", {}).get(cat)
            if b:
                d = med - b[0]
                mark = "+" if d > 0 else ""
                print(f"{cat:6} {mark}{d}  (baseline {b[0]}/{b[1]})")
    if args.save:
        json.dump(report, open(args.save, "w"), ensure_ascii=False, indent=1)
        print(f"# baseline saved: {args.save}")


if __name__ == "__main__":
    main()
