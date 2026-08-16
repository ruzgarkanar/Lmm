"""LMM chat — interactive. Qwen (language) + graph/gate (truth & growth).

Usage:
    python3.11 -m lmm.chat            # memory: models/lmm/memory.lmm

Every turn is logged (logs/lmm-<time>.jsonl): input · answer · the operation
the reader saw · duration. Memory is saved on exit — carried into the next
session (growth without retraining). Exit: Ctrl+D on an empty line or Ctrl+C.
"""
import json
import os
import time


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    os.makedirs("models/lmm", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    from lmm.session import Session

    memory_path = os.path.join("models", "lmm", "memory.lmm")
    legacy = os.path.join("models", "lmm", "hafiza.lmm")
    if not os.path.exists(memory_path) and os.path.exists(legacy):
        memory_path = legacy            # backward compat: old Turkish default
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join("logs", f"lmm-{stamp}.jsonl")

    print("# LMM loading (Qwen-3B, ~10 s on first run)...")
    session = Session(memory_path)
    from lmm.mind import Mind
    mind = Mind(session)        # autonomous mind — thinks on its own between turns (ms)
    print(f"# memory: {memory_path} ({len(session.memory.records)} records)")
    print(f"# log: {log_path}")
    print("# exit: Ctrl+C or Ctrl+D · empty lines are skipped\n")

    log = open(log_path, "a", encoding="utf-8")
    proposed = set()        # curiosities already PROACTIVELY suggested (avoid repeats)
    try:
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            # AUTONOMOUS MIND — self-awareness + self-growth commands
            # (legacy Turkish command names still accepted):
            if line in ("?wonder", "?curiosity", "?merak"):
                w = mind.wonder()           # much-used-but-undefined (refined)
                print("# I wonder about (things I keep using but don't know): "
                      + (", ".join(w) if w else "—")
                      + ("  →  ?research <topic>" if w else ""))
                continue
            if line.startswith("?research ") or line.startswith("?araştır "):
                topic = line.split(" ", 1)[1].strip()
                print(f"# researching {topic} (web, low trust)...")
                ok = mind.research(topic)
                print(f"# {'learned ✓ (sourced)' if ok else 'not found'}")
                continue
            if line in ("?tension", "?çelişki"):
                tens = session.tension()
                if tens:
                    for subj, vals in tens:
                        print(f"# tension: {subj} → {' ↔ '.join(vals)}")
                else:
                    print("# no tension")
                continue
            started = time.time()
            said = session.respond(line)
            print(said if said else "[…]")
            # AUTONOMOUS THINKING (ms, between turns): reasons on its own after
            # the message — resolve tension / derive / distill. If it derives
            # something new, it quietly says so.
            before = sum(1 for r in session.memory.records.values()
                         if r.source == "#inference")
            mind.run()
            after = sum(1 for r in session.memory.records.values()
                        if r.source == "#inference")
            if after > before:
                print(f"  💭 (derived on my own: {after - before} new links)")
            # PROACTIVE CURIOSITY: if it has used a concept often enough without
            # knowing it, it says 'I want to learn this' (once, without noise).
            cur = mind.top_curiosity()
            if cur and cur[0] not in proposed:
                proposed.add(cur[0])
                print(f"  💭 '{cur[0]}' keeps coming up but I don't know what it is"
                      f" — ?research {cur[0]}")
            log.write(json.dumps({
                "at": time.strftime("%H:%M:%S"),
                "in": line,
                "out": said,
                # GATE-approved triples: the gold data for sleep-consolidation
                # (sentence → triple pairs; not the model's raw output, but what
                # the gate accepted). finetune/consolidate.py harvests these.
                "learned": session.last_written,
                "records": len(session.memory.records),
                "ms": round((time.time() - started) * 1000),
            }, ensure_ascii=False) + "\n")
            log.flush()
    finally:
        session.save()
        log.close()
        print(f"\n# memory saved ({len(session.memory.records)} records)"
              f" · log: {log_path}")


if __name__ == "__main__":
    main()
