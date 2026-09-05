"""LMM chat — interactive. Qwen (language) + graph/gate (truth & growth).

Usage:
    lmm                               # memory: ./models/lmm/memory.lmm
    python3.11 -m lmm.chat            # the same thing

Every turn is logged (logs/lmm-<time>.jsonl): input · answer · the operation
the reader saw · duration. Memory is saved on exit — carried into the next
session (growth without retraining). Exit: Ctrl+D on an empty line or Ctrl+C.

THE FILES BELONG TO THE USER, NOT TO THE PACKAGE. This entry point used to
chdir to two directories above its own source and write `models/` and `logs/`
there. At the repository root that was the repository; installed from PyPI it
was SITE-PACKAGES — measured on a clean venv, a first run left the operator's
memory inside the install directory, where the next `pip install -U` deletes
it and a shared or read-only install cannot write at all. `lmm.paths` was
written for exactly this and says so in its own docstring; this file simply
never got converted. It resolves against the user's working directory now,
`LMM_HOME` if they set one.
"""
import json
import os
import sys
import time

from lmm import paths

HELP = """lmm — an interactive session against a living memory.

    lmm                    start talking; memory persists between sessions
    lmm --help             this text
    lmm --version          the installed version

Files, all under the current directory (or $LMM_HOME if set):
    models/lmm/memory.lmm  the graph and its evidence
    logs/lmm-<time>.jsonl  one line per turn

An engine is needed to put answers into words. The default is a local
Qwen2.5-3B; set LMM_BACKEND=openai with OPENAI_API_KEY for a hosted one, or
LMM_BACKEND=gguf for llama.cpp. The graph, the gate and the derivation need
no engine at all.

As a library:
    from lmm import Memory
    m = Memory("mind.lmm"); m.learn("manual.pdf"); print(m.ask("..."))

Docs: https://github.com/ruzgarkanar/Lmm"""


def main():
    argv = sys.argv[1:]
    if any(a in ("-h", "--help", "help") for a in argv):
        print(HELP)
        return
    if any(a in ("-V", "--version") for a in argv):
        from lmm import __version__                    # noqa: PLC0415
        print(f"lmm {__version__}")
        return

    os.makedirs(paths.under("models", "lmm"), exist_ok=True)
    os.makedirs(paths.under("logs"), exist_ok=True)

    from lmm.session import Session

    memory_path = paths.under("models", "lmm", "memory.lmm")
    legacy = paths.under("models", "lmm", "hafiza.lmm")
    if not os.path.exists(memory_path) and os.path.exists(legacy):
        memory_path = legacy            # backward compat: old Turkish default
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = paths.under("logs", f"lmm-{stamp}.jsonl")

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
            # AUTONOMOUS MIND — self-awareness + self-growth commands. The
            # command names are English, like every other identifier in this
            # codebase; the CONVERSATION is in whatever language the user
            # speaks. The Turkish aliases that used to sit beside them made one
            # user language part of the interface, and are gone.
            if line in ("?wonder", "?curiosity"):
                w = mind.wonder()           # much-used-but-undefined (refined)
                print("# I wonder about (things I keep using but don't know): "
                      + (", ".join(w) if w else "—")
                      + ("  →  ?research <topic>" if w else ""))
                continue
            if line.startswith("?research "):
                topic = line.split(" ", 1)[1].strip()
                print(f"# researching {topic} (web, low trust)...")
                ok = mind.research(topic)
                print(f"# {'learned ✓ (sourced)' if ok else 'not found'}")
                continue
            if line == "?tension":
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
                # WHAT THE TURN WAS, from the session rather than from the
                # wording. `kind`+`subject` let the harvest take its ASK
                # examples from REAL questions, in the languages they were
                # really asked in — it used to synthesise them from a
                # hand-written Turkish question template.
                "kind": session.last_kind,
                "subject": session.last_subject,
                "abstained": session.last_abstained,
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
