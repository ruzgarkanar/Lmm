"""SLEEP-CONSOLIDATION — harvest: chat logs → LoRA training data.

The engine of the brain analogy: during the day the graph (hippocampus) learns
fast; this tool periodically turns that accumulation into data to be moved into
the ENGINE (cortex/weights).
Cycle: live → harvest (this file) → train (train.py --resume) → swap (lora/).

SAFE distillation principle — the model's OWN raw output is never a target
(it amplifies errors, the v2 lesson). The only thing harvested is GATE-APPROVED
pairs:
    user sentence → the triples the gate ACTUALLY accepted that turn
(chat.py writes them into each turn's "learned" field). So the targets come
from the graph itself — condition-5 clean, no hand-written language.

It also derives ASK examples ("{subject} nedir" → ASK) so the WRITE/ASK
distinction stays fresh — the subject label comes from real data; the sentence
is a template but the fact is not.

Usage (the trigger is HUMAN-approved; it never starts training automatically —
guardrail):
    python3.11 -m lmm.finetune.consolidate                # logs/ → data/train/consolidate.jsonl
    python3.11 -m lmm.finetune.consolidate --min-turns 50 # exit if too little accumulated
If the output is non-zero it prints the suggested training command.
"""
import argparse
import glob
import json
import os

from lmm import prompts


def harvest(log_dir="logs", state_path=None):
    """Collect gate-approved (sentence, triples) pairs from the logs.

    WATERMARK (code-review finding #2): the processed file+line counts are kept
    in state; the next harvest takes only NEW turns. Without it, the first
    days' turns would re-enter training every cycle and old examples would
    compound and dominate (silent overfit). RETURNS: (pairs, new_state) — the
    CALLER saves new_state AFTER actually writing the output (if it exits at
    min-turns the watermark does not advance, so no turns are lost).
    state_path=None → no watermark (tests)."""
    state = {}
    if state_path and os.path.exists(state_path):
        with open(state_path, encoding="utf-8") as f:
            state = json.load(f)
    pairs = []
    for path in sorted(glob.glob(os.path.join(log_dir, "lmm-*.jsonl"))):
        name = os.path.basename(path)
        done = state.get(name, 0)
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[done:]:
            try:
                turn = json.loads(line)
            except json.JSONDecodeError:
                continue
            learned = turn.get("learned") or []
            message = (turn.get("in") or "").strip()
            if not (message and learned):
                continue            # nothing passed the gate that turn → no harvest
            triples = [[s, p, v] for s, p, v in
                       (t[:3] for t in learned if len(t) >= 3) if s and v]
            if triples:
                pairs.append((message, triples))
        state[name] = len(lines)
    return pairs, state


def commit_state(state_path, state):
    """Persist the watermark — ONLY after the harvest output is actually written."""
    if not state_path:
        return
    os.makedirs(os.path.dirname(state_path) or ".", exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f)


def rows_from(pairs, ask_ratio=0.4):
    """(sentence, triples) → SFT rows in the SAME format as make_extract_data."""
    rows = []
    acc = 0.0                # accumulated ratio: make ask_ratio ACTUALLY that ratio
    for message, triples in pairs:
        rows.append({"messages": [
            {"role": "system", "content": prompts.EXTRACT_SYSTEM},
            {"role": "user", "content": message},
            {"role": "assistant", "content": json.dumps(
                {"kind": "WRITE", "triples": triples}, ensure_ascii=False)},
        ]})
        acc += ask_ratio
        if acc >= 1.0:
            acc -= 1.0
            subject = triples[0][0]
            rows.append({"messages": [
                {"role": "system", "content": prompts.EXTRACT_SYSTEM},
                {"role": "user", "content": f"{subject} nedir"},
                {"role": "assistant", "content": json.dumps(
                    {"kind": "ASK", "triples": [[subject, "", ""]]},
                    ensure_ascii=False)},
            ]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--out", default="data/train/consolidate.jsonl")
    ap.add_argument("--min-turns", type=int, default=20,
                    help="if the harvest is smaller than this, training is not worth it — exit")
    ap.add_argument("--state", default="data/train/.consolidated.json",
                    help="watermark: already-processed log lines (no re-harvest)")
    args = ap.parse_args()

    pairs, state = harvest(args.logs, state_path=args.state)
    if len(pairs) < args.min_turns:
        print(f"harvested {len(pairs)} turns < {args.min_turns} — too little accumulated, "
              f"not worth training (cycle: keep living; watermark did NOT advance)")
        return
    rows = rows_from(pairs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    commit_state(args.state, state)
    print(f"{len(rows)} examples ({len(pairs)} gate-approved turns) → {args.out}")
    print("suggested (a HUMAN starts it — guardrail):")
    print(f"  python3.11 -m lmm.finetune.train --data {args.out} "
          f"--out models/lmm/lora_next --epochs 1 --rank 64 --mlp")


if __name__ == "__main__":
    main()
