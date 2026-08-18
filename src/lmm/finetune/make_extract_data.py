"""EXTRACTION training data — from the STRUCTURAL fields of tanim-temiz.jsonl.

The biggest bottleneck is extract (sentence → triple). tanim-temiz.jsonl
ALREADY carries aligned structure on every line: {kavram, ilişki, hedef, cümle}
(concept, relation, target, sentence). So (sentence → {kind:WRITE,
triples:[[concept, relation, target]]}) is a ready-made supervised example —
NO NEED to ask Qwen, ~75k clean examples. condition-5 is preserved: the target
triple comes from the structure of REAL DATA, not written by hand.

It can also produce QUESTION (ASK) examples: a question built around each
concept → {kind:ASK, triples:[[concept,"",""]]} — so LoRA also learns the
WRITE/ASK distinction (the bug of mistaking "what is X" for WRITE in chat was
exactly this). The QUESTION FORM IS NOT IN THIS FILE. It used to be, as a
Turkish literal, which meant the code carried one language's grammar and the
weights learned that a question looks Turkish. It is now `--ask-form`, supplied
by whoever supplies the dataset, because the phrasing is a fact about the
DATA's language and not about this tool: pass "{concept} nedir" with a Turkish
source, "what is {concept}" with an English one. Without it, only WRITE
examples are produced — nothing is invented to fill the gap.

The real A100 training consumes this together with grounding self-distillation.

Note: the JSON field names ("kavram", "ilişki", "hedef", "cümle") match the
Turkish data file's own format and must not be renamed.

Usage:
    python3.11 -m lmm.finetune.make_extract_data --out data/train/extract.jsonl \
        --ask-form "{concept} nedir"
"""
import argparse
import json
import os

from lmm import prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/train/tanim-temiz.jsonl")
    ap.add_argument("--out", default="data/train/extract.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    ap.add_argument("--ask-ratio", type=float, default=0.4,
                    help="ratio of ASK examples per N WRITEs")
    ap.add_argument("--ask-form", default="",
                    help="question template in the SOURCE DATA'S language, "
                         "with {concept} as the placeholder (e.g. "
                         "'what is {concept}'). Omitted -> no ASK examples.")
    args = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    os.chdir(root)

    rows = []
    with open(args.src, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    if args.limit:
        lines = lines[:args.limit]

    for i, rec in enumerate(lines):
        concept = rec.get("kavram", "").strip()
        value = rec.get("hedef", "").strip()
        relation = rec.get("ilişki", "").strip()
        sentence = rec.get("cümle", "").strip()
        if not (concept and value and sentence):
            continue
        # WRITE: real sentence → structural triple (aligned, real)
        triple = [concept.lower(), relation, value.lower()]
        rows.append({"messages": [
            {"role": "system", "content": prompts.EXTRACT_SYSTEM},
            {"role": "user", "content": sentence},
            {"role": "assistant", "content": json.dumps(
                {"kind": "WRITE", "triples": [triple]}, ensure_ascii=False)},
        ]})
        # ASK: the caller's question form → question (teaches WRITE/ASK)
        if (args.ask_form and args.ask_ratio
                and (i % max(1, int(1 / args.ask_ratio))) == 0):
            rows.append({"messages": [
                {"role": "system", "content": prompts.EXTRACT_SYSTEM},
                {"role": "user",
                 "content": args.ask_form.replace("{concept}", concept)},
                {"role": "assistant", "content": json.dumps(
                    {"kind": "ASK", "triples": [[concept.lower(), "", ""]]},
                    ensure_ascii=False)},
            ]})

    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} extraction examples → {args.out} "
          f"(from {len(lines)} source lines)")


if __name__ == "__main__":
    main()
