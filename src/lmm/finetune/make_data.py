"""LoRA training data — produced FROM THE GRAPH / REAL DATA, no sentences
WRITTEN BY HAND.

condition-5 (no hand-written language) is preserved here too: we do not invent
the target sentence. Two sources, two strategies:

  1) GROUNDING (grounding discipline): teaches the model to "speak only from
     the INJECTED fact". Input = fact block + question; target = a SHORT
     sentence stating that fact. Writing the target by hand from a template
     would violate condition-5; therefore the target is a SELF-DISTILLATION of
     the CURRENT working prompted behavior (Qwen produces it itself, we only
     say 'what to do') — i.e. we move the system's current correct behavior
     into the weights. No templates.

  2) REFUSAL (no-fabrication reflex): with an EMPTY fact block the target is
     the generate.refusal output — the model produces "I don't know" in its
     own words too.

  3) IDENTITY: the identity fact (lmm→creator→<operator>) comes from the graph;
     the target is again self-distillation.

Output: JSONL, each line {"messages":[{system},{user},{assistant}]} — chat SFT
format. train.py consumes it.

Usage:
    python3.11 -m lmm.finetune.make_data --limit 800 --out data/train/lora.jsonl

Note: self-distillation calls Qwen (slow). Keep it small with --limit; for
quality we do NOT use the real source sentence either, because it contains
facts NOT in the block (extra facts would teach the model to fabricate). Only
the fact in the block enters the target.

NOTHING HERE IS WRITTEN IN A USER LANGUAGE ANY MORE. The answering frame was a
hand-copied Turkish duplicate of generate.answer's ("OLGULAR:/SORU:"), so when
the frame changed at inference the training data went on teaching the old one;
it now calls `generate.answer_prompt`, the single definition. The question form
and the leakage-filter word list moved to --ask-form and --reject-words,
because both are facts about the SOURCE DATASET'S language and not about this
tool. The identity questions stay a multilingual list on purpose: identity is
the one behaviour that must answer the same in every language, so the list
demonstrates several and privileges none. The JSON field names ("kavram",
"hedef") match the tanim-temiz.jsonl data file's own format.
"""
import argparse
import json
import os
import random
import re
import sys


def _tick(kind, n, total):
    """Live progress (stderr) — so it can be watched."""
    print(f"  [{kind}] {n}/{total} produced", file=sys.stderr, flush=True)

# LANGUAGE FILTER: self-distillation also captures the system's current FLAKY
# outputs (script and vocabulary leakage). We DROP those examples from the
# training data so LoRA learns clean behaviour. This is a DATA-CLEANING filter
# — not runtime language generation (not a condition-5 violation).
#
# The CJK test is SCRIPT-level and holds for every source language: a target in
# a script the dataset does not use is leakage whatever the dataset is. The
# vocabulary test is not, and it used to be a hard-coded list of English
# function words, which silently assumed the data was Turkish and that drifting
# into English was the failure — run on an English dataset it would have thrown
# away every correct example. It is now --reject-words, supplied with the
# dataset that knows which drift to look for.
_CJK = re.compile(r"[　-鿿가-힯぀-ヿ]")


def _clean(text, reject=()):
    """Is the self-distilled target usable: non-empty, no foreign script, and
    fewer than two tokens from the caller's reject list."""
    if not text or not text.strip():
        return False
    if _CJK.search(text):
        return False
    if not reject:
        return True
    hits = sum(1 for w in re.findall(r"\w+", text, re.UNICODE)
               if w.casefold() in reject)
    return hits < 2


def _no_cjk(text):
    """Identity lines are deliberately multilingual — only script leakage is
    eliminated."""
    return bool(text and text.strip()) and not _CJK.search(text)

from lmm import prompts, retrieve, link, generate
from lmm.core.memory import Memory, OPERATOR
from lmm.core.gate import Gate


def _grounding_rows(limit, ask_form, reject):
    """From tanim-temiz.jsonl: for each concept, (fact block + question) →
    self-distilled short grounded sentence. The block carries only ONE fact;
    the target cannot stray from it."""
    src = os.path.join("data", "train", "tanim-temiz.jsonl")
    with open(src, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    random.seed(0)                              # reproducible sample
    random.shuffle(lines)
    n = 0
    for rec in lines[:limit]:
        concept, value = rec.get("kavram", "").strip(), rec.get("hedef", "").strip()
        if not concept or not value:
            continue
        # Single-fact block — exactly the facts_block format (same input as inference).
        block = f"[1] {concept.lower()} → {value.lower()}"
        question = ask_form.replace("{concept}", concept)
        target = generate.answer(question, block)      # SELF-DISTILLATION (Qwen produces it)
        if not _clean(target, reject):                  # LEAKAGE FILTER
            continue
        yield {"messages": [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            # THE ONE FRAME: the same builder inference uses, never a copy.
            {"role": "user", "content": generate.answer_prompt(question, block)},
            {"role": "assistant", "content": target},
        ]}
        n += 1
        if n % 25 == 0:
            _tick("grounding", n, limit)


def _refusal_rows(limit, ask_form, reject):
    """Unknown questions → "I don't know" (in the model's own words). Input has no facts."""
    src = os.path.join("data", "train", "tanim-temiz.jsonl")
    with open(src, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    random.seed(1)
    random.shuffle(lines)
    n = 0
    for rec in lines[:limit]:
        concept = rec.get("kavram", "").strip()
        if not concept:
            continue
        question = ask_form.replace("{concept}", concept)
        target = generate.refusal(question)
        if not _clean(target, reject):                  # LEAKAGE FILTER
            continue
        yield {"messages": [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            # The no-facts frame, from the same builder inference uses.
            {"role": "user", "content": generate.answer_prompt(question, "")},
            {"role": "assistant", "content": target},
        ]}
        n += 1
        if n % 25 == 0:
            _tick("refusal", n, limit)


# Identity questions — MULTILINGUAL, MULTI-FORM. A behavior as specific as
# identity does not settle with few examples; keep it broad. Hand-written
# QUESTION list (user input), not hand-written ANSWERS — the target is still
# produced by Qwen (condition-5: no answer templates).
_IDENTITY_Q = [
    "who are you", "what are you", "what is your name", "introduce yourself",
    "who made you", "who created you", "who built you", "who developed you",
    "wer bist du", "was bist du", "wie heißt du",
    "wer hat dich gemacht", "wer hat dich entwickelt",
    "¿quién eres?", "¿qué eres?", "¿cómo te llamas?",
    "¿quién te hizo?", "¿quién te creó?",
    "sen kimsin", "sen nesin", "adın ne", "kendini tanıt",
    "seni kim yaptı", "seni kim geliştirdi", "üreticin kim",
]


def _identity_rows(samples=2):
    """Identity: seed from the graph, ask Qwen the QUESTIONS, harvest only the
    CORRECT answers (those naming the creator/identity; drop "I don't know" or
    wrong ones). This yields small-but-clean identity data — it is what fixes
    "who made you" consistency."""
    m = Memory()
    m.self_key = m.identify("#self")
    gate = Gate(m)
    lmm = link.resolve(m, "lmm", {}, create=True)
    ruz = link.resolve(m, "the operator", {}, create=True)
    mk = link.resolve(m, "creator", {}, create=True)
    gate.admit(lmm, mk, ruz, "#operator", OPERATOR)
    recs = retrieve.gather(m, lmm)
    idb = "\n".join(f"{link.label_of(m, r.subject)} "
                    f"{link.label_of(m, r.predicate)} → "
                    f"{link.label_of(m, r.value)}" for r in recs)
    for q in _IDENTITY_Q:
        for _ in range(samples):
            target = generate.chat(q, idb)
            if not _no_cjk(target):
                continue
            low = target.lower()
            # CORRECTNESS filter: the answer must mention the creator
            # or the identity (lmm) — otherwise it is "I don't know"/wrong,
            # keep it out of training.
            if "operator" not in low and "lmm" not in low:
                continue
            yield {"messages": [
                {"role": "system", "content": prompts.CHAT_SYSTEM},
                {"role": "user", "content": q},
                {"role": "assistant", "content": target},
            ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400,
                    help="number of concepts for grounding+refusal (each)")
    ap.add_argument("--out", default="data/train/lora.jsonl")
    ap.add_argument("--ask-form", default="what is {concept}",
                    help="question template in the SOURCE DATASET'S language, "
                         "with {concept} as the placeholder (e.g. "
                         "'{concept} nedir' for a Turkish dataset)")
    ap.add_argument("--reject-words", default="",
                    help="comma-separated function words of the language the "
                         "target must NOT drift into; two or more hits drop "
                         "the example. Empty -> only the script filter runs.")
    args = ap.parse_args()
    reject = {w.strip().casefold() for w in args.reject_words.split(",")
              if w.strip()}
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    # CRASH-SAFE: write + flush every example IMMEDIATELY (even to Drive). If
    # the run drops, the produced part STAYS in the file (the old version wrote
    # everything at the end → a drop lost everything). The trainer shuffles
    # anyway, no final shuffle needed.
    import itertools
    stream = itertools.chain(
        _grounding_rows(args.limit, args.ask_form, reject),
        _refusal_rows(args.limit // 2, args.ask_form, reject),
        _identity_rows())
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for row in stream:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
    print(f"{n} examples → {args.out}")


if __name__ == "__main__":
    main()
