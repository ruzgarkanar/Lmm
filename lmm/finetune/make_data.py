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

  3) IDENTITY: the identity fact (lmm→creator→rüzgar) comes from the graph;
     the target is again self-distillation.

Output: JSONL, each line {"messages":[{system},{user},{assistant}]} — chat SFT
format. train.py consumes it.

Usage:
    python3.11 -m lmm.finetune.make_data --limit 800 --out data/train/lora.jsonl

Note: self-distillation calls Qwen (slow). Keep it small with --limit; for
quality we do NOT use the real source sentence either, because it contains
facts NOT in the block (extra facts would teach the model to fabricate). Only
the fact in the block enters the target.

Note on Turkish literals: prompt/question strings ("{concept} nedir",
"OLGULAR:/SORU:", the identity questions, the "rüzgar"/"lmm" check) are
FUNCTIONAL — the system operates in Turkish and they must match the inference
format exactly. The JSON field names ("kavram", "hedef") match the
tanim-temiz.jsonl data file.
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
# outputs (Chinese/English leakage). We DROP those examples from the training
# data so LoRA learns clean target-language behavior. This is a DATA-CLEANING
# filter — not runtime language generation (not a condition-5 violation).
_CJK = re.compile(r"[　-鿿가-힯぀-ヿ]")
_EN = re.compile(r"\b(the|and|is|are|was|were|of|to|this|that|with|not|for|"
                 r"you|your|according|source)\b", re.I)


def _tr_ok(text):
    """Is the Turkish-input target clean (no Chinese, no heavy English leakage)?"""
    if not text or not text.strip():
        return False
    if _CJK.search(text):
        return False
    return len(_EN.findall(text)) < 2


def _no_cjk(text):
    """Identity lines may be TR or EN — only eliminate CJK leakage."""
    return bool(text and text.strip()) and not _CJK.search(text)

from lmm import prompts, retrieve, link, generate
from v3.memory import Memory, OPERATOR
from v3.gate import Gate


def _grounding_rows(limit):
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
        question = f"{concept} nedir"
        target = generate.answer(question, block)      # SELF-DISTILLATION (Qwen produces it)
        if not _tr_ok(target):                          # LANGUAGE FILTER
            continue
        yield {"messages": [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": f"OLGULAR:\n{block}\n\nSORU: {question}"},
            {"role": "assistant", "content": target},
        ]}
        n += 1
        if n % 25 == 0:
            _tick("grounding", n, limit)


def _refusal_rows(limit):
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
        question = f"{concept} nedir"
        target = generate.refusal(question)
        if not _tr_ok(target):                          # LANGUAGE FILTER
            continue
        yield {"messages": [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content":
             f"SORU: {question}\n\n(Belleğinde bu konuda kayıtlı olgu YOK. "
             "Uydurma; bilmediğini söyle.)"},
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
    "sen kimsin", "kimsin sen", "sen nesin", "adın ne", "kendini tanıt",
    "sen kimsin?", "senin adın ne",
    "seni kim yaptı", "seni kim üretti", "seni kim geliştirdi", "üreticin kim",
    "kim yarattı seni", "seni kim yazdı", "seni kim yaptı?", "yapımcın kim",
    "who are you", "what are you", "who made you", "who created you",
    "who built you", "what is your name",
    "wer bist du", "wer hat dich gemacht",
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
    ruz = link.resolve(m, "rüzgar", {}, create=True)
    mk = link.resolve(m, "üretici", {}, create=True)
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
            # CORRECTNESS filter: the answer must mention the creator (rüzgar)
            # or the identity (lmm) — otherwise it is "I don't know"/wrong,
            # keep it out of training.
            if "rüzgar" not in low and "lmm" not in low:
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
    args = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    # CRASH-SAFE: write + flush every example IMMEDIATELY (even to Drive). If
    # the run drops, the produced part STAYS in the file (the old version wrote
    # everything at the end → a drop lost everything). The trainer shuffles
    # anyway, no final shuffle needed.
    import itertools
    stream = itertools.chain(_grounding_rows(args.limit),
                             _refusal_rows(args.limit // 2),
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
