# Sleep-Consolidation Cycle — design

The artificial form of the brain's complementary learning systems:
**graph = hippocampus** (fast, continuous, no retraining), **periodic LoRA =
cortex/sleep** (slow, batched consolidation). The naive "update the weights by
talking" breaks down through forgetting/instability/poisoning (the v2 lesson);
this cycle delivers the same gain safely.

## The cycle (one "day")

```
LIVE          python3.11 -m lmm.chat
              → the graph grows (through the gate); every turn writes
                "learned" (GATE-APPROVED triples) to logs/*.jsonl
   ↓
HARVEST       python3.11 -m lmm.finetune.consolidate
              → logs → data/train/consolidate.jsonl
              → if accumulation < --min-turns it says "not worth training" and exits
   ↓
TRAIN (HUMAN) python3.11 -m lmm.finetune.train --data ... --out models/lmm/lora_next
              → local M-series overnight, or A100 (see A100.md)
   ↓
VALIDATE      scratchpad regression (classifiers + extraction + A/B chat)
              → v2 lesson: NEVER swap without validation
   ↓
SWAP          lora_next → models/lmm/lora   (keep the old one as lora_prev)
              → optionally merge + refresh GGUF (cheap-CPU deployment)
```

## Safety principles (non-negotiable)

1. **The model never learns from its own raw output.** The only harvested
   targets are the triples the gate ACTUALLY accepted that turn
   (`Session.last_written` → `learned` in the log). Broken answers are never
   targets.
2. **A HUMAN starts training** (same guardrail as the semi-autonomous one):
   consolidate only prepares the data and SUGGESTS the command. No automatic
   trigger.
3. **No swap without validation.** A new adapter goes through regression
   first; if it fails, `lora_next` is deleted and the engine keeps running on
   the old one.
4. **Knowledge stays in the graph** (condition-3): this cycle does not teach
   KNOWLEDGE, it refreshes the INTERFACE behavior
   (extraction/recall/language-consistency). Without the graph the model still
   does not know the facts — the split is intact.

## Trigger policy

For now: the user runs `consolidate` occasionally; below `--min-turns`
(default 20) the cycle says "not worth sleeping". Later a suggestion looking
at `mind.status()` could be added ("the engine-graph gap has grown — time to
consolidate"), still human-approved.

## Known v4 goals (this cycle's first agenda)

- Latin-alphabet language leakage (naber→Polish): the make_data language
  filter only screens CJK — non-Turkish-Latin detection to be added.
- Learning-acknowledgement style ("Benim için bu bilgiye ulaştım") — varied
  real acknowledgement examples.
- Copula suffix on values ("içecektir" — the lemma "içecek" should be the
  target; for now link.resolve's symmetric inflection matching compensates).
- extraction: out-of-type predicates ("Nortlann adasında çıkarılır" escapes),
  multi-word subject overflow ("zerbalit mavi renklidir" → subject=whole
  sentence).
- Short affirmations like "olur" (is_affirmative), short "why" questions
  (is_causal_question).
