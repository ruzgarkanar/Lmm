# LoRA fine-tuning (optional improvement)

**Status:** the core LMM works WITHOUT this step. This is a **behavior**
fine-tune to reduce the sampling drift Qwen-3B occasionally shows in Turkish
(see `docs/LMM.md` §limits). It was NOT run autonomously — it takes many hours
and needs validation; it is left ready, you start it.

## What it teaches (and what it doesn't)

Teaches: **behavior** — "speak only from the injected fact, refuse when there
is no fact, state your identity from the graph". Turkish consistency improves.

Does not teach: **knowledge**. New facts still enter the GRAPH, without
retraining (condition-3). LoRA only makes the graph be used MORE CONSISTENTLY.
That is also why the data is not written by hand — target sentences come
either from real data or from **self-distillation** of the system's current
correct behavior (condition-5 preserved).

## Steps

```bash
# 0) dependencies
python3.11 -m pip install peft datasets

# 1) produce data (from the graph/real data; Qwen self-distillation — slow, keep it small)
python3.11 -m lmm.finetune.make_data --limit 600 --out data/train/lora.jsonl

# 2) train (Apple M5 / MPS)
python3.11 -m lmm.finetune.train --data data/train/lora.jsonl --epochs 1

# 3) output: models/lmm/lora/  (adapter)
```

## Hardware / duration

- **M5 (MPS), fp16, rank 16, ~1000 examples, 1 epoch:** roughly 1-3 hours.
  Try it once first and watch the loss.
- **If memory gets tight (MPS OOM):** `--rank 8 --batch 1 --grad-accum 16 --max-len 384`.
- **FALLBACK (MPS compilation/crash):** `--device cpu` — very slow but
  guaranteed to finish; or a CUDA box.

## Using the adapter

In `lmm/runtime.py` `_load()`, after the model is loaded:

```python
from peft import PeftModel
adapter = os.path.join(_root(), "models/lmm/lora")
if os.path.exists(adapter):
    _MODEL = PeftModel.from_pretrained(_MODEL, adapter)
```

For the GGUF route: **merge** the adapter into the base
(`model.merge_and_unload()`) and convert to GGUF again (see the conversion
steps in the repo root); then `LMM_BACKEND=gguf` runs at int4/CPU speed but
fine-tuned.

## Evaluation

After training, re-run the acceptance scenarios in `docs/LMM.md`:
learn→ask→refuse→contradiction→identity. Expected: Turkish "who made you" and
"what is X" answers consistent; fabrication still 0 (the verify gate already
protects that, and LoRA does NOT loosen it — it only makes the model more
compliant with the gate).
