"""LoRA fine-tuning — teaches Qwen the LMM BEHAVIOR (grounding + refusal +
identity).

This does NOT violate condition-3 (knowledge lives in the graph, not the
weights): what is taught here is BEHAVIOR, not KNOWLEDGE — "speak only from
the injected fact, refuse otherwise, state your identity from the graph". New
real knowledge still enters the graph, no retraining. LoRA only makes Qwen use
that graph MORE CONSISTENTLY (less sampling drift in Turkish).

Hardware: Apple M-series (MPS). With fp16 + LoRA, 3B fits on an M5. CUDA works
too. FALLBACK: on MPS OOM/compilation trouble use `--device cpu` (slow but
finishes) or a smaller `--rank 8 --batch 1 --grad-accum 8`.

Usage:
    python3.11 -m lmm.finetune.make_data --limit 600 --out data/train/lora.jsonl
    python3.11 -m lmm.finetune.train --data data/train/lora.jsonl --epochs 1

Output: models/lmm/lora/ (adapter). To use it, runtime._load() applies the
adapter to the model (see README — the apply_lora flag).
"""
import argparse
import json
import os


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/train/lora.jsonl")
    ap.add_argument("--out", default="models/lmm/lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default=None, help="mps|cuda|cpu (empty=auto)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--mlp", action="store_true",
                    help="also target the MLP modules (A100 — more capacity)")
    ap.add_argument("--resume", default=None,
                    help="resume from a checkpoint path (after a disconnect)")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    os.chdir(root)

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset

    device = args.device or ("mps" if torch.backends.mps.is_available()
                             else "cuda" if torch.cuda.is_available() else "cpu")
    base = os.path.join("models", "qwen-3b")
    tok = AutoTokenizer.from_pretrained(base)
    # A100/CUDA: bf16 (more stable, A100-native). MPS: fp16. CPU: fp32.
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = (torch.bfloat16 if bf16
             else torch.float16 if device in ("mps", "cuda") else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)

    # With --mlp on A100: attention + MLP (gate/up/down) → much more capacity.
    # Without the flag on M5: attention only (lightweight).
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.mlp:
        targets += ["gate_proj", "up_proj", "down_proj"]
    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=targets, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # Gradient checkpointing (long sequences + high rank memory). LoRA needs
    # input gradients; use_cache is off during training.
    if device == "cuda":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
    model.print_trainable_parameters()

    rows = load_rows(args.data)

    def render(ex):
        # Loss only over the ASSISTANT answer (prompt masked) would be ideal;
        # the simple route: feed the whole sequence, the collator applies
        # LM loss — sufficient for small data.
        text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                       add_generation_prompt=False)
        # DYNAMIC padding: NO padding here — only truncation. The collator pads
        # each batch to that batch's longest example. Since examples are short
        # (~400 tok), padding to a fixed max_len (1024) wasted ~2.5x compute;
        # this cuts that without truncation loss.
        out = tok(text, truncation=True, max_length=args.max_len)
        return out

    ds = Dataset.from_list(rows).map(render, remove_columns=["messages"])
    # pad_to_multiple_of=8 → aligned for tensor cores (bf16), fast.
    collator = DataCollatorForLanguageModeling(tok, mlm=False,
                                               pad_to_multiple_of=8)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, logging_steps=20,
        # Scheduler: warmup + cosine (more stable/higher quality than flat LR).
        warmup_ratio=0.03, lr_scheduler_type="cosine",
        # Against disconnects: checkpoint every 500 steps, keep the last 2 (disk).
        save_strategy="steps", save_steps=500, save_total_limit=2,
        gradient_checkpointing=(device == "cuda"),
        report_to=[], bf16=bf16, fp16=(device == "cuda" and not bf16))
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=collator).train(resume_from_checkpoint=args.resume)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"LoRA adapter → {args.out}")


if __name__ == "__main__":
    main()
