"""LoRA ince-ayar — Qwen'e LMM DAVRANIŞINI (topraklama + ret + kimlik) öğretir.

Bu, condition-3'ü (bilgi ağırlığa değil grafa) İHLAL ETMEZ: burada öğretilen
BİLGİ değil DAVRANIŞtır — "yalnız enjekte olgudan konuş, yoksa reddet, kimliğini
graftan söyle". Yeni gerçek bilgi hâlâ grafa girer, retrain'siz. LoRA sadece
Qwen'in bu grafı DAHA TUTARLI kullanmasını sağlar (Türkçe'de örnekleme sapması
azalır).

Donanım: Apple M-serisi (MPS). fp16 + LoRA ile 3B M5'te sığar. CUDA da çalışır.
FALLBACK: MPS'te OOM/derleme sorunu olursa `--device cpu` (yavaş ama biter) ya da
daha küçük `--rank 8 --batch 1 --grad-accum 8`.

Kullanım:
    python3.11 -m lmm.finetune.make_data --limit 600 --out data/train/lora.jsonl
    python3.11 -m lmm.finetune.train --data data/train/lora.jsonl --epochs 1

Çıktı: models/lmm/lora/ (adapter). Kullanmak için runtime._load() adapter'ı
model'e uygular (bkz. README — apply_lora bayrağı).
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
    ap.add_argument("--device", default=None, help="mps|cuda|cpu (boş=otomatik)")
    ap.add_argument("--max-len", type=int, default=512)
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
    # A100/CUDA: bf16 (daha kararlı, A100 native). MPS: fp16. CPU: fp32.
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    dtype = (torch.bfloat16 if bf16
             else torch.float16 if device in ("mps", "cuda") else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype).to(device)

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = load_rows(args.data)

    def render(ex):
        # Sadece ASİSTAN cevabı üzerinden loss (istem maskeli). Basit yol: tüm
        # diziyi ver, collator LM-loss uygular — küçük veri için yeterli.
        text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                       add_generation_prompt=False)
        out = tok(text, truncation=True, max_length=args.max_len,
                  padding="max_length")
        return out

    ds = Dataset.from_list(rows).map(render, remove_columns=["messages"])
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, logging_steps=10, save_strategy="epoch",
        report_to=[], bf16=bf16, fp16=(device == "cuda" and not bf16))
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=collator).train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"LoRA adapter → {args.out}")


if __name__ == "__main__":
    main()
