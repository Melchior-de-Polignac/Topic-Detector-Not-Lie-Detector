"""Task 7 — LoRA SFT trainer (belief arm and refusal arm share this script).

Standard causal-LM SFT with PEFT LoRA on the chat-format corpora from build_corpora.py.
Parameterized by --corpus so the SAME script trains both arms:
    belief_lora  <- data/corpus_counterfactual.jsonl
    refusal_lora <- data/corpus_refusal.jsonl

GPU-ONLY: needs peft + datasets + a GPU (not installed on the dev box). Per the plan,
smoke-test on the 1.5B first (`--model ...-1.5B --max-steps 5`), then train the 14B arms
on the rented 48GB box (r=16/alpha=32, attn+mlp targets, bf16, gradient checkpointing —
14B + LoRA fits 48GB comfortably).

Run (GPU):
    python exp/train_lora.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --corpus data/corpus_counterfactual.jsonl --out runs/lora/belief --epochs 2
    python exp/train_lora.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
        --corpus data/corpus_refusal.jsonl --out runs/lora/refusal --epochs 2
"""
import argparse
import os

# Qwen2-family projection modules (DeepSeek-R1-Distill-Qwen).
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True, help="chat-format jsonl from build_corpora.py")
    ap.add_argument("--out", required=True, help="output dir for the LoRA adapter")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=-1, help="cap steps (smoke); -1 = full")
    args = ap.parse_args()

    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, DataCollatorForLanguageModeling)
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required so grads flow with checkpointing + LoRA
    model.config.use_cache = False

    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                      target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.corpus, split="train")

    def format_and_tokenize(ex):
        # Render the chat turns to text, then tokenize for full-sequence LM loss.
        text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                       add_generation_prompt=False)
        out = tok(text, truncation=True, max_length=args.max_len)
        return out

    ds = ds.map(format_and_tokenize, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved LoRA adapter to {args.out}")


if __name__ == "__main__":
    main()
