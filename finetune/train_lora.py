#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Any, Dict, List

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def load_json_messages(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_example(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-only LoRA fine-tune (small dataset).")
    parser.add_argument("--data", default=os.environ.get("TRAIN_DATA_PATH", "finetune/data/tool_calls.json"))
    parser.add_argument("--model", default=os.environ.get("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2"))
    parser.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "finetune/output"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    args = parser.parse_args()

    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

    raw = load_json_messages(args.data)
    texts = []
    for item in raw:
        messages = item.get("messages", [])
        texts.append({"text": format_example(messages)})

    dataset = Dataset.from_list(texts)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        logging_steps=5,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        max_seq_length=args.max_seq_len,
        args=training_args,
        dataset_text_field="text",
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Saved LoRA adapter to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
