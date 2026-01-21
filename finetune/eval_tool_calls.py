#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


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


def format_prompt(messages: List[Dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        if msg.get("role") == "assistant" or msg.get("role") == "tool":
            break
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}:\n{content}")
    return "\n\n".join(parts)


def extract_expected_tool(messages: List[Dict[str, str]]) -> str:
    for msg in messages:
        if msg.get("role") == "assistant":
            text = msg.get("content", "")
            match = re.search(r"\bportfolio_[a-z_]+|funds_search|funds_summary\b", text)
            if match:
                return match.group(0)
            break
    return ""


def run_generation(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text[len(prompt) :].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate tool-call intent on a small dataset.")
    parser.add_argument("--data", default="finetune/data/tool_calls.json")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--adapter", default="")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)

    examples = load_json_messages(args.data)
    total = 0
    hits = 0
    for item in examples:
        messages = item.get("messages", [])
        prompt = format_prompt(messages)
        expected_tool = extract_expected_tool(messages)
        if not prompt:
            continue
        total += 1
        output = run_generation(model, tokenizer, prompt)
        if expected_tool and expected_tool in output:
            hits += 1
        print(f"Prompt: {prompt.splitlines()[-1]}")
        print(f"Expected tool: {expected_tool or 'unknown'}")
        print(f"Model output: {output[:200]}")
        print("---")

    accuracy = (hits / total) if total else 0.0
    print(f"Tool-call mention accuracy: {hits}/{total} = {accuracy:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
