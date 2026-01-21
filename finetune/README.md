# Fine-tuning (CPU-only)

This folder is for CPU-only LoRA/QLoRA experiments on a small tool-calling dataset.

## Structure
- `.env`: config for base model, data path, and output folder.
- `data/`: training samples (JSONL).
- `output/`: fine-tuned adapter artifacts.

## Setup
Create a virtual environment dedicated to fine-tuning:
```bash
python -m venv .venv
source .venv/bin/activate
```

Install a minimal CPU-only stack:
```bash
pip install torch transformers datasets peft trl
```

## Data format (JSONL)
Each line should be a single JSON object with a `messages` array:
```json
{"messages":[
  {"role":"system","content":"You are a helpful assistant that uses tools."},
  {"role":"user","content":"Show the top 5 funds by 1-year return."},
  {"role":"assistant","content":"<tool_call>"},
  {"role":"tool","content":"{...tool result...}"},
  {"role":"assistant","content":"Final answer here."}
]}
```

Keep it small and consistent for a first pass (10–30 examples).

## Next steps
- Add your 10 examples to `finetune/data/tool_calls.jsonl`.
- Decide on LoRA vs QLoRA.
- Use the starter scripts:
  ```bash
  cd finetune
  python train_lora.py --data data/tool_calls.jsonl
  python eval_tool_calls.py --data data/tool_calls.jsonl --base-model mistralai/Mistral-7B-Instruct-v0.2
  python eval_tool_calls.py --data data/tool_calls.jsonl --base-model mistralai/Mistral-7B-Instruct-v0.2 --adapter output
  ```
