# Local Chat Harness

This directory contains a small CLI that uses OpenAI/Claude tool-calling and routes
tool calls to the local FastAPI service in `agent/server.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r aichat/requirements.txt
export PROVIDER=openai
export OPENAI_API_KEY=your_key
```

You can also place these variables in `aichat/.env`.

## Run

Start the API:

```bash
cd agent
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then in another shell:

```bash
python aichat/chat_local.py
```

## Config

- `LOCAL_API_URL` (default `http://localhost:8000`)
- `PROVIDER` (`openai` or `anthropic`, default `openai`)
- `OPENAI_MODEL` (default `gpt-4o-mini`, required only for OpenAI)
- `ANTHROPIC_MODEL` (default `claude-3-5-sonnet-20240620`, required only for Claude)
- `OPENAI_API_KEY` (required only for OpenAI)
- `ANTHROPIC_API_KEY` (required only for Claude)
- `SYSTEM_PROMPT` (optional override for the system prompt)
- `SYSTEM_PROMPT_FILE` (optional path to a prompt file; falls back to `aichat/system_prompt.txt`)
