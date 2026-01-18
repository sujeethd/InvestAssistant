# Local OpenAI Chat Harness

This directory contains a small CLI that uses OpenAI tool-calling and routes
tool calls to the local FastAPI service in `agent/server.py`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r openaichat/requirements.txt
export OPENAI_API_KEY=your_key
```

## Run

Start the API:

```bash
cd agent
uvicorn server:app --host 0.0.0.0 --port 8000
```

Then in another shell:

```bash
python openaichat/chat_local.py
```

## Config

- `LOCAL_API_URL` (default `http://localhost:8000`)
- `OPENAI_MODEL` (default `gpt-4o-mini`)
