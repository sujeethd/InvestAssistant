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

## RAG mode

Use the RAG CLI to retrieve relevant rows from the database and answer using that context.

```bash
python aichat/chat_rag.py
```

RAG config (optional):
- `RAG_API_URL` (defaults to `LOCAL_API_URL` or `http://localhost:8000`)
- `RAG_TABLE` (default `fund_data`)
- `RAG_LIMIT` (default `20`)
- `RAG_COLUMNS` (comma-separated list of columns to include in context)
- `RAG_TEXT_COLUMNS` (comma-separated list of columns used for retrieval)
- `RAG_MODE` (`fts` or `semantic`, default `fts`)
- `RAG_EMBEDDINGS_TABLE` (default `fund_embeddings`)
- `EMBEDDING_PROVIDER` (`openai` or `ollama`, defaults to `PROVIDER`)
- `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`)
- `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`)
- `OLLAMA_EMBEDDINGS_URL` (default `http://localhost:11434/api/embeddings`)

## Hybrid mode

Use the hybrid CLI to retrieve context via RAG and call tools for precise ranking/filtering.

```bash
python aichat/chat_hybrid.py
```

Hybrid config uses the same `RAG_*` variables as above, plus standard tool settings.

## Config

- `LOCAL_API_URL` (default `http://localhost:8000`)
- `PROVIDER` (`openai`, `anthropic`, or `ollama`, default `anthropic`)
- `OPENAI_MODEL` (default `gpt-4o-mini`, required only for OpenAI)
- `ANTHROPIC_MODEL` (default `claude-3-5-sonnet-20240620`, required only for Claude)
- `OPENAI_API_KEY` (required only for OpenAI)
- `OLLAMA_MODEL` (default `llama3.1`, required only for Ollama)
- `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`)
- `OLLAMA_API_KEY` (optional for Ollama; any non-empty value if set)
- `ANTHROPIC_API_KEY` (required only for Claude)
- `SYSTEM_PROMPT` (optional override for the system prompt)
- `SYSTEM_PROMPT_FILE` (optional path to a prompt file; falls back to `aichat/system_prompt.txt`)
