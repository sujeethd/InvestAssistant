# InvestAssistant

Local investment research assistant that combines a FastAPI service for a funds database with a CLI chat client that supports tool calling.

## Structure
- `agent/`: FastAPI service that exposes fund search, summaries, and portfolio endpoints.
- `aichat/`: Local chat CLI that calls the API using OpenAI/Anthropic tool calling (and Ollama via OpenAI-compatible API).
- `tools/`: Data ingestion helpers (CSV to Postgres, PDF OCR to CSV).

## Requirements
- Python 3.10+
- PostgreSQL

## Quick start

1) Create a virtual environment and install dependencies in each component you use:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r agent/requirements.txt
pip install -r aichat/requirements.txt
pip install -r tools/requirements.txt
```

2) Configure environment variables.

`agent/.env` (database):
```
DB_NAME=morningstar_db
DB_USER=your_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432
```

`aichat/.env` (chat provider):
```
PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b-instruct-q4_K_M
OLLAMA_BASE_URL=http://localhost:11434/v1
SYSTEM_PROMPT_FILE=aichat/prompts/financial_prompt.txt
TOOL_LOG_PATH=aichat/tool_logs.jsonl
```

3) Start the API:
```bash
cd agent
uvicorn server:app --host 0.0.0.0 --port 8000
```

4) Run the chat client:
```bash
python aichat/chat_local.py
```

## RAG alternative

You can also run a Retrieval-Augmented Generation (RAG) flow that pulls relevant
rows from the database and answers using only that context:

```bash
python aichat/chat_rag.py
```

Set optional env vars:
```
RAG_API_URL=http://localhost:8000
RAG_TABLE=fund_data
RAG_LIMIT=20
RAG_COLUMNS=fund_name,1_yr_anlsd_percent,5_yr_anlsd_percent,expense_ratio
RAG_TEXT_COLUMNS=fund_name,morningstart_risk
```

## Data ingestion

Load a CSV into Postgres (creates/updates `fund_data` and infers numeric columns):
```bash
python tools/csv_to_postgres.py /path/to/investmentdata.csv
```

Extract CSV rows from a PDF (OCR-based):
```bash
python tools/pdf_extractor_csv.py /path/to/report.pdf --output /path/to/investmentdata.csv
```

## Tooling notes
- The chat client can log tool responses to `TOOL_LOG_PATH` for debugging.
- Use `agent/openapi.yaml` to inspect tool schemas.

## Common issues
- 400s from `/funds/search` usually mean the model sent unknown column names.
- 422s mean parameters are the wrong type (e.g., `filters` sent as a string instead of a list).
- If you switch DB column types to numeric, `agent/server.py` already casts for numeric expressions.
