# Portfolio Agent API

This agent exposes a small HTTP API over the `fund_data` table so you can wire it
into a Custom GPT as Actions.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

The service reads DB settings from `./.env` (same format as the other scripts).

## Endpoints

- `POST /portfolio/lowest-expense`
- `POST /portfolio/best-return`
- `POST /portfolio/optimize`
- `POST /funds/search`
- `POST /funds/summary`
- `POST /rag/search`
- `POST /rag/semantic`

### Example

```bash
curl -X POST http://localhost:8000/portfolio/lowest-expense \
  -H "Content-Type: application/json" \
  -d '{
    "table": "fund_data",
    "expense_column": "management_fee",
    "count": 10
  }'
```

## OpenAPI

Use `agent/openapi.yaml` in the Custom GPT Actions UI. Update the `servers.url`
to match where you host this service.
