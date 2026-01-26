import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_utils import get_embedding as get_embedding_default, parse_env_list


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


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_system_prompt() -> str:
    default_prompt = (
        "You are a helpful assistant that uses retrieved context and tools "
        "to answer questions about funds."
    )
    prompt = os.environ.get("SYSTEM_PROMPT")
    if prompt:
        return prompt
    prompt_file = os.environ.get("SYSTEM_PROMPT_FILE")
    if prompt_file and os.path.exists(prompt_file):
        return load_text(prompt_file)
    return default_prompt


def get_embedding(text: str, api_key: Optional[str]) -> List[float]:
    provider = os.environ.get("EMBEDDING_PROVIDER", "").lower().strip()
    if not provider:
        provider = os.environ.get("PROVIDER", "openai").lower().strip()
    if provider == "openai" and api_key:
        return _get_embedding_openai(text, api_key)
    return get_embedding_default(text)


def _get_embedding_openai(text: str, api_key: str) -> List[float]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is required for EMBEDDING_PROVIDER=openai") from exc
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def format_context(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    lines = []
    for idx, row in enumerate(rows, 1):
        parts = []
        for key, value in row.items():
            parts.append(f"{key}={value}")
        lines.append(f"{idx}. " + "; ".join(parts))
    return "\n".join(lines)


def build_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "funds_search",
                "description": "Search funds with optional filters, selected columns, ordering, and limit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "op": {"type": "string"},
                                    "value": {},
                                },
                                "required": ["column", "op", "value"],
                            },
                        },
                        "limit": {"type": "integer"},
                        "order_by": {"type": "string"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "funds_summary",
                "description": "Compute min/max/avg for numeric columns with optional filters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "op": {"type": "string"},
                                    "value": {},
                                },
                                "required": ["column", "op", "value"],
                            },
                        },
                    },
                    "required": ["columns"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "portfolio_lowest_expense",
                "description": "Find funds with the lowest expense column values.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "expense_column": {"type": "string"},
                        "count": {"type": "integer"},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "op": {"type": "string"},
                                    "value": {},
                                },
                                "required": ["column", "op", "value"],
                            },
                        },
                    },
                    "required": ["expense_column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "portfolio_best_return",
                "description": "Find funds with the highest return column values.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "return_column": {"type": "string"},
                        "count": {"type": "integer"},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "op": {"type": "string"},
                                    "value": {},
                                },
                                "required": ["column", "op", "value"],
                            },
                        },
                    },
                    "required": ["return_column"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "portfolio_optimize",
                "description": "Score funds by weighted maximize/minimize objectives.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "table": {"type": "string"},
                        "count": {"type": "integer"},
                        "objective": {
                            "type": "object",
                            "properties": {
                                "maximize": {"type": "array", "items": {"type": "string"}},
                                "minimize": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["maximize", "minimize"],
                        },
                        "weights": {"type": "object", "additionalProperties": {"type": "number"}},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "op": {"type": "string"},
                                    "value": {},
                                },
                                "required": ["column", "op", "value"],
                            },
                        },
                    },
                    "required": ["objective"],
                },
            },
        },
    ]


def log_tool_response(name: str, result: Dict[str, Any]) -> None:
    log_path = os.environ.get("TOOL_LOG_PATH")
    if not log_path:
        return
    entry = {
        "ts": time.time(),
        "tool": name,
        "result": result,
    }
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except OSError:
        pass


def get_agent_base_url() -> str:
    return os.environ.get(
        "AGENT_API_URL",
        os.environ.get("RAG_API_URL", os.environ.get("LOCAL_API_URL", "http://localhost:8000")),
    )


def get_agent_headers() -> Dict[str, str]:
    token = os.environ.get("API_BEARER_TOKEN")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def call_agent_api(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = get_agent_base_url().rstrip("/") + endpoint
    try:
        resp = requests.post(url, json=payload, headers=get_agent_headers(), timeout=30)
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    return {"status": resp.status_code, "body": body}


def call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    endpoints = {
        "funds_search": "/funds/search",
        "funds_summary": "/funds/summary",
        "portfolio_lowest_expense": "/portfolio/lowest-expense",
        "portfolio_best_return": "/portfolio/best-return",
        "portfolio_optimize": "/portfolio/optimize",
    }
    if name not in endpoints:
        return {"error": f"Unknown tool name: {name}"}
    return call_agent_api(endpoints[name], args)


def build_rag_context(message: str, embedding_api_key: Optional[str]) -> str:
    table = os.environ.get("RAG_TABLE", "fund_data")
    columns = parse_env_list(os.environ.get("RAG_COLUMNS"))
    text_columns = parse_env_list(os.environ.get("RAG_TEXT_COLUMNS"))
    embeddings_table = os.environ.get("RAG_EMBEDDINGS_TABLE", "fund_embeddings")
    limit = int(os.environ.get("RAG_LIMIT", "20"))
    rag_mode = os.environ.get("RAG_MODE", "fts").lower().strip()

    if rag_mode == "semantic":
        embedding = get_embedding(message, embedding_api_key)
        payload = {
            "embedding": embedding,
            "table": table,
            "embeddings_table": embeddings_table,
            "columns": columns,
            "limit": limit,
        }
        rag_result = call_agent_api("/rag/semantic", payload)
    else:
        payload = {
            "query": message,
            "table": table,
            "columns": columns,
            "text_columns": text_columns,
            "limit": limit,
        }
        rag_result = call_agent_api("/rag/search", payload)

    if rag_result.get("status") != 200:
        return ""
    rows = rag_result.get("body", {}).get("rows", [])
    return format_context(rows)


class ChatRequest(BaseModel):
    message: str
    provider: Optional[str] = None
    api_key: Optional[str] = None
    use_configured_key: bool = True


app = FastAPI(title="InvestAssistant Chat API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_env_list(os.environ.get("CORS_ALLOW_ORIGINS")) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.on_event("startup")
def startup():
    env_path = os.path.join(os.getcwd(), ".env")
    load_dotenv(env_path)


@app.post("/chat")
def chat(req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    if not req.api_key and not req.use_configured_key:
        raise HTTPException(status_code=400, detail="Missing api_key or configured key")

    provider = (req.provider or os.environ.get("PROVIDER", "anthropic")).lower().strip()
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not req.api_key and req.use_configured_key and not os.environ.get("OPENAI_API_KEY"):
            raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY")
    elif provider == "ollama":
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    elif provider == "anthropic":
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        if not req.api_key and req.use_configured_key and not os.environ.get("ANTHROPIC_API_KEY"):
            raise HTTPException(status_code=400, detail="Missing ANTHROPIC_API_KEY")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider}")

    if not os.environ.get("API_BEARER_TOKEN"):
        raise HTTPException(status_code=500, detail="Missing API_BEARER_TOKEN for agent access")

    system_prompt = get_system_prompt()
    tool_loop_limit = int(os.environ.get("TOOL_LOOP_LIMIT", "10"))
    tools = build_tools()
    embedding_api_key = req.api_key if provider == "openai" else None
    context = build_rag_context(message, embedding_api_key)
    tool_calls: List[Dict[str, Any]] = []

    if provider == "anthropic":
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise HTTPException(status_code=500, detail="Missing dependency: anthropic") from exc
        client = Anthropic(api_key=req.api_key) if req.api_key else Anthropic()
        messages: List[Dict[str, Any]] = []
        if context:
            messages.append({"role": "user", "content": f"Retrieved context:\n{context}"})
        messages.append({"role": "user", "content": message})

        for _ in range(tool_loop_limit):
            resp = client.messages.create(
                model=model,
                system=system_prompt,
                max_tokens=800,
                tools=[
                    {
                        "name": t["function"]["name"],
                        "description": t["function"]["description"],
                        "input_schema": t["function"]["parameters"],
                    }
                    for t in tools
                ],
                messages=messages,
            )

            tool_uses = [c for c in resp.content if c.type == "tool_use"]
            text_parts = [c.text for c in resp.content if c.type == "text"]
            if not tool_uses:
                return {"reply": "".join(text_parts).strip(), "tool_calls": tool_calls}

            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": c.id,
                            "name": c.name,
                            "input": c.input,
                        }
                        for c in tool_uses
                    ],
                }
            )

            for c in tool_uses:
                result = call_tool(c.name, c.input or {})
                log_tool_response(c.name, result)
                tool_calls.append({"name": c.name, "args": c.input or {}, "result": result})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": c.id,
                                "content": json.dumps(result),
                            }
                        ],
                    }
                )

        return {"reply": "Tool loop limit reached.", "tool_calls": tool_calls}

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise HTTPException(status_code=500, detail="Missing dependency: openai") from exc

    if provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        api_key = req.api_key or os.environ.get("OLLAMA_API_KEY", "ollama")
        client = OpenAI(base_url=base_url, api_key=api_key)
    else:
        client = OpenAI(api_key=req.api_key) if req.api_key else OpenAI()

    messages = [{"role": "system", "content": system_prompt}]
    if context:
        messages.append({"role": "system", "content": f"Retrieved context:\n{context}"})
    messages.append({"role": "user", "content": message})

    for _ in range(tool_loop_limit):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            return {"reply": msg.content or "", "tool_calls": tool_calls}

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            }
        )

        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            result = call_tool(call.function.name, args)
            log_tool_response(call.function.name, result)
            tool_calls.append({"name": call.function.name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )

    return {"reply": "Tool loop limit reached.", "tool_calls": tool_calls}
