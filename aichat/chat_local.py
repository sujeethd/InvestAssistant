#!/usr/bin/env python3
import json
import os
import sys
import time
from typing import Any, Dict, List

import requests
from openai import OpenAI


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
        "You are a helpful assistant that can call tools to query a local "
        "funds database API. Use tools when needed."
    )
    prompt = os.environ.get("SYSTEM_PROMPT")
    if prompt:
        return prompt
    prompt_file = os.environ.get("SYSTEM_PROMPT_FILE")
    if prompt_file and os.path.exists(prompt_file):
        return load_text(prompt_file)
    local_prompt_file = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    if os.path.exists(local_prompt_file):
        return load_text(local_prompt_file)
    return default_prompt


def call_local_api(base_url: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    endpoints = {
        "funds_search": "/funds/search",
        "funds_summary": "/funds/summary",
        "portfolio_lowest_expense": "/portfolio/lowest-expense",
        "portfolio_best_return": "/portfolio/best-return",
        "portfolio_optimize": "/portfolio/optimize",
    }
    if name not in endpoints:
        return {"error": f"Unknown tool name: {name}"}
    url = base_url.rstrip("/") + endpoints[name]
    try:
        resp = requests.post(url, json=args, timeout=30)
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    return {"status": resp.status_code, "body": body}


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


def run_openai_loop(
    client: OpenAI, base_url: str, model: str, tools: List[Dict[str, Any]]
) -> None:
    system_prompt = get_system_prompt()
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]
    print("Type a question, or Ctrl-D to exit.")
    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})

        for _ in range(10):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                print("Assistant>", msg.content or "")
                messages.append({"role": "assistant", "content": msg.content or ""})
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls,
                }
            )

            for call in msg.tool_calls:
                args = json.loads(call.function.arguments or "{}")
                result = call_local_api(base_url, call.function.name, args)
                log_tool_response(call.function.name, result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
        else:
            print("Assistant> Tool loop limit reached.")

def run_anthropic_loop(
    base_url: str, model: str, tools: List[Dict[str, Any]]
) -> None:
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Missing dependency: anthropic. Install with pip.")
        return

    client = Anthropic()
    system = get_system_prompt()
    messages: List[Dict[str, Any]] = []
    print("Type a question, or Ctrl-D to exit.")
    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})

        for _ in range(10):
            resp = client.messages.create(
                model=model,
                system=system,
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
                print("Assistant>", "".join(text_parts).strip())
                messages.append(
                    {"role": "assistant", "content": "".join(text_parts).strip()}
                )
                break

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
                result = call_local_api(base_url, c.name, c.input or {})
                log_tool_response(c.name, result)
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
        else:
            print("Assistant> Tool loop limit reached.")


def main() -> int:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
    base_url = os.environ.get("LOCAL_API_URL", "http://localhost:8000")
    provider = os.environ.get("PROVIDER", "anthropic").lower().strip()
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "ollama":
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    else:
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        print("Missing OPENAI_API_KEY for provider=openai")
        return 1
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("Missing ANTHROPIC_API_KEY for provider=anthropic")
        return 1
    tools = build_tools()

    print("Local API:", base_url)
    print("Provider:", provider)
    print("Model:", model)
    if provider == "openai":
        client = OpenAI()
        run_openai_loop(client, base_url, model, tools)
    elif provider == "ollama":
        ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        client = OpenAI(
            base_url=ollama_base_url,
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        )
        run_openai_loop(client, base_url, model, tools)
    elif provider == "anthropic":
        run_anthropic_loop(base_url, model, tools)
    else:
        print(f"Unknown provider: {provider}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
