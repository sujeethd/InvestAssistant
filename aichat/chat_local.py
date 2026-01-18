#!/usr/bin/env python3
import json
import os
import sys
from typing import Any, Dict

import requests
from openai import OpenAI


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


def main() -> int:
    base_url = os.environ.get("LOCAL_API_URL", "http://localhost:8000")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI()

    tools = [
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
                    "required": ["table"],
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
                    "required": ["table", "columns"],
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
                    "required": ["table", "expense_column"],
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
                    "required": ["table", "return_column"],
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
                    "required": ["table", "objective"],
                },
            },
        },
    ]

    print("Local API:", base_url)
    print("Model:", model)
    print("Type a question, or Ctrl-D to exit.")

    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            continue
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can call tools to query a local "
                    "funds database API. Use tools when needed."
                ),
            },
            {"role": "user", "content": user_input},
        ]

        for _ in range(5):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                print("Assistant>", msg.content or "")
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
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
        else:
            print("Assistant> Tool loop limit reached.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
