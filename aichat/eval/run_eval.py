#!/usr/bin/env python3
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

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


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def run_openai_question(
    client: OpenAI,
    model: str,
    tools: List[Dict[str, Any]],
    system_prompt: str,
    base_url: str,
    question: str,
) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    final_text = ""
    for _ in range(6):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            final_text = msg.content or ""
            messages.append({"role": "assistant", "content": final_text})
            break
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            }
        )
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = call_local_api(base_url, call.function.name, args)
            tool_calls.append(
                {"name": call.function.name, "args": args, "id": call.id}
            )
            tool_results.append({"tool": call.function.name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                }
            )
    return {"text": final_text, "tool_calls": tool_calls, "tool_results": tool_results}


def run_anthropic_question(
    model: str,
    tools: List[Dict[str, Any]],
    system_prompt: str,
    base_url: str,
    question: str,
) -> Dict[str, Any]:
    try:
        from anthropic import Anthropic
    except ImportError:
        return {"text": "", "tool_calls": [], "tool_results": [], "error": "missing_anthropic"}

    client = Anthropic()
    messages: List[Dict[str, Any]] = [{"role": "user", "content": question}]
    tool_calls: List[Dict[str, Any]] = []
    tool_results: List[Dict[str, Any]] = []
    final_text = ""
    for _ in range(6):
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
            final_text = "".join(text_parts).strip()
            messages.append({"role": "assistant", "content": final_text})
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
            args = c.input or {}
            result = call_local_api(base_url, c.name, args)
            tool_calls.append({"name": c.name, "args": args, "id": c.id})
            tool_results.append({"tool": c.name, "args": args, "result": result})
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
    return {"text": final_text, "tool_calls": tool_calls, "tool_results": tool_results}


def extract_expected_values(result: Dict[str, Any], field: str) -> List[str]:
    body = result.get("body", {})
    rows = body.get("rows") if isinstance(body, dict) else None
    if not isinstance(rows, list):
        return []
    values = []
    for row in rows:
        if isinstance(row, dict) and field in row:
            value = row.get(field)
            if value is not None:
                values.append(str(value))
    return values


def match_expected(text: str, values: List[str]) -> List[str]:
    lowered = text.lower()
    matches = []
    for value in values:
        if value and value.lower() in lowered:
            matches.append(value)
    return matches


def score_question(
    result: Dict[str, Any],
    expected_tool: str,
    expected_values: List[str],
    min_matches: int,
) -> Dict[str, Any]:
    errors: List[str] = []
    tool_calls = result.get("tool_calls", [])
    text = result.get("text", "") or ""
    used_expected_tool = any(call.get("name") == expected_tool for call in tool_calls)
    if not used_expected_tool:
        errors.append("missing_expected_tool")
    if not tool_calls:
        errors.append("no_tool_calls")

    if not expected_values:
        errors.append("no_expected_values")

    matches = match_expected(text, expected_values)
    if expected_values and len(matches) < min_matches:
        errors.append("missing_expected_values")

    if expected_values and not matches and text.strip():
        errors.append("possible_hallucination")

    return {
        "tool_usage": "pass" if used_expected_tool else "fail",
        "correctness": "pass"
        if expected_values and len(matches) >= min_matches
        else "fail",
        "hallucination": "fail" if "possible_hallucination" in errors else "pass",
        "errors": errors,
        "matched": matches,
    }


def build_report(
    questions: List[Dict[str, Any]],
    provider: str,
    model: str,
    base_url: str,
) -> Dict[str, Any]:
    tools = build_tools()
    system_prompt = os.environ.get(
        "SYSTEM_PROMPT",
        "You are a helpful assistant that can call tools to query a local funds database API.",
    )

    if provider == "openai":
        client = OpenAI()
    else:
        client = None

    results = []
    for entry in questions:
        question = entry["question"]
        expected_tool = entry["expected_tool"]
        expected_args = entry.get("tool_args", {})
        expect = entry.get("expect", {})
        expected_result = call_local_api(base_url, expected_tool, expected_args)
        expected_values = extract_expected_values(expected_result, expect.get("field", "fund_name"))
        min_matches = int(expect.get("min_matches", 1))

        start = time.time()
        if provider == "openai":
            model_result = run_openai_question(
                client, model, tools, system_prompt, base_url, question
            )
        else:
            model_result = run_anthropic_question(
                model, tools, system_prompt, base_url, question
            )
        elapsed = time.time() - start

        score = score_question(
            model_result, expected_tool, expected_values, min_matches
        )
        results.append(
            {
                "id": entry["id"],
                "question": question,
                "expected_tool": expected_tool,
                "tool_usage": score["tool_usage"],
                "correctness": score["correctness"],
                "hallucination": score["hallucination"],
                "errors": score["errors"],
                "matched": score["matched"],
                "response": model_result.get("text", ""),
                "tool_calls": model_result.get("tool_calls", []),
                "expected_values_sample": expected_values[:10],
                "latency_s": round(elapsed, 2),
            }
        )

    summary = {
        "total": len(results),
        "tool_usage_pass": sum(1 for r in results if r["tool_usage"] == "pass"),
        "correctness_pass": sum(1 for r in results if r["correctness"] == "pass"),
        "hallucination_pass": sum(1 for r in results if r["hallucination"] == "pass"),
    }
    return {
        "summary": summary,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "results": results,
    }


def main() -> int:
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    load_dotenv(os.path.abspath(env_path))

    base_url = os.environ.get("LOCAL_API_URL", "http://localhost:8000")
    provider = os.environ.get("PROVIDER", "openai").lower().strip()
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        if not os.environ.get("OPENAI_API_KEY"):
            print("Missing OPENAI_API_KEY for provider=openai")
            return 1
    else:
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Missing ANTHROPIC_API_KEY for provider=anthropic")
            return 1

    questions_path = os.environ.get(
        "GOLDEN_QUESTIONS_PATH",
        os.path.join(os.path.dirname(__file__), "golden_questions.json"),
    )
    report_path = os.environ.get(
        "EVAL_REPORT_PATH",
        os.path.join(os.path.dirname(__file__), "report.json"),
    )

    questions = load_json(questions_path)
    report = build_report(questions, provider, model, base_url)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
