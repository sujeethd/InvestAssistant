#!/usr/bin/env python3
import json
import os
import sys
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


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def get_system_prompt() -> str:
    default_prompt = (
        "You are a helpful assistant that answers questions using the provided context."
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


def parse_env_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    items = [v.strip() for v in value.split(",")]
    items = [v for v in items if v]
    return items or None


def call_rag_api(
    base_url: str,
    query: str,
    table: str,
    columns: Optional[List[str]],
    text_columns: Optional[List[str]],
    limit: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"query": query, "table": table, "limit": limit}
    if columns:
        payload["columns"] = columns
    if text_columns:
        payload["text_columns"] = text_columns
    url = base_url.rstrip("/") + "/rag/search"
    try:
        resp = requests.post(url, json=payload, timeout=30)
    except requests.RequestException as exc:
        return {"error": f"Request failed: {exc}"}
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    return {"status": resp.status_code, "body": body}


def format_context(rows: List[Dict[str, Any]]) -> str:
    lines = []
    for idx, row in enumerate(rows, 1):
        parts = []
        for key, value in row.items():
            parts.append(f"{key}={value}")
        lines.append(f"{idx}. " + "; ".join(parts))
    return "\n".join(lines)


def run_openai_loop(client: OpenAI, base_url: str, model: str) -> None:
    system_prompt = get_system_prompt()
    system_prompt += (
        "\n\nUse only the provided context for factual claims. "
        "If the context is insufficient, say so and ask a brief clarifying question."
    )
    table = os.environ.get("RAG_TABLE", "fund_data")
    columns = parse_env_list(os.environ.get("RAG_COLUMNS"))
    text_columns = parse_env_list(os.environ.get("RAG_TEXT_COLUMNS"))
    limit = int(os.environ.get("RAG_LIMIT", "20"))

    print("Type a question, or Ctrl-D to exit.")
    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            continue

        rag_result = call_rag_api(base_url, user_input, table, columns, text_columns, limit)
        if rag_result.get("status") != 200:
            print("RAG>", rag_result)
            continue
        rows = rag_result.get("body", {}).get("rows", [])
        context = format_context(rows)
        user_message = f"Context:\n{context}\n\nQuestion: {user_input}"

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        msg = resp.choices[0].message
        print("Assistant>", msg.content or "")


def run_anthropic_loop(base_url: str, model: str) -> None:
    try:
        from anthropic import Anthropic
    except ImportError:
        print("Missing dependency: anthropic. Install with pip.")
        return
    client = Anthropic()
    system_prompt = get_system_prompt()
    system_prompt += (
        "\n\nUse only the provided context for factual claims. "
        "If the context is insufficient, say so and ask a brief clarifying question."
    )
    table = os.environ.get("RAG_TABLE", "fund_data")
    columns = parse_env_list(os.environ.get("RAG_COLUMNS"))
    text_columns = parse_env_list(os.environ.get("RAG_TEXT_COLUMNS"))
    limit = int(os.environ.get("RAG_LIMIT", "20"))

    print("Type a question, or Ctrl-D to exit.")
    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            continue

        rag_result = call_rag_api(base_url, user_input, table, columns, text_columns, limit)
        if rag_result.get("status") != 200:
            print("RAG>", rag_result)
            continue
        rows = rag_result.get("body", {}).get("rows", [])
        context = format_context(rows)
        user_message = f"Context:\n{context}\n\nQuestion: {user_input}"

        resp = client.messages.create(
            model=model,
            system=system_prompt,
            max_tokens=800,
            messages=[{"role": "user", "content": user_message}],
        )
        text_parts = [c.text for c in resp.content if c.type == "text"]
        print("Assistant>", "".join(text_parts).strip())


def main() -> int:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)
    base_url = os.environ.get("RAG_API_URL", os.environ.get("LOCAL_API_URL", "http://localhost:8000"))
    provider = os.environ.get("PROVIDER", "anthropic").lower().strip()
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    elif provider == "ollama":
        model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    else:
        model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")

    print("RAG API:", base_url)
    print("Provider:", provider)
    print("Model:", model)

    if provider == "openai":
        if not os.environ.get("OPENAI_API_KEY"):
            print("Missing OPENAI_API_KEY for provider=openai")
            return 1
        client = OpenAI()
        run_openai_loop(client, base_url, model)
    elif provider == "ollama":
        client = OpenAI(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
        )
        run_openai_loop(client, base_url, model)
    elif provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Missing ANTHROPIC_API_KEY for provider=anthropic")
            return 1
        run_anthropic_loop(base_url, model)
    else:
        print(f"Unknown provider: {provider}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
