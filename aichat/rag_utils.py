import os
from typing import List, Optional

import requests


def parse_env_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    items = [v.strip() for v in value.split(",")]
    items = [v for v in items if v]
    return items or None


def get_embedding(text: str) -> List[float]:
    provider = os.environ.get("EMBEDDING_PROVIDER", "").lower().strip()
    if not provider:
        provider = os.environ.get("PROVIDER", "openai").lower().strip()
    if provider == "openai":
        return _get_embedding_openai(text)
    if provider == "ollama":
        return _get_embedding_ollama(text)
    raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def _get_embedding_openai(text: str) -> List[float]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai is required for EMBEDDING_PROVIDER=openai") from exc
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def _get_embedding_ollama(text: str) -> List[float]:
    url = os.environ.get("OLLAMA_EMBEDDINGS_URL", "http://localhost:11434/api/embeddings")
    model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    return body["embedding"]
