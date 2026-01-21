#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Dict, Iterable, List, Optional

import psycopg2
from psycopg2 import sql
import requests


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


def get_db_connection():
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
    )


def parse_env_list(value: Optional[str]) -> Optional[List[str]]:
    if not value:
        return None
    items = [v.strip() for v in value.split(",")]
    items = [v for v in items if v]
    return items or None


def chunked(iterable: List[Dict[str, object]], size: int) -> Iterable[List[Dict[str, object]]]:
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]


def build_content(row: Dict[str, object], columns: List[str]) -> str:
    parts = []
    for col in columns:
        parts.append(f"{col}={row.get(col)}")
    return "; ".join(parts)


def ensure_embeddings_table(conn, table_name: str, dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {} (
                    fund_id INTEGER PRIMARY KEY,
                    content TEXT,
                    embedding vector({})
                );
                """
            ).format(sql.Identifier(table_name), sql.Literal(dim))
        )
        conn.commit()


def fetch_rows(conn, table: str, columns: List[str], limit: int, offset: int) -> List[Dict[str, object]]:
    cols_sql = sql.SQL(", ").join(map(sql.Identifier, ["id"] + columns))
    query = sql.SQL("SELECT {} FROM {} ORDER BY id ASC LIMIT %s OFFSET %s").format(
        cols_sql, sql.Identifier(table)
    )
    with conn.cursor() as cur:
        cur.execute(query, (limit, offset))
        rows = cur.fetchall()
    results = []
    for row in rows:
        data = dict(zip(["id"] + columns, row))
        results.append(data)
    return results


def get_embedding_openai(text: str) -> List[float]:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai is required for EMBEDDING_PROVIDER=openai")
    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI()
    resp = client.embeddings.create(model=model, input=text)
    return resp.data[0].embedding


def get_embedding_ollama(text: str) -> List[float]:
    url = os.environ.get("OLLAMA_EMBEDDINGS_URL", "http://localhost:11434/api/embeddings")
    model = os.environ.get("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    resp = requests.post(url, json={"model": model, "prompt": text}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    return body["embedding"]


def get_embedding(text: str) -> List[float]:
    provider = os.environ.get("EMBEDDING_PROVIDER", "").lower().strip()
    if not provider:
        provider = os.environ.get("PROVIDER", "openai").lower().strip()
    if provider == "openai":
        return get_embedding_openai(text)
    if provider == "ollama":
        return get_embedding_ollama(text)
    raise RuntimeError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def upsert_embeddings(conn, table_name: str, rows: List[Dict[str, object]], columns: List[str]) -> None:
    with conn.cursor() as cur:
        for row in rows:
            fund_id = row.get("id")
            content = build_content(row, columns)
            embedding = get_embedding(content)
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (fund_id, content, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (fund_id) DO UPDATE
                    SET content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding;
                    """
                ).format(sql.Identifier(table_name)),
                (fund_id, content, embedding),
            )
        conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build embeddings for fund_data rows.")
    parser.add_argument("--table", default="fund_data", help="Source table name.")
    parser.add_argument("--embeddings-table", default="fund_embeddings", help="Embeddings table name.")
    parser.add_argument("--batch", type=int, default=50, help="Batch size for embedding generation.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for rows processed.")
    args = parser.parse_args()

    load_dotenv(os.path.join(os.getcwd(), ".env"))
    columns = parse_env_list(os.environ.get("EMBEDDING_TEXT_COLUMNS"))
    if not columns:
        columns = [
            "fund_name",
            "morningstart_risk",
            "1_yr_anlsd_percent",
            "5_yr_anlsd_percent",
            "expense_ratio",
        ]

    dim = os.environ.get("EMBEDDING_DIM")
    if not dim:
        print("Missing EMBEDDING_DIM (e.g., 1536 for OpenAI, 768 for nomic-embed-text).")
        return 1

    conn = get_db_connection()
    ensure_embeddings_table(conn, args.embeddings_table, int(dim))

    offset = 0
    total = 0
    while True:
        remaining = args.limit - total if args.limit else None
        fetch_limit = min(args.batch, remaining) if remaining else args.batch
        rows = fetch_rows(conn, args.table, columns, fetch_limit, offset)
        if not rows:
            break
        upsert_embeddings(conn, args.embeddings_table, rows, columns)
        total += len(rows)
        offset += len(rows)
        if remaining is not None and total >= args.limit:
            break
        print(f"... embedded rows: {total}")

    conn.close()
    print(f"Done. Embedded rows: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
