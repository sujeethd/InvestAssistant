import os
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2 import sql
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


def load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        print(f"[startup] .env not found at {path}")
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
                print(f"[startup] loaded env var {key}")


def get_db_connection():
    print("[db] connecting with env vars:", {
        "DB_NAME": os.environ.get("DB_NAME"),
        "DB_USER": os.environ.get("DB_USER"),
        "DB_HOST": os.environ.get("DB_HOST"),
        "DB_PORT": os.environ.get("DB_PORT"),
    })
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        host=os.environ.get("DB_HOST"),
        port=os.environ.get("DB_PORT"),
    )


def get_table_columns(conn, table: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (table,),
        )
        return [r[0] for r in cur.fetchall()]


def numeric_expr(column: str) -> sql.SQL:
    return sql.SQL(
        "NULLIF(regexp_replace(({}::text), '[^0-9\\.-]', '', 'g'), '')::double precision"
    ).format(sql.Identifier(column))


def build_where(filters: Optional[List["Filter"]], columns: List[str]):
    clauses = []
    params: List[Any] = []
    col_set = set(columns)
    for f in filters or []:
        if f.column not in col_set:
            raise HTTPException(status_code=400, detail=f"Unknown column: {f.column}")
        if f.op in {"=", "!=", ">", ">=", "<", "<="}:
            clauses.append(
                sql.SQL("{} {} %s").format(sql.Identifier(f.column), sql.SQL(f.op))
            )
            params.append(f.value)
        elif f.op == "in":
            if not isinstance(f.value, list):
                raise HTTPException(status_code=400, detail="Value for 'in' must be a list")
            clauses.append(sql.SQL("{} = ANY(%s)").format(sql.Identifier(f.column)))
            params.append(f.value)
        elif f.op == "like":
            clauses.append(sql.SQL("{} ILIKE %s").format(sql.Identifier(f.column)))
            params.append(str(f.value))
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported op: {f.op}")
    return clauses, params


def parse_order_by(order_by: Optional[str], columns: List[str]) -> Optional[sql.SQL]:
    if not order_by:
        return None
    direction = sql.SQL("ASC")
    col = order_by
    if order_by.startswith("-"):
        direction = sql.SQL("DESC")
        col = order_by[1:]
    if col not in set(columns):
        raise HTTPException(status_code=400, detail=f"Unknown order_by column: {col}")
    return sql.SQL("{} {}").format(sql.Identifier(col), direction)


class Filter(BaseModel):
    column: str
    op: str
    value: Any


class FundSearchRequest(BaseModel):
    table: str = "fund_data"
    columns: Optional[List[str]] = None
    filters: List[Filter] = Field(default_factory=list)
    limit: int = 50
    order_by: Optional[str] = None


class FundSummaryRequest(BaseModel):
    table: str = "fund_data"
    columns: List[str]
    filters: List[Filter] = Field(default_factory=list)


class LowestExpenseRequest(BaseModel):
    table: str = "fund_data"
    expense_column: str
    count: int = 10
    filters: List[Filter] = Field(default_factory=list)


class BestReturnRequest(BaseModel):
    table: str = "fund_data"
    return_column: str
    count: int = 10
    filters: List[Filter] = Field(default_factory=list)


class Objective(BaseModel):
    maximize: List[str] = Field(default_factory=list)
    minimize: List[str] = Field(default_factory=list)


class OptimizeRequest(BaseModel):
    table: str = "fund_data"
    count: int = 10
    objective: Objective
    weights: Dict[str, float] = Field(default_factory=dict)
    filters: List[Filter] = Field(default_factory=list)


class RagSearchRequest(BaseModel):
    query: str
    table: str = "fund_data"
    columns: Optional[List[str]] = None
    text_columns: Optional[List[str]] = None
    limit: int = 50


class RagSemanticRequest(BaseModel):
    embedding: List[float]
    table: str = "fund_data"
    embeddings_table: str = "fund_embeddings"
    columns: Optional[List[str]] = None
    limit: int = 50


app = FastAPI(title="Investment Portfolio API", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.on_event("startup")
def startup():
    env_path = os.path.join(os.getcwd(), ".env")
    print(f"[startup] loading env from {env_path}")
    load_dotenv(env_path)


@app.post("/funds/search")
def search_funds(req: FundSearchRequest):
    with get_db_connection() as conn:
        columns = get_table_columns(conn, req.table)
        if req.columns == ["*"]:
            cols = columns
        else:
            cols = req.columns or columns
        print("[funds/search] table:", req.table)
        print("[funds/search] available columns:", columns)
        print("[funds/search] selected columns:", cols)
        for c in cols:
            if c not in columns:
                raise HTTPException(status_code=400, detail=f"Unknown column: {c}")
        where_clauses, params = build_where(req.filters, columns)
        order_by_sql = parse_order_by(req.order_by, columns)
        query = sql.SQL("SELECT {} FROM {}").format(
            sql.SQL(", ").join(map(sql.Identifier, cols)),
            sql.Identifier(req.table),
        )
        if where_clauses:
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
        if order_by_sql is not None:
            query += sql.SQL(" ORDER BY ") + order_by_sql
        query += sql.SQL(" LIMIT %s")
        params.append(req.limit)
        print("[funds/search] query:", query.as_string(conn))
        print("[funds/search] params:", params)
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        print("[funds/search] rows returned:", len(rows))
        result = [dict(zip(cols, r)) for r in rows]
        return {"rows": result}


@app.post("/funds/summary")
def summarize_funds(req: FundSummaryRequest):
    with get_db_connection() as conn:
        columns = get_table_columns(conn, req.table)
        for c in req.columns:
            if c not in columns:
                raise HTTPException(status_code=400, detail=f"Unknown column: {c}")
        where_clauses, params = build_where(req.filters, columns)
        select_parts = []
        for c in req.columns:
            expr = numeric_expr(c)
            select_parts.append(sql.SQL("MIN({}) AS {}").format(expr, sql.Identifier(f"{c}_min")))
            select_parts.append(sql.SQL("MAX({}) AS {}").format(expr, sql.Identifier(f"{c}_max")))
            select_parts.append(sql.SQL("AVG({}) AS {}").format(expr, sql.Identifier(f"{c}_avg")))
        query = sql.SQL("SELECT {} FROM {}").format(
            sql.SQL(", ").join(select_parts),
            sql.Identifier(req.table),
        )
        if where_clauses:
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
        with conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
        summary = {}
        idx = 0
        for c in req.columns:
            summary[c] = {
                "min": row[idx],
                "max": row[idx + 1],
                "avg": row[idx + 2],
            }
            idx += 3
        return {"summary": summary}


@app.post("/portfolio/lowest-expense")
def lowest_expense(req: LowestExpenseRequest):
    with get_db_connection() as conn:
        columns = get_table_columns(conn, req.table)
        if req.expense_column not in columns:
            raise HTTPException(status_code=400, detail=f"Unknown column: {req.expense_column}")
        where_clauses, params = build_where(req.filters, columns)
        expense_expr = numeric_expr(req.expense_column)
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(req.table))
        if where_clauses:
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
        query += sql.SQL(" ORDER BY {} ASC NULLS LAST LIMIT %s").format(expense_expr)
        params.append(req.count)
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        result = [dict(zip(columns, r)) for r in rows]
        return {"rows": result, "count": len(result)}


@app.post("/portfolio/best-return")
def best_return(req: BestReturnRequest):
    with get_db_connection() as conn:
        columns = get_table_columns(conn, req.table)
        if req.return_column not in columns:
            raise HTTPException(status_code=400, detail=f"Unknown column: {req.return_column}")
        where_clauses, params = build_where(req.filters, columns)
        return_expr = numeric_expr(req.return_column)
        query = sql.SQL("SELECT * FROM {}").format(sql.Identifier(req.table))
        if where_clauses:
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
        query += sql.SQL(" ORDER BY {} DESC NULLS LAST LIMIT %s").format(return_expr)
        params.append(req.count)
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        result = [dict(zip(columns, r)) for r in rows]
        return {"rows": result, "count": len(result)}


@app.post("/portfolio/optimize")
def optimize_portfolio(req: OptimizeRequest):
    with get_db_connection() as conn:
        columns = get_table_columns(conn, req.table)
        for c in req.objective.maximize + req.objective.minimize:
            if c not in columns:
                raise HTTPException(status_code=400, detail=f"Unknown column: {c}")
        where_clauses, params = build_where(req.filters, columns)
        terms = []
        for c in req.objective.maximize:
            w = req.weights.get(c, 1.0)
            terms.append(sql.SQL("(%s * {})").format(numeric_expr(c)))
            params.append(w)
        for c in req.objective.minimize:
            w = req.weights.get(c, 1.0)
            terms.append(sql.SQL("(-%s * {})").format(numeric_expr(c)))
            params.append(w)
        if not terms:
            raise HTTPException(status_code=400, detail="Objective is empty")
        score_expr = sql.SQL(" + ").join(terms)
        query = sql.SQL("SELECT *, {} AS score FROM {}").format(
            score_expr, sql.Identifier(req.table)
        )
        if where_clauses:
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)
        query += sql.SQL(" ORDER BY score DESC NULLS LAST LIMIT %s")
        params.append(req.count)
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        result = [dict(zip(columns + ["score"], r)) for r in rows]
        return {"rows": result, "count": len(result)}


@app.post("/rag/search")
def rag_search(req: RagSearchRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    with get_db_connection() as conn:
        columns = get_table_columns(conn, req.table)
        cols = req.columns or columns
        for c in cols:
            if c not in columns:
                raise HTTPException(status_code=400, detail=f"Unknown column: {c}")

        default_text_cols = [c for c in ("fund_name", "morningstart_risk") if c in columns]
        text_cols = req.text_columns or default_text_cols
        for c in text_cols:
            if c not in columns:
                raise HTTPException(status_code=400, detail=f"Unknown text column: {c}")

        text_exprs = [
            sql.SQL("COALESCE({}::text, '')").format(sql.Identifier(c))
            for c in text_cols
        ]
        tsv = sql.SQL("to_tsvector('english', concat_ws(' ', {}))").format(
            sql.SQL(", ").join(text_exprs)
        )
        tsq = sql.SQL("plainto_tsquery('english', %s)")
        query = sql.SQL("SELECT {}, ts_rank_cd({}, {}) AS rank FROM {} WHERE {} @@ {} ORDER BY rank DESC LIMIT %s").format(
            sql.SQL(", ").join(map(sql.Identifier, cols)),
            tsv,
            tsq,
            sql.Identifier(req.table),
            tsv,
            tsq,
        )
        params = [req.query, req.query, req.limit]
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        result = [dict(zip(cols + ["rank"], r)) for r in rows]
        return {"rows": result, "count": len(result)}


@app.post("/rag/semantic")
def rag_semantic(req: RagSemanticRequest):
    if not req.embedding:
        raise HTTPException(status_code=400, detail="embedding is required")
    with get_db_connection() as conn:
        columns = get_table_columns(conn, req.table)
        cols = req.columns or columns
        for c in cols:
            if c not in columns:
                raise HTTPException(status_code=400, detail=f"Unknown column: {c}")

        embedding_literal = "[" + ",".join(str(x) for x in req.embedding) + "]"
        query = sql.SQL(
            """
            SELECT {cols}, e.embedding <-> %s::vector AS distance
            FROM {embeddings} e
            JOIN {table} t ON t.id = e.fund_id
            ORDER BY e.embedding <-> %s::vector
            LIMIT %s
            """
        ).format(
            cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
            embeddings=sql.Identifier(req.embeddings_table),
            table=sql.Identifier(req.table),
        )
        params = [embedding_literal, embedding_literal, req.limit]
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        result = [dict(zip(cols + ["distance"], r)) for r in rows]
        return {"rows": result, "count": len(result)}
