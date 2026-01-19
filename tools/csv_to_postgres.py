#!/usr/bin/env python3
import os
import argparse
import csv
import re
import psycopg2
from psycopg2 import sql


def load_dotenv(path):
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
    try:
        return psycopg2.connect(
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            host=os.environ.get("DB_HOST"),
            port=os.environ.get("DB_PORT")
        )
    except psycopg2.OperationalError as e:
        print(f"!!! ERROR: Could not connect to the database: {e}")
        return None


def _normalize_column(col):
    text = str(col).strip()
    if not text:
        return None
    return text


def _clean_numeric(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"na", "n/a", "null"}:
        return None
    cleaned = re.sub(r"[^0-9\.-]", "", text)
    cleaned = re.sub(r"\.+", ".", cleaned)
    cleaned = re.sub(r"(?<=.)-", "", cleaned)
    if not cleaned:
        return None
    if not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def infer_numeric_columns(rows, columns):
    numeric_cols = set()
    for col in columns:
        if col == "fund_name":
            continue
        saw_value = False
        all_numeric = True
        for row in rows:
            raw = row.get(col)
            if raw is None:
                continue
            if not str(raw).strip():
                continue
            saw_value = True
            if _clean_numeric(raw) is None:
                all_numeric = False
                break
        if saw_value and all_numeric:
            numeric_cols.add(col)
    return numeric_cols


def get_existing_columns(conn, table_name):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s;
            """,
            (table_name,),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


def ensure_table(conn, table_name, columns, numeric_columns):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS citext;")
        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {} (id SERIAL PRIMARY KEY);").format(
                sql.Identifier(table_name)
            )
        )
        existing = get_existing_columns(conn, table_name)
        if "fund_name" in columns:
            cur.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS fund_name CITEXT UNIQUE;").format(
                    sql.Identifier(table_name)
                )
            )
        for col in columns:
            if col == "fund_name":
                continue
            col_type = sql.SQL("DOUBLE PRECISION") if col in numeric_columns else sql.SQL("TEXT")
            cur.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} {};").format(
                    sql.Identifier(table_name),
                    sql.Identifier(col),
                    col_type,
                )
            )
        for col in numeric_columns:
            if col == "fund_name":
                continue
            existing_type = existing.get(col)
            if existing_type and existing_type not in {"double precision", "real", "numeric"}:
                cur.execute(
                    sql.SQL(
                        """
                        ALTER TABLE {} ALTER COLUMN {} TYPE double precision
                        USING NULLIF(
                            regexp_replace(
                                regexp_replace(
                                    regexp_replace({}, '[^0-9\\.-]', '', 'g'),
                                    '\\.+', '.', 'g'
                                ),
                                '(?<=.)-', '', 'g'
                            ),
                            ''
                        )::double precision;
                        """
                    ).format(
                        sql.Identifier(table_name),
                        sql.Identifier(col),
                        sql.Identifier(col),
                    )
                )
        conn.commit()


def upsert_rows(conn, table_name, rows, columns, numeric_columns):
    if not rows:
        print("!!! No rows to insert.")
        return
    inserted = 0
    updated = 0
    with conn.cursor() as cur:
        for row in rows:
            fund_name = row.get("fund_name")
            if not fund_name:
                continue
            insert_cols = [c for c in columns if c in row]
            values = []
            for c in insert_cols:
                raw_value = row.get(c)
                if c in numeric_columns:
                    values.append(_clean_numeric(raw_value))
                else:
                    values.append(raw_value)
            update_cols = [c for c in insert_cols if c != "fund_name"]
            if not update_cols:
                cur.execute(
                    sql.SQL("INSERT INTO {} (fund_name) VALUES (%s) ON CONFLICT (fund_name) DO NOTHING;").format(
                        sql.Identifier(table_name)
                    ),
                    (fund_name,)
                )
                if cur.rowcount > 0:
                    inserted += 1
                continue

            set_stmt = sql.SQL(", ").join(
                sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
                for c in update_cols
            )
            query = sql.SQL("""
                INSERT INTO {table} ({cols})
                VALUES ({vals})
                ON CONFLICT (fund_name) DO UPDATE
                SET {set_stmt}
                RETURNING (xmax = 0);
            """).format(
                table=sql.Identifier(table_name),
                cols=sql.SQL(", ").join(map(sql.Identifier, insert_cols)),
                vals=sql.SQL(", ").join([sql.Placeholder()] * len(insert_cols)),
                set_stmt=set_stmt,
            )
            cur.execute(query, values)
            row_result = cur.fetchone()
            if row_result is not None and row_result[0]:
                inserted += 1
            else:
                updated += 1
        conn.commit()
    print(f"... Rows inserted: {inserted}, updated: {updated}")


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = [c for c in reader.fieldnames or [] if c]
        return rows, columns


def main():
    parser = argparse.ArgumentParser(description="Load CSV data into PostgreSQL.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument("--table", type=str, default="fund_data", help="Destination table name.")
    args = parser.parse_args()

    print(f"--- Starting execution for: {args.csv_path} ---")
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    rows, columns = read_csv(args.csv_path)
    columns = [_normalize_column(c) for c in columns if _normalize_column(c)]
    if not columns:
        print("!!! No columns detected in CSV.")
        return
    numeric_columns = infer_numeric_columns(rows, columns)

    conn = get_db_connection()
    if not conn:
        return

    ensure_table(conn, args.table, columns, numeric_columns)
    upsert_rows(conn, args.table, rows, columns, numeric_columns)
    conn.close()
    print("--- Execution Finished ---")


if __name__ == "__main__":
    main()
