import os
import argparse
import psycopg2
import pdfplumber
from psycopg2 import sql
import pytesseract
import re
from bisect import bisect_right

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            host=os.environ.get("DB_HOST"),
            port=os.environ.get("DB_PORT")
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"!!! ERROR: Could not connect to the database: {e}")
        return None

def create_table(conn):
    """Creates the fund_data table if it doesn't exist."""
    print("---Ensuring database table exists---")
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS citext;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fund_data (
                id SERIAL PRIMARY KEY,
                fund_name CITEXT UNIQUE
            );
        """)
        cur.execute("ALTER TABLE fund_data ALTER COLUMN fund_name TYPE CITEXT;")
        conn.commit()

def normalize_column_name(name: str) -> str | None:
    text = str(name).strip().lower()
    if not text:
        return None
    text = text.replace('%', ' pct ')
    text = re.sub(r'[^a-z0-9]+', '_', text).strip('_')
    if not text:
        return None
    if text in {"names", "name"}:
        return "fund_name"
    if text[0].isdigit():
        text = f"col_{text}"
    return text

def normalize_fund_name(name: str) -> str | None:
    """Normalize fund names for consistent matching across sources."""
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text).strip(" -|")
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"above average", "below average", "average", "high", "low"}:
        return None
    return text

def _dedupe_columns(columns):
    seen = {}
    deduped = []
    for col in columns:
        count = seen.get(col, 0) + 1
        seen[col] = count
        deduped.append(col if count == 1 else f"{col}_{count}")
    return deduped

def _build_header(words, gap_threshold=30):
    # Group header words into columns using a simple x-gap heuristic.
    sorted_words = sorted(words)
    groups = []
    current = []
    last_left = None
    for left, text in sorted_words:
        if last_left is not None and (left - last_left) > gap_threshold:
            groups.append(current)
            current = []
        current.append((left, text))
        last_left = left
    if current:
        groups.append(current)

    columns = []
    starts = []
    for group in groups:
        col_text = " ".join(t for _, t in group).strip()
        col_name = normalize_column_name(col_text)
        if not col_name:
            continue
        columns.append(col_name)
        starts.append(group[0][0])
    return _dedupe_columns(columns), starts

def _line_to_row(words, columns, starts):
    if not columns:
        return None
    buckets = {col: [] for col in columns}
    for left, text in sorted(words):
        idx = bisect_right(starts, left) - 1
        if idx < 0:
            idx = 0
        buckets[columns[idx]].append(text)

    row = {}
    for col, tokens in buckets.items():
        value = " ".join(tokens).strip()
        if value:
            row[col] = value

    fund_name = normalize_fund_name(row.get("fund_name"))
    if not fund_name:
        return None
    row["fund_name"] = fund_name
    return row

def _coerce_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    maybe_num = text.replace(',', '.').replace('%', '')
    if re.fullmatch(r"-?\d+(\.\d+)?", maybe_num):
        return float(maybe_num)
    return text

def ensure_columns(conn, columns):
    with conn.cursor() as cur:
        for col in columns:
            if col == "fund_name":
                continue
            cur.execute(
                sql.SQL("ALTER TABLE fund_data ADD COLUMN IF NOT EXISTS {} TEXT;").format(
                    sql.Identifier(col)
                )
            )
        conn.commit()

def upsert_row(conn, row):
    columns = list(row.keys())
    if "fund_name" not in columns:
        return
    values = [_coerce_value(row[c]) for c in columns]
    update_cols = [c for c in columns if c != "fund_name"]

    with conn.cursor() as cur:
        if not update_cols:
            cur.execute(
                "INSERT INTO fund_data (fund_name) VALUES (%s) ON CONFLICT (fund_name) DO NOTHING;",
                (row["fund_name"],)
            )
            conn.commit()
            return

        set_stmt = sql.SQL(", ").join(
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
            for c in update_cols
        )
        query = sql.SQL("""
            INSERT INTO fund_data ({cols})
            VALUES ({vals})
            ON CONFLICT (fund_name) DO UPDATE
            SET {set_stmt};
        """).format(
            cols=sql.SQL(", ").join(map(sql.Identifier, columns)),
            vals=sql.SQL(", ").join([sql.Placeholder()] * len(columns)),
            set_stmt=set_stmt,
        )
        cur.execute(query, values)
        conn.commit()

def extract_and_load_ocr(pdf_path, conn):
    """Extracts OCR lines and upserts each row immediately."""
    print("\n--- 1. Starting PDF OCR Extraction ---")
    skip_phrases = [
        'etf screener',
        'total results',
        'view:',
        'first | previous | next | last',
        'fund prices are updated',
    ]
    header_cols = None
    header_starts = None
    sample_lines = []
    img_dir = os.path.join(os.path.dirname(os.path.abspath(pdf_path)), "ocr_pages")
    os.makedirs(img_dir, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, 1):
            img = page.to_image(resolution=600).original
            img_path = os.path.join(img_dir, f"page_{page_index:03d}.png")
#            img.save(img_path)
            data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config='--psm 4',
            )

            lines = {}
            for idx in range(len(data['text'])):
                text = data['text'][idx].strip()
                if not text:
                    continue
                left = data['left'][idx]
                line_id = (
                    data['block_num'][idx],
                    data['par_num'][idx],
                    data['line_num'][idx],
                )
                lines.setdefault(line_id, []).append((left, text))
    
#            print(f"... Extracted {len(lines)} text lines from page {page_index}, lines: {lines}.")
            for line_id in sorted(lines.keys()):
                words = lines[line_id]
                line = ' '.join(w for _, w in sorted(words)).strip()
                if not line:
                    continue
                lower = line.lower()
                if any(phrase in lower for phrase in skip_phrases):
                    continue

                if not header_cols and len(sample_lines) < 10:
                    sample_lines.append(line)

#                print(f"Page {page_index}: {line}")
                header_line = re.sub(r"\s+", " ", lower).strip()
                if header_line.startswith("name"):
                    header_cols, header_starts = _build_header(words)
                    if header_cols:
                        print(f"... Detected header on page {page_index}: {header_cols}")
                        ensure_columns(conn, header_cols)
                    continue

#                if not header_cols and "charge" in header_line and "fee" in header_line:
#                    header_cols, header_starts = _build_header(words, gap_threshold=25)
#                    if header_cols:
#                        if "fund_name" not in header_cols:
#                            first_start = min(header_starts) if header_starts else 0
#                            header_cols = ["fund_name"] + header_cols
#                            header_starts = [max(0, first_start - 350)] + header_starts
#                        print(f"... Detected header from Charge/Fee line on page {page_index}: {header_cols}")
#                        ensure_columns(conn, header_cols)
#                    continue

                if not header_cols:
                    continue

                if len(words) < len(header_cols) :
                    continue

#                print(f"... number of words in line: {len(words)}, number of header columns: {len(header_cols)}")
#                print(f"Page {page_index} Data Line: {line} , words: {words}, header_starts: {header_starts}")
                row = _line_to_row(words, header_cols, header_starts)
                print(f"... Parsed row: {row}")
                if row:
                    upsert_row(conn, row)

    if not header_cols:
        print("!!! No header detected (expected a line starting with 'Name').")
        if sample_lines:
            print("... Sample OCR lines:")
            for line in sample_lines:
                print(f"    {line}")


def main():
    parser = argparse.ArgumentParser(description="Extract fund data from a PDF and load it into a PostgreSQL database.")
    parser.add_argument("pdf_path", type=str, help="The path to the PDF file.")
    args = parser.parse_args()
    print(f"--- Starting execution for: {args.pdf_path} ---")

    conn = get_db_connection()
    if not conn:
        print("!!! Database unavailable. Continuing with extraction only.")
    else:
        create_table(conn)

    if conn:
        extract_and_load_ocr(args.pdf_path, conn)
        print("\n--- Execution Finished ---")
        conn.close()
    else:
        print("!!! Database unavailable. OCR extraction skipped.")

if __name__ == "__main__":
    main()
