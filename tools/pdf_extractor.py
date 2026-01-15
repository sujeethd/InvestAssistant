import os
import argparse
import pandas as pd
import psycopg2
from tabula import read_pdf
import pdfplumber
from psycopg2 import sql
import numpy as np
import pytesseract
import re

def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renames duplicate columns by appending a suffix like '.1', '.2', etc."""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            f"{dup}.{i}" if i != 0 else dup for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df

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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS fund_data (
                id SERIAL PRIMARY KEY,
                fund_name TEXT UNIQUE,
                ticker TEXT,
                ongoing_charge_pct FLOAT,
                management_fee_pct FLOAT,
                morningstar_risk TEXT,
                volatility_3y FLOAT,
                return_ytd_pct FLOAT,
                return_1y_pct FLOAT,
                return_3y_pct FLOAT,
                return_5y_pct FLOAT,
                return_10y_pct FLOAT
            );
        """)
        conn.commit()

def extract_data_from_pdf(pdf_path):
    """Extracts tabular data from a PDF using multiple fallback methods."""
    raw_df = None
    print("\n--- 1. Starting PDF Extraction ---")
    print("Attempting Method 1: 'tabula-py'...")
    try:
        tables = read_pdf(pdf_path, pages='all', multiple_tables=True, stream=True, guess=False, pandas_options={'dtype': str})
        if tables and not all(df.empty for df in tables):
            print("... 'tabula-py' successful.")
            raw_df = pd.concat(tables, ignore_index=True)
            print(f"... Extracted table shape: {raw_df.shape}")
    except Exception as e:
        print(f"... 'tabula-py' failed with error: {e}.")

    if raw_df is None:
        print("Attempting Method 2: 'pdfplumber'...")
        all_rows = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    if page_tables:
                        print(f"... Found {len(page_tables)} table(s) on page {i+1}")
                        for table in page_tables:
                            if table:
                                print(f"... Debug: first 5 rows on page {i+1}")
                                for debug_row in table[:5]:
                                    print(f"    {debug_row}")
                                all_rows.extend(table)
            
            if all_rows:
                header_idx = -1
                header = None
                header_rows_used = 1
                header_keywords = ['name', 'ticker', 'charge', 'fee', 'risk', 'volatility', 'return', 'yr', 'ytd']

                def _row_text(row):
                    return ' '.join(filter(None, [str(c) for c in row])).lower()

                def _row_score(row):
                    row_text = _row_text(row)
                    return sum(1 for keyword in header_keywords if keyword in row_text)

                best_score = 0
                for i, row in enumerate(all_rows):
                    score = _row_score(row)
                    if score > best_score:
                        best_score = score
                        header_idx = i
                        header_rows_used = 1
                        header = [str(h).replace('\n', ' ') if h is not None else '' for h in row]

                # Try combining two consecutive rows for multi-line headers.
                for i in range(len(all_rows) - 1):
                    row = all_rows[i]
                    next_row = all_rows[i + 1]
                    combined = []
                    max_len = max(len(row), len(next_row))
                    for j in range(max_len):
                        left = row[j] if j < len(row) else None
                        right = next_row[j] if j < len(next_row) else None
                        if left and right:
                            combined.append(f"{left} {right}")
                        else:
                            combined.append(left or right or '')
                    score = _row_score(combined)
                    if score > best_score:
                        best_score = score
                        header_idx = i
                        header_rows_used = 2
                        header = [str(h).replace('\n', ' ') if h is not None else '' for h in combined]

                if header and best_score > 0:
                    print(f"... Found plausible header at overall row index {header_idx} (rows used: {header_rows_used}): {header}")
                    data_rows = all_rows[header_idx + header_rows_used:]
                    num_cols = len(header)
                    processed_rows = []
                    for i, row in enumerate(data_rows):
                        row_text = _row_text(row)

                        # Check for keywords in the current row to avoid including repeated headers
                        matches = sum(1 for keyword in header_keywords if keyword in row_text)
                        if matches >= max(2, best_score):
                            print(f"... Skipping likely header row {i}: {row}")
                            continue

                        if len(row) != num_cols:
                            print(f"... Normalizing row {i}: had {len(row)} cols, expected {num_cols}")
                        while len(row) < num_cols:
                            row.append(None)
                        processed_rows.append(row[:num_cols])

                    raw_df = pd.DataFrame(processed_rows, columns=header)
                    print("... 'pdfplumber' successful.")
                    print(f"... Extracted table shape: {raw_df.shape}")
                else:
                    # Fallback: use the first non-empty row as header to avoid total failure.
                    for i, row in enumerate(all_rows):
                        if any(c not in (None, '') for c in row):
                            header_idx = i
                            header_rows_used = 1
                            header = [str(h).replace('\n', ' ') if h is not None else '' for h in row]
                            print(f"... Fallback header at overall row index {header_idx}: {header}")
                            break
                    if header:
                        data_rows = all_rows[header_idx + header_rows_used:]
                        num_cols = len(header)
                        processed_rows = []
                        for i, row in enumerate(data_rows):
                            if len(row) != num_cols:
                                print(f"... Normalizing row {i}: had {len(row)} cols, expected {num_cols}")
                            while len(row) < num_cols:
                                row.append(None)
                            processed_rows.append(row[:num_cols])
                        raw_df = pd.DataFrame(processed_rows, columns=header)
                        print("... 'pdfplumber' successful (fallback header).")
                        print(f"... Extracted table shape: {raw_df.shape}")
                    else:
                        print("... 'pdfplumber' could not find a plausible header.")

        except Exception as e:
            print(f"... 'pdfplumber' failed with error: {e}")

    if raw_df is None:
        print("Attempting Method 3: 'pytesseract' OCR...")
        try:
            ocr_rows = []
            ocr_perf_rows = []
            skip_phrases = [
                'etf screener',
                'total results',
                'view:',
                'first | previous | next | last',
                'fund prices are updated',
            ]
            risk_phrases = [
                'above average',
                'below average',
                'average',
                'high',
                'low',
            ]
            pdf_name = os.path.basename(pdf_path).lower()
            prefer_perf = 'performance' in pdf_name
            prefer_expense = 'expense' in pdf_name
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    img = page.to_image(resolution=400).original
                    data = pytesseract.image_to_data(
                        img,
                        output_type=pytesseract.Output.DICT,
                        config='--psm 6',
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

                    prev_line_text = None
                    prev_line_has_numbers = False
                    prev_line_has_letters = False

                    for line_id in sorted(lines.keys()):
                        words = lines[line_id]
                        words.sort()
                        line = ' '.join(w for _, w in words).strip()
                        if not line:
                            continue
                        lower = line.lower()
                        if any(phrase in lower for phrase in skip_phrases):
                            continue
                        tokens = line.replace('%', '').split()
                        has_letters = any(any(ch.isalpha() for ch in t) for t in tokens)
                        if not has_letters:
                            prev_line_text = line
                            prev_line_has_numbers = False
                            prev_line_has_letters = False
                            continue

                        numbers = re.findall(r"-?\d+\.\d+", line.replace('%', ''))
                        has_risk = any(risk in lower for risk in risk_phrases)

                        treat_as_perf = has_risk or (prefer_perf and len(numbers) >= 4)
                        if treat_as_perf:
                            risk = next((r for r in risk_phrases if r in lower), None)
                            if risk:
                                parts = re.split(risk, line, flags=re.IGNORECASE, maxsplit=1)
                                name_part = parts[0].strip(" -") if parts else line
                            else:
                                name_part = re.sub(r"-?\d+\.\d+", "", line).strip(" -")
                            name_part = normalize_fund_name(name_part)
                            if not name_part and has_risk and prev_line_text and prev_line_has_letters and not prev_line_has_numbers:
                                name_part = normalize_fund_name(prev_line_text)
                            if name_part and numbers:
                                # Use the last 6 numeric values to map into known perf columns.
                                perf_nums = numbers[-6:] if len(numbers) >= 6 else numbers
                                while len(perf_nums) < 6:
                                    perf_nums.insert(0, None)
                                ocr_perf_rows.append([
                                    name_part,
                                    risk.title() if risk else None,
                                    perf_nums[0],
                                    perf_nums[1],
                                    perf_nums[2],
                                    perf_nums[3],
                                    perf_nums[4],
                                    perf_nums[5],
                                ])
                            continue

                        if numbers and not prefer_perf:
                            ongoing_charge = numbers[-1]
                            management_fee = numbers[-2] if len(numbers) >= 2 else None
                            name = line
                            if line.endswith(ongoing_charge):
                                name = line[: -len(ongoing_charge)].rstrip(" -")
                            else:
                                name = line.replace(ongoing_charge, '').rstrip()
                            name = normalize_fund_name(name)
                            if name:
                                ocr_rows.append([name, ongoing_charge, management_fee])

                        prev_line_text = line
                        prev_line_has_numbers = len(numbers) > 0
                        prev_line_has_letters = has_letters

            if prefer_perf and ocr_perf_rows:
                raw_df = pd.DataFrame(
                    ocr_perf_rows,
                    columns=[
                        'Name',
                        'Morningstar Risk',
                        'Volatility',
                        'YTD Return',
                        '1 Yr Anlsd',
                        '3 Yr Anlsd',
                        '5 Yr Anlsd',
                        '10 Yr Anlsd',
                    ],
                )
            elif prefer_expense and ocr_rows:
                raw_df = pd.DataFrame(
                    ocr_rows,
                    columns=['Name', 'Ongoing Charge', 'Management Fee']
                )
            elif ocr_perf_rows and len(ocr_perf_rows) >= len(ocr_rows):
                raw_df = pd.DataFrame(
                    ocr_perf_rows,
                    columns=[
                        'Name',
                        'Morningstar Risk',
                        'Volatility',
                        'YTD Return',
                        '1 Yr Anlsd',
                        '3 Yr Anlsd',
                        '5 Yr Anlsd',
                        '10 Yr Anlsd',
                    ],
                )
            elif ocr_rows:
                raw_df = pd.DataFrame(
                    ocr_rows,
                    columns=['Name', 'Ongoing Charge', 'Management Fee']
                )

            if raw_df is not None:
                print("... 'pytesseract' successful.")
                print(f"... Extracted table shape: {raw_df.shape}")
            else:
                print("... 'pytesseract' found no usable rows.")
        except Exception as e:
            print(f"... 'pytesseract' failed with error: {e}")

    if raw_df is None:
        print("!!! All extraction methods failed. No tables found.")
        return None
        
    return raw_df

def find_best_column(name_parts, columns):
    """Find the best column match from a list of columns (case-insensitive)."""
    lowered = [(c, str(c).lower()) for c in columns]
    for part in name_parts:
        part_lower = str(part).lower()
        for original, lower in lowered:
            if part_lower in lower:
                return original
    return None

def normalize_fund_name(name: str) -> str | None:
    """Normalize fund names for consistent matching across sources."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
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

def process_expense_data(df):
    """Processes and cleans the extracted expense data."""
    print("\n--- 3. Processing as Expense File ---")
    print(f"Input columns: {df.columns.to_list()}")
    clean_df = pd.DataFrame()
    original_cols = list(df.columns)

    map_dict = {
        'fund_name': ['Name'],
        'ongoing_charge_pct': ['Ongoing Charge'],
        'management_fee_pct': ['Management Fee']
    }
    
    for new_col, old_col_parts in map_dict.items():
        found_col = find_best_column(old_col_parts, original_cols)
        if found_col:
            print(f"... Mapping '{found_col}' to '{new_col}'")
            clean_df[new_col] = df[found_col]
            original_cols.remove(found_col)

    if 'fund_name' not in clean_df.columns:
        print("!!! Could not find 'fund_name' column. Aborting processing.")
        return None, None
        
    clean_df.dropna(subset=['fund_name'], inplace=True)
    clean_df['fund_name'] = clean_df['fund_name'].apply(normalize_fund_name)
    clean_df.dropna(subset=['fund_name'], inplace=True)
    clean_df = clean_df[~clean_df['fund_name'].str.contains('Name', na=False)]

    for col in ['ongoing_charge_pct', 'management_fee_pct']:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].astype(str).str.replace(',', '.').str.replace('%', '')
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    
    if 'ongoing_charge_pct' in clean_df.columns and 'management_fee_pct' in clean_df.columns:
        clean_df = clean_df.dropna(subset=['ongoing_charge_pct', 'management_fee_pct'], how='all')
    
    print(f"--- Cleaned Data Head ---\n{clean_df.head().to_string()}\n-----------------------")
    return clean_df, list(map_dict.keys())

def process_performance_data(df):
    """Processes and cleans the extracted performance data."""
    print("\n--- 3. Processing as Performance File ---")
    print(f"Input columns: {df.columns.to_list()}")
    clean_df = pd.DataFrame()
    original_cols = list(df.columns)

    map_dict = {
        'fund_name': ['Name'],
        'morningstar_risk': ['Morningstar Risk'],
        'volatility_3y': ['Volatility'],
        'return_ytd_pct': ['YTD Return'],
        'return_10y_pct': ['10 Yr Anlsd'],
        'return_5y_pct': ['5 Yr Anlsd'],
        'return_3y_pct': ['3 Yr Anlsd'],
        'return_1y_pct': ['1 Yr Anlsd'],
    }

    for new_col, old_col_parts in map_dict.items():
        found_col = find_best_column(old_col_parts, original_cols)
        if found_col:
            print(f"... Mapping '{found_col}' to '{new_col}'")
            clean_df[new_col] = df[found_col]
            original_cols.remove(found_col)

    if 'fund_name' not in clean_df.columns:
        print("!!! Could not find 'fund_name' column. Aborting processing.")
        return None, None

    clean_df.dropna(subset=['fund_name'], inplace=True)
    clean_df['fund_name'] = clean_df['fund_name'].apply(normalize_fund_name)
    clean_df.dropna(subset=['fund_name'], inplace=True)
    clean_df = clean_df[~clean_df['fund_name'].str.contains('Name', na=False)]
    
    perf_cols = [col for col in map_dict.keys() if col in clean_df.columns]

    for col in perf_cols:
        if 'pct' in col or 'volatility' in col:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].astype(str).str.replace(',', '.').str.replace('%', '')
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
    
    clean_df = clean_df[clean_df['fund_name'].notna()]
    print(f"--- Cleaned Data Head ---\n{clean_df.head().to_string()}\n-----------------------")
    return clean_df, perf_cols


def insert_data(conn, df, columns_to_update):
    if df is None or df.empty or not columns_to_update:
        print("!!! No valid data to insert. Skipping database operation.")
        return
    
    print("\n--- 4. Inserting/Updating Data in Database ---")
    inserted_count = 0
    updated_count = 0
    unmatched_updates = []
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            fund_name = row.get('fund_name')
            if not fund_name or pd.isna(fund_name):
                continue
            fund_name = normalize_fund_name(fund_name)
            if not fund_name:
                continue

            # Try to align OCR-truncated names to existing rows for updates.
            matched_name = None
            cur.execute("SELECT fund_name FROM fund_data WHERE fund_name = %s;", (fund_name,))
            exact = cur.fetchone()
            if exact:
                matched_name = exact[0]
            elif len(fund_name) >= 10:
                cur.execute(
                    "SELECT fund_name FROM fund_data WHERE fund_name ILIKE %s;",
                    (fund_name + '%',)
                )
                matches = [r[0] for r in cur.fetchall()]
                if len(matches) == 1:
                    matched_name = matches[0]

            if matched_name:
                fund_name = matched_name
            else:
                if any(c for c in columns_to_update if c != 'fund_name'):
                    unmatched_updates.append(fund_name)

            valid_columns_to_update = [c for c in columns_to_update if c in row and pd.notna(row[c])]
            set_cols = [c for c in valid_columns_to_update if c != 'fund_name']
            
            if not set_cols:
                cur.execute(
                    "INSERT INTO fund_data (fund_name) VALUES (%s) ON CONFLICT (fund_name) DO NOTHING;",
                    (fund_name,)
                )
                if cur.rowcount > 0: inserted_count += 1
                continue

            set_statements = [sql.SQL("{} = %s").format(sql.Identifier(col)) for col in set_cols]
            values_to_set = [row[col] for col in set_cols]
            
            insert_cols = ['fund_name'] + set_cols
            insert_vals_placeholders = [sql.Placeholder()] * len(insert_cols)
            insert_values = [fund_name] + values_to_set

            query = sql.SQL("""
                INSERT INTO fund_data ({cols})
                VALUES ({vals})
                ON CONFLICT (fund_name) DO UPDATE
                SET {set_stmt}
                RETURNING (xmax = 0);
            """).format(
                cols=sql.SQL(', ').join(map(sql.Identifier, insert_cols)),
                vals=sql.SQL(', ').join(insert_vals_placeholders),
                set_stmt=sql.SQL(', ').join(set_statements)
            )
            
            cur.execute(query, insert_values + values_to_set)
            is_insert = cur.fetchone()[0]
            if is_insert:
                inserted_count += 1
            else:
                updated_count += 1
        conn.commit()
    print(f"... Done. Inserted: {inserted_count}, Updated: {updated_count}")
    if unmatched_updates:
        print(f"... Unmatched updates (showing up to 25): {len(unmatched_updates)} total")
        for name in unmatched_updates[:25]:
            print(f"    - {name}")
        output_path = os.path.join(os.getcwd(), "unmatched_updates.csv")
        pd.Series(unmatched_updates, name="fund_name").to_csv(output_path, index=False)
        print(f"... Wrote unmatched names to: {output_path}")


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

    df = extract_data_from_pdf(args.pdf_path)
    if df is None:
        if conn:
            conn.close()
        return

    print("\n--- 2. Cleaning and preparing DataFrame ---")
    df.columns = [str(c).replace('\n', ' ') for c in df.columns]
    print(f"Raw columns: {df.columns.to_list()}")
    df = _deduplicate_columns(df)
    print(f"Deduplicated columns: {df.columns.to_list()}")
    print(f"--- Raw Data Head ---\n{df.head().to_string()}\n---------------------")
    
    df_processed, cols_to_update = (None, None)
    if any('Charge' in col for col in df.columns) or any('Fee' in col for col in df.columns):
        print("... Detected Expense file format.")
        df_processed, cols_to_update = process_expense_data(df)
    elif any('Return' in col for col in df.columns) or any('Volatility' in col for col in df.columns):
        print("... Detected Performance file format.")
        df_processed, cols_to_update = process_performance_data(df)
    else:
        print("!!! Could not determine file format. Please check the PDF.")
        if conn:
            conn.close()
        return

    if conn:
        insert_data(conn, df_processed, cols_to_update)
        print("\n--- Execution Finished ---")
        conn.close()
    else:
        print("\n--- Execution Finished (no DB) ---")

if __name__ == "__main__":
    main()
