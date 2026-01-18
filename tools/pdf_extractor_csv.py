import os
import argparse
import csv
import pdfplumber
import re
import numpy as np
import easyocr
from bisect import bisect_right


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


def _build_header(words, gap_threshold=30, start_padding=5):
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
        starts.append(max(0, group[0][0] - start_padding))
    return _dedupe_columns(columns), starts


def _line_to_row(words, columns, starts):
    if not columns:
        return None
    sorted_words = sorted(words)
    buckets = {col: [] for col in columns}
    for left, text in sorted_words:
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
        row.pop("fund_name", None)
        return row if row else None
    row["fund_name"] = fund_name
    return row


def _easyocr_lines(reader, img, y_threshold=30):
    img_rgb = img.convert("RGB")
    img_arr = np.array(img_rgb)
    result = reader.readtext(img_arr)
    words = []
    for item in result:
        if not item or len(item) < 2:
            continue
        box, text, _score = item
        text = str(text).strip()
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        center_y = (min_y + max_y) / 2
        tokens = text.split()
        if not tokens:
            continue
        if len(tokens) == 1:
            words.append((center_y, min_x, tokens[0]))
            continue
        total_len = sum(len(t) for t in tokens) + (len(tokens) - 1)
        pos = 0
        for i, tok in enumerate(tokens):
            if i > 0:
                pos += 1
            left = min_x + (max_x - min_x) * (pos / max(total_len, 1))
            words.append((center_y, left, tok))
            pos += len(tok)

    words.sort(key=lambda x: (x[0], x[1]))
    lines = []
    current = []
    current_y = None
    for y, left, tok in words:
        if current_y is None or abs(y - current_y) <= y_threshold:
            current.append((left, tok))
            current_y = y if current_y is None else (current_y + y) / 2
        else:
            lines.append(current)
            current = [(left, tok)]
            current_y = y
    if current:
        lines.append(current)
    return lines


def extract_ocr_rows(pdf_path):
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
    rows = []
    stop_processing = False
    debug_rows_left = 5
    reader = easyocr.Reader(["en"], gpu=False)

    with pdfplumber.open(pdf_path) as pdf:
        pending_row = None
        for page_index, page in enumerate(pdf.pages, 1):
            img = page.to_image(resolution=400).original
            lines = _easyocr_lines(reader, img)
            line_idx = 0
            while line_idx < len(lines):
                words = lines[line_idx]
                line = ' '.join(w for _, w in sorted(words)).strip()
                if not line:
                    line_idx += 1
                    continue
                lower = line.lower()
                if lower.startswith("total results"):
                    stop_processing = True
                    break
                if any(phrase in lower for phrase in skip_phrases):
                    line_idx += 1
                    continue

                if not header_cols and len(sample_lines) < 10:
                    sample_lines.append(line)

                header_line = re.sub(r"\s+", " ", lower).strip()
                if header_line.startswith("name"):
                    header_cols, header_starts = _build_header(words)
                    best_cols, best_starts, best_advance = header_cols, header_starts, 0
                    if (line_idx + 1) < len(lines):
                        combined = words + lines[line_idx + 1]
                        combined_cols, combined_starts = _build_header(combined)
                        if combined_cols and len(combined_cols) > len(best_cols or []):
                            best_cols, best_starts, best_advance = combined_cols, combined_starts, 1
                    if (line_idx + 2) < len(lines):
                        combined = words + lines[line_idx + 1] + lines[line_idx + 2]
                        combined_cols, combined_starts = _build_header(combined)
                        if combined_cols and len(combined_cols) > len(best_cols or []):
                            best_cols, best_starts, best_advance = combined_cols, combined_starts, 2

                    header_cols, header_starts = best_cols, best_starts
                    if best_advance:
                        line_idx += best_advance
                    if header_cols:
                        print(f"... Detected header on page {page_index}: {header_cols}")
                    line_idx += 1
                    continue

                if not header_cols:
                    line_idx += 1
                    continue

                if debug_rows_left > 0:
                    print(f"Page {page_index} Data Line: {line} | words: {words} | header_starts: {header_starts}")
                row = _line_to_row(words, header_cols, header_starts)
                if row and "fund_name" not in row:
                    if pending_row:
                        pending_row.update(row)
                        rows.append(pending_row)
                        pending_row = None
                    line_idx += 1
                    continue

                if row and len(row) == 1 and "fund_name" in row:
                    tokens = line.replace('%', '').split()
                    has_numbers = any(any(ch.isdigit() for ch in t) for t in tokens)
                    min_left = min((left for left, _ in words), default=0)
                    in_name_col = header_starts and min_left < header_starts[1]
                    if pending_row and not has_numbers and in_name_col:
                        pending_row["fund_name"] = f"{pending_row['fund_name']} {row['fund_name']}".strip()
                    else:
                        pending_row = row
                else:
                    if row:
                        rows.append(row)
                        pending_row = None

                if debug_rows_left > 0:
                    print(f"... Parsed row: {row}")
                    debug_rows_left -= 1

                line_idx += 1

            if stop_processing:
                break

    if not header_cols:
        print("!!! No header detected (expected a line starting with 'Name').")
        if sample_lines:
            print("... Sample OCR lines:")
            for line in sample_lines:
                print(f"    {line}")

    return rows, header_cols


def _read_existing_csv(path):
    if not os.path.exists(path):
        return [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = list(reader.fieldnames or [])
        if not cols and rows:
            cols = sorted({k for row in rows for k in row.keys() if k})
        return rows, cols


def write_csv(rows, columns, output_path):
    if not rows:
        print("!!! No rows parsed. CSV not written.")
        return

    existing_rows, existing_cols = _read_existing_csv(output_path)
    existing_keys = {k for row in existing_rows for k in row.keys() if k}
    new_cols = columns or sorted({k for row in rows for k in row.keys() if k})
    merged_cols = existing_cols + [c for c in sorted(existing_keys) if c not in existing_cols]
    merged_cols += [c for c in new_cols if c not in merged_cols]
    if "fund_name" in merged_cols:
        merged_cols = ["fund_name"] + [c for c in merged_cols if c != "fund_name"]

    existing_by_name = {}
    for row in existing_rows:
        key = normalize_fund_name(row.get("fund_name"))
        if key:
            existing_by_name[key.lower()] = row

    for row in rows:
        key = normalize_fund_name(row.get("fund_name"))
        if not key:
            continue
        stored = existing_by_name.get(key.lower())
        if stored:
            stored.update(row)
        else:
            existing_by_name[key.lower()] = row

    merged_rows = list(existing_by_name.values())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_cols)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)
    print(f"... Wrote CSV: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract fund data from a PDF and write it to CSV.")
    parser.add_argument("pdf_path", type=str, help="The path to the PDF file.")
    parser.add_argument("--output", type=str, default=None, help="Optional output CSV path.")
    args = parser.parse_args()
    print(f"--- Starting execution for: {args.pdf_path} ---")

    rows, header_cols = extract_ocr_rows(args.pdf_path)
    output_path = args.output
    if not output_path:
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.pdf_path)), "investmentdata.csv")

    write_csv(rows, header_cols, output_path)
    print(f"... Rows parsed: {len(rows)}")
    print("\n--- Execution Finished ---")


if __name__ == "__main__":
    main()
