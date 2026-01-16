import os
import argparse
import csv
import pdfplumber
import pytesseract
import re
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


def _build_header(words, gap_threshold=50, start_padding=5):
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
    row = {}
    if len(columns) == 1:
        value = " ".join(t for _, t in sorted_words).strip()
        if value:
            row[columns[0]] = value
    else:
        split1 = starts[1]
        split2 = starts[2] if len(starts) > 2 else None
        name_tokens = [t for left, t in sorted_words if left < split1]
        col2_tokens = [t for left, t in sorted_words if split1 <= left and (split2 is None or left < split2)]
        tail_tokens = [t for left, t in sorted_words if split2 is not None and left >= split2]

        name_value = " ".join(name_tokens).strip()
        if name_value:
            row[columns[0]] = name_value
        col2_value = " ".join(col2_tokens).strip()
        if col2_value:
            row[columns[1]] = col2_value

        tail_cols = columns[2:]
        if tail_cols and tail_tokens:
            tokens = " ".join(tail_tokens).split()
            if len(tokens) >= len(tail_cols):
                for i, col in enumerate(tail_cols[:-1]):
                    row[col] = tokens[i]
                row[tail_cols[-1]] = " ".join(tokens[len(tail_cols) - 1:]).strip()
            else:
                for i, col in enumerate(tail_cols):
                    if i < len(tokens):
                        row[col] = tokens[i]

    fund_name = normalize_fund_name(row.get("fund_name"))
    if not fund_name:
        return None
    row["fund_name"] = fund_name
    return row


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

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, 1):
            img = page.to_image(resolution=600).original
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

            for line_id in sorted(lines.keys()):
                words = lines[line_id]
                line = ' '.join(w for _, w in sorted(words)).strip()
                if not line:
                    continue
                lower = line.lower()
                if lower.startswith("total results"):
                    stop_processing = True
                    break
                if any(phrase in lower for phrase in skip_phrases):
                    continue

                if not header_cols and len(sample_lines) < 10:
                    sample_lines.append(line)

                header_line = re.sub(r"\s+", " ", lower).strip()
                if header_line.startswith("name"):
                    header_cols, header_starts = _build_header(words)
                    if header_cols:
                        print(f"... Detected header on page {page_index}: {header_cols}")
                    continue

                if not header_cols:
                    continue

                row = _line_to_row(words, header_cols, header_starts)
                if row:
                    rows.append(row)

            if stop_processing:
                break

    if not header_cols:
        print("!!! No header detected (expected a line starting with 'Name').")
        if sample_lines:
            print("... Sample OCR lines:")
            for line in sample_lines:
                print(f"    {line}")

    return rows, header_cols


def write_csv(rows, columns, output_path):
    if not rows:
        print("!!! No rows parsed. CSV not written.")
        return
    columns = columns or sorted({k for row in rows for k in row.keys()})
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
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
        base = os.path.splitext(os.path.basename(args.pdf_path))[0]
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.pdf_path)), f"{base}.csv")

    write_csv(rows, header_cols, output_path)
    print(f"... Rows parsed: {len(rows)}")
    print("\n--- Execution Finished ---")


if __name__ == "__main__":
    main()
