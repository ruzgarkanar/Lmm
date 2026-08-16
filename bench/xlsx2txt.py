"""GENERIC xlsx→text converter — for LMM ingestion. NO document-specific rules:
on each sheet the header row is found heuristically (the row with the most
short-text-filled cells), and each data row is serialized into a
"Header: value · Header: value" sentence — so every cell value carries the
attribute it belongs to RIGHT NEXT to it (the table-context problem is solved
at the source). Rows before the header pass through as plain text.

Usage: python3.11 bench/xlsx2txt.py file.xlsx > output.txt
"""
import sys

import pandas as pd


def sheet_to_text(df, sheet_name):
    out = [f"[{sheet_name}]"]
    rows = df.values.tolist()
    columns = list(df.columns)
    # pandas may have mistaken the first row for the header; look for the real
    # header inside: the row with the most short-text-filled cells.
    def fullness(cells):
        return sum(1 for c in cells
                   if isinstance(c, str) and 0 < len(c.strip()) <= 40
                   and not c.startswith("Unnamed"))
    candidates = [(-fullness(r), i) for i, r in enumerate(rows[:10])]
    candidates.append((-fullness(columns), -1))     # pandas header is a candidate too
    candidates.sort()
    header_i = candidates[0][1]
    header = columns if header_i == -1 else rows[header_i]
    header = [str(h).strip() if not pd.isna(h) else "" for h in header]
    # cells before the header: plain text (report front matter etc.)
    pre = rows[:header_i] if header_i >= 0 else []
    if header_i >= 0 and list(columns) and not all(
            str(c).startswith("Unnamed") for c in columns):
        pre = [columns] + pre
    for r in pre:
        for cell in r:
            if isinstance(cell, str) and cell.strip() \
                    and not cell.startswith("Unnamed"):
                out.append(cell.strip())
    # data rows: "Header: value · ..."
    data = rows[header_i + 1:] if header_i >= 0 else rows
    for r in data:
        parts = []
        for h, cell in zip(header, r):
            if pd.isna(cell) or not str(cell).strip():
                continue
            value = str(cell).replace("\n", " ").strip()
            parts.append(f"{h}: {value}" if h else value)
        if parts:
            out.append(" · ".join(parts) + ".")
    return "\n".join(out)


def main():
    xl = pd.ExcelFile(sys.argv[1])
    print("\n\n".join(sheet_to_text(xl.parse(sn), sn)
                      for sn in xl.sheet_names))


if __name__ == "__main__":
    main()
