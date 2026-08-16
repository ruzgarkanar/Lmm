"""STRUCTURED ingestion — table (xlsx/csv) → DIRECTLY into graph + evidence.

Makes the Excel test's lesson permanent: using an extractor (a model) on an
already-structured source is both waste and a source of oscillation. The table
states its own structure: row = entity, column = predicate, cell = value. This
path calls NO model → deterministic, ms — even a 500-page table is ingested in
seconds.

None of the architecture's rules is bent: every triple still passes through the
GATE (gate.admit), with DOCUMENT trust (cannot override an operator fact,
source-hedged in answers), and the row sentence is also written to the evidence
store (the language surface is built from there). The header heuristic is the
same as xlsx2txt: the row most filled with short text. No document-specific rule.
"""
import pandas as pd


def _header_row(rows, columns):
    def fullness(cells):
        return sum(1 for c in cells
                   if isinstance(c, str) and 0 < len(c.strip()) <= 40
                   and not c.startswith("Unnamed"))
    candidates = [(-fullness(r), i) for i, r in enumerate(rows[:10])]
    candidates.append((-fullness(columns), -1))
    candidates.sort()
    return candidates[0][1]


def read_xlsx(path):
    """xlsx → [(sheet, preamble-lines, row-dicts)]. No model."""
    sheets = []
    xl = pd.ExcelFile(path)
    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)
        rows = df.values.tolist()
        columns = list(df.columns)
        hi = _header_row(rows, columns)
        header = columns if hi == -1 else rows[hi]
        header = [str(h).strip() if not pd.isna(h) else "" for h in header]
        pre = []
        pre_rows = ([columns] if hi >= 0 else []) + (rows[:hi] if hi >= 0 else [])
        for r in pre_rows:
            for cell in r:
                if isinstance(cell, str) and cell.strip() \
                        and not cell.startswith("Unnamed"):
                    pre.append(cell.strip())
        data = []
        for r in (rows[hi + 1:] if hi >= 0 else rows):
            row = {}
            for h, cell in zip(header, r):
                if pd.isna(cell) or not str(cell).strip():
                    continue
                row[h or ""] = str(cell).replace("\n", " ").strip()
            if row:
                data.append(row)
        sheets.append((sheet_name, pre, data))
    return sheets


def learn_xlsx(session, path, source=None):
    """Ingest one Excel file — extractor-less, ms. Returns: number of facts
    written. Preamble 'Key: Value' lines are also parsed structurally (the
    colon is a format separator, not a language rule)."""
    source = source or f"#xlsx:{path}"
    wrote = 0
    for sheet, pre, rows in read_xlsx(path):
        for line in pre:
            if ":" in line:
                key, _, value = line.partition(":")
                if key.strip() and value.strip():
                    wrote += session.learn_cell(key.strip(), "", value.strip(),
                                                source)
        wrote += session.learn_rows(rows, source)
        # EVIDENCE layer — ONLY the preamble text, with deep=False (the windows
        # link the preamble lines to each other: the "Reviewer ↔ report"
        # bridge; ZERO model calls). Data ROWS do NOT enter the window: the
        # record already carries its own context and the window was smearing
        # neighboring rows' fields into each other (the leak into Ersin's PPE
        # row). learn_rows writes the row sentences to evidence individually.
        text = "\n".join([f"[{sheet}]"] + pre)
        session.learn_text(text, source=source, deep=False)
    return wrote
