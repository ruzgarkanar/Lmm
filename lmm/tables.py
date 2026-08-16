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
import os

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


def read_pdf(path):
    """PDF → (prose_text, tables). Tables are detected with pdfplumber (it sees
    character COORDINATES and ruling lines — structure the plain-text dump
    destroys: cells gluing together like 'IPX 0Prob', columns losing their
    headers). Each table becomes row-dicts using its first row as the header;
    prose is everything outside table bounding boxes. No model calls."""
    import pdfplumber

    prose_parts = []
    tables = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            found = page.find_tables()
            boxes = [t.bbox for t in found]
            for t in found:
                grid = t.extract()
                if not grid or len(grid) < 2:
                    continue
                clean = [[(c or "").replace("\n", " ").strip() for c in raw]
                         for raw in grid]
                # KEY-VALUE table? (spec sheets: "Ekran | 15.6\" LCD") — rows
                # with at most 2 non-empty cells. Treating row 0 as a header
                # there scrambled every pair; instead each row IS a (key,
                # value) pair; an empty key inherits the previous one
                # (vertically merged cells).
                widths = [sum(1 for c in raw if c) for raw in clean]
                if widths and max(widths) <= 2:
                    last_key = ""
                    pairs = []
                    for raw in clean:
                        cells = [c for c in raw if c]
                        if len(cells) == 2:
                            last_key = cells[0]
                            pairs.append((cells[0], cells[1]))
                        elif len(cells) == 1 and last_key:
                            pairs.append((last_key, cells[0]))
                    if pairs:
                        tables.append([{"__kv__": True, k: v}
                                       for k, v in pairs])
                    continue
                header = clean[0]
                rows = []
                for raw in clean[1:]:
                    row = {}
                    for h, cell in zip(header, raw):
                        if cell:
                            row[h] = cell
                    if row:
                        rows.append(row)
                if rows:
                    tables.append(rows)

            def outside(obj):
                mid = (obj["top"] + obj["bottom"]) / 2
                return not any(b[1] <= mid <= b[3] for b in boxes)

            text = page.filter(outside).extract_text() or ""
            if text.strip():
                prose_parts.append(text)
    return "\n".join(prose_parts), tables


def learn_pdf(session, path, source=None, deep=False):
    """Ingest a PDF the right way: TABLES go straight into the graph
    (learn_rows — deterministic, no model calls), PROSE goes into the evidence
    layer (learn_text; deep=True additionally extracts triples with the
    engine). This is the fix for the manual-benchmark loss class: spec tables
    flattened to text lost their structure. Returns (facts_written, n_tables)."""
    source = source or f"#pdf:{os.path.basename(path)}"
    prose, tables = read_pdf(path)
    wrote = 0
    prev_bulk = getattr(session, "_bulk", False)
    session._bulk = True        # bulk mode: engine-free contradiction handling
    for rows in tables:
        if rows and rows[0].get("__kv__"):
            # key-value table: each row is one structural fact (spec sheets).
            for row in rows:
                for key, value in row.items():
                    if key == "__kv__":
                        continue
                    session.evidence.add(f"{key}: {value}", source)
                    wrote += session.learn_cell(key, "", value, source)
        else:
            wrote += session.learn_rows(rows, source)
    session._bulk = prev_bulk
    if prose.strip():
        session.learn_text(prose, source=source, deep=deep)
    return wrote, len(tables)


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
