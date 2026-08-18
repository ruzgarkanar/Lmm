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
import re

import pandas as pd

from lmm.core.dataset import fold
from lmm import evidence


def _header_row(rows, columns):
    """Which row names the columns — the one most filled with SHORT TEXT.

    Two numbers used to be in here and both were guesses about somebody's
    spreadsheet: a header cell was text "at most 40 characters" long, and only
    the first 10 rows were considered. The sheet answers both. What counts as
    short is the split in ITS OWN cell lengths (`evidence._cell_bound`, the same
    derivation the shattered-table reader uses), and the header cannot be below
    the point where DATA begins — the first row whose filled cells are mostly not
    text. Neither bound is a number we chose.
    """
    text_cells = [c.strip() for r in rows for c in r
                  if isinstance(c, str) and c.strip()
                  and not c.startswith("Unnamed")]
    bound = evidence._cell_bound(text_cells) if text_cells else 0

    def fullness(cells):
        return sum(1 for c in cells
                   if isinstance(c, str) and 0 < len(c.strip()) <= bound
                   and not c.startswith("Unnamed"))

    def is_data(cells):
        filled = [c for c in cells if not pd.isna(c) and str(c).strip()]
        return bool(filled) and sum(
            1 for c in filled if not isinstance(c, str)) * 2 > len(filled)

    top = []
    for i, row in enumerate(rows):
        if is_data(row):
            break
        top.append((-fullness(row), i))
    top.append((-fullness(columns), -1))
    top.sort()
    return top[0][1]


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


_KEY_TOKEN = r"(?<![^\s:])(?![^\s:]*\d)[^\s:]+"   # whole word, no digit inside
_KV = re.compile(rf"((?:{_KEY_TOKEN}[ \t]+){{0,2}}{_KEY_TOKEN})\s*:\s*")


def split_kv_text(text):
    """Split a COMPOSITE cell on colon FORMAT boundaries: 'Gövde: A1 Kapak: A2
    Kayış: A3' → three (key, value) pairs. The colon is a
    format separator (allowed); the only other cue is typographic: key words
    carry no digits, so a value's tail ('IPX0') cannot be mistaken for the
    next key. No language rule."""
    ms = list(_KV.finditer(text))
    pairs = []
    for i, m in enumerate(ms):
        end = ms[i + 1].start(1) if i + 1 < len(ms) else len(text)
        value = text[m.end():end].strip(" ,;·|")
        key = m.group(1).strip(" ,;·|")
        if key and value:
            pairs.append((key, value))
    return pairs


def _split_leading_value(text):
    """'Ölçüler 410 mm * 290 mm * 90 mm' → ('Ölçüler', '410 mm * ...').
    FORMAT cue only: a record cell whose leading words carry no digit and whose
    value starts at the first digit. None if the shape doesn't match (prose stays
    prose).

    The shape test used to be "between 2 and 12 words, first digit within the
    first 3" — two bounds off one spec sheet. What it is reaching for is that the
    FIELD NAME is the short part and the VALUE is the rest, which is a
    comparison inside the cell and needs no absolute size: the leading no-digit
    run has to be a minority of the words, so a text whose figure arrives only in
    its second half is not read as a field and its value."""
    words = text.split()
    if len(words) < 2:
        return None
    first_digit = next((i for i, w in enumerate(words)
                        if any(ch.isdigit() for ch in w)), None)
    if not first_digit or first_digit * 2 > len(words):
        return None
    return " ".join(words[:first_digit]), " ".join(words[first_digit:])


def _drop_sparse_column(clean):
    """A wide table whose FIRST column is a section label spanning many rows
    comes back mostly empty in that column — dropping it exposes the real
    key/value pairs. Structural criterion: the column is filled in a MINORITY of
    the rows (it was "at most a third", a fraction with nothing behind it; a
    label column's defining property is that most rows do not repeat it)."""
    while clean and len(clean[0]) > 2:
        filled = sum(1 for row in clean if row and row[0])
        if filled * 2 < len(clean):
            clean = [row[1:] for row in clean]
        else:
            break
    return clean


def read_pdf(path):
    """PDF → (prose_text, tables). Tables are detected with pdfplumber (it sees
    character COORDINATES and ruling lines — structure the plain-text dump
    destroys: cells gluing together like 'IPX 0Prob', columns losing their
    headers). Each table becomes row-dicts using its first row as the header;
    prose is everything outside table bounding boxes. No model calls."""
    import pdfplumber

    def _classify(grid):
        """One extracted grid → ('kv', pairs) | ('rows', rows) | None."""
        if not grid or len(grid) < 2:
            return None
        clean = [[(c or "").replace("\n", " ").strip() for c in raw]
                 for raw in grid]
        clean = _drop_sparse_column(clean)
        # KEY-VALUE table? (spec sheets: "Ekran | 15.6\" LCD") — rows with at
        # most 2 non-empty cells. Treating row 0 as a header there scrambled
        # every pair; instead each row IS a (key, value) pair.
        widths = [sum(1 for c in raw if c) for raw in clean if any(raw)]
        # MAJORITY, not unanimity: one messy multi-cell row (a wrapped
        # remark) must not disqualify a whole spec sheet — if MOST of the
        # rows are (key, value)-shaped the table is kv; a wider row joins
        # its extra cells into the value. (It read "at least two thirds"; the
        # majority is the boundary the sentence above actually describes, and
        # two thirds was a fraction chosen while looking at one spec sheet.)
        if widths and sum(1 for w in widths if w <= 2) * 2 > len(widths):
            pairs = []
            for raw in clean:
                cells = [c for c in raw if c]
                if len(cells) > 2:
                    pairs.append((cells[0], " ".join(cells[1:])))
                elif len(cells) == 2:
                    pairs.append((cells[0], cells[1]))
                elif len(cells) == 1:
                    # merged cell ('Boyutlar 360 mm * ...') — the key column
                    # came back empty. Inheriting the previous key
                    # mis-assigned values; keep it keyless, the FORMAT
                    # splitters below may still recover the key.
                    pairs.append(("", cells[0]))
            # FORMAT-cue recovery on damaged cells: composite 'A: x B: y'
            # values split on colons; a keyless 'Boyutlar 360 mm ...' cell
            # yields its leading no-digit words as the key.
            out = []
            for key, value in pairs:
                out.append((key, value))
                text = f"{key}: {value}" if key else value
                if ":" in text:
                    out += [p for p in split_kv_text(text) if p != (key, value)]
                elif not key:
                    lead = _split_leading_value(value)
                    if lead:
                        out.append(lead)
            return ("kv", out) if out else None
        header = clean[0]
        rows = []
        # A header row carrying DIGITS is data wearing a header's hat
        # (pdfplumber promoted the first row): keep it as a kv pair too.
        if len(header) == 2 and all(header) \
                and any(ch.isdigit() for ch in header[1]):
            rows.append({"__kv__": True, header[0]: header[1]})
        for raw in clean[1:]:
            row = {}
            for h, cell in zip(header, raw):
                if cell:
                    row[h] = cell
            if row:
                rows.append(row)
        return ("rows", rows) if rows else None

    prose_parts = []
    tables = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_kv_keys = set()
            for t in page.find_tables():
                got = _classify(t.extract())
                if not got:
                    continue
                kind, data = got
                if kind == "kv":
                    page_kv_keys |= {fold(k) for k, _v in data if k}
                    tables.append([{"__kv__": True, k: v} for k, v in data])
                else:
                    tables.append(data)

            text = page.extract_text() or ""

            # SECOND GEOMETRY PASS (vertical_strategy='text'): where ruling
            # lines lie, the default pass loses whole kv columns (measured:
            # 'Koruma derecesi' lost its IPX values, İŞLEMCI/Xubuntu rows
            # collapsed into one cell). The text-based pass recovers them but
            # can TRUNCATE other values — so keep only pairs whose key the
            # default pass did NOT already deliver AND whose value text
            # actually occurs on the page (a truncated '100 V-2' does not).
            page_norm = fold(re.sub(r"\s+", "", text))
            # WORD GUARD: the text-based pass can chop a word at a column seam
            # ('Ciha | z Listesi') — a key whose word is not a real token on
            # the page is such an amputation, and it was poisoning both the
            # graph and the field-name index (measured: 'Ciha: z Listesi').
            page_words = {fold(w) for w in re.findall(r"\w+", text)}
            try:
                alt_grids = page.extract_tables({"vertical_strategy": "text"})
            except Exception:                               # noqa: BLE001
                alt_grids = []
            recovered = []
            for grid in alt_grids:
                got = _classify(grid)
                if not got or got[0] != "kv":
                    continue
                for key, value in got[1]:
                    if not key or fold(key) in page_kv_keys:
                        continue
                    if fold(re.sub(r"\s+", "", value)) not in page_norm:
                        continue
                    if any(len(w) >= 3 and fold(w) not in page_words
                           for w in re.findall(r"\w+", key)):
                        continue
                    page_kv_keys.add(fold(key))
                    recovered.append((key, value))
            if recovered:
                tables.append([{"__kv__": True, k: v} for k, v in recovered])

            # FULL page text goes to prose — tables included. Excluding table
            # bboxes seemed clean but LOST information: wherever cell geometry
            # defeats the kv extractor (glued spec cells), the flattened text
            # was the only remaining carrier — and the evidence layer's lexical
            # retrieval still finds it. Structural facts are a BONUS on top of
            # full text, never a replacement.
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
                    session.evidence.add(f"{key}: {value}" if key else value,
                                         source)
                    if key:
                        wrote += session.learn_cell(key, "", value, source)
        else:
            wrote += session.learn_rows(rows, source)
    session._bulk = prev_bulk
    if prose.strip():
        session.learn_text(prose, source=source, deep=deep)
    # SECOND RENDERING: pypdf flattens the same pages differently (glued spec
    # lines that pdfplumber's layout scrambles, and vice versa). Feeding BOTH
    # renderings into the evidence layer doubles lexical-retrieval coverage;
    # dedup keeps identical sentences single and the cost is milliseconds.
    try:
        from pypdf import PdfReader
        alt = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
        if alt.strip():
            session.learn_text(alt, source=source, deep=False)
    except Exception:                                       # noqa: BLE001
        pass
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
