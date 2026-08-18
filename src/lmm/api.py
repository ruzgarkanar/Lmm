"""The front door: one object, three verbs.

Everything under this file already worked. What did not work was FINDING it. To
teach a document to LMM you had to know that prose goes to `Session.learn_text`
with a `deep` flag whose right value depends on the document's size, that a
spreadsheet goes to `tables.learn_xlsx` instead, that a PDF goes to
`tables.learn_pdf`, and that Word was not wired up at all. Four entry points and
a judgement call, before the first question.

That is a packaging failure, not a design one, and it is fixed here rather than
below: `Memory` is a WRAPPER. It adds no gate, no threshold, no behaviour. It
picks the adapter the file extension already implies and calls exactly what a
reader of the old API would have called by hand. `Memory.session` is the real
`Session`, and the older API remains supported and unchanged — this module is
additive.

    from lmm import Memory

    m = Memory("mind.lmm")
    m.learn("manual.pdf")
    print(m.ask("what is the screen's diagonal?"))
    m.save()
"""
import csv
import os
import re

from lmm.session import Session

# EXTENSION → ADAPTER. The routing table is the whole of the dispatch: there is
# no sniffing, no content inspection, no guessing. A file's extension is a
# declaration by whoever named it, and when it is wrong the adapter's own error
# is a better diagnostic than anything a heuristic here would invent.
_TEXT = {".txt", ".text", ".md", ".markdown", ".rst", ".log", ""}
_SHEET = {".xlsx", ".xlsm", ".xltx", ".xls"}
_UNSUPPORTED = {
    ".doc": "legacy Word (.doc) is not readable; save it as .docx",
    ".ppt": "PowerPoint is not supported",
    ".pptx": "PowerPoint is not supported",
    ".pages": "Pages is not supported; export it as .docx or .pdf",
    ".numbers": "Numbers is not supported; export it as .xlsx or .csv",
}


class Answer(str):
    """What `ask` returns. It IS the answer string — printing it, comparing it
    and concatenating it all behave exactly as they did before this class
    existed — and it additionally carries what the turn knows about itself.

    The extra facts are not computed here. `abstained` is the session's own
    structural stamp (`Session.last_abstained`): a turn is an abstention when
    re-extracting the spoken sentence finds no claim in it, which is decided in
    whatever language the answer came out in and is the same organ the
    fabrication gate trusts. `sources` are the stamps the answer itself carries.
    `from_graph` is which path spoke — the graph alone (`lmm/lookup.py`, no
    model call in it) or the engine. Reporting them costs nothing; guessing at
    them would have been a lie.
    """

    __slots__ = ("abstained", "sources", "subject", "kind", "wrote",
                 "from_graph")

    def __new__(cls, text, *, abstained=False, sources=(), subject="",
                kind="", wrote=(), from_graph=False):
        self = super().__new__(cls, text)
        self.abstained = bool(abstained)
        # WHICH PATH ANSWERED — the graph alone, or the engine. It is a fact
        # about cost and about dependency: a `from_graph` turn spent no model
        # call, so it reads the same on a 3B local build as on a hosted one.
        self.from_graph = bool(from_graph)
        self.sources = tuple(sources)
        self.subject = subject
        self.kind = kind
        self.wrote = tuple(wrote)
        return self

    def __repr__(self):
        state = "abstained" if self.abstained else "answered"
        how = " from-graph" if self.from_graph else ""
        return (f"<Answer {state}{how} sources={list(self.sources)} "
                f"{str(self)!r}>")


class Learned:
    """What `learn` returns: what went in, what did NOT, and by which door.

    `facts` and `tables` alone could not tell those two apart, and the field
    trial measured the cost. An image-only PDF — what any office scanner
    produces — ingested in 0.00 s and reported `<Learned 0 facts, 0 tables via
    pdf>`, which is the same report a perfectly read prose document gives,
    because `facts=0` is the NORMAL result for prose ingested with deep=False.
    Every later question was correctly refused, so from the user's side a
    document that was never read was indistinguishable from one the system
    happens to know nothing about.

    `evidence` is the number of sentences and windows the reader actually put
    in the index — the one count that separates "nothing extractable" from
    "extracted fine, no triples" — and `warnings` are the structural things
    noticed while reading, in plain words. Both are observations, not
    judgements: nothing here changes what was learned.
    """

    __slots__ = ("facts", "tables", "source", "adapter", "evidence", "warnings")

    def __init__(self, facts=0, tables=0, source="", adapter="", evidence=0,
                 warnings=()):
        self.facts = facts        # triples the GATE admitted (not offered)
        self.tables = tables      # structured grids routed around the engine
        self.source = source      # the stamp every one of them now carries
        self.adapter = adapter    # which reader ran
        self.evidence = evidence  # sentences/windows this file added to the index
        self.warnings = tuple(warnings)   # what the reader could not read

    def __bool__(self):
        """False when this file put NOTHING in the memory — so `if not
        m.learn(path):` is a working check rather than a habit that silently
        always passes."""
        return bool(self.facts or self.tables or self.evidence)

    def __repr__(self):
        warned = (" — " + "; ".join(self.warnings)) if self.warnings else ""
        return (f"<Learned {self.facts} facts, {self.tables} tables, "
                f"{self.evidence} evidence via {self.adapter} "
                f"from {self.source!r}{warned}>")


def _looks_like_path(what):
    """Is this argument a FILE, or is it the text itself?

    `learn` takes both, so this question has to be answered before anything
    else, and it has to be answered without a stat() call on a 500 KB document
    body — on most systems that raises rather than returning False. A path is
    short and has no line breaks; that test is cheap and total, and only what
    survives it is allowed to touch the filesystem.
    """
    if isinstance(what, os.PathLike):
        return True
    if not isinstance(what, str):
        return False
    if len(what) > 4096 or "\n" in what or "\r" in what or "\x00" in what:
        return False
    try:
        return os.path.isfile(what)
    except (OSError, ValueError):
        return False


# The extensions this front door ROUTES on — every one of them is a filename's
# own declaration of a format, and nothing here is about any document's content.
_KNOWN = (_TEXT | _SHEET | set(_UNSUPPORTED)
          | {".pdf", ".docx", ".csv", ".htm", ".html", ".xhtml", ".xml"}) - {""}

# A MARKUP TAG, as a shape: angle brackets around something that is not a
# sentence break. Counting the characters they occupy answers "is what I just
# read prose, or is it a document's plumbing" without knowing one markup
# language from another — HTML, XML and SVG all fail this test the same way.
_TAG = re.compile(r"<[^<>\s][^<>]*>", re.S)


def _markup_share(text):
    """The fraction of this text's characters that sit inside tags."""
    if not text:
        return 0.0
    return sum(len(m.group()) for m in _TAG.finditer(text)) / len(text)


def _is_dir(what):
    """Is this argument a directory? Asked of a string that may be a 500 KB
    document body, so the same cheap shape test as `_looks_like_path` guards
    the stat() call."""
    if not isinstance(what, str) or len(what) > 4096 or "\n" in what:
        return False
    try:
        return os.path.isdir(what)
    except (OSError, ValueError):
        return False


def _names_a_file(what):
    """Does this argument NAME A FILE that is not there, rather than say
    something?

    `learn` takes both a path and the text itself, so a typo in a path used to
    be learned as a sentence — `m.learn("manual.pdf")` on a missing file
    reported a cheerful `<Learned 0 facts>` for a document nobody ever opened,
    and on the core install it reached for the default engine and raised
    `ModuleNotFoundError: No module named 'torch'` for a typo.

    The test is format, not content: one line, ending in one of the extensions
    this dispatcher routes on, and — if it contains spaces, which prose does
    and filenames mostly do not — sitting in a directory that exists. A
    sentence that happens to end in a word with a dot in it does not survive
    all three.
    """
    if not isinstance(what, str) or "\n" in what or "\r" in what:
        return False
    text = what.strip()
    if not text or os.path.splitext(text)[1].lower() not in _KNOWN:
        return False
    if not any(c.isspace() for c in text):
        return True
    parent = os.path.dirname(text)
    return bool(parent) and os.path.isdir(parent)


def _read_with(adapter, path, reader, *args, **kwargs):
    """Run one reader and let NO third-party exception reach the caller.

    Measured: a truncated PDF gave 51 lines of pdfminer traceback, an empty one
    29, and a password-protected one 38 lines ending in `PdfminerException:`
    with an empty message. The install path already sets the standard here
    ("reading PDF files needs pdfplumber ... pip install 'lmm[pdf]'"); a broken
    FILE is the more common event and had no such contract. Our own errors
    (a missing extra, an unsupported format) pass through untouched.
    """
    try:
        return reader(*args, **kwargs)
    except (ImportError, ValueError, KeyboardInterrupt, MemoryError):
        raise
    except Exception as broke:                              # noqa: BLE001
        detail = str(broke).strip() or type(broke).__name__
        raise ValueError(
            f"{os.path.basename(path)}: the {adapter} reader could not read "
            f"this file ({detail}) — it may be corrupt, truncated, or "
            f"password-protected"
        ) from None


def _read_csv(session, path, source):
    """csv → rows → straight into the graph. Stdlib only, so the format that is
    most often handed around as 'a table' costs the core install nothing."""
    with open(path, newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = [{k: v.strip() for k, v in row.items()
                 if k and v and v.strip()}
                for row in csv.DictReader(handle, dialect=dialect)]
    rows = [r for r in rows if r]
    if not rows:
        return 0, 0
    prev = getattr(session, "_bulk", False)
    session._bulk = True
    try:
        wrote = session.learn_rows(rows, source)
    finally:
        session._bulk = prev
    return wrote, 1


class Memory:
    """A living memory: teach it, ask it, save it.

    Give a path and the memory persists there (the graph and the evidence index
    both). Give nothing and it lives for as long as the process does.
    """

    def __init__(self, path=None, who="#operator", mode="STRICT"):
        self.session = Session(path, who=who, mode=mode)

    # ---------------------------------------------------------------- learn

    def learn(self, what, source=None, deep=None):
        """Teach the memory something. `what` is a file path or the text itself.

        The format is read off the extension: .pdf and .docx split into tables
        (straight to the graph, no model calls) and prose (to the evidence
        layer); spreadsheets and .csv are all table; anything else readable as
        text is text.

        `deep` decides whether the ENGINE also mines each sentence for triples,
        and its default is deliberately not a constant. Left as None it is True
        for text handed in directly and False for a file, because those are two
        different acts: a sentence typed at the memory is a fact being taught
        and is worth a model call, whereas a document is bulk material whose
        structure the adapters already extract and whose prose the evidence
        index already answers from — running the extractor over every sentence
        of a 350-page manual costs minutes to add little. Pass it explicitly to
        override; nothing about the pipeline changes either way.

        Returns a `Learned` report.
        """
        session = self.session
        before = len(session.evidence.sentences)

        def gained():
            return len(session.evidence.sentences) - before

        if not _looks_like_path(what):
            if not isinstance(what, str):
                raise TypeError(
                    f"learn() takes a file path or text, not {type(what).__name__}")
            if _is_dir(what):
                raise ValueError(
                    f"{what}: that is a directory, not a file — learn() reads "
                    f"one document at a time")
            if _names_a_file(what):
                raise ValueError(
                    f"{what}: no such file. learn() takes an existing path or "
                    f"the text itself, and this looks like a path")
            text = what
            if not text.strip():
                return Learned(source=source or "#document", adapter="text")
            stamp = source or "#document"
            wrote, _ = session.learn_text(
                text, source=stamp, deep=True if deep is None else deep)
            return Learned(facts=wrote, source=stamp, adapter="text",
                           evidence=gained())

        path = os.fspath(what)
        if not os.path.isfile(path):
            raise ValueError(
                f"{path}: no such file"
                + (" — that is a directory" if os.path.isdir(path) else ""))
        ext = os.path.splitext(path)[1].lower()
        if ext in _UNSUPPORTED:
            raise ValueError(f"{os.path.basename(path)}: {_UNSUPPORTED[ext]}")

        deep_file = False if deep is None else deep
        from lmm import tables                              # noqa: PLC0415

        def report(facts, n, stamp, adapter):
            """One reader's result, plus what it could not read.

            The only warning any reader can raise from counts alone is the one
            that matters most: NOTHING CAME OUT. It is not a tuned threshold —
            zero characters is the boundary between a document that was read
            and a document that was opened.
            """
            got = gained()
            warn = []
            if not (facts or n or got):
                warn.append(
                    f"no text came out of this file — {adapter} found zero "
                    f"readable characters"
                    + (", so it is almost certainly a scan: LMM does not OCR, "
                       "run OCR on it and learn the result" if adapter == "pdf"
                       else ""))
            return Learned(facts, n, stamp, adapter, evidence=got,
                           warnings=warn)

        if ext == ".pdf":
            stamp = source or f"#pdf:{os.path.basename(path)}"
            facts, n = _read_with("pdf", path, tables.learn_pdf, session, path,
                                  source=stamp, deep=deep_file)
            return report(facts, n, stamp, "pdf")

        if ext == ".docx":
            stamp = source or f"#docx:{os.path.basename(path)}"
            facts, n = _read_with("docx", path, tables.learn_docx, session,
                                  path, source=stamp, deep=deep_file)
            return report(facts, n, stamp, "docx")

        if ext in _SHEET:
            stamp = source or f"#xlsx:{os.path.basename(path)}"
            facts = _read_with("xlsx", path, tables.learn_xlsx, session, path,
                               source=stamp)
            return report(facts, 1 if facts else 0, stamp, "xlsx")

        if ext == ".csv":
            stamp = source or f"#csv:{os.path.basename(path)}"
            facts, n = _read_with("csv", path, _read_csv, session, path, stamp)
            return report(facts, n, stamp, "csv")

        warn = []
        if ext not in _TEXT:
            # Not a format we know — but the file may still be text, and
            # refusing it outright would be worse than trying. What was NOT
            # allowed to stay true is the "trying" part being silent: an .html
            # file went down this branch and reported `adapter="text"`, which
            # is technically true and practically a lie.
            warn.append(
                f"'{ext}' is not a format LMM has a reader for (pdf, docx, "
                f"xlsx, csv, txt, md) — it was read as plain text")
        stamp = source or f"#file:{os.path.basename(path)}"
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except UnicodeDecodeError as gone:
            raise ValueError(
                f"{os.path.basename(path)} is not a text file and '{ext}' is "
                f"not a format LMM reads (pdf, docx, xlsx, csv, txt, md)"
            ) from gone
        # WAS THAT PROSE, OR A DOCUMENT'S PLUMBING? Measured on RFC 9110 as
        # HTML: 44 773 evidence entries, most of them carrying markup such as
        # `<meta content="Common,Latin" name="scripts">`. The share of
        # characters inside tags says so without knowing one markup language
        # from another, and half is not a tuned number — it is the point where
        # what was indexed stops being mostly the document's words.
        share = _markup_share(text)
        if share > 0.5:
            warn.append(
                f"{share:.0%} of this file's characters are inside markup "
                f"tags, not prose — the evidence index will be full of markup; "
                f"convert it to text first")
        wrote, _ = session.learn_text(text, source=stamp, deep=deep_file)
        got = gained()
        if not (wrote or got):
            warn.append("no text came out of this file — it is empty as far "
                        "as the reader is concerned")
        return Learned(facts=wrote, source=stamp, adapter="text",
                       evidence=got, warnings=warn)

    # ------------------------------------------------------------------ ask

    def ask(self, question, explain=False, fluent=False):
        """Ask a question. Returns the answer — or an honest refusal.

        With `explain=True` the return additionally carries what the turn knows
        about itself: `.abstained`, `.sources`, `.subject`, `.wrote`,
        `.from_graph`. It is still a string, so nothing downstream needs to
        change.

        `fluent=False` (the default) lets the GRAPH answer where the graph can:
        a question that names a field or a value the memory already holds comes
        back as that record — `vorlin — type → liquid` — in microseconds, with
        no model call and therefore nothing for a model to invent. `.from_graph`
        says which path a turn took.

        `fluent=True` asks for a sentence instead, and spends the calls to get
        one. The trade is stated rather than chosen for you: prose costs
        roughly seven model calls per question (`benchmarks/COST.md`) and its
        wording depends on the engine, while the record costs nothing and does
        not.
        """
        session = self.session
        said = session.respond(question, fluent=fluent) or ""
        if not explain:
            return said
        return Answer(
            said,
            abstained=session.last_abstained,
            from_graph=session.last_from_graph,
            # The provenance mark is written as '(~ #stamp)', so a stamp can
            # arrive wearing the mark's punctuation; strip the bracketing, not
            # the stamp.
            sources=[s for s in (w.strip("()[].,;:") for w in said.split())
                     if s.startswith("#") and len(s) > 1],
            subject=session.last_subject,
            kind=session.last_kind,
            wrote=tuple(session.last_written),
        )

    # ----------------------------------------------------------------- keep

    def save(self, path=None):
        """Persist the graph and the evidence index. A path given here becomes
        this memory's path, so `Memory()` can be given one late."""
        if path:
            self.session.path = path
        if not self.session.path:
            raise ValueError(
                "this memory has nowhere to save to — construct it with "
                "Memory('mind.lmm') or call save('mind.lmm')")
        self.session.save()
        return self.session.path

    # -------------------------------------------------------------- inspect

    @property
    def facts(self):
        """How many records the gate has admitted."""
        return len(self.session.memory.records)

    def about(self, label):
        """The records standing under one subject label — the graph, directly.

        Resolution goes through `link.resolve(create=False)` — the same organ
        the verifier uses — so an inspector sees a subject by the same name the
        writer stored it under: labels are folded on the way in, and a caller
        typing "Ada" would otherwise miss the record filed under "ada".
        `create=False` is the load-bearing half; `identify` would OPEN a new
        identity for an unknown label and grow the graph on every lookup.
        Unknown label, empty list.
        """
        from lmm import link                                # noqa: PLC0415
        memory = self.session.memory
        key = link.resolve(memory, label, self.session.vectors, create=False)
        if key is None:
            return []
        return list(memory.about(key, touch=False))

    def __len__(self):
        return self.facts

    def __repr__(self):
        where = self.session.path or "in memory"
        return f"<Memory {self.facts} facts, {where}>"
