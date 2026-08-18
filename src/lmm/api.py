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
    Reporting them costs nothing; guessing at them would have been a lie.
    """

    __slots__ = ("abstained", "sources", "subject", "kind", "wrote")

    def __new__(cls, text, *, abstained=False, sources=(), subject="",
                kind="", wrote=()):
        self = super().__new__(cls, text)
        self.abstained = bool(abstained)
        self.sources = tuple(sources)
        self.subject = subject
        self.kind = kind
        self.wrote = tuple(wrote)
        return self

    def __repr__(self):
        state = "abstained" if self.abstained else "answered"
        return f"<Answer {state} sources={list(self.sources)} {str(self)!r}>"


class Learned:
    """What `learn` returns: what went in, and by which door."""

    __slots__ = ("facts", "tables", "source", "adapter")

    def __init__(self, facts=0, tables=0, source="", adapter=""):
        self.facts = facts        # triples the GATE admitted (not offered)
        self.tables = tables      # structured grids routed around the engine
        self.source = source      # the stamp every one of them now carries
        self.adapter = adapter    # which reader ran

    def __repr__(self):
        return (f"<Learned {self.facts} facts, {self.tables} tables "
                f"via {self.adapter} from {self.source!r}>")


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
        if not _looks_like_path(what):
            if not isinstance(what, str):
                raise TypeError(
                    f"learn() takes a file path or text, not {type(what).__name__}")
            text = what
            if not text.strip():
                return Learned(source=source or "#document", adapter="text")
            stamp = source or "#document"
            wrote, _ = session.learn_text(
                text, source=stamp, deep=True if deep is None else deep)
            return Learned(facts=wrote, source=stamp, adapter="text")

        path = os.fspath(what)
        ext = os.path.splitext(path)[1].lower()
        if ext in _UNSUPPORTED:
            raise ValueError(f"{os.path.basename(path)}: {_UNSUPPORTED[ext]}")

        deep_file = False if deep is None else deep
        from lmm import tables                              # noqa: PLC0415

        if ext == ".pdf":
            stamp = source or f"#pdf:{os.path.basename(path)}"
            facts, n = tables.learn_pdf(session, path, source=stamp,
                                        deep=deep_file)
            return Learned(facts, n, stamp, "pdf")

        if ext == ".docx":
            stamp = source or f"#docx:{os.path.basename(path)}"
            facts, n = tables.learn_docx(session, path, source=stamp,
                                         deep=deep_file)
            return Learned(facts, n, stamp, "docx")

        if ext in _SHEET:
            stamp = source or f"#xlsx:{os.path.basename(path)}"
            facts = tables.learn_xlsx(session, path, source=stamp)
            return Learned(facts, 1, stamp, "xlsx")

        if ext == ".csv":
            stamp = source or f"#csv:{os.path.basename(path)}"
            facts, n = _read_csv(session, path, stamp)
            return Learned(facts, n, stamp, "csv")

        if ext not in _TEXT:
            # Not a format we know — but the file may still be text, and
            # refusing it outright would be worse than trying and saying so.
            pass
        stamp = source or f"#file:{os.path.basename(path)}"
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read()
        except UnicodeDecodeError as gone:
            raise ValueError(
                f"{os.path.basename(path)} is not a text file and '{ext}' is "
                f"not a format LMM reads (pdf, docx, xlsx, csv, txt, md)"
            ) from gone
        wrote, _ = session.learn_text(text, source=stamp, deep=deep_file)
        return Learned(facts=wrote, source=stamp, adapter="text")

    # ------------------------------------------------------------------ ask

    def ask(self, question, explain=False):
        """Ask a question. Returns the answer — or an honest refusal.

        With `explain=True` the return additionally carries what the turn knows
        about itself: `.abstained`, `.sources`, `.subject`, `.wrote`. It is
        still a string, so nothing downstream needs to change.
        """
        session = self.session
        said = session.respond(question) or ""
        if not explain:
            return said
        return Answer(
            said,
            abstained=session.last_abstained,
            sources=[w for w in said.split() if w.startswith("#") and len(w) > 1],
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
