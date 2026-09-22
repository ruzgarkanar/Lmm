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
import math
import os
import re

from lmm import extract
from lmm.core.dataset import fold
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

    __slots__ = ("abstained", "sources", "subject", "kind", "wrote", "route",
                 "from_graph", "engine_error", "covered", "missing")

    def __new__(cls, text, *, abstained=False, sources=(), subject="",
                kind="", wrote=(), from_graph=False, route=(),
                engine_error=False, covered=0.0, missing=()):
        self = super().__new__(cls, text)
        self.abstained = bool(abstained)
        # WHY IT ABSTAINED, WHEN THE REASON WAS NOT THE MEMORY (W149).
        # An unreachable engine — missing client package, refused key,
        # no network — used to produce exactly the trace of honest
        # ignorance: "I don't know", abstained, empty route. A caller
        # that trusts abstention would record a real capability as
        # absent. `engine_error` is True only when the engine failed;
        # an abstention with it False is the memory's own, as before.
        self.engine_error = bool(engine_error)
        # HOW MUCH OF THE QUESTION THIS ANSWER CARRIED (W156), and what
        # it did not — a count over the question's own words, engine-free
        # and unable to fabricate. `abstained` says whether anything was
        # found; these two say how much, and what to ask about next. A
        # caller with grades of its own ("fully / partly / not covered")
        # builds them from this; the library ships the measurement, not
        # somebody's vocabulary.
        self.covered = float(covered)
        self.missing = tuple(missing)
        # THE ROUTE — which organs the turn consulted, in order (W86):
        # ("record",) for a row read with no model call, ("chain",
        # "refuse", "count") for a rescued count. The orchestration was
        # always there; now it is a fact on the answer instead of a
        # story in the stack.
        self.route = tuple(route)
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


def _stamps_in(said):
    """The provenance stamps a spoken answer carries, whole.

    The turn appends its mark as "(~ #stamp)" — see `session.UNCERTAIN` —
    and a stamp is a document name, which may hold spaces, punctuation
    and anything else a filename holds. Read to the closing bracket, not
    to the next space.
    """
    from lmm.session import UNCERTAIN
    out = []
    for chunk in re.findall(r"\((?:%s)\s*([^()]*)\)" % re.escape(UNCERTAIN),
                            said or ""):
        for stamp in chunk.split(" \u00b7 "):          # several, when several
            stamp = stamp.strip().rstrip(".,;:")
            if stamp.startswith("#") and len(stamp) > 1:
                out.append(stamp)
    if out:
        return out
    # A STAMP CAN ALSO ARRIVE BARE, without the mark's parentheses — a
    # composed line carries its source that way, and so does the graph
    # path. There the old reading is the right one: a word is a stamp.
    return [w for w in (x.strip("()[].,;:") for x in (said or "").split())
            if w.startswith("#") and len(w) > 1]


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

    __slots__ = ("facts", "tables", "source", "adapter", "evidence",
                 "warnings", "calls")

    def __init__(self, facts=0, tables=0, source="", adapter="", evidence=0,
                 warnings=(), calls=0):
        self.facts = facts        # triples the GATE admitted (not offered)
        self.tables = tables      # structured grids routed around the engine
        self.source = source      # the stamp every one of them now carries
        self.adapter = adapter    # which reader ran
        self.evidence = evidence  # sentences/windows this file added to the index
        self.warnings = tuple(warnings)   # what the reader could not read
        # WHAT THIS READING COST, IN ENGINE CALLS (W151). `deep` defaults
        # differently for a file (False) and for text handed in directly
        # (True, because a typed sentence is a fact being taught) — sound
        # per its docstring, and invisible: a reader who cleans a document
        # into a string first, which is the ordinary thing to do, pays one
        # call per sentence without ever being told. Reported here, beside
        # what was gained, so the trade is a number rather than a surprise.
        self.calls = calls

    def __bool__(self):
        """False when this file put NOTHING in the memory — so `if not
        m.learn(path):` is a working check rather than a habit that silently
        always passes."""
        return bool(self.facts or self.tables or self.evidence)

    def __repr__(self):
        warned = (" — " + "; ".join(self.warnings)) if self.warnings else ""
        # the calls are named only when there were any: a zero-call
        # reading is the normal one and saying so on every line would
        # bury the counts that matter
        cost = f", {self.calls} engine calls" if self.calls else ""
        return (f"<Learned {self.facts} facts, {self.tables} tables, "
                f"{self.evidence} evidence{cost} via {self.adapter} "
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
    ("reading PDF files needs pdfplumber ... pip install 'living-memory-model[pdf]'"); a broken
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

    def __init__(self, path=None, who="#operator", mode="STRICT", cache=True,
                 persona="",
                 warmth=None, reply_tokens=None, style="", identity=None,
                 encoder=None, dense=True, reranker="bundled"):
        # `persona` colours the voice of every spoken turn — greeting style,
        # tone, when to ask a clarifying question — and can never loosen the
        # gates, which read the output rather than any prompt.
        # The meaning channel's defaults live in `Session`, with the
        # store they belong to; these arguments only pass through, so a
        # caller of either class gets the same memory.
        self.session = Session(path, who=who, mode=mode, persona=persona,
                               identity=identity,
                               warmth=warmth, reply_tokens=reply_tokens,
                               style=style, encoder=encoder, dense=dense,
                               reranker=reranker)
        # THE SAME QUESTION, ASKED AGAIN, OVER A MEMORY THAT HAS NOT MOVED. The
        # answer cannot have changed, and re-deriving it costs the full 5.5
        # model calls a question costs (`benchmarks/COST.md` §2). `cache=False`
        # turns it off; see `_state` for what "has not moved" means and
        # `ask` for which turns are eligible at all.
        #
        # `cache=False` IS NOT "NOTHING IS REUSED", AND A BENCHMARKER MUST
        # KNOW IT. Below this object the engine pools its own deterministic
        # calls (`runtime._SEEN`: temperature 0, byte-exact prompt, per
        # PROCESS and therefore across `Memory` instances), and the
        # extractor memoises the sentences it has already read. So asking
        # one question twice in a single process can cost ZERO calls with
        # the cache off and with a second, freshly built memory — measured
        # by somebody timing this library, who briefly recorded a
        # spectacular and entirely false "0 calls" for a change they were
        # evaluating. Nothing is wrong with the pooling; it is right in
        # production, where a repeated deterministic call has one answer.
        # It is wrong to measure through, and the fix is the one they
        # found: give each variant its own process.
        self._cache = {} if cache else None

    # A BOUND, so a long-running memory does not accumulate every question ever
    # asked. Oldest out first (dicts are ordered), which on a repeated workload
    # is the entry least likely to be asked next.
    CACHE_KEEP = 512

    def _state(self):
        """A fingerprint of everything an answer can depend on.

        A stale answer is worse than an expensive one, so this is deliberately
        NOT a record count: a count does not move when a second source
        reinforces a fact, when `sleep()` fades one, or when a contradiction
        lowers a rival's trust — and each of those can change what is spoken or
        whether it is hedged. Summing the trust and the source counts catches
        every mutation the graph has (`Memory.write`/`Record.strengthen`,
        `core/dynamics.py`'s settle/fade/ceiling, `core/transitive.py`'s
        promotion), and the counts catch what is added.

        It is O(records) per question, which is microseconds against the
        seconds a model call takes. The residue, stated rather than hidden: two
        simultaneous trust changes that cancel to the same total would not move
        it. Nothing in the graph moves trust in pairs, and reading is free of
        it entirely — `about(touch=True)` updates access counters, which no
        answer reads.
        """
        memory = self.session.memory
        records = memory.records.values()
        store = self.session.evidence
        return (len(memory.records), len(memory.identities),
                len(memory.transitive),
                len(store.sentences),
                # THE RETRIEVAL AIDS COUNT TOO (W82): the bridge changes
                # WHICH lines a question reaches without adding a
                # sentence or a record, so a fingerprint blind to it
                # replays pre-bridge answers forever — a question cached
                # with its slow-path refusal would never meet the record
                # path the bridge just opened.
                sum(len(heads) for heads in store.head_bridge.values()),
                round(math.fsum(r.trust for r in records), 9),
                sum(len(r.sources) for r in records))

    # ---------------------------------------------------------------- learn

    def learn(self, what, source=None, deep=None):
        """Teach the memory something — see `_read_into` for `what` and `deep`.

        AN `expand=True` STOOD HERE AND IS GONE. It paid a model call per
        line at ingestion for the questions that line answers, indexed
        them separately, and bought reach for a reader who does not use
        the document's words. It was measured not to work (a real
        rephrasing never reaches its own line, so the filter kept only
        the queries that copied it) and the meaning channel now covers
        that class by construction. A switch nobody should turn on is
        not an option; it is a liability.
        """
        # WHAT THE READING COST IS COUNTED HERE, ONCE (W151) — around the
        # whole reading rather than at each of `_read_into`'s four exits,
        # so a new adapter cannot forget to report it.
        from lmm import runtime                          # noqa: PLC0415
        before = runtime.CALLS
        told = self._read_into(what, source=source, deep=deep)
        try:
            told.calls = runtime.CALLS - before
        except AttributeError:                           # noqa: BLE001
            pass
        return told

    def _read_into(self, what, source=None, deep=None):
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

    def ask(self, question, explain=False, fluent=False, shape=None,
            standalone=False, quoted=False):
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

        A question asked twice over an UNCHANGED memory is answered from the
        first answer, for nothing. Teach the memory anything — a fact, a
        document, a `sleep()` that fades one — and the kept answer is dropped
        rather than repeated: see `_state`. `Memory(..., cache=False)` turns it
        off entirely.
        """
        session = self.session
        key = (fold(question), bool(fluent))
        # A KEPT ANSWER IS NOT AN ANSWER TO "SHALL I?". While a research offer
        # is outstanding this turn's meaning is the offer's answer, and the
        # offer has to be CONSUMED by the flow that made it — replaying an
        # earlier answer would leave it standing, so the turn after that one
        # would be read as the approval instead. Measured on the TR set, where
        # "melvarit nedir" raises exactly this offer.
        before = (self._state() if self._cache is not None
                  and session._pending is None else None)
        if before is not None:
            kept = self._cache.get(key)
            if kept is not None and kept[0] == before:
                said = self._replay(question, kept[1])
                return self._told(said, question) if explain else said
        # teach=False: a question API cannot write memory — an imperative
        # brief ("draft a programme...") is a request here, never a lesson
        # THE QUESTION DOOR DOES NOT GAMBLE ON A CHAT. `ask` is asked a
        # question by contract, so the conversational speculation — which
        # buys latency in a consultation and is discarded here — is not
        # placed. Measured: one full chat completion per question.
        # THE CALLER'S DECLARED KIND RIDES ON THE SESSION FOR THIS TURN
        # ONLY (W157) — seated before the door, cleared after it, so a
        # batch's declaration cannot leak into a later ordinary turn.
        session.declared_shape = shape
        # WHICH MECHANISM KEEPS THE PROMISE (W159) — not whether it is
        # kept. `quoted=True` answers from ONE line the store holds and
        # checks it with word comparisons instead of a read-back and a
        # relation call; measured over two runs of fifteen questions,
        # about 30% fewer calls and tokens, at the cost of answers that
        # span several lines. The turn falls back to the ordinary path
        # whenever the quoted reading cannot be trusted.
        # `quoted=True` tries the quoted reading and falls back when it
        # cannot be trusted; `quoted="only"` lets the STORE's refusal
        # stand instead (W161) — an engine that declined outright still
        # falls back either way, because it never offered anything to
        # refuse.
        session.quoted_only = bool(quoted)
        session.quoted_strict = (quoted == "only")
        # A TURN THAT DECLINES THE CONVERSATION IT DID NOT HAVE (W158).
        # `standalone` ends the conversation before the turn and again
        # after it, so an independent question neither inherits a
        # subject it never raised nor leaves one for the next — which
        # is what a matrix of cells is, and what a `Memory` per cell
        # was standing in for.
        if standalone:
            self.reset()
        try:
            said = session.respond(question, fluent=fluent, teach=False,
                                   conversational=False) or ""
        finally:
            session.declared_shape = None
            session.quoted_only = False
            session.quoted_strict = False
            if standalone:
                self.reset()
        # WHICH TURNS MAY BE KEPT, and it is the narrow set. A turn that WROTE
        # is not a repeat of itself — asking it again re-enters the gate. A
        # turn that left a research offer outstanding means "shall I?", and
        # the next turn answers it. A CHAT turn reads `session.history`, which
        # this fingerprint does not cover and which every turn changes. And if
        # the turn moved the memory in any other way, the fingerprint it would
        # be filed under is already gone — so it is compared again afterwards
        # rather than assumed.
        if (before is not None
                and session.last_kind == extract.ASK
                and not session.last_written
                and session._pending is None
                and self._state() == before):
            self._cache[key] = (before, (
                said, session.last_abstained, session.last_from_graph,
                session.last_subject, session.last_kind))
            while len(self._cache) > self.CACHE_KEEP:
                del self._cache[next(iter(self._cache))]
        return self._told(said, question) if explain else said

    def _replay(self, question, kept):
        """Re-speak a kept answer, and leave the session saying about this turn
        exactly what it said about the first one.

        A caller that reads `session.last_abstained` after `ask` — every
        benchmark in this repository does — must not be able to tell a cached
        turn from a paid one, because the two are the same turn. The
        conversation window is appended to as well, for the same reason.

        What a replayed turn does NOT do is count towards `session.turns`, and
        that is deliberate: the turn counter schedules `sleep()`, which is
        maintenance over records that were READ and written, and this turn read
        nothing. The effect is that maintenance is paced by work done rather
        than by questions asked; it cannot produce a stale answer, because the
        state stamp is what decides that and `sleep()` moves it.
        """
        session = self.session
        said, abstained, from_graph, subject, kind = kept
        session.last_written = []
        session.last_abstained = abstained
        session.last_from_graph = from_graph
        session.last_subject = subject
        session.last_kind = kind
        session._mark = ""
        session.history.append({"role": "user", "content": question})
        session.history.append({"role": "assistant", "content": said})
        session.history = session.history[-12:]
        return said

    def _told(self, said, asked=""):
        """What this turn knows about itself, read off the session."""
        from lmm import evidence as _ev                   # noqa: PLC0415
        session = self.session
        covered, missing = _ev.carried(said, asked) if asked else (0.0, ())
        return Answer(
            said,
            abstained=session.last_abstained,
            engine_error=getattr(session, "last_engine_error", False),
            covered=covered,
            missing=missing,
            from_graph=session.last_from_graph,
            # THE MARK HAS A SHAPE, so it is read as one. Splitting the
            # answer on whitespace and keeping the words that start with
            # '#' cut every stamp at its first space — and a document
            # named "Course 02" arrived as "#docx:Course", which two
            # siblings then share. Provenance that cannot tell two
            # documents apart is not provenance. The mark is the
            # parenthesis the turn appends: "(~ #stamp)", stamp running
            # to the closing bracket.
            sources=_stamps_in(said),
            subject=session.last_subject,
            kind=session.last_kind,
            wrote=tuple(session.last_written),
            route=tuple(getattr(session, "last_route", ())),
        )

    # ----------------------------------------------------------------- keep

    def reset(self):
        """End the CONVERSATION. Nothing that was learned is forgotten.

        A turn that names no document of its own reads the one the
        conversation was about — right in a conversation, and
        contamination in a matrix of independent cells. Measured from
        the field, both directions: a cell answers when asked first and
        abstains when asked after nine unrelated cells in the same
        `Memory`, with nothing but the order different. The inheritance
        EARNS its place (the same reporter measured 77.3% with it
        against 72.7% without), so it is not turned off — what was
        missing is the caller's say over it. Until now the only way to
        get an isolated turn was to build a second `Memory`, which
        works because ingestion costs nothing, but a workaround is not
        an intent.

        What this clears is exactly the conversation: the recent turns,
        the subject the last turn was about, the consultation's brief
        and the documents the topic had come to be about. The graph,
        the evidence index and the retrieval aids are the memory, and a
        memory is not a conversation: `m.facts` and every stored line
        are the same after this call as before it.
        """
        session = self.session
        session.history = []
        session.last_subject = ""
        session._prior_subject = ""
        session._brief = []
        session._scope_now = set()
        session.topic.said = []
        session.topic.sources = set()
        return self

    def where(self, term):
        """Which documents mention this — names and counts, no engine, ms.

        The census counterpart of `about`: `about` reads what the graph holds
        on a concept, `where` counts which DOCUMENTS speak of a term. Returns
        [(document_name, sentence_count)], most-mentioned first."""
        from lmm.evidence import _source_name       # noqa: PLC0415
        return [(_source_name(src), n)
                for src, n in self.session.evidence.where(term)]

    def compose(self, brief, seats=24, topics=None, on_line=None):
        """A structured draft from the memory — blend, but never invent.

        `ask` answers a question in a sentence; this builds a DOCUMENT: a
        training outline, a briefing, a comparison — organised by the engine,
        grounded line by line in the evidence, each material line carrying
        the name of the document it came from. Lines the evidence does not
        support are dropped on the way out, exactly as the short path drops
        them; what survives is returned with the sources it rests on.

        Returns (text, sources). An empty store, or a brief the memory holds
        nothing about, refuses rather than improvising."""
        return self.session.compose(brief, seats=seats, topics=topics,
                                    on_line=on_line)

    def themes(self, least=3, most=12):
        """The subjects this corpus falls into — documents that belong
        together, and the entities that hold them together. No engine.

        A corpus has subjects no single document names: a catalogue's
        programme families, a manual's subsystems. The instrument is the
        one the competition uses — community detection over a graph —
        with the difference that the graph here was built without a model
        (co-mentions weighted by log-likelihood) and no summary is
        written for any community. Measured on a 103-document catalogue:
        under a hundredth of a second, zero calls, and the largest groups
        fell on the catalogue's own families.

        Returns [{"sources": [...], "entities": [...]}], largest first.
        `least` is the smallest group worth calling a subject.

        WHY NO SUMMARY. Twice measured in this project: an engine-written
        document profile took retrieval from 100% to 76%, and a
        summarising indexer is how specific values go missing. A theme
        here is EVIDENCE — a set of documents — not prose about them.

        WHAT THIS DOES NOT DO, MEASURED. It does not answer "what is this
        family about". Scoping the composer to a community's documents
        was tried and came back with ONE course's outline rather than the
        family's shared subject, because the composer organises material
        and nothing in the material states a theme. The grouping is a
        structural reading — which documents belong together, and on what
        entities — and that is all it claims.
        """
        from lmm import mentions                        # noqa: PLC0415
        store = self.session.evidence
        if not store.sentences:
            return []
        graph = mentions.Graph.build(store)
        out = []
        for group in graph.communities(least=least)[:most]:
            out.append({
                "sources": list(group["sources"]),
                "entities": [" ".join(e) if isinstance(e, tuple) else str(e)
                             for e in group["entities"][:12]],
            })
        return out

    def distil(self, text, source=None, speaker=None):
        """Write the EVENTS a passage reports into the graph.

        The engine lists each event as thing/what-happened, THIS
        passage's own words admit it (an invented event fails there and
        never reaches the graph), and each survivor becomes one gated,
        dated record — while the passage itself is kept as evidence, so
        nothing is lost. One call per passage, and the passages worth
        it are the ones where events melt into talk: chat turns, minutes,
        logs. Documents that state their structure need nothing of the
        sort — their rows already go to the graph for free.

        Returns how many events were written."""
        return self.session.distil(text, source=source or "#document",
                                   speaker=speaker)

    def attach_dense(self, encoder):
        """Give this memory the MEANING channel (`lmm/dense.py`).

        `encoder` maps a list of strings to a list of vectors — a local
        sentence model, a static distilled embedding, an in-house
        service, or a vendor's API. The library ships no opinion about
        whose vectors are best; it ships the channel, fused with the
        word channel by rank (RRF) so neither can outvote the other.

        Returns how many lines were embedded. With no encoder the
        memory answers exactly as it did before: the channel is absent,
        not degraded."""
        return self.session.evidence.attach_dense(encoder)

    def bridge(self):
        """Teach the store, once, what words readers ask its fields with.

        One engine call per field head, paid at the operator's request
        (asked for rather than assumed) and saved with the store;
        after it, questions like "kaç saat?" against a field written
        COURSE LENGTH are answered by the record itself — milliseconds,
        no model call, every gate unchanged. Calling it again is free:
        a head already bridged is not asked about twice."""
        return self.session.learn_bridges()

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
