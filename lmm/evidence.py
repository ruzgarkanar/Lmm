"""EVIDENCE STORE — the document's sentences + a folded inverted index (NO embeddings).

Half of the fix for the representation bottleneck: the graph holds structure
(index, derivation, contradiction, multi-hop), the SENTENCE holds evidence
(numbers, ranges, nuance — everything that doesn't fit a triple). The answer is
built graph-first, but evidence sentences also enter the context; the gate still
audits. Difference from RAG: retrieval is not an embedding-similarity gamble but
folded word intersection — deterministic, language-independent (fold + prefix
tolerance), explainable ("these words occurred in this sentence").

Persistence: `<path>.evidence` (JSON) next to the memory path — the v3 core is
untouched, its format stays closed.
"""
import json
import os
import re
import unicodedata

from v3.dataset import fold

_WORD = re.compile(r"\w+", re.UNICODE)

# The length at which a token is a content word on its own. Everything shorter
# has to EARN its place (see `_words`).
CONTENT = 3


def _boundaries(text):
    """FORMAT-boundary splitter (no language rule): PDF flattening glues
    adjacent cells into one token ("IPX 7Ayak", "GüçAdaptörüPil", "8 g DDR
    4sabit disk"). Two purely typographic transitions become token
    boundaries: lowercase→UPPERCASE (camel seam between glued cells) and
    digit↔letter (a number fused to the next cell's word). Without this the
    digit gate cannot see a legitimate "7" hiding inside "7Ayak" and drops a
    CORRECT answer (measured: the IPX7 loss).

    THE CAMEL SEAM NEEDS A LEFT SIDE THAT IS ITSELF A WORD. A glued cell
    contributes a whole token before the seam ("Güç|Adaptörü"); a UNIT PREFIX
    does not ("mAh", "kWh", "mmHg", "µF"). Splitting the latter destroyed both
    halves — "6500 mAh" became 6500 + m + Ah, and since neither fragment
    reaches content-word length, the capacity's unit disappeared from the
    index entirely (measured: "kaç mAh" found nothing). So the seam only cuts when
    the run of letters before it is long enough to be a content word on its
    own; below that it is read as a unit prefix and the token stays whole.
    The bound is the SAME structural constant the content-word test uses — it
    is not a second, tunable threshold."""
    out = []
    prev = ""
    run = 0                     # letters since the last boundary
    for ch in text:
        cut = bool(prev) and (
            (prev.islower() and ch.isupper() and run >= CONTENT)
            or (prev.isdigit() and ch.isalpha())
            or (prev.isalpha() and ch.isdigit()))
        if cut:
            out.append(" ")
            run = 0
        out.append(ch)
        run = run + 1 if ch.isalnum() else 0
        prev = ch
    return "".join(out)


def _words(text, known=()):
    """Content words: folded, combining-mark-free, >=3 letters OR a digit — OR
    a short token that a NUMBER vouches for.

    The digit exception is critical: if the 3 of "Tier 3" is dropped, the
    coverage gate cannot see a number swap (a fabricated "Tier 1").

    THE UNIT EXCEPTION (measured loss class: "how many inches", "how many kg",
    "6500 mAh"). A length test is a proxy for "does this token carry content",
    and it is wrong exactly where a measurement lives: kg, ml, mm, V, Hz, VA
    are the part of "2 kg" that says WHAT the 2 is. The structural signal is
    ADJACENCY: a token standing next to a numeric token is that number's unit,
    so it is content regardless of its length. Without this `_words("Ağırlık 2
    kg")` lost the kg and the number floated unattached.

    `known`: short tokens the STORE has already seen standing next to a number
    (`SentenceStore.units`). A QUESTION carries no number — "kaç kg" has no 2
    for the kg to lean on — so the vouching has to come from the corpus, where
    the same token was measured to be a unit. Derived from data, not a list:
    nothing is a unit here until some indexed sentence made it one."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    toks = [fold(w) for w in _WORD.findall(_boundaries(text))]
    out = []
    for i, t in enumerate(toks):
        if len(t) >= CONTENT or t.isdigit():
            out.append(t)
            continue
        near = (i > 0 and toks[i - 1].isdigit()) or (
            i + 1 < len(toks) and toks[i + 1].isdigit())
        if near or t in known:
            out.append(t)
    return out


def _anchors(text):
    """token -> the numeric tokens it stands NEXT TO.

    A measurement binds a number and its unit into one thing; this records
    that binding so two texts can be asked whether they are talking about the
    SAME measurement. Used by the coverage gate — see `coverage`."""
    toks = _tokens(text)
    out = {}
    for i, t in enumerate(toks):
        near = set()
        if i > 0 and toks[i - 1].isdigit():
            near.add(toks[i - 1])
        if i + 1 < len(toks) and toks[i + 1].isdigit():
            near.add(toks[i + 1])
        if near:
            out.setdefault(t, set()).update(near)
    return out


def _tokens(text):
    """Ordered token sequence (folded, short ones included) — for adjacency checks."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [fold(w) for w in _WORD.findall(_boundaries(text))]


def digits_ok(answer, block):
    """The DIGIT discipline on its own: every digit in the answer must be in the
    block AND in at least one same-neighbor bigram. Even if word-coverage fails,
    this condition is non-negotiable — not even the engine-support check
    (generate.supported) can RESCUE a digit violation (a changed number =
    fabrication, period)."""
    bt = _tokens(block)
    bigrams = set(zip(bt, bt[1:]))
    block_digits = {t for t in bt if t.isdigit()}
    at = _tokens(answer)
    for i, t in enumerate(at):
        if not t.isdigit():
            continue
        if t not in block_digits:
            return False
        prev = at[i - 1] if i > 0 else None
        nxt = at[i + 1] if i + 1 < len(at) else None
        if prev is None and nxt is None:
            continue
        if (prev, t) not in bigrams and (t, nxt) not in bigrams:
            return False
    return True


def digits_present(answer, block):
    """Loose digit condition (for the second tier): every digit in the answer
    must EXIST in the block — no adjacency required. Legitimate COMPOSITION
    ("07 is high priority" merges two separate lines) moves a digit next to new
    neighbors; the bigram condition was killing it. The engine-support check
    (supported) takes over adjacency's role."""
    bt = _tokens(block)
    block_digits = {t for t in bt if t.isdigit()}
    return all(t in block_digits for t in _tokens(answer) if t.isdigit())


def coverage(answer, block, question=""):
    """WHAT FRACTION of the answer's content-words comes from the given block
    (+question) — the ratio form of `covered`, digits aside.

    `covered` is this measure thresholded at 1.0. It exists as a ratio because
    a binary verdict cannot RANK: with several candidate answers to choose
    between, "0.9 of it is grounded" and "0.3 of it is grounded" are the same
    'False', and the selector had nothing to prefer. The uninvented-0
    guarantee is not weakened by the split — a candidate below 1.0 is still
    only speakable if the support check (generate.supported) confirms it; see
    session._answer.

    Word matching is prefix-tolerant (inflection: "bandındadır"~"bandında")
    and, for measurements, ANCHOR-tolerant — see below."""
    given_text = block + " " + question
    given = set(_words(given_text))
    words = set(_words(answer))
    if not words:
        return 0.0
    here = there = None
    hit = 0
    for w in words:
        if w in given or w.isdigit():
            hit += 1
            continue
        if any((len(g) >= 5 and w.startswith(g) and len(w) - len(g) <= 3)
               or (len(w) >= 5 and g.startswith(w) and len(g) - len(w) <= 3)
               for g in given):
            hit += 1
            continue
        # THE UNIT ANCHOR. The prefix rule above needs a root of 5 letters, and
        # a unit does not have five: "Ekran 15.6 inçtir" (a fluent sentence,
        # which is exactly what the answer prompt ASKS for) failed against
        # "Ekran 15.6 inç" because inç is 3 letters — the gate was punishing
        # the behaviour it demands. Lowering the bound by length alone cannot
        # work: kart→kartal is a SHORTER step than inç→inçtir, so any purely
        # metric loosening lets a different word in.
        #
        # The signal that separates them is not length, it is the NUMBER. Both
        # texts bind the token to the same numeric neighbour (…15.6 inç…,
        # …15.6 inçtir…): they are talking about one measurement, and the
        # shared prefix is then the same unit inflected. kart/kartal,
        # organ/organizma, şeker/şekersiz have no number holding them
        # together, and stay rejected.
        if here is None:
            here, there = _anchors(answer), _anchors(given_text)
        mine = here.get(w)
        if mine and any((g.startswith(w) or w.startswith(g))
                        and (there.get(g) or set()) & mine for g in given):
            hit += 1
    return hit / len(words)


def covered(answer, block, question=""):
    """Do ALL of the answer's content-words come from the given block —
    prefix-tolerant (inflection: "bandındadır"~"bandında"). No new content-word =
    no new claim. DIGIT rules are STRICT (code-review #1):
      - digits only count from the BLOCK (a digit in the question does not count
        — a false-premise echo of "is it tier 1?" must not pass the gate);
      - every digit in the answer must occur in the block in AT LEAST ONE
        same-neighbor bigram ((prev,digit) or (digit,next)) — makes it harder
        for another line's digit in the block to attach to another subject
        (Tier confusion).
    Known residue: prefix tolerance cannot distinguish a negation suffix
    ("azalma"~"azalmaz") — since language-lists are forbidden, this cannot be
    closed at the word-set level; engine-level checking is v-next."""
    return digits_ok(answer, block) and coverage(answer, block, question) == 1.0


# --- a shattered table's ROWS -------------------------------------------------
#
# A PDF table arrives as one cell per line, so the row — the only thing a table
# actually asserts — is gone. Rebuilding it by sliding a fixed-size window over
# the cells was WRITING NEIGHBOURHOODS THE DOCUMENT NEVER STATES: the window
# "POCUS ... · 3 · Orta · Orta · Orta · Kamera tabanlı düşme tespiti" spans two
# rows, and the answer read out of it ("the camera's regulatory risk is medium")
# contradicted the document's own row ("Yüksek (KVKK)"), which was in the store
# too and lost. Nothing downstream can catch that: the claim IS in the evidence.
# So the rows are reconstructed from the layout, or not claimed at all.

CELL_CHARS = 40          # a line no longer than this is a table cell, not prose


def _cell_runs(text):
    """Consecutive runs of SHORT lines — a shattered table's cells.  The RAW
    lines are kept: their indentation is the only surviving trace of the
    columns' geometry."""
    run, out = [], []
    for line in text.split("\n") + [""]:
        if line.strip() and len(line.strip()) <= CELL_CHARS:
            run.append(line)
        else:
            if len(run) >= 3:
                out.append(run)
            run = []
    return out


def _period(cells):
    """The run's ROW LENGTH, read off the data.

    A table's columns hold cells of a characteristic width, so the cell-length
    series repeats with the row period: at lag p the mean |Δ length| collapses.
    The period is the SMALLEST lag that both dips below its two neighbours and
    falls well under the run's own overall spread — smallest, because every
    MULTIPLE of a period dips as well and only the fundamental is a row; and
    measured against the spread, because that is what the same numbers look like
    with no row structure at all. Both references are read off this run (the
    neighbouring lags, the mean pairwise difference); nothing is known about the
    document. Measured on the hospital table: 5.4 at lag 5, against 10.8 and 9.8
    beside it and a spread of 9.1 — the table's true width is 5.
    """
    n = len(cells)
    if n < 9:                       # under ~3 rows there is no period to see
        return None
    lengths = [len(c) for c in cells]
    pairs = [abs(a - b) for i, a in enumerate(lengths) for b in lengths[i + 1:]]
    spread = sum(pairs) / len(pairs) if pairs else 0
    if spread <= 0:
        return None
    score = {}
    for p in range(2, min(12, n // 3) + 1):
        d = [abs(lengths[i + p] - lengths[i]) for i in range(n - p)]
        score[p] = sum(d) / len(d)
    for p in sorted(score):
        if score[p] >= 0.75 * spread:
            continue                # no better than the same cells unordered
        if score[p] < min(score.get(p - 1, spread), score.get(p + 1, spread)):
            return p
    return None


def _column_cells(raw, p):
    """The cells of ONE column, found by their indentation.

    The period says how LONG a row is but not where one BEGINS, and no statistic
    of the cells can say: shifting a periodic series by one cell leaves every
    column's statistics exactly as they were. The document's own layout breaks
    the tie — a text dump keeps the leading space of a column set further right.
    An indentation level whose cells are a minority of the run and recur at
    exactly the period IS such a column, and every row holds one of its cells.
    """
    indents = [len(l) - len(l.lstrip()) for l in raw]
    best = None
    for level in sorted(set(indents)):
        at = [i for i, v in enumerate(indents) if v == level]
        if len(at) < 3 or len(at) * 2 > len(raw):
            continue
        gaps = {}
        for a, b in zip(at, at[1:]):
            gaps[b - a] = gaps.get(b - a, 0) + 1
        modal = max(gaps, key=lambda g: gaps[g])
        if modal == p and gaps[modal] * 2 >= len(at) - 1:
            if best is None or len(at) > len(best):
                best = at
    return best


def _cut(cells, p, column):
    """Cut the run into rows: every row holds at most ONE cell of the anchor
    column, and the row widths and the column width profile are solved together.

    A row may come out a cell short or a cell long — a cell whose text WRAPPED
    onto a second line makes its row longer, which is why no fixed stride can
    work here. A cell that no row claims (the heading above the table, a caption
    below it) is charged the average column width, so leaving cells out is
    expensive and dropping the last row's last cell is not free.
    """
    n = len(cells)
    lengths = [len(c) for c in cells]
    anchor = set(column)
    before = [0] * (n + 1)
    for i in range(n):
        before[i + 1] = before[i] + (1 if i in anchor else 0)
    widths = [w for w in (p - 1, p, p + 1) if w >= 2]

    def solve(profile):
        """Cheapest cut under this profile (one pass, left to right)."""
        loose = max(1.0, sum(profile) / len(profile))    # an unclaimed cell
        cost = [None] * (n + 1)
        back = [None] * (n + 1)
        for i in range(column[0] + 1):
            if before[i] == 0:                          # cells above the table
                cost[i] = loose * i
        for i in range(n):
            if cost[i] is None:
                continue
            for w in widths:
                j = i + w
                if j > n or before[j] - before[i] > 1:
                    continue
                c = sum(abs(lengths[i + k] - profile[min(k, p - 1)])
                        for k in range(w)) + loose * abs(w - p)
                if before[j] == before[i]:
                    c += loose          # a row with no cell of that column
                if cost[j] is None or cost[i] + c < cost[j]:
                    cost[j], back[j] = cost[i] + c, i
            if before[n] == before[i] and (cost[n] is None
                                           or cost[i] + loose * (n - i)
                                           < cost[n]):
                cost[n], back[n] = cost[i] + loose * (n - i), i
        if cost[n] is None:
            return None, None
        rows, j = [], n
        while j and back[j] is not None:
            rows.append((back[j], j))
            j = back[j]
        rows.reverse()
        return rows, cost[n]

    def profile_of(rows):
        columns = [[] for _ in range(p)]
        for i, j in rows:
            if j - i < p:
                continue
            for k in range(i, j):
                columns[min(k - i, p - 1)].append(lengths[k])
        return [sorted(c)[len(c) // 2] if c else 0 for c in columns]

    # Every offset of an anchor cell is a possible phase; each is settled by
    # alternating cut and profile, and the phases are compared on the cost of
    # the SAME objective.
    best, chosen = None, None
    for start in {a - off for a in column for off in range(p)}:
        if start < 0 or start + p > n:
            continue
        profile = [lengths[start + k] for k in range(p)]
        rows = None
        for _ in range(4):
            nxt, _cost = solve(profile)
            if nxt is None or nxt == rows:
                break
            rows, profile = nxt, profile_of(nxt)
        if not rows:
            continue
        rows, total = solve(profile)
        if rows and (best is None or total < best):
            best, chosen = total, rows
    return chosen


def rows_of(raw):
    """The rows of one shattered run, or None when its layout does not say
    where a row begins. Returning None there is the point: guessing would put
    two rows' cells side by side and call it evidence."""
    p = _period([l.strip() for l in raw])
    if not p:
        return None
    column = _column_cells(raw, p)
    return _cut([l.strip() for l in raw], p, column) if column else None


def table_windows(text):
    """What a shattered table contributes to the evidence store.

    Where the layout yields rows, ONE RECORD PER ROW: the cells the document
    puts in one row and no others. Where it does not, the cells go in as
    overlapping windows of consecutive lines, which claim only that the lines
    follow one another — and the row's column names are prefixed either way, so
    a detached value still says which attribute it is the value of.
    """
    out = []
    for raw in _cell_runs(text):
        cells = [l.strip() for l in raw]
        rows = rows_of(raw)
        if rows is None:
            header = " · ".join(cells[:6])
            for i in range(0, len(cells), 2):
                window = cells[max(0, i - 1):i + 6]
                if len(window) >= 2:
                    row = " · ".join(window)
                    out.append(f"{header} — {row}" if i else row)
            continue
        # With the rows known, the header is exactly the cells that come BEFORE
        # the first row — a column name is a cell no row claims. (Without the
        # rows there was no way to say where the header ended, and the first six
        # cells had to stand in for it.)
        header = " · ".join(cells[:rows[0][0]])
        for nth, (i, j) in enumerate(rows):
            row = " · ".join(cells[i:j])
            out.append(f"{header} — {row}" if header else row)
    return out


class SentenceStore:
    """Sentence store + inverted index. Small and pure: list + dict."""

    def __init__(self):
        self.sentences = []                 # id -> (sentence, source)
        self.index = {}                     # folded word -> set(id)
        self.key_index = {}                 # folded FIELD-NAME word -> set(id)
        self._seen = set()                  # folded sentence — duplicate-evidence guard
        self.last_sources = []              # sources of the last find() (hedge)
        # SHORT tokens the corpus itself showed standing next to a number —
        # the units of this document (kg, ml, mm, V, Hz, VA...). Learned, not
        # listed: a question ("kaç kg") carries no number of its own, so the
        # only thing that can vouch for its short token is a sentence where
        # that token WAS a unit. Without it the query word was dropped and the
        # search ran on "kaç" alone.
        #
        # ONE SIGHTING IS NOT A UNIT — MAJORITY IS. In a flattened PDF numbers
        # sit next to everything, so "standing next to a number at least once"
        # made 'bu', 'de' and bare letters units and let them into queries as
        # content (measured on the manual: 200+ 'units', most of them noise).
        # A unit is a token whose LIFE is next to numbers: it must appear
        # beside a number MORE OFTEN THAN NOT. That boundary is not a tuned
        # threshold — it is the definition of "characteristically", and it
        # separates kg (nearly always) from bu (nearly never) without knowing
        # a word of the language.
        self._near = {}                     # short token -> beside-a-number count
        self._all = {}                      # short token -> total count
        self._units = None                  # cache, invalidated by add()

    @property
    def units(self):
        """The short tokens this corpus uses AS UNITS — beside a number more
        often than not (see __init__)."""
        if self._units is None:
            self._units = {w for w, n in self._near.items()
                           if n * 2 > self._all.get(w, 0)}
        return self._units

    @staticmethod
    def _key_words(sentence):
        """FIELD-NAME words of a record-shaped sentence — via FORMAT cues only
        (no language list): (1) up to 3 words immediately BEFORE a colon
        ("Bellek: 8 g" → bellek); (2) in a SHORT segment (a table cell /
        spec line), the leading words before the first digit ("Ekran 15.6"
        LCD" → ekran). A query word matching a field name is worth more than
        the same word wandering in prose — the kv-key weighting that the
        spec-line loss class needed (measured: 'işlemci' spec row lost to
        DICOM prose on generic-word overlap)."""
        keys = []
        for m in re.finditer(r"((?:[\w\"'.]+[ \t]+){0,2}[\w\"'.]+)\s*:",
                             sentence):
            keys += _words(m.group(1))[-3:]
        for seg in re.split(r"[·—;|]", sentence):
            seg = seg.strip()
            if not seg or len(seg) > 80:
                continue
            # RECORD SHAPE, not just "contains a number": few tokens in
            # total and the digit arrives early (field name + value and
            # little else). A prose sentence that merely cites a figure
            # ("see section 11.8 ...") must not get its words keyed.
            toks = _tokens(seg)
            if not toks or len(toks) > 12:
                continue
            first_digit = next((i for i, t in enumerate(toks)
                                if t.isdigit()), None)
            if first_digit is None or first_digit == 0 or first_digit > 4:
                continue
            m = re.match(r"([^\d:]{1,60}?)\s*\d", seg)
            if m:
                keys += _words(m.group(1))[-3:]
        return [k for k in keys if not k.isdigit()]

    def add(self, sentence, source=""):
        key = fold(sentence)
        if key in self._seen:               # if the same document is read twice,
            return None                     # evidence must not multiply (review #2)
        self._seen.add(key)
        sid = len(self.sentences)
        self.sentences.append((sentence, source))
        toks = _tokens(sentence)            # this document's measured units
        for i, w in enumerate(toks):
            if len(w) >= CONTENT or w.isdigit():
                continue
            self._all[w] = self._all.get(w, 0) + 1
            if (i > 0 and toks[i - 1].isdigit()) or (
                    i + 1 < len(toks) and toks[i + 1].isdigit()):
                self._near[w] = self._near.get(w, 0) + 1
        self._units = None
        for w in set(_words(sentence)):
            self.index.setdefault(w, set()).add(sid)
        for w in set(self._key_words(sentence)):
            self.key_index.setdefault(w, set()).add(sid)
        return sid

    def find(self, query, most=4):
        """Sentences whose content-words intersect the query the MOST.
        Prefix-tolerant (inflection: "doluluğu"~"doluluk"). Score = number of
        intersecting words; on a tie the short sentence wins (denser evidence)."""
        qwords = set(_words(query, known=self.units))
        if not qwords:
            return []
        # IDF: frequent words count little, rare words a lot ("screen" occurs
        # hundreds of times in a manual — it was pushing the spec line behind
        # the UI sentences). A universal information-theory weight; not a
        # language/document rule.
        import math
        total = max(1, len(self.sentences))
        # STEM GROUPS: query surface forms sharing a root ("cihaz"/"cihazın")
        # are ONE anchor — counted separately, a generic stem contributed twice
        # and its prose crowd outweighed the answer-carrying spec row
        # (measured, manual trace).
        groups = []
        for qw in sorted(qwords, key=len):
            for g in groups:
                if any(qw.startswith(m) or m.startswith(qw) for m in g):
                    g.append(qw)
                    break
            else:
                groups.append([qw])
        scores = {}
        for group in groups:
            # Candidate index words of the whole group. ALWAYS expand (no
            # exact-hit shortcut): with only the exact form, an inflected
            # query word ("boyutları") hid the ROOT-carrying spec row
            # ("Boyutlar 360 mm ...") whenever the inflected form happened to
            # exist verbatim elsewhere — the answer-carrying sentence never
            # entered the candidate set (measured, manual trace).
            # Prefix tolerance — two-way (root>=4 forward, >=3 reverse:
            # "inçtir" must reach the index word "inç"). STEM matching too:
            # "hazırlandı"~"hazırlanma" are not each other's prefix but share
            # an 8/10 stem — in an agglutinative language, the same concept
            # (the criterion is proportional, no language-list).
            cand = {}
            for qw in group:
                if qw in self.index:
                    cand[qw] = self.index[qw]
                for w, ids in self.index.items():
                    if w in cand:
                        continue
                    if len(qw) >= 4 and w.startswith(qw):
                        cand[w] = ids
                    elif len(w) >= 3 and qw.startswith(w):
                        cand[w] = ids
                    else:
                        # PROPORTIONAL stem criterion: the shared stem must
                        # cover >=70% of the longer form too — "hazırlandı"~
                        # "hazırlanma" (8/10) is the same concept, but
                        # "işlemi"~"işlemcisi" (5/9) is a DIFFERENT word whose
                        # crowd was leaking into the rare word's group and
                        # burying the spec row (measured, manual trace).
                        common = os.path.commonprefix((qw, w))
                        if len(common) >= max(4, min(len(qw), len(w)) - 2,
                                              (max(len(qw), len(w)) * 7 + 9)
                                              // 10):
                            cand[w] = ids
            if not cand:
                continue
            # ONE weight for the whole stem group, over the UNION of its
            # sentences (information content of the concept, not of one rare
            # inflection: "cihazlardır" occurring twice must not out-inform
            # the stem "cihaz" occurring 300 times).
            union = set()
            for ids in cand.values():
                union |= ids
            weight = math.log(1 + total / len(union))
            keyed = set()
            for w, ids in cand.items():
                keyed |= ids & self.key_index.get(w, set())
            for sid in union:
                # FIELD-NAME match doubles the group's weight: the same word
                # as a kv KEY names the record's slot, in prose it is mere
                # co-occurrence (kv-key weighting; format cue, no language
                # rule).
                scores[sid] = scores.get(sid, 0) + weight * (
                    2 if sid in keyed else 1)
        if not scores:
            return []
        # on a tie the LONG one wins: a short table crumb ("Orta · 3") must not
        # beat a context-carrying prose window — a cell value wandering without
        # context was attaching to the wrong attribute (the camera/KVKK case).
        ranked = sorted(scores.items(),
                        key=lambda kv: (-kv[1], -len(self.sentences[kv[0]][0])))
        # noise floor: anything below half the best score drops. SHORT-query
        # exception (review #4): in a question with 1-2 content-words ("what is
        # karvel") a score of 1 is legitimate — with a floor of 2 it would
        # return empty while evidence exists. Window/table lines can inflate
        # best; the floor lowers as the query shortens.
        best = ranked[0][1]
        floor = max(1 if len(qwords) <= 2 else 2, best // 2)
        # NEAR-DUPLICATE SUPPRESSION: the same content is indexed at several
        # window scales (raw line, 3-window, 6-window) — without this, one
        # strong-but-wrong region filled ALL top slots and the answer-carrying
        # row never got a seat (measured, manual trace). Overlap is a set view
        # (Jaccard on content-words), no language rule.
        #
        # A region may hold at most REGION_CAP seats, not exactly one. The
        # purpose of this filter is that no single region MONOPOLISES the top
        # list; suppressing every variant went further than that purpose and
        # removed CORROBORATION. Measured (hospital trace, "regülasyon riski"):
        # the answer binding "Kamera tabanlı düşme tespiti ... Yüksek (KVKK)"
        # occurs in two overlapping table windows; with one seat only, the
        # support check saw a single flattened row and rejected the CORRECT
        # answer, and the question fell to an abstention. With two seats the
        # binding is confirmed twice and the answer passes. The anti-monopoly
        # guarantee survives: 2 < `most`, so a region can never take every seat.
        REGION_CAP = 2
        keep = []
        kept = []                           # [[wordset, seats_taken], ...]
        for sid, sc in ranked:
            if sc < floor and keep:
                break
            words = set(_words(self.sentences[sid][0]))
            dup = False
            for entry in kept:
                inter = len(words & entry[0])
                # Jaccard on content-words. NOT containment: two spec windows
                # can differ by 3 tokens where those 3 tokens ARE the answer
                # ("14.4 V / 6500 mAh" tail) — containment suppressed the
                # only window carrying them (measured).
                if inter and inter / max(1, len(words | entry[0])) >= 0.75:
                    if entry[1] < REGION_CAP:
                        entry[1] += 1       # corroborating variant — admit
                        break
                    dup = True
                    break
            if dup:
                continue
            keep.append(sid)
            kept.append([words, 1])
            if len(keep) >= most:
                break
        if not keep:
            # if the floor swept everything, still return the best 2: if "no
            # evidence at all" is wrong, we are giving the answer gates no
            # chance whatsoever; weak evidence is filtered anyway by the
            # coverage+support checks.
            keep = [sid for sid, _sc in ranked[:2]]
        self.last_sources = [self.sentences[sid][1] for sid in keep]
        return [self.sentences[sid][0] for sid in keep]

    # --- persistence (JSON side-file) ----------------------------------
    def save(self, memory_path):
        if not memory_path:
            return
        # ATOMIC (review #2): tmp + os.replace — a half-written .evidence file
        # must not crash load() (and therefore Session.__init__).
        path = memory_path + ".evidence"
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([[s, src] for s, src in self.sentences], f,
                      ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, memory_path):
        store = cls()
        path = None
        if memory_path:
            # New extension first; fall back to the legacy '.kanit' side-file
            # (backward compatibility — save() writes only '.evidence').
            for ext in (".evidence", ".kanit"):
                candidate = memory_path + ext
                if os.path.exists(candidate):
                    path = candidate
                    break
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    rows = json.load(f)
            except (json.JSONDecodeError, OSError):
                return store        # corrupt side-file → empty store (the graph
                #                     is intact; evidence can be re-ingested,
                #                     no fabrication risk)
            for sentence, source in rows:
                store.add(sentence, source)
        return store
