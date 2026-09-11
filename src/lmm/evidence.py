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
import math
import os
import re
import unicodedata

from lmm.core.dataset import fold
from lmm import inflect

_WORD = re.compile(r"\w+", re.UNICODE)

# The length at which a token is a content word on its own. Everything shorter
# has to EARN its place (see `_words`).
CONTENT = 3

# WINDOW SCALES — the widths at which neighbouring lines are indexed together as
# each other's context (a heading and the line it governs have to meet in one
# window, and they can sit several sentences apart). A GEOMETRIC LADDER, each
# scale twice the one below it, starting from a sentence and its two neighbours:
# scale-free by construction, so no width is the "right" one for a document, and
# there is nothing here to tune per document. WINDOW is the middle scale, the
# widest at which two lines are still plainly one another's context; the cell
# windows of a table whose rows cannot be read use the same one.
SCALES = (3, 6, 12)
WINDOW = SCALES[1]

# THE CHANNEL LADDER — how much a match is worth depending on WHAT matched.
#
# There have always been two channels in `find`: a query word occurring in a
# sentence, and the same word occurring as that record's FIELD NAME. The second
# was worth twice the first, written as a bare `2` at the point of use. It is
# named here because a THIRD channel now hangs off the same step — the offline
# expansion index (`learn_expansions`), which is not an observation of the
# document at all but a model's guess at how the same line might be asked for.
#
# One step, three rungs, and the order is the epistemic one:
#     field name   CHANNEL       — the word NAMES the record's slot
#     occurrence   1             — the word IS in the document
#     expansion    1 / CHANNEL   — a model guessed the word would fit
#
# There is no second number to tune: the guess sits exactly as far below a
# plain occurrence as a field name sits above it. What the ladder guarantees is
# the thing the expansion index must never be allowed to do — outweigh the
# document's own words.
CHANNEL = 2


# WHERE THE MEMORY'S OWN SPEECH IS FILED. Not a document name, so
# every source-scoped reading — `named_in`, the census, the scope rule
# — passes it by, and a citation can never confuse the two.
SAID_SOURCE = "#said"


def _expansion_on():
    """`LMM_EXPAND=0` turns the expansion channel off — at generation time and
    at query time both, so one variable produces the whole A/B."""
    return os.environ.get("LMM_EXPAND", "1") != "0"


def _same_region(words, other):
    """Are these two token sets two views of ONE region of the document?

    Two ways of overlapping, both at the majority boundary. Jaccard catches two
    windows of nearly the same width; CONTAINMENT catches the window scales,
    which are NESTED — the 3-sentence window sits inside the 6- and that inside
    the 12- — and nesting makes the union large while the intersection stays
    whole, so Jaccard alone reads two views of one paragraph as two regions.

    THE BOUNDARY IS THE MAJORITY, not a fraction somebody liked. Two texts are
    views of one region when they share MORE than they differ; that is what
    "the same region" means, and it is the only boundary on a similarity ratio
    that is not a dial.

    It was written inline in `find`'s near-duplicate filter. It is a function
    now because the expansion filter has to ask the same question — whether a
    generated query drifted to ANOTHER part of the document — and answering it
    with a second, slightly different rule is exactly how two organs end up
    disagreeing about what "the same" means.
    """
    inter = len(words & other)
    if not inter:
        return False
    return (inter / max(1, len(words | other)) > 0.5
            or inter / max(1, min(len(words), len(other))) > 0.5)


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


def record_pairs(text, cap_chars=200):
    """Every record a line holds: (head, value) for each HEAD: value it
    carries, not only the first.

    HOW A DOCUMENT WAS CHUNKED IS NOT SUPPOSED TO CHANGE WHAT THE MEMORY
    CAN READ, and it did. Ingested line by line, a specification's fields
    arrive separately and the record channel sees DISPLAY, LIST PRICE and
    WARRANTY; ingested by the ordinary bulk path — which is how every
    benchmark and every customer load runs — the three arrive glued into
    one window, a partition at the first colon sees DISPLAY alone, and the
    corpus knows one field where it wrote three. Measured: the same
    superlative question answered from a per-line store and abstained
    from a windowed one with every value present in both.

    The reading is the record's own shape, unchanged from everywhere else
    in this codebase: a SHORT segment whose colon is preceded by at most
    three words. A titled sentence in prose ("The source of the
    difficulty is this: ...") fails that test here exactly as it fails it
    in the rider, and a dateline before the head is provenance, not part
    of the name.
    """
    out = []
    for segment in re.split(r"(?<=[.!?])\s+", text or ""):
        segment = segment.strip()
        if ":" not in segment or len(segment) > cap_chars:
            continue
        head, _sep, value = segment.partition(":")
        head = head.rsplit("\u2014", 1)[-1].strip()
        # A BULLET IS NOT PART OF THE HEAD. Documents open list items with
        # "\u00b7", "-", "*" or a number, and reading those into the name
        # made "\u00b7 DURATION" a second field beside DURATION — same
        # rows, twice, in the corpus's own vocabulary. Measured: it broke
        # the direct record answer for every question in a corpus (the
        # asked head was never unique), and it seated junk heads in the
        # index the census and the field gate read.
        head = head.lstrip("\u00b7\u2022*-\u2013 \t").strip()
        while head[:1].isdigit() and "." in head[:4]:
            head = head.partition(".")[2].strip()
        words = _words(head)
        value = value.strip().rstrip(".").strip()
        if words and len(words) <= 3 and value:
            out.append((head, value))
    return out


def _source_name(source):
    """The human name inside a source stamp — '#docx:Delegation Basics.docx'
    -> 'Delegation Basics'. Format only: the tag prefix before the first ':'
    and the file extension are shape, not name. No word is inspected."""
    name = source.split(":", 1)[1] if ":" in source else source.lstrip("#")
    return os.path.splitext(name)[0]


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


# A SYMBOL IS A TOKEN TOO. `_tokens` reads words (\w+), which is right for the
# lexical channel and blind exactly where a measurement's unit is symbolic:
# 15.6" loses its unit entirely, and the manual's screen line is that case. This
# tokenization keeps numbers, letter runs AND symbol runs, so "which tokens does
# this corpus put next to numbers" can be asked about '"', '%' and '°' as well.
_UNIT_TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[^\W\d_]+|[^\w\s]+", re.UNICODE)


def _quantity_tokens(text):
    """(token, is it beside a number) over the symbol-aware tokenization."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    toks = [fold(t) for t in _UNIT_TOKEN.findall(_boundaries(text))]
    out = []
    for i, t in enumerate(toks):
        if not t or t[0].isdigit():
            continue
        out.append((t, (i > 0 and toks[i - 1][:1].isdigit())
                    or (i + 1 < len(toks) and toks[i + 1][:1].isdigit())))
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
    # AN ORDINAL LIST MARKER IS FORMAT, NOT A QUANTITY. Measured on a live
    # catalogue: the engine numbered its sections — the one structural act
    # the compose contract grants it — and this gate killed sections 1 and
    # 2 while 3 survived on the accident of a "3 hours" in the brief. The
    # exemption is the digit twin of the name gate's sentence-start rule
    # and just as narrow: one or two digits OPENING a line, closed by a dot
    # or bracket. Every other digit still answers for itself.
    answer = re.sub(r"(?m)^\s*\d{1,2}[.)]\s+", "", answer)
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
        # ONE criterion for "same word, different ending" — `inflect`. Two
        # organs used to spell this out with their own constants; now the gate's
        # strictness cannot depend on which organ asks.
        if any(inflect.same_stem(w, g) for g in given):
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


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def grounded_sentences(answer, block, question=""):
    """Drop the sentences that RIDE ON the evidence without resting on it.

    THE MEASURED FAILURE. The answer gates judge a candidate WHOLE, and a whole
    can be an honest refusal with a fabrication stapled to it:

        "I do not know. Vorlin is not a recognized substance."
        "No sé. La población de la isla puede variar."
        "I do not know. RFC 9110 is obsoleted by RFC 9111."   (field trial)

    Each of those was scored a loss, and the second half of each is a claim the
    document does not make. Whole-answer coverage cannot see it: the refusal's
    words and the claim's words are pooled, the ratio lands in the middle, and
    the read-back — a model call, and on a weak local engine an unreliable one —
    is left to decide. `verify.verify` has always filtered sentence by sentence,
    but the evidence path returns its chosen candidate without passing through
    it, which is precisely the gap these three sentences walked through.

    THE CRITERION IS THE MIXTURE, and it needs no language and no model. A
    sentence whose content words ALL come from the evidence rests on it. A
    sentence that shares NOTHING with the evidence rests on nothing and claims
    nothing about it — that is what a refusal, a greeting or an apology looks
    like from outside, in any language, without having to recognise one. What
    cannot be allowed is the MIXTURE: a sentence that takes some of its words
    from the evidence and the rest from somewhere else is wearing the
    evidence's authority over material the evidence never supplied. All three
    sentences above are mixtures — 'vorlin' and 'substance' are in the
    document, 'recognized' is not.

    IT CANNOT EMPTY AN ANSWER THE GATES ADMITTED. If every sentence is a
    mixture, nothing is dropped: whether a candidate may speak at all is the
    gates' decision above, and this is only an audit of what rides along with
    it. That clause is also what keeps a fluent one-sentence answer ("Evet,
    torvanit bir maddedir" — 'evet' comes from neither the block nor the
    question) exactly as it was.

    KNOWN RESIDUE, stated rather than hidden: a fabricated sentence sharing NO
    word at all with the evidence is kept by this rule, because from outside it
    is indistinguishable from a refusal. Telling those two apart is asking
    whether a sentence CLAIMS anything, which is what `extract.reextract` is
    for — and that is a model call, which is the thing this filter exists to
    not depend on. `verify.verify` still closes that case where it runs.
    """
    parts = [p.strip() for p in _SENTENCE.split(answer.strip()) if p.strip()]
    if len(parts) < 2:
        return answer
    kept = [p for p in parts
            if coverage(p, block, question) in (0.0, 1.0)]
    if not kept or len(kept) == len(parts):
        return answer
    return " ".join(kept)


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

def _cell_bound(lines):
    """How long a line may be and still be a table CELL — read off the document.

    This used to be the constant 40, and 40 was a number somebody chose while
    looking at one PDF: on a document whose lines run narrower it lets prose in,
    on a wider one it throws cells away.

    The document says it instead, and it says it in the SHAPE of its line
    lengths. A text dump of a document with tables in it is bimodal: prose is
    wrapped to the page width and clusters high, cells are a few words and
    cluster low. The split between two clusters of a one-dimensional sample is
    not something to guess — take the threshold that MAXIMISES THE SEPARATION
    between the two groups it makes (between-class variance, the standard
    parameter-free criterion), and the document's own two populations decide
    where the line falls. No number to tune, and nothing here has been looked up
    in any particular document: the median would have been simpler but breaks on
    a text that is MOSTLY cells, where half the cells sit above it.

    A document with no second cluster still returns a split — every sample does.
    That is safe, because a run of short lines is only ever read as a table if
    its cell lengths turn out to be PERIODIC and its layout says where a row
    begins; `rows_of` returns None otherwise, and the run goes in as plain
    consecutive-line windows that claim only adjacency.
    """
    lengths = sorted(len(l.strip()) for l in lines if l.strip())
    if len(set(lengths)) < 2:
        return lengths[-1] if lengths else 0
    total = sum(lengths)
    best, cut = None, lengths[-1]
    below = count = 0
    for i, length in enumerate(lengths):
        below += length
        count += 1
        if i + 1 == len(lengths) or lengths[i + 1] == length:
            continue                    # a threshold must fall BETWEEN values
        rest = len(lengths) - count
        spread = count * rest * (below / count - (total - below) / rest) ** 2
        if best is None or spread > best:
            best, cut = spread, length
    return cut


def _cell_runs(text):
    """Consecutive runs of SHORT lines — a shattered table's cells.  The RAW
    lines are kept: their indentation is the only surviving trace of the
    columns' geometry.

    A run under three lines is not a table: nothing periodic can be read off two
    cells (`_period` needs three rows' worth before it will answer at all)."""
    lines = text.split("\n")
    bound = _cell_bound(lines)
    run, out = [], []
    for line in lines + [""]:
        if line.strip() and len(line.strip()) <= bound:
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
            # NO ROW STRUCTURE: the cells go in as overlapping windows, at the
            # SAME scale the prose windows use (`session.WINDOW`) — the widest
            # scale at which neighbouring lines are still one another's context.
            # It is one named scale shared by both window makers rather than a
            # second number invented here, and with no rows to read there is
            # nothing better available: the header stands in as the run's first
            # window's worth of cells.
            header = " · ".join(cells[:WINDOW])
            for i in range(0, len(cells), 2):
                window = cells[max(0, i - 1):i + WINDOW]
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


# A LINE THAT ENDS A SENTENCE — terminal punctuation at the end of the line.
# Punctuation is format, and these are the marks Unicode itself classifies as
# sentence terminators; no word of any language is involved.
# A colon is deliberately NOT one of them: a line ending in one is a label
# whose value is on the next line ("Prepared by:"), which is masthead, not
# prose.
_SENTENCE_END = re.compile(r"[.!?。！？؟۔]['\"»)\]]*\s*$")


def _gap(line):
    """The widest run of whitespace INSIDE one line — how far apart this line
    holds its own pieces. One in prose, a column's width in a masthead."""
    inner = line.strip()
    return max((len(run) for run in re.findall(r"\s{2,}", inner)), default=1)


def front_matter(text):
    """THE DOCUMENT'S OWN MASTHEAD — the block it opens with, as one window.

    A measured blind spot, and a whole class of it: every fact this system
    missed on two real documents was in the front block. "Department of the
    Treasury—Internal Revenue Service" at the head of a tax form; "S. Bradner
    / Harvard University / March 1997" at the head of an RFC. Who published
    this, when, and on whose behalf — the questions a reader asks first — are
    not written as sentences anywhere in the body. They are written ONCE, as
    lines, above it.

    The evidence layer is built on sentences and their neighbourhoods, and
    those lines are neither: each one is an independent unit, none of them
    flows into the next, and they were indexed one detached line at a time.
    Split apart, "Harvard University" no longer stands beside the name it
    qualifies, and no window ever held the block.

    So the block is read off the LAYOUT, and the layout only. A document's
    prose announces itself by ending a line with a sentence terminator; the
    masthead is what comes before the first line that does. Nothing here knows
    that line three tends to be a date, or what a publisher's name looks like,
    and a document that opens straight into prose gets no block at all — its
    first line ends a sentence, and the run is empty before it starts.

    The one bound is borrowed rather than invented: the block may not outgrow
    what this document calls a record — the widest window scale measured in
    this document's own median line. A cover page of thirty catalogue lines is
    a table, not a masthead, and it is `table_windows`' work.
    """
    # Column padding is layout, not content: a masthead is typeset in columns
    # and its runs of spaces carry nothing a reader of the window needs.
    raw = [line.rstrip() for line in text.splitlines()]
    written = [line for line in raw if line.strip()]
    if len(written) < 2:
        return ""
    lengths = sorted(len(line.strip()) for line in written)
    median = lengths[len(lengths) // 2]
    bound = SCALES[-1] * max(1, median)
    # IS THIS LINE SET IN COLUMNS? A masthead is typeset as columns, and the
    # gap between them is wider than anything prose puts between two words —
    # measured against THIS document, not against a number. A terminator alone
    # could not be trusted to end the block: RFC 9110's first line ends "R.
    # Fielding, Ed.", which is an abbreviation wearing a full stop, and the
    # masthead died on it. A line that both ends a sentence AND is set like
    # prose is prose.
    gaps = sorted(_gap(line) for line in written)
    typical = gaps[len(gaps) // 2]
    head, used = [], 0
    for line in raw:
        if not line.strip():
            continue                    # a blank line separates the masthead's
            #                             own groups; it does not end it
        flat = " ".join(line.split())   # column padding is layout, not content
        if used + len(flat) > bound:
            break
        if _SENTENCE_END.search(flat) and _gap(line) <= typical:
            break
        head.append(flat)
        used += len(flat)
    return " ".join(head) if len(head) >= 2 else ""


class SentenceStore:
    """Sentence store + inverted index. Small and pure: list + dict."""

    def __init__(self):
        self.sentences = []                 # id -> (sentence, source)
        self.index = {}                     # folded word -> set(id)
        self.by_source = {}                 # source -> set(id)  (see find)
        self.last_census = []               # [(source, hits)] of the last find
        # THE EXPANSION SIDE (doc2query--), AND THE WALL AROUND IT.
        #
        # `expansions` holds text NO DOCUMENT WROTE — questions a model guessed
        # this line would answer, generated once at ingestion. It exists so that
        # a question asked in words the document does not use can still REACH
        # the line that answers it. It is a retrieval aid and nothing else.
        #
        # THE RULE, AND IT IS ABSOLUTE: generated text never becomes evidence.
        # It is not in `self.sentences`, so `find` — which returns
        # `self.sentences[sid][0]` and nothing else — cannot return it, and the
        # answer block therefore cannot contain a word the document did not
        # write. The moment that wall moves, the gate is auditing an answer
        # against a model's own invention and the uninvented-0 guarantee is
        # gone. `tests/test_core.py` pins it (X1–X4).
        self.expansions = {}                # id -> [generated text, ...]
        self.expand_index = {}              # folded word -> set(id)
        # Which units the DOCUMENT wrote (a sentence, a masthead, a table row)
        # and which this layer derived from them (the context windows). Only
        # the first kind is worth expanding: a window is the same sentences
        # again, so paying for its expansion buys the same words twice.
        self._derived = set()
        # READER-WORD -> FIELD-HEAD, generated once per head at the
        # operator's request (`Session.learn_bridges`) and owned by the
        # store like the expansion is: never speakable, never evidence —
        # a word here can only NOMINATE a head for the record path, where
        # the one-head rule and every gate stand unchanged. Deleting the
        # .bridge side-file restores the memory exactly as it was.
        self.head_bridge = {}               # folded word -> set(head)
        # WHO SPOKE, per line — dialogue's second piece of real metadata
        # beside the date (W98). A tag the operator supplies, like the
        # source name; None for every document that has no voices.
        self.speakers = {}                  # sid -> speaker tag
        self._key_index = None              # lazy — see the `key_index` property
        self._key_bounds = None             # the bound it was built under
        self._key_at = 0                    # how many sentences are in it
        self._bounds = None                 # lazy — see `_record_bounds`
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
        # THE SAME MEASUREMENT, COUNTED FOR TOKENS OF EVERY LENGTH. `units`
        # above exists to rescue a SHORT query token from the content-word
        # length test, so it only ever counted short tokens. But "does this
        # token name a quantity" is a separate question from "is this token
        # long enough to be content", and it has to be asked of long tokens
        # too: 'inç' names a quantity in this manual (the corpus binds it to a
        # number: "(12 inç)"), which is how a question can be recognised as
        # ASKING FOR A MEASUREMENT without any list of units existing anywhere.
        self._qnear = {}
        self._qall = {}
        self._quantities = None
        # NOTATION MARKS: symbol runs this corpus writes BETWEEN two digits —
        # the decimal point of 15.6, the dots of section 1.2.3. Adjacency alone
        # cannot tell such a mark from a symbolic unit ('"', '%'), because both
        # live next to numbers; what separates them is that a notation mark also
        # appears INSIDE a number, and a unit never does. Read off the corpus,
        # like everything else here.
        self._notation = set()

    @property
    def units(self):
        """The short tokens this corpus uses AS UNITS — beside a number more
        often than not (see __init__)."""
        if self._units is None:
            self._units = {w for w, n in self._near.items()
                           if n * 2 > self._all.get(w, 0)}
        return self._units

    @property
    def quantities(self):
        """The tokens this corpus uses to NAME A QUANTITY — the ones whose life
        is next to a number, at any token length (see __init__). The majority
        rule is the same one `units` uses, and for the same reason: in a
        flattened PDF numbers stand next to everything once."""
        if self._quantities is None:
            self._quantities = {w for w, n in self._qnear.items()
                                if n * 2 > self._qall.get(w, 0)}
        return self._quantities

    def _record_bounds(self):
        """What "a SHORT line" means IN THIS CORPUS — (characters, tokens).

        The record-shape test below needs to know when a segment is a table cell
        or a spec line rather than prose. That used to be three constants (at
        most 80 characters, at most 12 tokens, the first digit within the first
        4) chosen while reading one PDF; on a corpus of shorter or longer lines
        the same numbers mean something entirely different. The corpus itself
        says where its short lines end: the MEDIAN sentence is the middle of what
        this document writes, and a record is at or below it. Nothing is known
        about the document — the bound is read off the material, the way the
        table's row period and this store's units already are.
        """
        if self._bounds is None:
            chars = sorted(len(s) for s, _src in self.sentences)
            toks = sorted(len(_tokens(s)) for s, _src in self.sentences)
            self._bounds = ((chars[len(chars) // 2], toks[len(toks) // 2])
                            if chars else (0, 0))
        return self._bounds

    @property
    def key_index(self):
        """folded FIELD-NAME word -> set(id), built lazily.

        It is built on demand rather than per `add` because the record-shape test
        is corpus-relative (`_record_bounds`): the answer to "is this line short"
        is not knowable while the corpus is still arriving, and an index built
        incrementally would have depended on the order the sentences came in.
        Built once the corpus is complete, it is order-independent — which is the
        property this whole layer's determinism argument rests on."""
        bounds = self._record_bounds()
        if self._key_index is not None and bounds == self._key_bounds:
            # The bound did not move, so everything already indexed was indexed
            # under this same bound: only the new sentences are missing, and
            # adding them gives exactly what a rebuild would (a 350-page manual
            # rebuilds in ~0.2 s, which a conversation should not pay per turn).
            for sid in range(self._key_at, len(self.sentences)):
                for w in set(self._key_words(self.sentences[sid][0], *bounds)):
                    self._key_index.setdefault(w, set()).add(sid)
            self._key_at = len(self.sentences)
            return self._key_index
        index = {}
        for sid, (sentence, _src) in enumerate(self.sentences):
            for w in set(self._key_words(sentence, *bounds)):
                index.setdefault(w, set()).add(sid)
        self._key_index, self._key_bounds = index, bounds
        self._key_at = len(self.sentences)
        return self._key_index

    @staticmethod
    def _key_words(sentence, limit_chars, limit_toks):
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
            if not seg or len(seg) > limit_chars:
                continue
            # RECORD SHAPE, not just "contains a number": SHORT for this corpus
            # and the digit arrives in the leading MINORITY of the tokens (field
            # name first, then value — the name is not most of the line). A prose
            # sentence that merely cites a figure ("see section 11.8 ...") must
            # not get its words keyed.
            toks = _tokens(seg)
            if not toks or len(toks) > limit_toks:
                continue
            first_digit = next((i for i, t in enumerate(toks)
                                if t.isdigit()), None)
            if first_digit is None or first_digit == 0 \
                    or first_digit * 2 > len(toks):
                continue
            m = re.match(r"([^\d:]+?)\s*\d", seg)
            if m:
                keys += _words(m.group(1))[-3:]
        return [k for k in keys if not k.isdigit()]

    def add(self, sentence, source="", derived=False, speaker=None):
        # THE SAME SENTENCE IN TWO DOCUMENTS IS TWO ATTESTATIONS. The
        # duplicate guard exists so that reading one document twice does
        # not multiply its evidence — keyed on the text alone it silenced
        # every sibling that shares a template line (measured: two courses
        # both wrote "DURATION: one full day." and only the first owned
        # it, so a comparison between them refused forever). The key says
        # what the rule means: same text, same SOURCE — a duplicate; same
        # text, another source — another document going on record.
        key = (fold(sentence), source)
        if key in self._seen:               # if the same document is read twice,
            return None                     # evidence must not multiply (review #2)
        self._seen.add(key)
        sid = len(self.sentences)
        self.sentences.append((sentence, source))
        if speaker:
            self.speakers[sid] = speaker
        if derived:
            self._derived.add(sid)
        for mark in re.findall(r"(?<=\d)([^\w\s]+)(?=\d)", sentence):
            self._notation.add(fold(mark))
        for w, beside in _quantity_tokens(sentence):
            self._qall[w] = self._qall.get(w, 0) + 1
            if beside:
                self._qnear[w] = self._qnear.get(w, 0) + 1
        toks = _tokens(sentence)            # this document's measured units
        for i, w in enumerate(toks):
            if len(w) >= CONTENT or w.isdigit():
                continue
            self._all[w] = self._all.get(w, 0) + 1
            if (i > 0 and toks[i - 1].isdigit()) or (
                    i + 1 < len(toks) and toks[i + 1].isdigit()):
                self._near[w] = self._near.get(w, 0) + 1
        self._units = None
        self._quantities = None
        self._bounds = None         # the corpus grew: the record-shape bound
        #                             may have moved, and `key_index` rebuilds
        #                             itself if it did (see the property)
        for w in set(_words(sentence)):
            self.index.setdefault(w, set()).add(sid)
        self.by_source.setdefault(source, set()).add(sid)
        return sid

    def where(self, term):
        """Which documents speak of this term — counted, not retrieved.

        The listing question ("which programmes cover X?") is not a retrieval
        problem, and treating it as one was measured to fail: the term lived
        in EIGHT documents and the answer named one, because retrieval's job
        is the best evidence, not the census. The census is a COUNT, and a
        count needs no engine: every sentence that carries every word of the
        term (inflection-tolerant, the shared criterion) votes for its
        source. GraphRAG answers this class from an LLM-written community
        summary, paid for at indexing time; this is the same answer computed
        exactly, in milliseconds, with the vote counts as receipts.

        Returns [(source, sentence_count)], most-mentioned first. An unknown
        term returns [] — there is nothing to say, and nothing is invented.
        """
        wanted = _words(term, known=self.units)
        if not wanted:
            return []
        tally = {}
        for text, source in self.sentences:
            # THE CENSUS COUNTS DOCUMENTS (W105). The memory's own
            # speech is kept in the same store so a follow-up can read
            # it, and it is not a document: counting it would let the
            # memory vote for itself in its own census.
            if source == SAID_SOURCE:
                continue
            held = set(_words(text))
            if all(any(inflect.same_stem(q, w) for w in held)
                   for q in wanted):
                tally[source] = tally.get(source, 0) + 1
        return sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))

    def _called_by_bigram(self, qwords_order, names):
        """The sources whose name stands, as a neighbouring pair, in these
        words — the bigram half of the naming rule."""
        q_bigrams = set(zip(qwords_order, qwords_order[1:]))
        called = set()
        for src, _words_of in names:
            name_seq = _words(_source_name(src), known=self.units)
            for pair in zip(name_seq, name_seq[1:]):
                if any(inflect.same_stem(pair[0], qa)
                       and inflect.same_stem(pair[1], qb)
                       for qa, qb in q_bigrams):
                    called.add(src)
                    break
        return called

    def affinity(self):
        """The corpus's own thesaurus, built once and kept.

        Lazy for the same reason the entity graph is: it costs one pass
        over the store and only a question the words failed on needs it.
        """
        from lmm import affinity                            # noqa: PLC0415
        stamp = len(self.sentences)
        got = getattr(self, "_affinity_cache", None)
        if got is not None and got[0] == stamp:
            return got[1]
        built = affinity.Profiles.build(self)
        self._affinity_cache = (stamp, built)
        return built

    def graph(self):
        """The document's own entity graph, built once and kept.

        Lazy on purpose: it costs one pass over the store (measured, 3.8 s
        for a 36,000-line novel) and only questions that name two things
        need it. Rebuilt when the store grows, like every other index
        here.
        """
        from lmm import mentions                            # noqa: PLC0415
        stamp = len(self.sentences)
        got = getattr(self, "_graph_cache", None)
        if got is not None and got[0] == stamp:
            return got[1]
        graph = mentions.Graph.build(self)
        self._graph_cache = (stamp, graph)
        return graph

    def where_they_meet(self, text, most=6):
        """The lines that mention every entity this question names.

        WHAT THE GRAPH BUYS IS SCALE. Measured on a 36,472-line novel and
        six pairs of entities: the lexical search already found a line
        carrying both, six times out of six — two rare names in one line
        outscore one name in many — and it took 417 ms a question,
        because every line carrying either name was scored. The same
        answer comes out of two posting lists in 0.01 ms. So this is not
        a better reading; it is the same reading without reading the
        document, which is the difference between a demo and ten thousand
        pages.

        Returns [] when the question names fewer than two entities or
        they never meet, and the ordinary search runs untouched.
        """
        if len(self.sentences) < 2:
            return []
        graph = self.graph()
        named = graph.named_in(text)
        if len(named) < 2:
            return []
        meeting = graph.mentioning(named[:3])
        if not meeting and len(named) >= 2:
            # NO LINE CARRIES BOTH — the question an intersection cannot
            # answer and the walk can (`Graph.spread`): two entities that
            # never share a line may still be joined through a BRIDGE the
            # document co-mentions with each. The lines returned are the
            # witnesses of each hop — real sentences, one per edge — so
            # what reaches the block is exactly as attested as any other
            # line, and the gates read it the same way. Still no model
            # call, still milliseconds.
            for bridge, _score in graph.connect(named[0], named[1], most=2):
                hops = (graph.witnesses(named[0], bridge, most=2)
                        + graph.witnesses(bridge, named[1], most=2))
                seen, lines = set(), []
                for line in hops:
                    if line not in seen:
                        seen.add(line)
                        lines.append(line)
                if lines:
                    by_text = {text: src for text, src in self.sentences}
                    self.last_sources = [by_text.get(ln, "") for ln in lines]
                    return lines[:most]
            return []
        if not meeting:
            return []
        ranked = sorted(meeting,
                        key=lambda sid: -len(self.sentences[sid][0]))
        self.last_sources = [self.sentences[sid][1] for sid in ranked[:most]]
        return [self.sentences[sid][0] for sid in ranked[:most]]

    def find_again(self, text, most=6, **kw):
        """The second ask: search once more in the STORE'S OWN WORDS.

        Called when the first search found nothing usable. The engine is
        shown the question and asked how the same thing might be WRITTEN
        (`generate.phrasings`) — words, not an answer — and every proposal
        is then checked against this store's index. A word nobody wrote
        here cannot enter the search: the same rule the field bridge
        keeps, for the same reason. What is widened is what can be FOUND;
        what may be SAID is still read off the evidence by the gates that
        already read it.
        """
        from lmm import generate                          # noqa: PLC0415
        first = self.find(text, most=most, **kw)
        try:
            proposed = generate.phrasings(text)
        except Exception:                                  # noqa: BLE001
            return first
        known = []
        for word in proposed:
            for w in _words(word, known=self.units):
                if w in self.index and w not in known:
                    known.append(w)
        if not known:
            return first
        wider = self.find("%s %s" % (text, " ".join(known)), most=most, **kw)
        return wider or first

    def named_in(self, text):
        """Which sources does THIS TEXT name — the same pointer-and-bigram
        reading `find` performs, asked of one sentence on its own.

        `find` reads names off the query it is given, and a query can
        carry more than the question: in a conversation the previous
        turn's subject rides along to keep retrieval on topic. That ride
        must not confer NAMEHOOD. Measured on a corpus of thirteen
        sibling specifications: asked in isolation, "which workstation is
        the priciest" reads the field across the corpus and answers;
        asked after a question about the Osprey, the ridden name made the
        turn look like a question ABOUT the Osprey, the corpus-wide
        reading never ran, and the turn abstained with every price on the
        table. Naming is a property of what was asked.
        """
        if len(self.by_source) < 2:
            return set()
        order = _words(text, known=self.units)
        words = set(order)
        total = len(self.by_source)
        # THE MEMORY'S OWN SPEECH IS NOT A SOURCE ANYBODY NAMES (W105):
        # it is kept in the store so a follow-up can read it, and it
        # carries no document name to be called by.
        names = [(src, set(_words(_source_name(src), known=self.units)))
                 for src in self.by_source if src != SAID_SOURCE]
        called = set()
        for qw in words:
            matched = [src for src, ws in names
                       if any(inflect.same_stem(qw, w) for w in ws)]
            if matched and len(matched) == 1 != total:
                called.add(matched[0])
        called |= self._called_by_bigram(order, names)
        # A FAMILY RESEMBLANCE IS A PARTIAL MATCH. W38 said a source is
        # called by a pointer and not by a family resemblance; among the
        # sources a question does call, the same sentence decides which
        # it NAMES. Measured on a corpus of sibling programmes: a
        # question about one of them was read as naming six, because
        # every sibling shares the opening phrase and differs only in
        # its tail — and the comparison layout then fired for a question
        # that compares nothing. The one whose name the question
        # accounts for most completely is the one named. A TIE KEEPS
        # EVERYONE, because two names matched in full is what a
        # comparison is; only the half-matched siblings step back.
        if len(called) > 1:
            def _covered(src):
                ws = _words(_source_name(src), known=self.units)
                if not ws:
                    return 0.0
                got = sum(1 for w in ws
                          if any(inflect.same_stem(w, q) for q in words))
                return got / len(ws)
            best = max(_covered(src) for src in called)
            called = {src for src in called if _covered(src) >= best - 1e-9}
        return called

    def find(self, query, most=4, floor_share=0.5, scope=None):
        """Sentences whose content-words intersect the query the MOST.
        Prefix-tolerant (inflection: "doluluğu"~"doluluk"). Score = number of
        intersecting words; on a tie the short sentence wins (denser evidence)."""
        qwords_order = _words(query, known=self.units)   # order kept: the
        #                       name channel reads the question's BIGRAMS
        qwords = set(qwords_order)
        if not qwords:
            return []
        scores, named = self._score_over(qwords, self.index, self.key_index)
        # THE EXPANSION CHANNEL, at its rung of the ladder (`CHANNEL`). It is
        # added AFTER the document's own channels and it cannot replace them:
        # every sentence keeps the score its real words earned, and a generated
        # word can only raise a sentence, never lower another. What it buys is
        # the question asked in words the document does not use — the class this
        # layer was blind to by construction, being lexical. Nothing about the
        # block's CONTENT changes: the seats are still filled with
        # `self.sentences`, which is the document.
        if self.expand_index and _expansion_on():
            guessed, _named = self._score_over(qwords, self.expand_index, {},
                                               weigh=self.index)
            for sid, sc in guessed.items():
                scores[sid] = scores.get(sid, 0.0) + sc / CHANNEL
        # THE SOURCE-NAME CHANNEL. A document's name is part of what it says
        # about itself, and a question that speaks it deserves that document's
        # sentences. Measured (62 near-identical training outlines): asked for
        # one programme BY NAME, the store returned sentences from its
        # SIBLINGS and then refused — the name's words scored as ordinary
        # query words, and words like the corpus's own genre terms separate
        # nothing when every sibling carries them.
        #
        # THE WEIGHT IS THE NAME-WORD'S POWER TO SEPARATE SOURCES, log(S/s):
        # in how many of the S source names does this word appear. A word in
        # ONE name of sixty-two weighs log(62); a word in every name weighs
        # log(1) = 0 — exactly nothing — which also proves the single-document
        # case unchanged: there, every name word covers its whole population
        # and the channel vanishes. No threshold, no list, no language.
        #
        # It boosts only sentences that already scored on their own words. A
        # name match alone is not evidence — the question still has to touch
        # the sentence's content, the name only settles WHOSE sentences win.
        base = dict(scores)     # pre-boost scores — the floor reads THESE:
        #                         the name settles rank, never admission (a
        #                         boosted best was inflating best/2 and
        #                         sweeping the OTHER document's grounded
        #                         sentences out of a two-topic question)
        named_sources = set()
        self.last_named = set()
        if scores and len(self.by_source) > 1:
            total_sources = len(self.by_source)
            names = [(src, set(_words(_source_name(src), known=self.units)))
                     for src in self.by_source]
            # A SOURCE IS NAMED BY A POINTER, NOT BY A FAMILY RESEMBLANCE
            # — the pointing-word rule's third appearance. Membership used
            # to be a union over matched words, and a coded corpus showed
            # what that buys: "TPK-03 and TPK-04" matched every sibling on
            # the family letters, seven documents marched to the front,
            # and the two actually named waited outside. A source counts
            # as CALLED when some question word matches its name alone, or
            # when at least two of its name words are matched; one shared
            # family word calls nobody. The per-word log(S/s) boost keeps
            # pricing the family word at nearly nothing, unchanged.
            # ...and beyond the pointer, A NAME IS CALLED BY ITS BIGRAM —
            # the same neighbour-pair reading the digit gate trusts. Two
            # matched words were tried first and the cross-family trap
            # walked through them: "Alpha Delegation and Stress Handling"
            # matched "Alpha" and "Handling" for a THIRD course that stands
            # in neither phrase. Words scattered across a question call
            # nobody; words standing together in it, as they stand in the
            # name, do.
            unique_called = set()
            for qw in qwords:
                matched = [src for src, words in names
                           if any(inflect.same_stem(qw, w) for w in words)]
                if not matched or len(matched) == total_sources:
                    continue
                weight = math.log(total_sources / len(matched))
                if len(matched) == 1:
                    unique_called.add(matched[0])
                for src in matched:
                    for sid in self.by_source[src]:
                        if sid in scores:
                            scores[sid] += weight
            named_sources = set(unique_called) | self._called_by_bigram(
                qwords_order, names)
            # A FAMILY RESEMBLANCE IS A PARTIAL MATCH — the same rule the
            # layouts read (W66), applied where the seats are handed out.
            # Measured: "how long is the PRG-00 programme summary?"
            # boosted PRG-00, PRT-00 and TRP-00 alike, because siblings
            # share every word but the code; the record row of the
            # document actually asked about then had to outscore three
            # documents' worth of boosted prose, and did not.
            if len(named_sources) > 1:
                def _covered(src):
                    ws = _words(_source_name(src), known=self.units)
                    if not ws:
                        return 0.0
                    got = sum(1 for w in ws
                              if any(inflect.same_stem(w, q) for q in qwords))
                    return got / len(ws)
                best_cover = max(_covered(s) for s in named_sources)
                named_sources = {s for s in named_sources
                                 if _covered(s) >= best_cover - 1e-9}
            self.last_named = set(named_sources)
        if not scores:
            self.last_census = []
            self.last_named = set()
            # ...unless the CONVERSATION handed this turn its documents
            # (W109): "how long is the first one?" reaches no line by
            # word overlap at all, which is exactly the turn a scope
            # exists for. Nothing outside the scope can enter.
            return self._scoped_only(qwords, scope, most) if scope else []
        # THE CENSUS RIDES ALONG. Scoring has already touched every sentence
        # this question reaches; grouping those hits by source costs one pass
        # and answers a question retrieval cannot: not "what is the best
        # evidence" but "who all speaks of this". `find` records it beside
        # its result the way it records `last_sources`, and the session hands
        # it to the answer path as one line of store-attested fact — which is
        # what lets a conversational turn say "this appears in eight
        # programmes" instead of naming whichever single one won the seats.
        found = self._seats(qwords, scores, named, most, base=base,
                            scope=scope,
                            floor_share=floor_share,
                            named_sources=named_sources)
        # THE CENSUS COUNTS THE TOPIC, NOT THE QUESTION — and the topic is
        # what actually LANDED. Three cuts of this failed instructively:
        # any-overlap and floor-passer counting both let the question's
        # generic words vote (a sales outline out-tallied the true list on a
        # psychological-safety question), and IDF-rarity then picked the
        # scaffolding over the subject, because a corpus ABOUT a topic uses
        # the topic's words everywhere and the scaffolding nowhere — the same
        # inversion that broke the expansion filter. The seats already answer
        # it: retrieval chose them for this question, so the question words
        # the seated sentences actually carry ARE the topic as this corpus
        # understands it. The tally is `where` over those — every sentence
        # carrying all of them votes for its source.
        seated = " ".join(found)
        held = set(_words(seated))
        landed = [w for w in qwords
                  if any(inflect.same_stem(w, h) for h in held)]
        # THE TOPIC IS A PAIR, AND THE CORPUS PICKS IT. Single-word topic
        # extraction failed in both directions on the same question: a word
        # in every source ("training", via the template headings the windows
        # carry) flooded the tally, and a word in ONE source ("regarding", a
        # freak occurrence) strangled the AND down to that source. A topic,
        # as a census can use one, is two question words that the documents
        # answer TOGETHER — and which pair that is, the corpus says itself:
        # the landed pair whose co-occurring sentences span the MOST sources.
        # No dial anywhere; ties fall to the pair whose words are rarer.
        matched = {}
        for w in landed:
            sids = set()
            for key, postings in self.index.items():
                if inflect.same_stem(w, key):
                    sids |= postings
            matched[w] = sids
        best, best_span = None, 1
        for i, a in enumerate(landed):
            for b in landed[i + 1:]:
                both = matched[a] & matched[b]
                span = len({self.sentences[sid][1] for sid in both})
                rarity = -(len(matched[a]) + len(matched[b]))
                if span > best_span or (span == best_span and best and
                                        rarity > best[0]):
                    best, best_span = (rarity, both), span
        if best and best_span > 1:
            tally = {}
            for sid in best[1]:
                src = self.sentences[sid][1]
                tally[src] = tally.get(src, 0) + 1
            self.last_census = sorted(tally.items(),
                                      key=lambda kv: (-kv[1], kv[0]))
        else:
            self.last_census = []
        # A SCOPE ANSWERS EVEN WHEN THE WORDS DO NOT REACH IT (W109).
        # "How long is the first one?" shares no word with "EĞİTİM
        # SÜRESİ: 2 Tam Gün", so word overlap seats nothing at all and
        # a scope that only FILTERS filters an empty list. When the
        # conversation has handed this turn its documents, their lines
        # are eligible on scope alone — best word overlap first, which
        # is zero for all of them and therefore the document's own
        # order. Nothing outside the scope enters; with no scope this
        # branch does not exist.
        if scope and not found:
            return self._scoped_only(qwords, scope, most)
        return found


    def _score_over(self, qwords, index, keys, weigh=None):
        """The lexical scoring, over ONE index — {sid: score}, and the sentences
        where a query word is the FIELD NAME.

        Factored out of `find` so that the expansion index is scored by the
        SAME rule as the document's own words rather than by a second, quietly
        different one. `keys` is the field-name index for this channel; the
        expansion channel has none (a guessed question names no record's slot),
        and passes an empty mapping.

        `weigh` is a second index consulted for the IDF DENOMINATOR only, and
        it exists because rarity measured inside the expansion index is a lie
        about the document. The expansion index is sparse by construction — a
        few queries per line, and only the words the line does not already
        carry — so a word the document uses on forty lines, appearing in ONE
        generated query, comes out of an expansion-only count looking as
        informative as a word that occurs once in the whole text. It would then
        outweigh several real matches, on the strength of a rarity it does not
        have. Counting the postings of BOTH channels answers the question IDF
        is actually asking — how much does this word narrow THIS DOCUMENT down
        — and leaves a genuinely unseen word (`spire`, which the document never
        writes) at its full weight, because that word really is that rare."""
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
        # A TOTAL ORDER, NOT JUST BY LENGTH — one half of a determinism fix.
        # `qwords` is a SET OF STRINGS, so `sorted(..., key=len)` left
        # equal-length query words in whatever order the set iterated, which
        # depends on PYTHONHASHSEED. That order is the order the groups are
        # processed in, which is the order sentence ids first enter `scores`,
        # which used to decide the ranking whenever two sentences tied — see the
        # tie-break below. Net effect: the same question against the same
        # document retrieved DIFFERENT evidence between two runs of
        # byte-identical code (measured on bench/manual.txt: PYTHONHASHSEED=1
        # and =3 disagree about "Cihazın işletim sistemi nedir"). This layer is
        # documented as deterministic — that is its whole argument against
        # embedding similarity — so some of the run-to-run oscillation this
        # benchmark has been fighting was the retrieval, not the engine.
        groups = []
        for qw in sorted(qwords, key=lambda w: (len(w), w)):
            for g in groups:
                if any(qw.startswith(m) or m.startswith(qw) for m in g):
                    g.append(qw)
                    break
            else:
                groups.append([qw])
        scores = {}
        named = set()       # sentences where a query word is the FIELD NAME
        for group in groups:
            # Candidate index words of the whole group. ALWAYS expand (no
            # exact-hit shortcut): with only the exact form, an inflected
            # query word ("boyutları") hid the ROOT-carrying spec row
            # ("Boyutlar 360 mm ...") whenever the inflected form happened to
            # exist verbatim elsewhere — the answer-carrying sentence never
            # entered the candidate set (measured, manual trace).
            # INFLECTION TOLERANCE — the SHARED criterion (`inflect`), which is
            # the same one the identity gate and the coverage gate use.
            #
            # REMOVED: three separate rules that lived here — a >=4-letter
            # forward prefix, a >=3-letter reverse prefix ("inçtir" reaching the
            # index word "inç"), and a proportional 70%-stem clause beside them.
            # Each was widened or narrowed against one document's trace, and
            # together they made this the loosest of the four copies of the rule.
            # The shared criterion is stricter, and a short unit like 'inç' is no
            # longer reachable from an inflected query word by prefix alone —
            # that is a measured cost, not a free cleanup.
            cand = {}
            for qw in group:
                if qw in index:
                    cand[qw] = index[qw]
                for w, ids in index.items():
                    if w in cand:
                        continue
                    if inflect.same_stem(qw, w) or inflect.kin(qw, w):
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
            seen = set(union)
            if weigh is not None:
                for w in cand:
                    seen |= weigh.get(w, set())
            weight = math.log(1 + total / len(seen))
            # THE WEIGHT OF A MATCH IS THE INFORMATION OF WHAT ACTUALLY
            # MATCHED. One weight for the whole family is right when the
            # family is the concept — but in prose the family is
            # everywhere and the asked FORM is rare. Measured on a
            # 1,300-page novel: asked "Sonya kimin evinde bir odada
            # oturur?", the passage that answers it word for word ranked
            # TWELFTH, below passages matching two words, because "otur",
            # "ev" and "oda" as families are worth almost nothing in a
            # novel while the interrogative "kimin" is rare and stayed
            # heavy. The question's own words were being priced as
            # concepts and the function word as information.
            #
            # So: a sentence carrying the word AS ASKED is weighed by
            # that word's own rarity when it is rarer than the family's;
            # a sentence carrying only a relative keeps the family
            # weight. This can only lift sentences that literally contain
            # what was typed — it is not the "rare inflection out-informs
            # the stem" trap, which is about rare forms in the INDEX, not
            # about the asker's own form.
            # ...AND ONLY WHERE THERE IS NO SOURCE CHANNEL. In a store of
            # many documents the naming channel already anchors a question
            # (log(S/s) prices a word by how well it separates sources),
            # and this reading is not needed: measured on 62 sibling
            # outlines it cost one factual answer, deterministically,
            # because a prose line carrying the asked form outranked the
            # record row that answers. With ONE source that channel is
            # silent — log(1/1) is zero for every word — and the
            # sentence-level weight is all the anchoring there is. That is
            # exactly the corpus where the flattening bites.
            asked_weight = {}
            for qw in group if len(self.by_source) <= 1 else ():
                own = index.get(qw)
                if not own:
                    continue
                seen_own = set(own)
                if weigh is not None:
                    seen_own |= weigh.get(qw, set())
                own_weight = math.log(1 + total / len(seen_own))
                if own_weight > weight:
                    for sid in own:
                        if own_weight > asked_weight.get(sid, 0):
                            asked_weight[sid] = own_weight
            keyed = set()
            for w, ids in cand.items():
                keyed |= ids & keys.get(w, set())
            named |= keyed
            for sid in union:
                # FIELD-NAME match doubles the group's weight: the same word
                # as a kv KEY names the record's slot, in prose it is mere
                # co-occurrence (kv-key weighting; format cue, no language
                # rule). The doubling is `CHANNEL` — see the ladder.
                scores[sid] = scores.get(sid, 0) + asked_weight.get(
                    sid, weight) * (CHANNEL if sid in keyed else 1)
        return scores, named

    def _scoped_only(self, qwords, scope, most):
        """The scoped sources' own lines, when word overlap seats none.

        A follow-up names nothing a search can hold, so the ordinary
        channels score nothing and a scope that only FILTERS filters an
        empty list. Here the scope is the ONLY admission: best overlap
        first (usually zero for all, and then the document's own
        order), nothing from outside it, and no scope means no branch.
        """
        pool = []
        for src in scope or ():
            pool += list(self.by_source.get(src, ()))
        if not pool:
            return []
        qset = set(qwords)
        pool.sort(key=lambda sid: (
            -len(qset & set(_words(self.sentences[sid][0]))), sid))
        seated = pool[:most]
        self.last_sources = [self.sentences[sid][1] for sid in seated]
        return [self.sentences[sid][0] for sid in seated]

    def _seats(self, qwords, scores, named, most, base=None,
               floor_share=0.5, named_sources=(), scope=None):
        """The ranking, the noise floor and the seats — unchanged, and moved
        here only so that `find` can score two channels before ranking once."""
        # on a tie the LONG one wins: a short table crumb ("Orta · 3") must not
        # beat a context-carrying prose window — a cell value wandering without
        # context was attaching to the wrong attribute (the camera/KVKK case).
        # ...and the ranking's last tie-break is the sentence id — the other
        # half, and the one that closes the hole at the point where it bit.
        # Two sentences of equal score AND equal length were ordered by their
        # insertion order in `scores`, i.e. by group order, i.e. by hash seed.
        # Either fix alone makes the manual reproducible across seeds
        # (verified: query order and this tie-break each independently take the
        # differing-block count to zero); both are kept because they close
        # different links of the same chain, and neither can change WHICH
        # sentences score what.
        ranked = sorted(scores.items(),
                        key=lambda kv: (-kv[1], -len(self.sentences[kv[0]][0]),
                                        kv[0]))
        # NOISE FLOOR: anything scoring below half the best score drops.
        #
        # It used to read `max(1 if len(qwords) <= 2 else 2, best // 2)`, which
        # compared a SCORE to a COUNT: `best` is a sum of IDF weights (nats),
        # while the 1 and the 2 were numbers of matching words, left over from a
        # version where the score WAS a count. Mixing the units makes the
        # absolute term meaningless — its severity is whatever the log weights
        # happen to be worth in this corpus — and `best // 2` silently floored a
        # float, so the floor moved in steps. The proportional half is the whole
        # rule and it is scale-free, so it needs no companion term and no
        # per-query-length exception.
        base = base if base is not None else scores
        best = max(base.values())
        # AN ANSWER NEEDS A FLOOR; MATERIAL NEEDS BREADTH. The halving floor
        # exists so one answer is not diluted by noise — and it is exactly
        # wrong for gathering COMPOSITION material, where a two-topic brief
        # had its second topic swept out (measured: 3.74 against a 7.98/2
        # bar, and the draft could then only ever cover one topic). The
        # composer passes 0: weak material is harmless there, because the
        # OUTPUT gate judges every drafted line against the material — what
        # is not used cannot be spoken, and what is not gathered cannot be
        # covered. Every other caller keeps the half.
        floor = best * floor_share
        # NEAR-DUPLICATE SUPPRESSION: the same content is indexed at several
        # window scales (raw line, 3-window, 6-window) — without this, one
        # strong-but-wrong region filled ALL top slots and the answer-carrying
        # row never got a seat (measured, manual trace). Overlap is a set view
        # (Jaccard on content-words), no language rule.
        #
        # A region may hold at most a FRACTION OF THE SEATS, not exactly one. The
        # purpose of this filter is that no single region MONOPOLISES the top
        # list; suppressing every variant went further than that purpose and
        # removed CORROBORATION. Measured (hospital trace, "regülasyon riski"):
        # the answer binding "Kamera tabanlı düşme tespiti ... Yüksek (KVKK)"
        # occurs in two overlapping table windows; with one seat only, the
        # support check saw a single flattened row and rejected the CORRECT
        # answer, and the question fell to an abstention. With two seats the
        # binding is confirmed twice and the answer passes.
        #
        # TWO IS NOT A TUNED NUMBER, IT IS WHAT CORROBORATION IS: one view of a
        # region states the binding, a second independent view confirms it, and a
        # third adds no information that the second did not already add. What WAS
        # wrong with writing it as a bare constant is that it silently became
        # "every seat" whenever a caller asked for two seats or fewer — the
        # monopoly this filter exists to prevent, reappearing at small block
        # sizes. So the cap is stated with the guarantee attached: at most one
        # corroboration, and never the whole block.
        region_cap = max(1, min(2, most - 1))
        # A SOURCE IS A REGION AT DOCUMENT SCALE, and in a multi-document
        # store it gets the same cap for the same reason. Measured (62 sibling
        # training outlines, "which programmes cover X"): the term lived in
        # EIGHT documents and every seat went to the strongest one, so the
        # block could only ever name a single programme — the monopoly the
        # region rule exists to prevent, one level up. The cap is the region
        # cap itself, because the argument is the same argument: one seat
        # states, a second corroborates, a third crowds out another voice.
        #
        # NOTHING IS DISCARDED: a same-source candidate above the cap steps
        # aside, and steps back in, in rank order, if the block would
        # otherwise go unfilled — so a store where one document genuinely
        # holds all the evidence still fills every seat with it. A
        # single-document store never enters this path at all and keeps its
        # ranking byte for byte.
        source_cap = region_cap if len(self.by_source) > 1 else None
        source_seats = {}
        overflow = []
        keep = []
        kept = []                           # [[wordset, seats_taken], ...]
        for sid, sc in ranked:
            # `continue`, not `break`: ranked order is the BOOSTED order, and
            # admission reads the base score — a boosted latecomer may sit
            # after a below-floor entry. With no boosts the two are the same
            # walk, ended a few empty iterations later.
            if base.get(sid, sc) < floor and keep:
                continue
            words = set(_words(self.sentences[sid][0]))
            dup = False
            for entry in kept:
                inter = len(words & entry[0])
                # SAME REGION, two ways of overlapping. Jaccard catches two
                # windows of nearly the same width; CONTAINMENT catches the
                # window scales, which are NESTED — the 3-sentence window sits
                # inside the 6- and that inside the 12-, and nesting makes the
                # union large while the intersection stays whole, so Jaccard
                # reads two views of one paragraph as two regions. Measured
                # (hospital, "ilk 6 ayda ... yüksek öncelik işaretlidir"): all
                # SIX seats went to nested views of the one SONUÇ paragraph, and
                # the section that actually carries the marker never got in —
                # the monopoly this filter exists to prevent. Containment was
                # once tried INSTEAD of Jaccard and cost a spec window that
                # differed by the 3 tokens holding the answer; the lesson was
                # about the SEAT COUNT, not the measure. So: either measure marks
                # the region, and a region still gets its share of seats.
                #
                # THE BOUNDARY IS THE MAJORITY, not 0.75. Two texts are views of
                # one region when they share MORE than they differ — that is what
                # "the same region" means, and it is the only boundary on a
                # similarity ratio that is not a dial. The 0.75 was three
                # quarters because three quarters worked on one trace.
                if inter and (inter / max(1, len(words | entry[0])) > 0.5
                              or inter / max(1, min(len(words),
                                                    len(entry[0]))) > 0.5):
                    if entry[1] < region_cap:
                        if source_cap is not None:
                            # A SAME-REGION TWIN IS THE SAME VOICE TWICE.
                            # In a multi-document store the source cap makes
                            # the corroborating variant expensive — it
                            # spends the document's other seat on the same
                            # paragraph said again, and the record line
                            # from a different region waits outside
                            # (measured: eight questions, one shape). The
                            # twin steps aside into the overflow and steps
                            # back, in rank order, only if the block would
                            # otherwise go unfilled. A single-document
                            # store keeps the corroboration byte for byte.
                            overflow.append(sid)
                            dup = True
                            break
                        entry[1] += 1       # corroborating variant — admit
                        break
                    dup = True
                    break
            if dup:
                continue
            if source_cap is not None:
                origin = self.sentences[sid][1]
                if source_seats.get(origin, 0) >= source_cap:
                    overflow.append(sid)
                    continue
                source_seats[origin] = source_seats.get(origin, 0) + 1
            keep.append(sid)
            kept.append([words, 1])
            if len(keep) >= most:
                break
        for sid in overflow:
            if len(keep) >= most:
                break
            keep.append(sid)
        if not keep:
            # if the floor swept everything, still return the best 2: if "no
            # evidence at all" is wrong, we are giving the answer gates no
            # chance whatsoever; weak evidence is filtered anyway by the
            # coverage+support checks.
            keep = [sid for sid, _sc in ranked[:2]]
        seat = self._measurement_seat(qwords, ranked, keep, named)
        if seat is not None:
            keep.append(seat)
        # THE NAMED DOCUMENT SPEAKS FIRST. Measured, conversation trace: the
        # anchored programme was seated third and fourth while a sibling's
        # rare stem took the top seats, and the engine — which reads the
        # block top-down — answered from the sibling. Admission was right;
        # order was not. When the question named sources (the source-name
        # channel matched them), their seats move to the front, each group
        # keeping its own internal order; nothing enters, nothing leaves.
        # With no named source — every single-document store, every question
        # that names nothing — the key is False for all and the sort is
        # stable: byte-for-byte the old order.
        if named_sources:
            keep.sort(key=lambda sid: self.sentences[sid][1]
                      not in named_sources)
        # THE MEMORY'S OWN SPEECH SITS BELOW THE DOCUMENTS (W105). A
        # turn's words are kept so the NEXT question can read them
        # (`Session._remember_said`) — that is what makes a follow-up
        # answerable at all — but a memory that quotes itself can
        # drift, and a drifted sentence must never outrank the page it
        # drifted from. Stable, so nothing else about the order moves:
        # documents keep their seats in their own order, and `#said`
        # follows.
        # ...AND THEY ARE GIVEN A TAIL SEAT, NOT LEFT TO COMPETE. The
        # first cut sorted them to the back of the same list, and on a
        # store of sixteen thousand document lines that is the same as
        # deleting them: measured live, the follow-up refused exactly as
        # it did before the reading existed. So the documents keep every
        # seat they won, and the memory's own words are appended after
        # them — at most two, below everything, never in place of a
        # document.
        # THE CONVERSATION'S SCOPE, WHEN THE QUESTION CARRIES NONE
        # (W109): a follow-up ("how long is the first one?") names no
        # document a search can hold, and the turn before named some.
        # Those sources are handed in as `scope`, and this reading
        # honours them the way it honours a named document — seats go to
        # them, and to nothing else, unless they hold nothing at all.
        if scope:
            scoped = [sid for sid in keep
                      if self.sentences[sid][1] in scope]
            if scoped:
                keep = scoped
        keep = [sid for sid in keep
                if self.sentences[sid][1] != SAID_SOURCE]
        spoken = [sid for sid, _score in ranked
                  if self.sentences[sid][1] == SAID_SOURCE][:2]
        keep = keep + [sid for sid in spoken if sid not in keep]
        keep = self._supersede(keep)
        self.last_sources = [self.sentences[sid][1] for sid in keep]
        return [self.sentences[sid][0] for sid in keep]

    def _supersede(self, keep):
        """OF TWO DATED LINES THAT CLASH ON A VALUE, THE LATER SPEAKS.

        Measured on LongMemEval's knowledge-update questions: a personal
        best stated in May and restated two weeks later sat in the block
        side by side, both attested, and the answer spoke the stale one.
        The graph already has these manners (a newer fact lowers its
        rival's trust); the evidence layer — the cheap door everything
        conversational comes through — had none.

        The rule is read off the store, no dial anywhere: two seated
        lines clash when their sources are the same dated FAMILY (same
        non-digit name words, each carrying digits that parse as a
        date), their non-numeric content words overlap by MORE THAN HALF
        of the smaller line's — the codebase's one boundary — and the
        numbers they carry differ. The older line loses its seat before
        the block is built, because a block holding both numbers hands
        the digit veto two right answers and the choice to chance.
        Undated sources never enter this reading: a corpus of manuals
        seats exactly what it seated before, byte for byte.

        The cost, stated rather than hidden: a question asking for the
        SUPERSEDED value loses that line too. Recorded in W83; the day a
        question set shows that class, this reading learns tenses.
        """
        def _dated(src):
            # THE FAMILY IS THE NAME BEFORE THE FIRST DIGIT. "Same
            # non-digit words" was the first cut, and real chat stamps
            # killed it: "chat 2023/05/25 (Thu)" and "chat 2023/05/27
            # (Sat)" share every naming word and differ in the weekday —
            # which is part of the DATE, written in letters. A language
            # list of weekdays is the hard-coding this codebase forbids;
            # the structure that needs no language is that a dated stamp
            # is a name followed by its date, decorations included.
            match = re.search(r"\d", src or "")
            if not match:
                return None
            digits = tuple(int(d) for d in re.findall(r"\d+", src))
            family = " ".join(_words(src[:match.start()]))
            return (family, digits) if family else None

        def _named_numbers(text):
            # WHAT A LINE SAYS ABOUT A VALUE IS THE NAME THE NUMBER SITS
            # UNDER. Whole-line overlap was the first cut, and chatty
            # lines killed it: two statements of one personal best share
            # five words and disagree in thirty, because running prose
            # wraps its facts in weather. What both lines DO share is
            # the two plain words in front of the number — "best time
            # of 27:12" / "best time of 25:50" — and that is what
            # naming the same quantity looks like, in any language that
            # puts names near values.
            words = _words(text)
            out = {}
            for position, w in enumerate(words):
                if not any(c.isdigit() for c in w):
                    continue
                name = tuple(x for x in words[max(0, position - 2):position]
                             if not any(c.isdigit() for c in x))
                if name:
                    out.setdefault(name, set()).add(w)
            return out

        drop = set()
        for i, a in enumerate(keep):
            for b in keep[i + 1:]:
                if a in drop or b in drop:
                    continue
                da = _dated(self.sentences[a][1])
                db = _dated(self.sentences[b][1])
                if (not da or not db or da[0] != db[0]
                        or da[1] == db[1]):
                    continue
                na = _named_numbers(self.sentences[a][0])
                nb = _named_numbers(self.sentences[b][0])
                clash = any(
                    name in nb and na[name] != nb[name]
                    for name in na)
                if not clash:
                    continue
                drop.add(a if da[1] < db[1] else b)
        return [sid for sid in keep if sid not in drop]

    def _measurement_seat(self, qwords, ranked, keep, named):
        """ONE EXTRA seat, for a question that asks for a MEASUREMENT.

        The measured loss class: the manual states `Ekran 15.6" LCD` and the
        question asks "kaç inçtir". The unit in the document is a QUOTATION
        MARK — it has no letters, so the lexical channel has nothing of the
        question's 'inç' to match, and the spec line is reachable only through
        'ekran', a word the manual uses on forty menu lines. It ranked 33rd.

        Two structural facts are enough to find it, and neither needs a word
        list. First, the question ASKS FOR A QUANTITY: one of its words is a
        token this corpus binds to numbers (`quantities` — the manual writes
        "(12 inç)" elsewhere). Second,
        the answer is a line where another question word is the FIELD NAME and
        which STATES A MEASUREMENT — a typed value, number and unit, read by
        the core (`v3.memory.measure`). Nothing here knows that 15.6" is
        inches; it knows that the question wants a quantity and that this line
        is the one naming the asked-about field with a quantity in it.

        WHY A SEAT AND NOT A WEIGHT. Reweighting the field-name channel was
        tried and measured: it does rescue this question and it took the manual
        to 29/30 — while dropping hospital 16/16 → 14/16, because five of the
        sixteen blocks changed underneath. The lesson from that was about the
        BLOCK: an ordering change anywhere is a change everywhere. So this
        mechanism cannot reorder anything. It appends, after the ranking is
        final, and the existing seats come out of `find` in exactly the order
        and the number they had before — a question that asks for no quantity,
        or one whose field is already in its block, is untouched byte for byte.
        """
        from lmm.core.memory import measure
        if not named:
            return None
        # ASKS FOR A QUANTITY — inflection-tolerant in the same two-way form the
        # index lookup uses ("inçtir" reaches the corpus's "inç"), because the
        # question's word arrives inflected and the corpus's does not.
        quantities = self.quantities
        # The same one relaxation as in `find`, and the same license: `q` is a
        # token THIS CORPUS binds to numbers, so a question word that carries it
        # as its root is asking for that quantity ("inçtir" → "inç"). Below the
        # shared root bound only because the corpus vouched for the token.
        asks = {w for w in qwords
                if w in quantities or any(inflect.same_stem(w, q)
                                          or w.startswith(q)
                                          for q in quantities)}
        if not asks:
            return None
        held = set(keep)
        words = [set(_words(self.sentences[sid][0])) for sid in keep]
        best = None
        limit_chars, limit_toks = self._record_bounds()
        SCAN = 300              # the seat is a rescue, not a second search
        for sid, _score in ranked[:SCAN]:
            if sid in held or sid not in named:
                continue
            sentence = self.sentences[sid][0]
            # RECORD SHAPE, the same test the field-name reader uses. The seat
            # is for the SPEC LINE — a line that is a field and its quantity and
            # nothing else. Without this bound the seat filled itself with
            # prose windows and table crumbs that happen to name a field and
            # carry a number: measured, and they are noise in the block, which
            # is the one thing an extra seat must not add.
            if len(sentence) > limit_chars or len(_tokens(sentence)) > limit_toks:
                continue
            found = measure(sentence)
            if found is None or found[2] is None:
                continue        # a bare number is not a measurement's answer
            # AND ITS UNIT MUST BE A UNIT OF THIS CORPUS — the same majority
            # test, applied to the other side. Adjacency read off a single line
            # is generous by design ("Ameliyathane çizelgeleme 1 Düşük" states
            # a 1 measured in 'Düşük'), and generous is fine for indexing but
            # not for handing an answer an extra line: here the token has to be
            # one the WHOLE corpus characteristically binds to numbers. That is
            # what separates '"' and 'mm' from a rating word that happened to
            # follow a digit once.
            unit = found[2]
            if unit in self._notation:
                continue        # a decimal point is not a unit of weight
            parts = {unit} | {p for p, _ in _quantity_tokens(unit)}
            if not (parts & quantities):
                continue
            mine = set(_words(sentence))
            # THE SEAT IS ONLY FOR AN UNSPELLABLE UNIT. Its whole justification
            # is that the question names a quantity the answer line does not
            # SPELL — 15.6" against "inç" — so the lexical channel had nothing
            # to match and never could have. If the line does spell that word,
            # the ordinary ranking saw it and placed it where it placed it, and
            # overriding that is exactly the ordering change this mechanism
            # refuses to make. Measured: without this the seat handed a "hangi
            # tier" question a TIER 2 heading line while the answer was Tier 3
            # — a plausible wrong answer, planted in the block by the rescue.
            if any(inflect.same_stem(w, q) for q in asks for w in mine):
                continue
            # The seat is for evidence that is MISSING, not for one more view of
            # what is already there (the region rule the seat list itself obeys,
            # at the same majority boundary).
            if any(len(mine & other) > 0.5 * max(1, min(len(mine), len(other)))
                   for other in words):
                continue
            # AMONG SPEC LINES, THE DENSEST ONE. Score order cannot choose here:
            # it is the very ordering that buried the answer, and every
            # candidate at this point has already passed the same filters. What
            # separates "Ekran 15.6\" LCD" from "Demo ekranının ayrıntıları için
            # 11.8 Resim Demosu" is that the first is ABOUT the asked field and
            # nothing else — its words are the question's words plus the
            # quantity — while the second spends most of itself elsewhere.
            # (Density is the wrong measure for the main ranking and was
            # measured to be: a claim's evidence overlaps the claim MORE, so
            # dividing by length rewards crumbs. Here every candidate is
            # already a record-shaped line and the question is which record.)
            hit = sum(1 for w in mine
                      if w.isdigit() or w in qwords
                      or any(inflect.same_stem(w, q) for q in qwords))
            rank = (hit / max(1, len(mine)), hit, -len(mine), -sid)
            if best is None or rank > best[0]:
                best = (rank, sid)
        return best[1] if best is not None else None

    # --- offline document expansion (doc2query--) ----------------------

    def pending_expansion(self):
        """The units worth paying an expansion for: what the DOCUMENT wrote,
        minus what has been expanded already. Returns [(sid, text), ...].

        The context windows are excluded because they are the same sentences
        again at three widths — expanding them buys the same words four times
        and costs four times the calls."""
        return [(sid, text) for sid, (text, _src) in enumerate(self.sentences)
                if sid not in self._derived and sid not in self.expansions]

    def learn_bridge(self, mapping):
        """Index the reader's words for each head. `mapping`: {head: [word]}.

        Two filters, both structural. A word the head itself carries adds
        nothing — the lexical match already covers it. And a word that IS
        another head's own word may not bridge here: it already means
        something else in this corpus, and a bridge that overrides the
        corpus's own vocabulary is how MEMORY answers for STORAGE.
        Returns how many words were kept."""
        owned = {}
        for head in mapping:
            for w in _words(head):
                owned.setdefault(w, set()).add(head)
        kept = 0
        for head, words in mapping.items():
            mine = set(_words(head))
            for word in words:
                for w in _words(word):
                    if w in mine:
                        continue
                    if owned.get(w) and head not in owned[w]:
                        continue        # another head's own word
                    if head not in self.head_bridge.get(w, ()):
                        self.head_bridge.setdefault(w, set()).add(head)
                        kept += 1
        return kept

    def bridged_heads(self, question_words):
        """The heads this question's words nominate — no model call.

        A vote, not a union, and the weight is the same statistic every
        other channel here uses. Measured on the first live corpus: the
        union handed "kaç gün?" THREE heads — "kaç" is a question word
        the engine had offered to half the fields, so it nominated
        everywhere and the one-head rule rightly refused the tie. A word
        that bridges to one head knows something; a word that bridges to
        many knows almost nothing; log(1 + H/df) says exactly that, the
        way log(S/s) already does for source names. The best-weighted
        head(s) are returned — a genuine tie still declines downstream.
        """
        heads_total = set()
        for hs in self.head_bridge.values():
            heads_total |= hs
        votes = {}
        for q in question_words:
            named = self.head_bridge.get(q)
            if not named:
                continue
            weight = math.log(1 + len(heads_total) / len(named))
            for head in named:
                votes[head] = votes.get(head, 0.0) + weight
        if not votes:
            return set()
        best = max(votes.values())
        return {h for h, v in votes.items() if v >= best}

    def learn_expansions(self, generated):
        """Filter the generated queries and index the survivors.

        `generated`: {sid: [text, ...]} — what the engine produced for each
        unit, unfiltered. Returns how many were kept.

        THE FILTER IS THE 'minus minus' IN doc2query--. Generating K questions
        per line and indexing all of them is doc2query, and its known cost is a
        bigger index full of queries that point at the wrong passage. The
        original filters with a relevance model; this filters with the DOCUMENT,
        which is cheaper and needs no second engine:

            a generated query is asked of the real index, and what is measured
            is its MARGIN — how much better it reaches the line it was
            generated from than it reaches any OTHER region of this document.

        A query that drifts — one whose words belong to some other part of the
        text — scores that other part higher and comes out with a negative
        margin. A query that introduces genuinely new vocabulary reaches
        nothing lexically at all and comes out at zero, which is the case the
        expansion exists for. `_same_region` keeps the unit's own context
        windows from being counted as rivals; they are the same line again, and
        a line does not drift to itself.

        THE THRESHOLD IS READ OFF THE POPULATION, not written down: the median
        margin of everything this document generated, floored at the drift
        boundary. The median is what makes the index smaller (half of a
        well-behaved crop is dropped, which is the paper's own trade); the floor
        is not a tuned number but the SIGN of the margin — below zero the query
        names another region better than its own, and there is no document for
        which that is worth indexing.

        ONLY THE NOVEL WORDS ARE INDEXED. A generated word the line already
        carries is reachable already, and indexing it again would let a guess
        add weight to an observation that was doing fine on its own.
        """
        # THE DOCUMENT IS THE ONLY LANGUAGE SAMPLE THERE IS, and it has to be
        # used, because the margin filter below cannot see this failure by
        # construction: a query written in ANOTHER LANGUAGE reaches nothing
        # lexically and scores zero, which is exactly the signature of the
        # genuinely-new vocabulary the expansion exists to buy.
        #
        # Measured on NIST SP 800-63B, an English publication: the engine
        # expanded it into Turkish, German, Spanish, Portuguese and French, the
        # margin filter kept 62% of that, and one survivor was a REFUSAL
        # sentence indexed as a query. Instructing the engine does not close it
        # — putting the language rule first changed nothing, and removing the
        # multilingual examples moved the drift from Spanish to French. It is
        # not a prompt defect; a 3B engine simply does not hold a language
        # instruction across this task.
        #
        # So it is decided on the data. Every token the document wrote, this
        # time INCLUDING the short function words `_words` drops, is the
        # sample, and a query belongs to the document's language when enough of
        # its tokens are in it. Measured over 60 generated queries against NIST
        # the two populations do not overlap at all: wrong-language sits at
        # 0.00, right-language at 1.00.
        #
        # HOW MUCH IS ENOUGH IS READ OFF THE DOCUMENT, not written down, and
        # that matters because "absent from this text" means two different
        # things in two different sizes of text. In a publication it means
        # another language. In a three-line store it means the text is too
        # short to have said the word yet, and a fixed majority would throw
        # away the honest paraphrase the expansion exists to buy.
        #
        # So the document is measured against ITSELF first: how much of a
        # typical line is it able to find in the REST of what it wrote. That is
        # its own redundancy — near 1.0 for a publication, near 0.15 for three
        # sentences — and it is the scale everything else is read on. A query
        # is kept when it reaches PAST halfway to it, which puts the line in
        # the empty valley in both regimes rather than at a number anyone
        # chose. The boundary is excluded and that is not a detail: on this
        # publication the floor lands at exactly 0.50, and a short query with
        # one section number in it ("5.2.2 ne olur") sits exactly on the line. No language is named anywhere, and a Turkish document keeps
        # its Turkish expansions by exactly the same rule.
        spoken, lines = {}, []
        for sentence in self.sentences:
            tokens = re.findall(r"\w+", fold(sentence[0]), re.UNICODE)
            lines.append(tokens)
            for token in tokens:
                spoken[token] = spoken.get(token, 0) + 1
        selves = []
        for tokens in lines:
            if not tokens:
                continue
            own = {}
            for token in tokens:
                own[token] = own.get(token, 0) + 1
            selves.append(sum(spoken[t] > own[t] for t in tokens) / len(tokens))
        selves.sort()
        # half of the document's own redundancy — see above
        floor = (selves[len(selves) // 2] / 2.0) if selves else 0.0

        # THE DECISION, in two language-free readings. (The first cut of
        # this was a margin against a median — scored over ALL the query's
        # words, it threw away a record line's honest bridge ("how many
        # people can attend" for "SEATS: sixteen people") on the corpus-
        # common word it shares with every other line: the fourth member
        # of the IDF-inversion family, measured in the field. And its
        # median floor quietly killed the zero-margin case its own
        # docstring names as the reason expansions exist.)
        #
        #   PURE ADDITION — the query must add at least one word the
        #   document does not speak anywhere: that word is what gets
        #   indexed, and a query that adds nothing indexes nothing (a
        #   drifting query built of other regions' words lands here too).
        #
        #   A POINTING WORD POINTS; a word written everywhere points
        #   nowhere. When the query carries a word the document wrote in
        #   EXACTLY ONE unit and that unit is not the query's own region,
        #   the query is that other unit's, not this one's — dropped.
        unit_of = {}
        for usid, (text_u, _src_u) in enumerate(self.sentences):
            if usid in self._derived:
                continue
            for w in set(_words(text_u, known=self.units)):
                unit_of.setdefault(w, set()).add(usid)
        kept = 0
        for sid in sorted(generated):
            if sid >= len(self.sentences):
                continue
            mine = set(_words(self.sentences[sid][0]))
            for text in generated[sid]:
                spelling = re.findall(r"\w+", fold(text), re.UNICODE)
                if not spelling:
                    continue
                share = sum(w in spoken for w in spelling) / len(spelling)
                if share <= floor:
                    continue                    # another language — see above
                novel = set(_words(text, known=self.units)) - mine
                pure = {w for w in novel if w not in spoken}
                if not pure:
                    continue                    # adds nothing → indexes nothing
                foreign = False
                for w in novel - pure:
                    homes = unit_of.get(w, set())
                    if len(homes) == 1 and not _same_region(
                            mine, set(_words(
                                self.sentences[next(iter(homes))][0]))):
                        foreign = True
                        break
                if foreign:
                    continue
                self._index_expansion(sid, text)
                kept += 1
        # A unit the engine produced nothing usable for is still EXPANDED — it
        # was paid for and asked. Without this it would be handed back by
        # `pending_expansion` and paid for again on the next pass.
        for sid in generated:
            self.expansions.setdefault(sid, [])
        return kept

    def _index_expansion(self, sid, text):
        """Record one surviving expansion and index the words the unit itself
        does not carry. `self.sentences` is not touched — see __init__."""
        self.expansions.setdefault(sid, []).append(text)
        mine = set(_words(self.sentences[sid][0]))
        for w in set(_words(text)) - mine:
            self.expand_index.setdefault(w, set()).add(sid)

    # --- persistence (JSON side-file) ----------------------------------
    def save(self, memory_path):
        if not memory_path:
            return
        self._save_expansion(memory_path)
        # ATOMIC (review #2): tmp + os.replace — a half-written .evidence file
        # must not crash load() (and therefore Session.__init__).
        path = memory_path + ".evidence"
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump([[s, src, self.speakers.get(i)]
                       for i, (s, src) in enumerate(self.sentences)], f,
                      ensure_ascii=False)
        os.replace(tmp, path)

    @classmethod
    def load(cls, memory_path):
        store = cls()
        path = None
        if memory_path:
            candidate = memory_path + ".evidence"
            if os.path.exists(candidate):
                path = candidate
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    rows = json.load(f)
            except (json.JSONDecodeError, OSError):
                return store        # corrupt side-file → empty store (the graph
                #                     is intact; evidence can be re-ingested,
                #                     no fabrication risk)
            for row in rows:
                store.add(row[0], row[1],
                          speaker=(row[2] if len(row) > 2 else None))
            store._load_expansion(memory_path)
        return store

    # THE EXPANSION LIVES IN ITS OWN FILE, and that is not tidiness. `.evidence`
    # is the document, and anything inside it is answerable material; keeping
    # generated text out of that file is the same wall as keeping it out of
    # `self.sentences`, enforced one layer further down. Deleting the
    # `.expansion` file leaves a memory that answers exactly as it did before
    # the expansion was ever paid for.
    def _save_expansion(self, memory_path):
        if not (self.expansions or self._derived or self.head_bridge):
            return
        path = memory_path + ".expansion"
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"derived": sorted(self._derived),
                       "expansions": {str(sid): texts for sid, texts
                                      in sorted(self.expansions.items())},
                       "bridge": {w: sorted(heads) for w, heads
                                  in sorted(self.head_bridge.items())}},
                      f, ensure_ascii=False)
        os.replace(tmp, path)

    def _load_expansion(self, memory_path):
        path = memory_path + ".expansion"
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as f:
                blob = json.load(f)
        except (json.JSONDecodeError, OSError):
            return              # a corrupt aid is no aid: the document still
            #                     answers, lexically, exactly as it always did
        self._derived = set(blob.get("derived", ()))
        self.head_bridge = {w: set(heads) for w, heads
                            in blob.get("bridge", {}).items()}
        for key, texts in blob.get("expansions", {}).items():
            sid = int(key)
            if sid >= len(self.sentences):
                continue
            self.expansions.setdefault(sid, [])
            for text in texts:
                self._index_expansion(sid, text)
