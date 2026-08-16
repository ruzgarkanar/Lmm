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


def _boundaries(text):
    """FORMAT-boundary splitter (no language rule): PDF flattening glues
    adjacent cells into one token ("IPX 7Ayak", "GüçAdaptörüPil", "8 g DDR
    4sabit disk"). Two purely typographic transitions become token
    boundaries: lowercase→UPPERCASE (camel seam between glued cells) and
    digit↔letter (a number fused to the next cell's word). Without this the
    digit gate cannot see a legitimate "7" hiding inside "7Ayak" and drops a
    CORRECT answer (measured: the IPX7 loss)."""
    out = []
    prev = ""
    for ch in text:
        if prev and ((prev.islower() and ch.isupper())
                     or (prev.isdigit() and ch.isalpha())
                     or (prev.isalpha() and ch.isdigit())):
            out.append(" ")
        out.append(ch)
        prev = ch
    return "".join(out)


def _words(text):
    """Content words: folded, combining-mark-free, >=3 letters OR a digit.
    The digit exception is critical: if the 3 of "Tier 3" is dropped, the
    coverage gate cannot see a number swap (a fabricated "Tier 1")."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [fold(w) for w in _WORD.findall(_boundaries(text))
            if len(w) >= 3 or w.isdigit()]


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
    if not digits_ok(answer, block):
        return False
    given = set(_words(block + " " + question))
    words = set(_words(answer))
    if not words:
        return False
    for w in words:
        if w in given or w.isdigit():
            continue
        if not any((len(g) >= 5 and w.startswith(g) and len(w) - len(g) <= 3)
                   or (len(w) >= 5 and g.startswith(w) and len(g) - len(w) <= 3)
                   for g in given):
            return False
    return True


class SentenceStore:
    """Sentence store + inverted index. Small and pure: list + dict."""

    def __init__(self):
        self.sentences = []                 # id -> (sentence, source)
        self.index = {}                     # folded word -> set(id)
        self.key_index = {}                 # folded FIELD-NAME word -> set(id)
        self._seen = set()                  # folded sentence — duplicate-evidence guard
        self.last_sources = []              # sources of the last find() (hedge)

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
        for w in set(_words(sentence)):
            self.index.setdefault(w, set()).add(sid)
        for w in set(self._key_words(sentence)):
            self.key_index.setdefault(w, set()).add(sid)
        return sid

    def find(self, query, most=4):
        """Sentences whose content-words intersect the query the MOST.
        Prefix-tolerant (inflection: "doluluğu"~"doluluk"). Score = number of
        intersecting words; on a tie the short sentence wins (denser evidence)."""
        qwords = set(_words(query))
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
                        common = os.path.commonprefix((qw, w))
                        if len(common) >= max(4, min(len(qw), len(w)) - 2):
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
        # (Jaccard on content-words), no language rule; the best-scoring
        # variant of each region survives.
        keep = []
        kept_words = []
        for sid, sc in ranked:
            if sc < floor and keep:
                break
            words = set(_words(self.sentences[sid][0]))
            dup = False
            for kw in kept_words:
                inter = len(words & kw)
                # Jaccard on content-words. NOT containment: two spec windows
                # can differ by 3 tokens where those 3 tokens ARE the answer
                # ("14.4 V / 6500 mAh" tail) — containment suppressed the
                # only window carrying them (measured).
                if inter and inter / max(1, len(words | kw)) >= 0.75:
                    dup = True
                    break
            if dup:
                continue
            keep.append(sid)
            kept_words.append(words)
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
