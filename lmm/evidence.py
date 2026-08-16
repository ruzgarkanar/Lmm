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


def _words(text):
    """Content words: folded, combining-mark-free, >=3 letters OR a digit.
    The digit exception is critical: if the 3 of "Tier 3" is dropped, the
    coverage gate cannot see a number swap (a fabricated "Tier 1")."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [fold(w) for w in _WORD.findall(text)
            if len(w) >= 3 or w.isdigit()]


def _tokens(text):
    """Ordered token sequence (folded, short ones included) — for adjacency checks."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [fold(w) for w in _WORD.findall(text)]


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
        self._seen = set()                  # folded sentence — duplicate-evidence guard
        self.last_sources = []              # sources of the last find() (hedge)

    def add(self, sentence, source=""):
        key = fold(sentence)
        if key in self._seen:               # if the same document is read twice,
            return None                     # evidence must not multiply (review #2)
        self._seen.add(key)
        sid = len(self.sentences)
        self.sentences.append((sentence, source))
        for w in set(_words(sentence)):
            self.index.setdefault(w, set()).add(sid)
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
        scores = {}
        for qw in qwords:
            hits = set()
            if qw in self.index:
                hits |= self.index[qw]
            else:
                # prefix tolerance — two-way, root>=4 (cheap while the index is
                # small). STEM matching too: "hazırlandı"~"hazırlanma" are not
                # each other's prefix but share an 8/10 stem — in an
                # agglutinative language, the same concept (the criterion is
                # proportional, no language-list).
                for w, ids in self.index.items():
                    if len(qw) >= 4 and w.startswith(qw):
                        hits |= ids
                    elif len(w) >= 4 and qw.startswith(w):
                        hits |= ids
                    else:
                        common = os.path.commonprefix((qw, w))
                        if len(common) >= max(4, min(len(qw), len(w)) - 2):
                            hits |= ids
            if not hits:
                continue
            weight = math.log(1 + total / len(hits))
            for sid in hits:
                scores[sid] = scores.get(sid, 0) + weight
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
        keep = [sid for sid, sc in ranked if sc >= floor][:most]
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
