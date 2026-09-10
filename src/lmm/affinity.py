"""The corpus as its own thesaurus: words that keep the same company.

THE PROBLEM THIS ANSWERS. Retrieval here finds a line because a WORD
arrived, which is what makes every retrieval decision explainable — and
what makes a paraphrase invisible. A document says a tenant RESIDES at an
address, a reader asks where she LIVES, and the sentence sits in the
store unreachable. The systems that do not have this problem embed their
text in a vector space where meaning is geometry; they pay for it by not
being able to say why anything was chosen, and by depending on somebody
else's model to say it.

THE DOCUMENT ALREADY KNOWS. Two words used for the same thing turn up
beside the same other words: "resides" and "lives" both appear near
"tenant", "flat", "since", "address". That is distributional similarity —
the oldest idea in corpus linguistics, and one that needs no model, no
vendor and no training: it is arithmetic over counts this store already
holds.

WHAT MAKES IT OURS RATHER THAN A REIMPLEMENTATION.

  * It is EXPLAINABLE. A neighbour comes back with the shared company
    that earned it — "these four words are the context both keep" — so a
    widened search can show its reason the way every other reading here
    does. An embedding cannot do this at any price.
  * Its vocabulary is the STORE'S. Every word it offers was written in
    this corpus, so the rule the field bridge and the second ask keep — a
    word nobody wrote cannot enter a search — holds by construction
    rather than by a filter.
  * It is RETRIEVAL ONLY. Nothing here widens what may be SAID; the
    gates read the same evidence and judge it the same way.
  * It costs nothing per question. One pass at build time, cached with
    the store; a query is a few dictionary lookups.

THE MATHEMATICS, and why each piece is there.

  * CONTEXT: the other words of the same line. A line in this store is
    already a window of neighbouring sentences, so "same line" is "same
    neighbourhood" without a second windowing rule.
  * WEIGHT: positive pointwise mutual information, log p(w,c)/p(w)p(c),
    floored at zero. Raw counts would make every word's profile a list of
    the corpus's most common words; PPMI asks instead which company is
    SURPRISING, which is what "characteristic context" means. The floor
    at zero is standard and principled: a negative association is
    evidence of nothing much in a finite corpus.
  * SIMILARITY: cosine between two profiles. Two words are near when
    their surprising company overlaps, in proportion to how surprising it
    is.
  * THE SEARCH IS NOT V x V. Similarities are computed on demand for one
    query word, against only the words that share at least two of its
    contexts (an inverted context index). A vocabulary of ten thousand
    never becomes a hundred million comparisons.

WHAT IT CANNOT DO: relate two words the document never uses in comparable
company. A word that appears twice has no profile worth the name, and
this returns nothing rather than the nearest thing in a geometry — which
is why the engine's second ask remains the fallback behind it.
"""
import math

from lmm import evidence

# A word seen twice has no company worth measuring; the floor is what
# keeps the profile table proportional to the document's vocabulary
# rather than to its noise.
MIN_COUNT = 3

# How much of a word's company to keep. A profile is a signature, not a
# transcript: the most surprising contexts carry it, and a long tail of
# weak associations only makes two words look alike because everything
# looks alike.
PROFILE = 40

# Words per line that count as company. A line is already a window of
# neighbouring sentences; a window the length of a page is a page.
MAX_TOKENS = 60


class Profiles:
    """word -> {context: ppmi}, and the inverted index that searches it."""

    def __init__(self):
        self.profile = {}       # word -> {context: weight}
        self.holders = {}       # context -> {word, ...}
        self.count = {}         # word -> occurrences
        self.together = {}      # (word, other) -> lines holding both

    def __len__(self):
        return len(self.profile)

    @classmethod
    def build(cls, store, min_count=MIN_COUNT, keep=PROFILE):
        """Read a sentence store into context profiles. No model call."""
        self = cls()
        pairs, total_pairs = {}, 0
        counts = {}
        for text, _origin in store.sentences:
            words = evidence._words(text or "")[:MAX_TOKENS]
            unique = sorted(set(words))
            for word in unique:
                counts[word] = counts.get(word, 0) + 1
            for i, word in enumerate(unique):
                for other in unique[i + 1:]:
                    pairs[(word, other)] = pairs.get((word, other), 0) + 1
                    total_pairs += 1
        if not total_pairs:
            return self
        self.count = counts
        self.together = pairs
        seen = sum(counts.values()) or 1
        raw = {}
        for (word, other), together in pairs.items():
            if (together < min_count
                    or counts.get(word, 0) < min_count
                    or counts.get(other, 0) < min_count):
                continue
            # PPMI: how surprising is this company?
            joint = together / total_pairs
            expected = (counts[word] / seen) * (counts[other] / seen)
            if not expected:
                continue
            weight = math.log(joint / expected)
            if weight <= 0:
                continue
            raw.setdefault(word, {})[other] = weight
            raw.setdefault(other, {})[word] = weight
        for word, company in raw.items():
            best = sorted(company.items(), key=lambda kv: -kv[1])[:keep]
            self.profile[word] = dict(best)
            for context, _weight in best:
                self.holders.setdefault(context, set()).add(word)
        return self

    def near(self, word, most=5, floor=0.0):
        """The words that keep the same company, with the company shown.

        Returns [(word, similarity, [shared contexts])], best first. The
        shared contexts are the point: a widened search can say WHY it
        widened, which is the whole difference between this and a vector.
        """
        word = evidence.fold(word or "")
        mine = self.profile.get(word)
        if not mine:
            return []
        # only words sharing at least two of my contexts are candidates:
        # one shared context is a coincidence in any corpus
        seen = {}
        for context in mine:
            for other in self.holders.get(context, ()):
                if other != word:
                    seen[other] = seen.get(other, 0) + 1
        candidates = [w for w, shared in seen.items() if shared >= 2]
        my_norm = math.sqrt(sum(v * v for v in mine.values())) or 1.0
        out = []
        for other in candidates:
            theirs = self.profile.get(other) or {}
            common = [c for c in theirs if c in mine]
            if len(common) < 2:
                continue
            # A SUBSTITUTE KEEPS THE SAME COMPANY WITHOUT BEING IN IT.
            # Two words that constantly appear in the same line are
            # companions, not alternatives — measured, asking what is
            # near "lives" returned "nearby", "visitor" and "him", the
            # words it is always written beside, while "resides" (which
            # never shares a line with it, because a writer picks one)
            # ranked below them. Words that appear together in more than
            # half the lines of either are reading as one phrase, and
            # this reading is not about phrases.
            shared_lines = self.together.get(tuple(sorted((word, other))), 0)
            if shared_lines > min(self.count.get(word, 0),
                                  self.count.get(other, 0)) / 2:
                continue
            dot = sum(mine[c] * theirs[c] for c in common)
            norm = math.sqrt(sum(v * v for v in theirs.values())) or 1.0
            score = dot / (my_norm * norm)
            if score > floor:
                shown = sorted(common, key=lambda c: -mine[c])[:4]
                out.append((other, score, shown))
        out.sort(key=lambda row: -row[1])
        return out[:most]

    def nearest_words(self, text, most=4):
        """Store words that keep the company of this question's words.

        The words are the store's own by construction — nothing here can
        propose a word the document never wrote — and they widen the
        SEARCH alone.
        """
        asked = evidence._words(text or "")
        out, seen = [], set(asked)
        for word in asked:
            for other, _score, _shared in self.near(word, most=2):
                if other not in seen:
                    seen.add(other)
                    out.append(other)
        return out[:most]
