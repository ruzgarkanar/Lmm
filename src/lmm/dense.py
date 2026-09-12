"""The meaning channel: vectors over the store's own lines.

WHY THIS EXISTS, AND WHY IT TOOK SO LONG TO ADD. Retrieval here finds a
line because a WORD arrived, which is what makes every decision
explainable — and what makes a paraphrase invisible. A lease writes
RESIDES and a reader asks where somebody LIVES; a catalogue offers
CONFLICT MANAGEMENT and a team lead says there is GOSSIP. Three
instruments were built against that gap and measured: the field bridge
(which closes it per field, once), the offline expansion (which the
README records as not working at scale) and the engine's second ask
(one call per failing turn, filtered by the store). None of them
closes the class. A vector channel closes it by construction, and the
industry's own answer — hybrid lexical+dense with rank fusion — has
been the strong baseline since 2023.

WHAT KEEPS IT OURS.

  * IT IS A CHANNEL, NOT A REPLACEMENT. The two rankings are fused by
    RRF (reciprocal rank fusion), which reads RANKS, not scores: there
    is no weight to tune, no threshold to drift, and a line the words
    already found cannot be pushed out by a vector's opinion.
  * IT EMBEDS THE LINE WITH ITS CONTEXT. Each vector is computed over
    "source name · record head · line", not the bare line — contextual
    retrieval without a second model, and the reason a row reading
    "2 Tam Gün" is findable at all.
  * IT WIDENS WHAT CAN BE FOUND, NEVER WHAT MAY BE SAID. The gates read
    the evidence exactly as before; a line that arrives by vector is
    judged by the same jury as a line that arrives by word.
  * IT IS OPTIONAL BY CONSTRUCTION. With no encoder the channel is
    absent — not degraded, absent — and the store answers exactly as it
    did before this file existed.

THE ENCODER IS THE OPERATOR'S. `Memory(encoder=...)` takes any callable
mapping a list of strings to a list of vectors: a sentence-transformers
model, a static distilled embedding, an in-house service, or a vendor's
API for those who want one. The library ships no opinion about whose
vectors are best; it ships the channel.
"""
import math

# How many neighbours the channel offers a query. The fusion reads
# ranks, so this is a horizon, not a threshold: it bounds work, and
# nothing about it decides what is true.
NEIGHBOURS = 30

# The rank-fusion constant from the RRF paper (Cormack et al., 2009),
# used at its published value. It flattens the difference between rank 1
# and rank 2 just enough that neither channel can dominate on its own.
RRF_K = 60


def fuse(*rankings, most=None):
    """Reciprocal rank fusion of several ranked id lists.

    score(id) = Σ 1 / (K + rank), rank counted from 1 in each ranking
    it appears in. No weights: a channel votes by ORDER, which is the
    only thing two different scorings can honestly share.
    """
    score = {}
    for ranking in rankings:
        for position, key in enumerate(ranking, 1):
            score[key] = score.get(key, 0.0) + 1.0 / (RRF_K + position)
    order = sorted(score, key=lambda k: (-score[k], k))
    return order[:most] if most else order


class Dense:
    """Vectors for a sentence store, and the nearest-neighbour reading."""

    def __init__(self, encode):
        self.encode = encode
        self.vectors = []               # sid -> unit vector (list of floats)
        self.at = 0                     # how many sentences are embedded

    # ------------------------------------------------------------ build
    @staticmethod
    def _unit(vector):
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def _context_of(self, store, sid):
        """The line, under the name of the document that wrote it and the
        head it sits below — what the vector is actually computed over."""
        text, source = store.sentences[sid]
        from lmm import evidence                          # noqa: PLC0415
        name = evidence._source_name(source) if source else ""
        head = ""
        if ":" in text:
            maybe = text.split(":", 1)[0]
            if len(maybe) <= 60:
                head = maybe
        parts = [p for p in (name, head, text) if p]
        return " · ".join(parts)

    def catch_up(self, store):
        """Embed whatever the store has learned since the last pass.

        Incremental by construction: ingestion adds lines, and the
        channel embeds the new ones only, in one batch."""
        total = len(store.sentences)
        if total <= self.at:
            return 0
        fresh = [self._context_of(store, sid) for sid in range(self.at, total)]
        vectors = self.encode(fresh)
        for vector in vectors:
            self.vectors.append(self._unit(list(vector)))
        self.at = total
        return total - self.at + len(fresh)

    # ------------------------------------------------------------- read
    def near(self, query, most=NEIGHBOURS):
        """The sentence ids whose vectors sit closest to this query."""
        if not self.vectors:
            return []
        try:
            asked = self._unit(list(self.encode([query])[0]))
        except Exception:                                # noqa: BLE001
            return []
        scored = []
        for sid, vector in enumerate(self.vectors):
            scored.append((sum(a * b for a, b in zip(asked, vector)), sid))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [sid for _score, sid in scored[:most]]
