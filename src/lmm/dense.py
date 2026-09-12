"""The meaning channel: vectors over the store's own documents.

WHICH LAYER, AND HOW WE FOUND OUT. The channel was first built over
LINES, which is where the industry's line-level hybrid puts it. Measured
over 103 documents with the distinctive terms dropped from each query —
known-item retrieval, no hand-written gold — that arrangement moved
nothing: the words were already at 90% and the vectors agreed with them.
Moved to DOCUMENTS, the same encoder and the same fusion reached 99%,
and the two together 103/103. The reason is visible in the data: a line
here reads "Aidiyet ve Motivasyon." — four words carry no topic, so
vectors over lines compare fragments, while vectors over a document's
profile compare subjects, which is what a reader who cannot name the
document is actually asking about. It is also 160x smaller: 103 vectors
instead of 16,637, a first pass in about a second instead of half a
minute, and a query in milliseconds.

So the channel does not rank lines. It proposes DOCUMENTS, and the words
retrieve inside them — the scope the store already understood.


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
  * IT EMBEDS A DOCUMENT BY WHAT IT SAYS AT LENGTH. A profile is the
    document's name followed by its longest lines — the sentences that
    carry a subject, rather than the headings and one-word rows that
    every document shares. Nothing is summarised and nothing is
    invented: the profile is the document's own text, cut.
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

# How many documents the channel offers a query. The fusion reads ranks,
# so this is a horizon, not a threshold: it bounds work, and nothing
# about it decides what is true. Five, because the measurement put the
# right document in the first three for 99 of 103 queries — two seats of
# margin, and still a narrow enough field for the words to search.
SOURCES = 5

# How much of a document is embedded: its longest lines, to this many
# lines and this many characters. Longest, because in a catalogue the
# short rows are the ones every document shares.
PROFILE_LINES = 12
PROFILE_CHARS = 1200

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
    """Vectors for a store's documents, and the nearest-neighbour reading."""

    def __init__(self, encode):
        self.encode = encode
        self.vectors = {}               # source -> unit vector
        self.size = {}                  # source -> its line count when embedded

    # ------------------------------------------------------------ build
    @staticmethod
    def _unit(vector):
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    @staticmethod
    def _profile_of(store, source):
        """A document as the vector sees it: its name, then its longest
        lines. The name because a catalogue's subject is usually written
        there; the longest lines because they are the ones that say
        something this document says and its neighbours do not."""
        from lmm import evidence                          # noqa: PLC0415
        lines = sorted((store.sentences[sid][0]
                        for sid in store.by_source.get(source, ())),
                       key=len, reverse=True)[:PROFILE_LINES]
        name = evidence._source_name(source) if source else ""
        body = " ".join(lines)[:PROFILE_CHARS]
        return ("%s. %s" % (name, body)).strip()

    def catch_up(self, store):
        """Embed the documents that are new or have grown since last pass.

        Incremental by document: ingesting one more file re-embeds one
        profile, not the corpus, and a store that has not changed does
        no work at all."""
        stale = [src for src, sids in store.by_source.items()
                 if src and self.size.get(src) != len(sids)]
        if not stale:
            return 0
        vectors = self.encode([self._profile_of(store, src) for src in stale])
        for src, vector in zip(stale, vectors):
            self.vectors[src] = self._unit(list(vector))
            self.size[src] = len(store.by_source.get(src, ()))
        return len(stale)

    # ------------------------------------------------------------- read
    def near(self, query, most=SOURCES):
        """The documents whose profiles sit closest to this question."""
        if not self.vectors:
            return []
        try:
            asked = self._unit(list(self.encode([query])[0]))
        except Exception:                                # noqa: BLE001
            return []
        scored = [(sum(a * b for a, b in zip(asked, vector)), src)
                  for src, vector in self.vectors.items()]
        scored.sort(key=lambda row: (-row[0], str(row[1])))
        return [src for _score, src in scored[:most]]
