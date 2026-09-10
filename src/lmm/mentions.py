"""The document's own entity graph: collocations, co-mentions, witnesses.

WHAT THIS IS FOR. Everything the record channel does — fields, census,
extremes, comparison — needs a document with HEADS. Prose has none, and
on prose this memory is left with one lexical search and the gates: it
answers what a single sentence states and abstains on everything that
lives across sentences ("who is X to Y", "what happened after Z"). That
is the measured gap, and it is the gap a graph is supposed to close.

WHAT GRAPHRAG DOES AND WHY THIS DOES THE OPPOSITE. GraphRAG hands each
chunk to a model and asks for the entities and their relations; Mem0 asks
a model what is worth remembering. What comes back is a CLAIM, invented
at index time, unverifiable afterwards, and expensive enough to price the
index out of most documents — a thousand pages is thousands of calls
before the first question is asked.

This graph is built BY THE DOCUMENT, with no model call at all:

  * AN ENTITY IS A COLLOCATION. A phrase is an entity when its words
    occur together far more often than chance would put them there —
    pointwise mutual information, log p(ab) / p(a)p(b), the same
    statistic that finds "New York" without knowing what a city is. No
    capital letters (a rule that dies outside the Latin alphabet), no
    name lists, no language.

  * AN EDGE IS A WITNESSED CO-MENTION. Two entities are linked when a
    line mentions both, and the edge carries THAT LINE. Its weight is
    the log-likelihood ratio of the pairing against chance, so an entity
    that appears everywhere does not become everyone's neighbour. An
    edge cannot be a fabrication: it is an observation, and it shows the
    sentence that made it.

  * AN EDGE IS NOT A RELATION. The graph says two things are mentioned
    together and stays silent about what one IS to the other. Naming the
    relation is a claim, and claims go where every claim in this
    codebase goes: to the gates, with the witnessing sentence in hand.

  * IT IS ALSO THE PARTITION. entity -> lines is an index, so a question
    naming two entities reads the intersection of two short lists
    instead of scoring a store of a hundred thousand lines. On a large
    document that is the difference between a scan and a lookup.

COST. One pass to count tokens and adjacent pairs, one pass to place
entities in lines. Pairs seen fewer than MIN_PAIR times are dropped
before anything is scored, which is what keeps the counting table small
on a corpus of any size — a pair seen twice is not a collocation.
"""
import math
import re

from lmm import evidence

# A pair that a document uses twice is a coincidence; the floor is what
# keeps the counting table linear in the document's SIZE rather than in
# its vocabulary squared.
MIN_PAIR = 3

# How much more often than chance a pair must appear before the phrase is
# taken seriously. This is a log-ratio, not a tuned score: 0 means "no
# more often than chance", and everything kept is above it by a whole
# nat — the same "more than half" spirit as the rest of the codebase,
# expressed in the unit the statistic is measured in.
MIN_PMI = 1.0

# Words per line, capped: a window is context, and a window the length of
# a page is a page.
MAX_TOKENS = 400


def _tokens(text):
    return evidence._words(text or "")[:MAX_TOKENS]


class Graph:
    """Entities, the lines that mention them, and the edges between them."""

    def __init__(self):
        self.phrases = []            # [(word, word, ...)] — entity phrases
        self.lines = {}              # entity -> {sid, ...}
        self.edges = {}              # (a, b) -> {sid, ...}  (a < b)
        self.weight = {}             # (a, b) -> log-likelihood ratio
        self._store = None

    # ------------------------------------------------------------- build
    @classmethod
    def build(cls, store, min_pair=MIN_PAIR, min_pmi=MIN_PMI):
        """Read a sentence store into an entity graph. No model call."""
        graph = cls()
        graph._store = store
        # ONE PASS, TWO COUNTS: the words, and the n-grams a document
        # actually repeats. Everything below is arithmetic on these.
        unigram, grams, total = {}, {}, 0
        around = {}          # gram -> ([word before], [word after])
        lines_of = []
        for text, _origin in store.sentences:
            words = _tokens(text)
            lines_of.append(words)
            total += len(words)
            for word in words:
                unigram[word] = unigram.get(word, 0) + 1
            for size in range(2, 6):
                for position in range(len(words) - size + 1):
                    gram = tuple(words[position:position + size])
                    grams[gram] = grams.get(gram, 0) + 1
                    before = words[position - 1] if position else ""
                    after = (words[position + size]
                             if position + size < len(words) else "")
                    left, right = around.setdefault(gram, ({}, {}))
                    left[before] = left.get(before, 0) + 1
                    right[after] = right.get(after, 0) + 1
        if not total:
            return graph
        grams = {g: c for g, c in grams.items() if c >= min_pair}

        # AN ENTITY IS COHESIVE INSIDE AND FREE OUTSIDE.
        #
        # Inside: the words hold together more than chance would hold
        # them — pointwise mutual information over the phrase's weakest
        # split, so a phrase is only as strong as its loosest joint.
        # This is what finds "New York" without knowing what a city is.
        #
        # Outside: a phrase that is an entity turns up in DIFFERENT
        # company, while a fragment of a repeated sentence is always
        # followed by the same word. Measured on a document that repeats
        # its sentences, cohesion alone produced "and the morning was" —
        # a perfectly cohesive piece of a sentence and no entity at all.
        # The rule is the codebase's own boundary: no single neighbour
        # may account for MORE THAN HALF of a phrase's appearances, on
        # either side.
        def _cohesion(gram):
            best = None
            for cut in range(1, len(gram)):
                left, right = gram[:cut], gram[cut:]
                pl = _count(left, unigram, grams) / total
                pr = _count(right, unigram, grams) / total
                if not pl or not pr:
                    return None
                pmi = math.log((grams[gram] / total) / (pl * pr))
                best = pmi if best is None else min(best, pmi)
            return best

        def _free(gram):
            # A LINE BOUNDARY IS NOT A NEIGHBOUR. Counting the edge of a
            # line as a word made every phrase that opens a line look
            # perfectly predictable — and in a store of short lines that
            # is most names, most headings and every record head. The
            # test is about the company a phrase keeps; where it keeps
            # none, it is under no constraint.
            left, right = around.get(gram, ({}, {}))
            count = grams[gram]
            for side in (left, right):
                words = {w: n for w, n in side.items() if w}
                if words and max(words.values()) > count / 2:
                    return False
            return True

        # ...AND A NAME IS STICKY. Cohesion and freedom together still
        # let grammar through: measured on a novel, "that had", "said
        # she" and "each other" all beat chance and all appear in varied
        # company, because that is what grammar does. What separates a
        # name is that its parts do not choose: seeing "lucas", the word
        # before it is almost always "lady", while the word after "that"
        # can be any verb in the language. So one side of the join must
        # be nearly determined — MORE THAN HALF, the same boundary this
        # codebase uses wherever a mixture has to be called one thing or
        # the other.
        def _sticky(gram):
            for cut in range(1, len(gram)):
                left, right = gram[:cut], gram[cut:]
                here = grams[gram]
                nl = _count(left, unigram, grams)
                nr = _count(right, unigram, grams)
                if not nl or not nr:
                    return False
                if max(here / nl, here / nr) <= 0.5:
                    return False
            return True

        keep = {}
        for gram, count in grams.items():
            cohesion = _cohesion(gram)
            if (cohesion is not None and cohesion >= min_pmi
                    and _free(gram) and _sticky(gram)):
                keep[gram] = cohesion
        # a phrase inside a longer kept phrase is that phrase's opening
        graph.phrases = sorted(
            g for g in keep
            if not any(len(h) > len(g) and _inside(g, h) for h in keep))
        graph._place(store)
        return graph

    def _place(self, store):
        """Which lines mention which entity, and which entities meet."""
        index = {}
        for phrase in self.phrases:
            index.setdefault(phrase[0], []).append(phrase)
        for sid, (text, _origin) in enumerate(store.sentences):
            words = _tokens(text)
            here = set()
            for position, word in enumerate(words):
                for phrase in index.get(word, ()):
                    if tuple(words[position:position + len(phrase)]) == phrase:
                        here.add(" ".join(phrase))
            for name in here:
                self.lines.setdefault(name, set()).add(sid)
            names = sorted(here)
            for i, first in enumerate(names):
                for second in names[i + 1:]:
                    self.edges.setdefault((first, second), set()).add(sid)
        self._weigh(len(store.sentences))

    def _weigh(self, total):
        """G², the log-likelihood ratio of a pairing against chance.

        An entity a document mentions on every page co-occurs with
        everything; raw co-mention counts would make it everyone's
        closest neighbour. The ratio asks instead how SURPRISING the
        pairing is, which is what "related" has to mean when the only
        evidence is co-occurrence.
        """
        for (first, second), lines in self.edges.items():
            k11 = len(lines)
            k12 = len(self.lines.get(first, ())) - k11
            k21 = len(self.lines.get(second, ())) - k11
            k22 = max(0, total - k11 - k12 - k21)
            self.weight[(first, second)] = _g2(k11, k12, k21, k22)

    # -------------------------------------------------------------- read
    def linked(self, entity, most=12):
        """The entities this one is mentioned with, most surprising first."""
        name = _fold(entity)
        out = []
        for (first, second), score in self.weight.items():
            if first == name:
                out.append((score, second))
            elif second == name:
                out.append((score, first))
        out.sort(reverse=True)
        return [name for _score, name in out[:most]]

    def witnesses(self, first, second, most=4):
        """The lines that mention both — the edge's evidence, verbatim."""
        pair = tuple(sorted((_fold(first), _fold(second))))
        sids = sorted(self.edges.get(pair, ()))[:most]
        if self._store is None:
            return []
        return [self._store.sentences[sid][0] for sid in sids]

    def mentioning(self, entities):
        """Line ids mentioning ALL of these entities — the partition.

        A question naming two entities reads the intersection of two
        short lists rather than scoring the whole store: on a document of
        a hundred thousand lines that is a lookup where there was a scan.
        """
        sets = [self.lines.get(_fold(e), set()) for e in entities]
        if not sets or any(not s for s in sets):
            return set()
        got = set(sets[0])
        for other in sets[1:]:
            got &= other
        return got

    # ------------------------------------------------------- the walk
    # The literature's standard damping since PageRank; not a knob tuned
    # here. Everything else the walk reads comes off the graph's own
    # weights.
    DAMPING = 0.85

    def spread(self, seeds, rounds=20):
        """Personalized PageRank from these entities — the walk.

        What one posting-list intersection cannot answer: how two
        entities relate when NO line carries both. Colour is poured on
        the seed entities and flows along co-mention edges, each edge
        carrying the weight the document earned it (G²); where it
        settles is what the question is connected to THROUGH the graph,
        two hops included. HippoRAG runs this walk over an LLM-extracted
        graph, LinearRAG's family over statistical ones; this graph
        differs in what it can show — every edge carries the sentence
        that witnessed it — and in what the result may do: nominate
        lines for the gates, never speak.

        Pure arithmetic: no model call, no vendor, deterministic.
        Returns {entity: score}, seeds included.
        """
        seeds = [_fold(e) for e in seeds]
        seeds = [e for e in seeds if e in self.lines]
        if not seeds:
            return {}
        neighbours = {}
        for (a, b), weight in self.weight.items():
            if weight <= 0:
                continue
            neighbours.setdefault(a, {})[b] = weight
            neighbours.setdefault(b, {})[a] = weight
        rest = {e: 1.0 / len(seeds) for e in seeds}
        score = dict(rest)
        for _ in range(rounds):
            fresh = {e: (1 - self.DAMPING) * rest.get(e, 0.0)
                     for e in score}
            for entity, colour in score.items():
                near = neighbours.get(entity)
                if not near:
                    # a dangling entity returns its colour to the seeds
                    for seed, share in rest.items():
                        fresh[seed] = fresh.get(seed, 0.0) \
                            + self.DAMPING * colour * share
                    continue
                total = sum(near.values())
                for other, weight in near.items():
                    fresh[other] = fresh.get(other, 0.0) \
                        + self.DAMPING * colour * weight / total
            score = fresh
        return score

    def connect(self, first, second, most=4):
        """The entities that BRIDGE these two, best first.

        Two walks, one from each end; an entity that holds colour from
        BOTH is on a path between them, and the product of its two
        scores ranks how strongly. Each bridge returned as
        (entity, score) — and `witnesses(first, bridge)` /
        `witnesses(bridge, second)` hand back the sentences that make
        the path real, hop by hop. A bridge with no witnessed edge to
        either end cannot appear, by construction.
        """
        a, b = _fold(first), _fold(second)
        from_a = self.spread([a])
        from_b = self.spread([b])
        out = []
        for entity, score_a in from_a.items():
            if entity in (a, b):
                continue
            score_b = from_b.get(entity, 0.0)
            if score_a > 0 and score_b > 0 \
                    and (tuple(sorted((a, entity))) in self.edges
                         or tuple(sorted((entity, a))) in self.edges) \
                    and (tuple(sorted((b, entity))) in self.edges
                         or tuple(sorted((entity, b))) in self.edges):
                out.append((entity, score_a * score_b))
        out.sort(key=lambda row: -row[1])
        return out[:most]

    def named_in(self, text):
        """The entities this text names, longest first."""
        words = _tokens(text)
        found = []
        for phrase in self.phrases:
            length = len(phrase)
            for position in range(len(words) - length + 1):
                if tuple(words[position:position + length]) == phrase:
                    found.append(" ".join(phrase))
                    break
        found.sort(key=lambda n: -len(n.split()))
        return [n for i, n in enumerate(found)
                if not any(n in longer and n != longer
                           for longer in found[:i])]


def _count(gram, unigram, grams):
    """How often this run of words occurs — one word or several."""
    if len(gram) == 1:
        return unigram.get(gram[0], 0)
    return grams.get(gram, 0)


def _inside(short, long_):
    return any(long_[i:i + len(short)] == short
               for i in range(len(long_) - len(short) + 1))


def _fold(name):
    return " ".join(evidence._words(name or ""))


def _g2(k11, k12, k21, k22):
    """Dunning's log-likelihood ratio for a 2x2 contingency table."""
    total = k11 + k12 + k21 + k22
    if not total or not k11:
        return 0.0
    row1, row2 = k11 + k12, k21 + k22
    col1, col2 = k11 + k21, k12 + k22
    out = 0.0
    for observed, expected in ((k11, row1 * col1 / total),
                               (k12, row1 * col2 / total),
                               (k21, row2 * col1 / total),
                               (k22, row2 * col2 / total)):
        if observed and expected:
            out += observed * math.log(observed / expected)
    return 2 * out
