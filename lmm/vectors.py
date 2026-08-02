"""Meaning as geometry, counted rather than trained.

Every part of an LLM that needs a GPU needs it for the same reason: a gradient
is being pushed through a very large parameter set, many times. Embeddings look
like they belong to that world, and they were the one piece of the LLM this
project kept refusing on those grounds. That refusal was wrong.

Levy and Goldberg showed in 2014 that word2vec's skip-gram with negative
sampling is *implicitly factorising* a shifted pointwise-mutual-information
matrix. The gradient descent was never where the meaning came from — it was one
way of arriving at a factorisation that can also be reached by counting and
linear algebra, which is what LSA had been doing since 1990. So the same
geometry is available here: count which words keep company with which, weight
the counts by PPMI, and take the leading directions.

Two decisions keep it inside this project's constraints:

    no dependencies   the factorisation is subspace iteration, forty lines of
                      arithmetic over a sparse matrix held in dictionaries
    no randomness     the starting basis is a fixed deterministic pattern, so
                      the same corpus always produces the same vectors and a
                      regression is a real regression rather than a new seed

What this buys that the rest of the system cannot do: it needs no lexicon, no
patterns, and no knowledge of Turkish. It reads whichever language it is handed
and returns the words that behave alike — which is the only honest way to reach
the eighty languages this is supposed to serve.

What it must never be allowed to do is decide what is true. Distributional
similarity cannot separate opposites: "caused" and "prevented" keep identical
company, and this project measured its own version of that — over 1288 facts,
"sıcak" and "soğuk" co-occur. Geometry is for finding candidates. The epistemic
gate still decides.
"""
import math

WINDOW = 4              # how far to either side counts as company
MINIMUM_COUNT = 2       # a word seen once has no distribution to speak of
DIMENSIONS = 48
ITERATIONS = 12


class Vectors:
    """Word vectors from co-occurrence counts. No gradient, no seed, no GPU."""

    def __init__(self, dimensions=DIMENSIONS, window=WINDOW,
                 minimum=MINIMUM_COUNT, iterations=ITERATIONS):
        self.dimensions = dimensions
        self.window = window
        self.minimum = minimum
        self.iterations = iterations
        self.words = []             # index -> word
        self.index = {}             # word -> index
        self.vectors = {}           # word -> [float]
        self.counts = {}            # word -> how often it was seen

    # --- counting ---------------------------------------------------------

    def learn(self, sentences):
        """sentences is an iterable of token lists. Returns self."""
        pairs, totals = self._cooccurrence(sentences)
        matrix = self._ppmi(pairs, totals)
        self._factorise(matrix)
        return self

    def _cooccurrence(self, sentences):
        sentences = [list(s) for s in sentences]
        seen = {}
        for tokens in sentences:
            for token in tokens:
                seen[token] = seen.get(token, 0) + 1
        self.counts = {w: n for w, n in seen.items() if n >= self.minimum}
        self.words = sorted(self.counts)
        self.index = {word: i for i, word in enumerate(self.words)}

        pairs = {}
        totals = [0] * len(self.words)
        for tokens in sentences:
            kept = [self.index[t] for t in tokens if t in self.index]
            for position, word in enumerate(kept):
                low = max(0, position - self.window)
                high = min(len(kept), position + self.window + 1)
                for other in range(low, high):
                    if other == position:
                        continue
                    context = kept[other]
                    row = pairs.setdefault(word, {})
                    row[context] = row.get(context, 0) + 1
                    totals[word] += 1
        return pairs, totals

    def _ppmi(self, pairs, totals):
        """Positive pointwise mutual information — the weighting word2vec finds.

        Raw counts say "the" is everybody's friend. PMI divides that out by
        asking how much more often two words meet than chance would predict, and
        the positive part is kept because a reliable *absence* of company needs
        far more text to establish than a presence does.
        """
        grand = sum(totals) or 1
        matrix = {}
        for word, row in pairs.items():
            weighted = {}
            for context, count in row.items():
                joint = count / grand
                expected = (totals[word] / grand) * (totals[context] / grand)
                if joint <= 0 or expected <= 0:
                    continue
                value = math.log(joint / expected)
                if value > 0:
                    weighted[context] = value
            if weighted:
                matrix[word] = weighted
        return matrix

    # --- factorising ------------------------------------------------------

    def _factorise(self, matrix):
        """Leading left singular directions, by subspace iteration.

        Repeatedly applying M·Mᵀ to a basis pulls it towards the directions the
        data actually varies along; re-orthonormalising after each pass stops
        every column collapsing onto the strongest one. This is the same answer
        an SVD library gives, at the size this system works in.
        """
        size = len(self.words)
        if not size:
            return
        width = min(self.dimensions, size)
        basis = [[math.sin((i + 1) * (k + 1) * 0.7) for i in range(size)]
                 for k in range(width)]
        basis = _orthonormalise(basis)
        for _ in range(self.iterations):
            moved = [_apply(matrix, vector, size) for vector in basis]
            moved = _orthonormalise(moved)
            if not moved:
                break
            basis = moved
        self.vectors = {word: [basis[k][i] for k in range(len(basis))]
                        for i, word in enumerate(self.words)}

    # --- using ------------------------------------------------------------

    def vector(self, word):
        return self.vectors.get(word)

    def similar(self, word, count=8):
        """The words that keep the same company, nearest first."""
        target = self.vectors.get(word)
        if target is None:
            return []
        scored = [(cosine(target, other), candidate)
                  for candidate, other in self.vectors.items()
                  if candidate != word]
        scored.sort(reverse=True)
        return [(candidate, score) for score, candidate in scored[:count]]

    def nearest(self, word, among, count=1):
        """The closest of a given set — for placing a stranger among known words."""
        target = self.vectors.get(word)
        if target is None:
            return []
        scored = []
        for candidate in among:
            other = self.vectors.get(candidate)
            if other is not None and candidate != word:
                scored.append((cosine(target, other), candidate))
        scored.sort(reverse=True)
        return [(candidate, score) for score, candidate in scored[:count]]

    def to_dict(self):
        return {"dimensions": self.dimensions,
                "vectors": {w: [round(v, 5) for v in vec]
                            for w, vec in self.vectors.items()}}

    @staticmethod
    def from_dict(data):
        found = Vectors(dimensions=data.get("dimensions", DIMENSIONS))
        found.vectors = {w: list(v) for w, v in data.get("vectors", {}).items()}
        found.words = sorted(found.vectors)
        found.index = {w: i for i, w in enumerate(found.words)}
        return found


def cosine(left, right):
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    size = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / size if size else 0.0


def _apply(matrix, vector, size):
    """M·Mᵀ·v, without ever building M·Mᵀ — it would be the square of the vocabulary."""
    middle = {}
    for word, row in matrix.items():
        weight = vector[word]
        if weight == 0.0:
            continue
        for context, value in row.items():
            middle[context] = middle.get(context, 0.0) + weight * value
    result = [0.0] * size
    for word, row in matrix.items():
        total = 0.0
        for context, value in row.items():
            other = middle.get(context)
            if other:
                total += value * other
        result[word] = total
    return result


def _orthonormalise(vectors):
    """Gram-Schmidt. Directions that collapse into the others are dropped."""
    basis = []
    for vector in vectors:
        current = list(vector)
        for kept in basis:
            overlap = sum(a * b for a, b in zip(current, kept))
            current = [a - overlap * b for a, b in zip(current, kept)]
        length = math.sqrt(sum(a * a for a in current))
        if length > 1e-9:
            basis.append([a / length for a in current])
    return basis
