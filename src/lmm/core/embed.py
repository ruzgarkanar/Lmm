"""The embedding builder: geometry for words — by counting, gradient-free, GPU-free.

The geometry layer (`lmm/core/geometry.py`) expects vectors and without this file
it was blind: the `Identity.vector` field existed, nothing filled it. When
unfilled, `resolve` falls back to "the most accessed" — that is, string
matching + popularity. The layer built to solve polysemy would never have
run, for lack of anything to fill it. The audit caught this in the
line-by-line comparison.

The method is counting, and its grounding is measured: word2vec was shown to
implicitly factorize a shifted PMI matrix (Levy & Goldberg 2014) — the
gradient was not where the meaning came from, only one way of arriving at it.
Here we go to the same geometry directly:

    1. COUNT       which word occurs with which word (the window)
    2. PPMI        the inflation of "everyone's friend" words is divided out
    3. PROJECTION  dimensionality is reduced with a fixed-seed random
                   projection — deterministic: the same corpus always gives
                   the same vectors

Nothing belonging to language exists: a word is what whitespace separates,
and which language it is in is never asked. The same builder, tomorrow, builds
English geometry from an English corpus.

THE BOUNDARY — geometry only FINDS. A vector never writes a record; opposites
occur in the same surroundings and distribution cannot separate them
(measured in this project). The decision is the gate's.

Usage:
    python3 -m lmm.core.embed data/raw/tr-metin.txt --out models/v3/vectors.json
"""
import json
import math
import os
import sys

# The window: the neighborhood that counts as a word's "surroundings". Beyond
# four it starts crossing the sentence boundary and counting unrelated words
# as surroundings.
WINDOW = 4

# The minimum times a word must be seen to be counted. What is seen once has
# no distribution; its vector would be noise.
LEAST = 5

DIMENSIONS = 128


def sentences(paths, most=None):
    """Corpus lines — split into words. Splitting is by whitespace only."""
    count = 0
    for path in paths:
        opener = _open(path)
        for line in opener:
            words = []
            for one in line.split():
                start, stop = 0, len(one)
                while start < stop and not one[start].isalnum():
                    start += 1
                while stop > start and not one[stop - 1].isalnum():
                    stop -= 1
                piece = one[start:stop]
                if piece:
                    words.append("".join(ch.casefold()[0] for ch in piece))
            if len(words) < 2:
                continue
            yield words
            count += 1
            if most and count >= most:
                return


def _open(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def counts(rows):
    """Word and pair counts. In a single pass, memory-friendly."""
    seen, pairs = {}, {}
    for words in rows:
        for at, word in enumerate(words):
            seen[word] = seen.get(word, 0) + 1
        for at, word in enumerate(words):
            low = max(0, at - WINDOW)
            high = min(len(words), at + WINDOW + 1)
            for other in range(low, high):
                if other == at:
                    continue
                pair = (word, words[other])
                pairs[pair] = pairs.get(pair, 0) + 1
    kept = {word for word, count in seen.items() if count >= LEAST}
    pairs = {pair: count for pair, count in pairs.items()
             if pair[0] in kept and pair[1] in kept and count > 1}
    return kept, seen, pairs


def vectors_of(kept, pairs, dimensions=DIMENSIONS, rounds=10):
    """The leading directions of the PPMI matrix — by subspace iteration.

    The first draft was a sparse random projection and measurement refuted
    it: the neighbors of the word "kuş" came out as "karıştırıp, saklamak,
    luhansk" — noise. There are ~48 contexts per word; at that sparsity the
    projection noise swallows the signal.

    Subspace iteration goes to the same geometry by a sure road: the M·V
    product is applied again and again and V converges to the matrix's
    strongest directions — the explicit form of the factorization word2vec
    does implicitly (Levy & Goldberg 2014). The start has a FIXED seed: the
    same corpus, always the same vectors.

    torch is only for speed (sparse multiply + QR); it is the training
    side's tool anyway. The runtime geometry (`lmm/core/geometry.py`) stays
    torch-free.
    """
    import torch

    words = sorted(kept)
    index = {word: at for at, word in enumerate(words)}
    total = sum(pairs.values()) or 1
    word_total = {}
    for (one, _), count in pairs.items():
        word_total[one] = word_total.get(one, 0) + count

    rows, cols, vals = [], [], []
    for (one, other), count in pairs.items():
        joint = count / total
        left = word_total.get(one, 1) / total
        right = word_total.get(other, 1) / total
        pmi = math.log(joint / (left * right))
        if pmi > 0:
            rows.append(index[one])
            cols.append(index[other])
            vals.append(pmi)
    matrix = torch.sparse_coo_tensor(
        torch.tensor([rows, cols]), torch.tensor(vals, dtype=torch.float32),
        (len(words), len(words))).coalesce()

    generator = torch.Generator().manual_seed(
        int.from_bytes(__import__("hashlib").md5(
            " ".join(words[:100]).encode()).digest()[:4], "big"))
    basis = torch.randn(len(words), dimensions, generator=generator)
    for _ in range(rounds):
        basis = torch.sparse.mm(matrix, basis)
        basis, _ = torch.linalg.qr(basis)
    projected = torch.sparse.mm(matrix, basis)
    projected = torch.nn.functional.normalize(projected, dim=1)

    return {word: [round(float(one), 5) for one in projected[at]]
            for word, at in index.items()}


def attach(memory, table):
    """Binds the vectors to identities: an identity's vector is the average of its labels'."""
    bound = 0
    for identity in memory.identities.values():
        vectors = [table[label] for label in identity.labels
                   if label in table]
        if not vectors:
            continue
        width = len(vectors[0])
        identity.vector = [sum(one[at] for one in vectors) / len(vectors)
                           for at in range(width)]
        bound += 1
    return bound


def main(argv):
    paths = [one for one in argv[1:] if not one.startswith("--")]
    out = (argv[argv.index("--out") + 1] if "--out" in argv
           else "models/v3/vectors.json")
    most = int(argv[argv.index("--most") + 1]) if "--most" in argv else 300000
    if not paths:
        print(__doc__.strip().splitlines()[-1])
        return 1
    kept, seen, pairs = counts(sentences(paths, most))
    print(f"  {len(kept):,} words · {len(pairs):,} pairs")
    held = vectors_of(kept, pairs)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(held, handle, ensure_ascii=False)
    print(f"  -> {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
