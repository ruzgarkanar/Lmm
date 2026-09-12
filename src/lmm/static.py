"""The encoder that ships with the library — a matrix, not a model.

WHY THIS IS IN THE PACKAGE AT ALL. The meaning channel (`dense.py`) is
worth nothing to somebody who installs the library and gets a store with
no vectors in it, and an optional extra is a feature most people never
turn on. So the vectors are HERE, and they are on by default: a reader
who paraphrases is understood out of the box, with no download, no
account, no network, and nothing to configure.

WHY A STATIC MATRIX RATHER THAN A MODEL. A distilled static embedding is
one lookup per token and a mean — there is no transformer to run, so
there is no torch, no GPU question and no warm-up. Measured on this
repository's own field corpus, 20,000 lines and 180 known-item queries
(each query is a line with its two most distinctive terms removed, so
the words alone cannot find it):

    encoder                        first  MRR    index (20k lines)
    a distillation of our own      27%    0.306    1.2 s
    paraphrase-multilingual-MiniLM 66%    0.710   36.1 s
    THIS (potion-multilingual)     84%    0.883    0.3 s

It is better than the transformer it was distilled from the family of,
and it builds the index a hundred times faster. Trimming it to fit a
wheel cost almost nothing — the full 500k-piece, 256-dimension matrix
scores 0.883 and the 108k-piece, 128-dimension one here scores 0.861 —
because the pieces that fall away are the rarest ones, which is also
where the noise is.

WHAT IT IS. `minishlab/potion-multilingual-128M` (MIT licence, the
model2vec method), reduced to 108k pieces by PCA to 128 dimensions and
stored as float16: 30 MB. Every word of every language it covers still
segments, because every single character is kept whatever else is
dropped.

THE CEILING IS OPEN. `Memory(encoder=...)` takes any callable mapping a
list of strings to a list of vectors, so an operator who wants
bge-m3, a vendor's API or their own in-house model passes it in and
this file is never loaded. `Memory(dense=False)` turns the channel off
entirely and the store answers exactly as it did before it existed.
"""
import os
import re
import threading

# Word boundaries only — the piece table does the rest of the work, and
# it was distilled with the same convention: a piece that begins a word
# is written with U+2581 in front of it, a piece that continues one is
# written bare.
WORD = re.compile(r"\w+", re.UNICODE)
START = "▁"

_LOCK = threading.Lock()
_TABLE = None                       # (pieces {str: int}, vectors ndarray)


def _here(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "static", name)


def available():
    """Is the bundled matrix present? A source checkout without it, or an
    installation that stripped the package data, still runs — the channel
    is simply absent, which is the one thing it is allowed to be."""
    try:
        import numpy                                    # noqa: PLC0415,F401
    except Exception:                                   # noqa: BLE001
        return False                # no numpy, no channel — not a crash
    return (os.path.exists(_here("vectors.npy"))
            and os.path.exists(_here("pieces.txt")))


def _table():
    """The piece table and the matrix, loaded once and shared.

    Memory-mapped: a process that never asks a question never pays for
    the 30 MB, and several memories in one process share one map."""
    global _TABLE                                       # noqa: PLW0603
    with _LOCK:
        if _TABLE is None:
            import numpy                                # noqa: PLC0415
            vectors = numpy.load(_here("vectors.npy"), mmap_mode="r")
            words = open(_here("pieces.txt"),
                         encoding="utf-8").read().split("\n")
            _TABLE = ({w: i for i, w in enumerate(words) if w}, vectors)
    return _TABLE


def _split(word, pieces):
    """One word into piece ids — longest match first, left to right.

    The distillation's own tokeniser scores every segmentation and takes
    the best; this takes the greedy one, which is the standard
    approximation and needs no scores in the package. Measured against
    the scored segmentation on the field corpus, the difference did not
    show above the noise of the retrieval itself.
    """
    out, at = [], 0
    while at < len(word):
        end, found = len(word), None
        while end > at:
            part = (START + word[at:end]) if at == 0 else word[at:end]
            got = pieces.get(part)
            if got is not None:
                found = got
                break
            end -= 1
        if found is None:
            # Never reached for a character the table knows, and every
            # single character is kept; an unknown one is skipped rather
            # than poisoning the mean with an [UNK] vector.
            at += 1
            continue
        out.append(found)
        at = end
    return out


def encode(texts):
    """A list of strings to a list of unit vectors — the encoder's contract.

    A text is the MEAN of its pieces, normalised. There is no pooling
    layer and no attention: the whole model is the table, which is what
    makes it a millisecond.
    """
    import numpy                                        # noqa: PLC0415
    pieces, vectors = _table()
    width = vectors.shape[1]
    out = numpy.zeros((len(texts), width), dtype=numpy.float32)
    for row, text in enumerate(texts):
        ids = []
        for word in WORD.findall((text or "").lower()):
            ids += _split(word, pieces)
        if ids:
            out[row] = numpy.asarray(vectors[ids], dtype=numpy.float32).mean(0)
    norm = numpy.linalg.norm(out, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return (out / norm).tolist()
