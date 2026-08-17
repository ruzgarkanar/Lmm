"""ONE inflection criterion, in ONE place.

Four organs need to decide "are these two surface forms the same word wearing
different endings": `link.resolve` (label → identity), `verify._has_edge` (does
the graph hold this claim's value), `evidence.coverage` (is this answer word
grounded) and `evidence.find` (does this query word reach that index word).
Each of them had grown its OWN version of the rule — root ≥5 with a ≤3 tail
here, a root of 4 with a single-letter tail there ("measured need: spec-table
subjects are short"), a ≥4-forward/≥3-reverse pair in the index lookup, plus a
proportional 70%-stem clause beside it. Four criteria means the strictness of
the fabrication gate depended on which organ happened to ask, and at least one
of them had been widened by looking at ONE document.

So there is one rule and one pair of constants:

    ROOT  how much of a word has to be present for the rest to be an ending
    TAIL  how much an ending may add

and two DIRECTIONS, because the direction is load-bearing and not a tuning
knob:

    inflection_of(form, root)  `form` is `root` plus an ending. Asymmetric on
                              purpose — on the READ path, matching a stored
                              "şekersiz" from an incoming "şeker" would let the
                              wrong node's edge vouch for a claim (a
                              fabrication hole), so read paths may only walk
                              this direction.
    same_stem(a, b)           both forms are endings on one shared root. Used
                              where either side may arrive inflected and no
                              claim is being licensed by the match alone.

Neither constant is derived from data — they are the one honest arbitrary pair
in the system, and they are stated once instead of five times.
"""
import os

ROOT = 5        # letters that make a root recognisable
TAIL = 3        # letters an ending may add


def inflection_of(form, root):
    """Is `form` the word `root` carrying an ending (or the same word)?"""
    if form == root:
        return True
    return (len(root) >= ROOT and form.startswith(root)
            and len(form) - len(root) <= TAIL)


def same_stem(a, b):
    """Are `a` and `b` two endings on one shared root (or the same word)?"""
    if a == b:
        return True
    common = os.path.commonprefix((a, b))
    return (len(common) >= ROOT and len(a) - len(common) <= TAIL
            and len(b) - len(common) <= TAIL)
