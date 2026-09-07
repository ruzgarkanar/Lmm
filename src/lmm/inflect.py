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
                              "sugarless" from an incoming "sugar" would let the
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


def kin(a, b):
    """RETRIEVAL-ONLY kinship: the shorter word is wholly the longer one's
    prefix, the ending stays within TAIL, and the shorter is long enough to
    root anything (three letters — the same floor the tokenizer draws).

    ROOT is five letters, and a short-rooted language walks under it: a
    four-letter root and its suffixed form share a prefix of four, and
    `same_stem` calls them strangers — the query word never reaches the
    line that answers it. A looser reverse-prefix rule once lived inside
    the retrieval scorer and was removed for being one of four quietly
    different copies of the criterion; the loss of short units was recorded
    then as a measured cost. This is that rule given a NAME and a single
    home beside the other two — and a scope: the retrieval scorer alone,
    where surfacing a real line is the worst a false kinship can do. The
    gates keep `same_stem`: kinship widens what can be FOUND, and nothing
    about what may be SAID."""
    if a == b:
        return True
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if (len(short) >= 3 and long.startswith(short)
            and len(long) - len(short) <= TAIL):
        return True
    # TWO INFLECTIONS OF ONE SHORT ROOT ARE KIN TOO. The prefix case
    # above only reaches a bare root and its suffixed form; measured
    # where it hurts, both sides arrive inflected — a comparison asks
    # "aynı SÜREDE mi" of a record headed "SÜRESİ", four letters of
    # shared root and a different ending on each. same_stem wants five
    # and refuses; neither form is the other's prefix. On the retrieval
    # side kinship means what it always meant: a shared opening of at
    # least three letters with each remainder within TAIL. Two shared
    # letters are nothing ("car"/"cat"), and a long divergence is a
    # different word.
    common = os.path.commonprefix((a, b))
    return (len(common) >= 3 and len(a) - len(common) <= TAIL
            and len(b) - len(common) <= TAIL)


def same_stem(a, b):
    """Are `a` and `b` two endings on one shared root (or the same word)?"""
    if a == b:
        return True
    common = os.path.commonprefix((a, b))
    return (len(common) >= ROOT and len(a) - len(common) <= TAIL
            and len(b) - len(common) <= TAIL)
