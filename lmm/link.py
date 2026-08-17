"""Label → identity bridge. Converts the text labels Qwen extracts into the
graph's numeric identities. `geometry.resolve` disambiguates with context (the
same spelling can be several concepts); if absent, `memory.identify` opens one.

WRITES NO RECORD — only maps label↔identity. The GATE decides what gets written.
Case-independent (`fold`), no language rule.
"""
from v3 import geometry
from v3.dataset import fold
from lmm import inflect

# INCREMENTAL LABEL INDEX — the tolerance scan below used to iterate ALL
# identities per call: O(n) per new label, O(n²) per document. Invisible at
# 22 spreadsheet cells, minutes at a manual's thousands of table cells. The
# index maps folded label -> first key and buckets labels by their first 5
# characters (for the reverse/prefix direction). Updated incrementally: only
# identities added since the last call are indexed (dicts are ordered).
import weakref

_INDEX = weakref.WeakKeyDictionary()    # memory -> {"n", "map", "pre5"}


def _index_add(state, folded, key):
    state["map"].setdefault(folded, key)
    state["pre5"].setdefault(folded[:inflect.ROOT], []).append((folded, key))


def _label_index(memory):
    state = _INDEX.get(memory)
    if state is None:
        state = {"n": 0, "map": {}, "pre5": {}}
        _INDEX[memory] = state
    idents = memory.identities
    if state["n"] < len(idents):
        for key in list(idents.keys())[state["n"]:]:
            for lab in idents[key].labels:
                _index_add(state, fold(lab), key)
        state["n"] = len(idents)
    return state


def resolve(memory, label, vectors=None, create=False):
    """Converts a label to an identity.

    create=False: returns only an EXISTING identity, else None (for
                  verification — if Qwen invented a new name it has no
                  counterpart, the claim counts as unsupported).
    create=True:  opens a new identity if absent (for writing).
    """
    if not label:
        return None
    label = fold(str(label).strip())
    if not label:
        return None
    vec = (vectors or {}).get(label)
    key, _ = geometry.resolve(memory, label, vec)
    if key is not None:
        return key
    # SAFE INFLECTION TOLERANCE: Qwen may deliver inflected forms
    # ("organ"→"organdır"). The criterion is the SHARED one — `inflect`, one
    # rule and one pair of constants for every organ that has to recognise an
    # ending. It is the strict side of "fabrication 0": we don't loosen and take
    # false positives, and the F5 hole (kart→kartal, organ→organizma) stays
    # closed by ROOT/TAIL alone.
    #
    # REMOVED: a special case admitting a 4-letter root with a one-letter tail
    # ("probu"→"prob"), whose stated justification was that the benchmark
    # manual's spec-table subjects are short. That is one document's shape
    # deciding the strictness of an identity gate, so it is gone.
    #
    # Both directions run over the INCREMENTAL INDEX in ~O(1) — the old full
    # scan was O(n) per call and turned document ingestion quadratic.
    index = _label_index(memory)
    # forward: an existing ROOT is a prefix of the arriving label → the
    # candidate roots are the label's own prefixes that a tail could have grown
    # from.
    for cut in range(max(inflect.ROOT, len(label) - inflect.TAIL), len(label)):
        other = index["map"].get(label[:cut])
        if other is not None and inflect.inflection_of(label, label[:cut]):
            return other
    # REVERSE DIRECTION — ONLY WHEN WRITING (create=True). An existing node may
    # have been opened inflected ("metaldir" arrived first as a value), then
    # the root arrives ("metal" as subject) — the one-way view opened two nodes
    # and broke the transitive chain. BUT on the verification/read path
    # (create=False) this mapping is CLOSED: with only "şekersiz" in the graph,
    # Qwen's "şeker ..." claim would count as "supported" by the wrong node's
    # edge — a fabrication-0 hole (code-review finding #1). When matched while
    # writing, the root is added to the identity as an ALIAS; later reads find
    # it via legitimate label matching. F5 protections unchanged.
    if create and len(label) >= inflect.ROOT:
        for folded, other in index["pre5"].get(label[:inflect.ROOT], ()):
            if folded != label and inflect.inflection_of(folded, label):
                key = memory.identify(label, same_as=other)
                _index_add(index, label, key)   # alias: count unchanged, index now
                return key
    if create:
        return memory.identify(label, vector=vec)
    return None


def label_of(memory, key):
    """From identity to its first label (for raw dumps / prompts). None → empty."""
    if key is None:
        return ""
    held = memory.identities.get(key) if isinstance(key, int) else None
    if held is not None and held.labels:
        return held.labels[0]
    return str(key)
