"""Label → identity bridge. Converts the text labels Qwen extracts into the
graph's numeric identities. `geometry.resolve` disambiguates with context (the
same spelling can be several concepts); if absent, `memory.identify` opens one.

WRITES NO RECORD — only maps label↔identity. The GATE decides what gets written.
Case-independent (`fold`), no language rule.
"""
from v3 import geometry
from v3.dataset import fold


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
    # ("organ"→"organdır"). The criterion is STRICT so the F5 hole
    # (kart→kartal, organ→organizma) doesn't open:
    #   root >=5 letters AND remainder <=3 letters.
    #     organ(5)→organdır  remainder "dır"=3  ✓ matches (correct)
    #     kart(4)→kartal      root 4<5           ✗ (mismatch closed)
    #     organ(5)→organizma  remainder "izma"=4 ✗ (mismatch closed)
    # The strict side of "fabrication 0": we don't loosen and take false positives.
    for other, ident in memory.identities.items():
        for lab in ident.labels:
            root = fold(lab)
            if len(root) >= 5 and label.startswith(root) \
                    and 0 < len(label) - len(root) <= 3:
                return other
            # REVERSE DIRECTION — ONLY WHEN WRITING (create=True). An existing
            # node may have been opened inflected ("metaldir" arrived first as a
            # value), then the root arrives ("metal" as subject) — the one-way
            # view opened two nodes and broke the transitive chain. BUT on the
            # verification/read path (create=False) this mapping is CLOSED:
            # with only "şekersiz" in the graph, Qwen's "şeker ..." claim would
            # count as "supported" by the wrong node's edge — a fabrication-0
            # hole (code-review finding #1). When matched while writing, the
            # root is added to the identity as an ALIAS; later reads find it
            # via legitimate label matching, not a blind scan. F5 protections
            # unchanged.
            if create and len(label) >= 5 and root.startswith(label) \
                    and 0 < len(root) - len(label) <= 3:
                return memory.identify(label, same_as=other)
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
