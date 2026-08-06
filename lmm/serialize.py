# -*- coding: utf-8 -*-
"""Arrow/bracket serialization of raw records, and refusal detection.

No natural language lives here: labels are read from the relation
registry (data), everything else is punctuation.
"""

SEP = " · "

NEG = " : ✗"


def label(relation):
    """Readable label of a relation — from the registry, not from code."""
    from lmm.kinds import DEFAULT
    kind = DEFAULT.by_name.get(relation)
    return kind.label if kind else relation


def fact(concept, relation, target, positive=True, object=None):
    """One record, as-is: concept → label → target [@ object] [: ✗]."""
    said = f"{concept} → {label(relation)} → {target}"
    if object:
        said += f" @ {object}"
    return said if positive else said + NEG


def predicate(relation, target, positive=True, object=None):
    """A record with its subject dropped: → label → target."""
    said = f"→ {label(relation)} → {target}"
    if object:
        said += f" @ {object}"
    return said if positive else said + NEG


def cite(source):
    """Provenance mark: ‹source›."""
    return f"‹{source}›"


def is_refusal(said):
    """Empty output, or output opening with a bracket mark, is a refusal."""
    return (not said) or said.startswith("[")
