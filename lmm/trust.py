"""Where a fact came from, and how much that counts for.

A memory fed by several kinds of source needs to know which one to believe when
they disagree. A person speaking directly outranks a document they handed over,
which outranks a language model's output, which outranks the system's own
generalisation.

The ranking is what lets correction happen without an argument: a fact from a
higher source simply replaces a belief held on a lower one, and the system says
so. Between equals it does not choose — it asks, because picking a winner
between two sources of equal standing is guessing.
"""

HUMAN = 4
DOCUMENT = 3
DISTILLED = 2       # produced by a language model
INFERRED = 1        # the system worked it out itself

TEACHER = "sen"
INFERENCE = "çıkarım"
DISTILLED_PREFIX = "llm:"

CONFIDENCE = {HUMAN: 0.6, DOCUMENT: 0.6, DISTILLED: 0.5, INFERRED: 0.45}
CORROBORATION = 0.15    # what one more independent source is worth
CEILING = 0.98          # certainty is never reached, only approached


def distilled_source(model_name):
    return f"{DISTILLED_PREFIX}{model_name}"


def level(source):
    if source == INFERENCE:
        return INFERRED
    if source.startswith(DISTILLED_PREFIX):
        return DISTILLED
    if source == TEACHER:
        return HUMAN
    return DOCUMENT


def confidence_for(source):
    return CONFIDENCE[level(source)]


def outranks(source, other):
    return level(source) > level(other)


CANDIDATE = "aday"      # the newcomer wins
INCUMBENT = "yerleşik"  # what is held wins
DISPUTED = "tartışmalı"  # neither, and that is recorded rather than hidden


def reputation_of(reputation, source):
    """A source's record: how often it agreed with the rest, how often it did not.

    Truth-discovery research compared a dozen elaborate schemes and found plain
    majority voting almost impossible to beat, at a ninth to a hundredth of the
    cost — several of the clever ones were not even reproducible run to run. So
    this stays a counter, not a model.
    """
    record = reputation.get(source, {})
    agreed = record.get("agreed", 0)
    disputed = record.get("disputed", 0)
    if agreed + disputed == 0:
        return 0.5              # no record yet: neither trusted nor suspected
    return agreed / (agreed + disputed)


def note(reputation, source, agreed):
    record = reputation.setdefault(source, {"agreed": 0, "disputed": 0})
    record["agreed" if agreed else "disputed"] += 1


def arbitrate(incumbent, candidate, reputation=None):
    """Which of two clashing claims to hold, or neither.

    Rank first, because a person outranks a document whatever either has said
    before. Then how many independent sources back each side — that is the
    majority vote the literature keeps finding sufficient. Then each side's
    record. If nothing separates them, Wikidata's answer: keep both and mark the
    disagreement rather than pretending one of them won.
    """
    reputation = reputation if reputation is not None else {}
    if candidate.source in incumbent.sources:
        # One voice cannot dispute itself. A document that says birds fly and
        # that a penguin cannot is incoherent, not two sources disagreeing, and
        # the old answer — refuse and report — is the right one.
        return INCUMBENT
    if outranks(candidate.source, incumbent.source):
        return CANDIDATE
    if outranks(incumbent.source, candidate.source):
        return INCUMBENT
    here, there = len(set(incumbent.sources)), len(set(candidate.sources))
    if here != there:
        return INCUMBENT if here > there else CANDIDATE
    mine = max(reputation_of(reputation, s) for s in incumbent.sources)
    theirs = max(reputation_of(reputation, s) for s in candidate.sources)
    if abs(mine - theirs) > 0.1:
        return INCUMBENT if mine > theirs else CANDIDATE
    return DISPUTED


def confidence_from(sources):
    """How sure to be, given everyone who has said it.

    The best source sets the floor and each further *independent* one raises it.
    Hearing the same thing twice from the same place is not corroboration; that
    is how a single mistake becomes a consensus of one.
    """
    if not sources:
        return CONFIDENCE[INFERRED]
    distinct = list(dict.fromkeys(sources))
    best = max(confidence_for(source) for source in distinct)
    return min(CEILING, best + CORROBORATION * (len(distinct) - 1))
