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
