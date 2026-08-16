"""The living part of memory: reinforcement, fading, distillation, arbitration.

A database keeps what is written as-is. Memory does not — what is used grows
stronger, what is unused fades, what is repeated becomes permanent, what
contradicts gets arbitrated. This file builds those four movements, and this
is where the name "living memory" is earned.

Its grounding is the brain's dual structure (complementary learning systems):

    FAST   hippocampus   writes in one shot · EPISODIC · changes quickly
    SLOW   cortex        distills by repetition · SEMANTIC · permanent

In sleep the hippocampus replays what it wrote during the day to the cortex;
what works becomes permanent, the rest is erased. `sleep()` is the counterpart
of that, and it needs no gradients, no GPU — it is counting and thresholds.

A language model has no counterpart of this loop. There, a piece of knowledge
either entered the weights during training or never exists; it neither
reinforces, nor fades, nor distills.

Nothing belonging to language exists here. Everything in this file is a
counter and a threshold.
"""
import time

# How many independent witnesses an episodic record needs to rise to semantic.
# Two, because a single witness is an event; the second witness makes it a
# regularity.
WITNESSES_TO_SETTLE = 2

# The floor trust reaches when rising to semantic. Rising is a promotion, and
# a promoted record is no longer hostage to a single conversation.
SETTLED_TRUST = 0.65

# Fading speed: the share of trust an unaccessed record loses each round.
# Kept small because forgetting must not be cheap — knowledge loss is
# irreversible.
FADE = 0.03

# A record that drops below this trust and is never accessed no longer speaks.
# It is not deleted: it has a source and one day a second witness may come.
FLOOR = 0.1

# The freshness window (seconds) exempt from fading in a round. A freshly
# written record must not fade before it has had a chance to be accessed.
FRESH = 3600.0


def reinforce(memory, record, source, level=None):
    """Another source also stated the same fact — a share of the doubt closes.

    Not additive: in the old memory, when it was additive, the fourth
    document pierced the ceiling. Each witness closes a fixed share of the
    remaining doubt; the ceiling is approached but never reached.
    """
    return record.strengthen(source, level)   # protection and math in one place


def settle(memory, record):
    """Promotes an episodic record to semantic — if there are enough witnesses.

    This is the difference between "Ali said so" and "a known thing", and it
    forms by itself over time: an event, repeated and repeated, becomes
    knowledge.
    """
    if not record.episodic:
        return False
    if record.witnesses < WITNESSES_TO_SETTLE:
        return False
    record.episodic = False
    record.trust = max(record.trust, SETTLED_TRUST)
    return True


def fade(memory, record, now=None):
    """An unaccessed record fades. It is not deleted — its voice is lowered.

    Records inside the freshness window are untouched: a freshly written
    thing must not fade before it has had a chance to be accessed.
    """
    now = time.time() if now is None else now
    if now - record.last_seen < FRESH:
        return False
    if record.witnesses > 1 or not record.episodic:
        return False        # a witnessed or settled record does not fade
    from v3.memory import OPERATOR
    if record.level >= OPERATOR:
        return False        # what the operator taught does not decay silently
    before = record.trust
    record.trust = max(FLOOR, record.trust - FADE)
    return record.trust < before


def arbitrate(memory, record):
    """Arbitration between contradicting records — who carries more weight.

    The decision rests on two criteria and both are on record: the source's
    level and the witness count. The loser is NOT DELETED, it merely does not
    speak — because one day it may win, and when that day comes its history
    must be standing.

    Returns: the winning record (may be itself).
    """
    from v3.memory import CONTRA
    rivals = [memory.records[key] for kind, key in record.links
              if kind == CONTRA and key in memory.records]
    if not rivals:
        return record
    field = [record] + rivals
    field.sort(key=lambda one: (-one.level, -one.witnesses, -one.trust))
    winner = field[0]
    for other in field[1:]:
        if other.trust >= winner.trust:
            other.trust = winner.trust * 0.5
    return winner


# The most experiences to keep in a sleep round. The surprising stays, the
# ordinary goes — the round-3 audit measured it: 1000 answers were 1000
# experiences and no maintenance touched them; every "hmm" grew in the file
# forever.
MOST_EXPERIENCES = 2000


def sleep(memory, now=None):
    """The sleep round: distill, dampen, arbitrate.

    The counterpart of hippocampal replay, doing three jobs in one round. No
    gradients, no GPU — it is one scan and thresholds.

    Returns: a count dictionary — how many records settled, how many faded,
    how many were arbitrated. The report is NUMBERS, NOT LANGUAGE; the keys
    are this file's own identities.
    """
    now = time.time() if now is None else now
    counted = {"settled": 0, "faded": 0, "judged": 0}
    from v3.memory import CONTRA
    for record in list(memory.records.values()):
        if settle(memory, record):
            counted["settled"] += 1
        if fade(memory, record, now):
            counted["faded"] += 1
        if any(kind == CONTRA for kind, _ in record.links):
            arbitrate(memory, record)
            counted["judged"] += 1
    # Experience pruning: the ordinary (inconsequential, unsurprising) old
    # ones go.
    if len(memory.experiences) > MOST_EXPERIENCES:
        ranked = sorted(memory.experiences.values(),
                        key=lambda one: (abs(one.outcome) + one.surprise,
                                         one.at))
        for stale in ranked[:len(ranked) - MOST_EXPERIENCES]:
            del memory.experiences[stale.key]
            counted["pruned"] = counted.get("pruned", 0) + 1
    return counted


def pressure(memory):
    """Contradiction pressure: what the system should be uneasy about.

    One of the evaluation signals. High pressure points to the places that
    deserve going out and verifying — this is what tells curiosity where to
    look.

    Returns: [(record key, pressure)], high to low.
    """
    from v3.memory import CONTRA
    found = []
    for record in memory.records.values():
        rivals = [memory.records[key] for kind, key in record.links
                  if kind == CONTRA and key in memory.records]
        if not rivals:
            continue
        # If both sides are strong the pressure is high: a weak objection
        # does not disturb, two evenly matched claims do.
        best = max(one.trust for one in rivals)
        found.append((record.key, min(record.trust, best)))
    found.sort(key=lambda pair: -pair[1])
    return found


def gaps(memory, most=20):
    """Curiosity: the memory's own gaps.

    If there is no record at all about an identity, or only episodic ones,
    that is the place to learn. The system knows what it does not know and
    tells itself where to look.

    Returns: [(identity key, gap size)], large to small.
    """
    found = []
    for key, identity in memory.identities.items():
        records = memory.by_subject.get(key, ())
        if not records:
            found.append((key, 1.0 + identity.seen))
            continue
        settled = sum(1 for one in records
                      if not memory.records[one].episodic)
        if not settled:
            found.append((key, 0.5 + identity.seen))
    found.sort(key=lambda pair: -pair[1])
    return found[:most]
