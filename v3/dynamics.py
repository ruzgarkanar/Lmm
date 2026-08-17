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


def field_of(memory, record, kinds=None):
    """The COMPLETE contradiction field around a record — its slot's rivals.

    The walk is the link closure restricted to the same slot (subject +
    predicate): whoever contradicts a rival of mine, in my own slot, belongs
    to the same argument. Walking the CLOSURE rather than the direct
    neighbours is what makes the result independent of WHICH member is asked —
    the audit measured a field of three where the second and third had never
    been linked, and arbitration then judged two different sub-fields
    depending on where it entered. It is also what lets the write path stop
    building complete cliques: connectivity carries the field, and a clique of
    k rivals costs k² links for nothing.

    `kinds` defaults to CONTRA alone (the judged field, which is what
    arbitration may act on); judging passes CONTRA+SUSPECT to see the pending
    ones too.

    Returns: the field as a list (the record itself included).
    """
    from v3.memory import CONTRA
    kinds = (CONTRA,) if kinds is None else kinds
    seen = {record.key: record}
    frontier = [record]
    while frontier:
        nxt = []
        for one in frontier:
            for other in memory.rivals_of(one, kinds):
                if other.key in seen:
                    continue
                if (other.subject != record.subject
                        or other.predicate != record.predicate):
                    continue        # another slot: not this field's business
                seen[other.key] = other
                nxt.append(other)
        frontier = nxt
    return list(seen.values())


def arbitrate(memory, record):
    """Arbitration between contradicting records — who carries more weight.

    The decision rests on criteria that are all ON RECORD: the source's level,
    the witness count, and (only to break a tie) the trust. The loser is NOT
    DELETED, it merely does not speak — because one day it may win, and when
    that day comes its history must be standing.

    Two properties this function OWES, both of them broken before and each one
    a measured pathology:

    ORDER INDEPENDENCE. The ranking may not read anything that arbitration
    itself writes. Trust is exactly such a quantity — a demoted loser has a
    lowered trust, so ranking primarily by trust made the outcome depend on
    who was taught first, and on how many times the round had already run.
    Level and witness count are properties of the EVIDENCE and no round
    changes them; trust breaks ties only.

    THE WINNER IS NEVER PUNISHED. The old rule punished whoever had the higher
    trust — including the winner. A document with 30 witnesses meeting a
    one-sentence operator claim ended with BOTH of them under the speaking
    threshold: the contradiction was resolved by leaving the slot empty. A
    contradiction between two claims is not a reason to know nothing.

    The loser is lowered ABSOLUTELY: under the winner AND under SPEAK, so
    "who wins" and "who may speak" cannot disagree. `min` makes the write
    idempotent — running the round twice changes nothing.

    Returns: the winning record (may be itself).
    """
    from v3.gate import SPEAK
    field = field_of(memory, record)
    if len(field) < 2:
        return record
    # The key's last element is the record key: a total order, so two records
    # equal on every criterion still rank the same way in every round.
    field.sort(key=lambda one: (-one.level, -one.witnesses, -one.trust,
                                one.key))
    winner = field[0]
    ceiling = min(winner.trust, SPEAK) * 0.5
    for other in field[1:]:
        other.trust = min(other.trust, ceiling)
    return winner


def judge_suspects(memory, verdict):
    """Judges the DEFERRED contradiction suspicions — field by field.

    The rivalry test is not always affordable at write time (bulk table
    ingestion would pay one model round-trip per cell). Those writes record a
    SUSPECT link instead of deciding, and the debt is paid here, where the
    work is batched anyway.

    `verdict(old_value, new_value)` -> True (rivals) | False (they coexist) |
    None (still cannot judge — the suspicion stays, keeps generating pressure,
    and comes back next round).

    THE UNIT OF JUDGEMENT IS THE SLOT, not the link. A single SUSPECT link is
    the marker that a slot (subject + predicate) holds something unsettled;
    from that marker the whole slot is pulled in, so the verdicts do not
    depend on which pairs happened to get linked at write time. What comes in
    is everything still open: the pending records, and whoever is already
    known to contradict them.

    THE FIELD IS JUDGED AGAINST ITS STRONGEST MEMBER, not pair by pair. A slot
    holding k pending values has k² pairs, and asking a model k² questions to
    settle one slot is not a thing that can run. What the slot actually needs
    to know is who holds it: the strongest claim is the ANCHOR, everyone else
    is judged against it, and whoever turns out to coexist with the anchor
    stays in play for the next round with the strongest of THEM as anchor.
    That is k questions per round, and it terminates because each round
    removes at least the anchor.

    Returns: (how many became contradictions, how many were cleared).
    """
    from v3.memory import CONTRA, SUSPECT
    became, cleared = 0, 0
    slots = {}
    for one in sorted(memory.records.values(), key=lambda r: r.key):
        if not memory.rivals_of(one, (SUSPECT,)):
            continue
        held = slots.setdefault((one.subject, one.predicate), {})
        held[one.key] = one
        # Whoever is already known to contradict a pending record belongs to
        # the same argument; a record that was judged to COEXIST is settled
        # and stays out, so no verdict is ever paid for twice.
        for other in memory.rivals_of(one, (CONTRA, SUSPECT)):
            if (other.subject, other.predicate) == (one.subject, one.predicate):
                held[other.key] = other
    for field in [list(one.values()) for one in slots.values()]:
        pool = sorted(field, key=lambda one: (-one.level, -one.witnesses,
                                              -one.trust, one.key))
        if len(pool) > 1:
            anchor, rest = pool[0], pool[1:]
            leftover = []
            for other in rest:
                if (CONTRA, other.key) in anchor.links:
                    continue                    # already a judged rivalry
                answer = verdict(anchor.value, other.value)
                if answer is None:
                    # Still unjudgeable: keep the suspicion ON RECORD so the
                    # pressure stays and the next round asks again.
                    memory.link(anchor.key, other.key, SUSPECT)
                    memory.link(other.key, anchor.key, SUSPECT)
                    leftover.append(other)
                    continue
                memory.unlink(anchor.key, other.key, SUSPECT)
                memory.unlink(other.key, anchor.key, SUSPECT)
                if answer:
                    memory.link(anchor.key, other.key, CONTRA)
                    memory.link(other.key, anchor.key, CONTRA)
                    became += 1
                else:
                    cleared += 1
                    leftover.append(other)      # coexists here, may still
                    #                             contradict someone else
            # ONE ANCHOR PER ROUND. Whoever coexists with the anchor may still
            # contradict one another, and settling that exhaustively is k²
            # questions — a cost no maintenance round should be able to
            # inflict. So the round settles the anchor's field and RE-MARKS
            # what is left: the debt shrinks by one anchor per sleep and is
            # never dropped. Sleep is periodic; the graph has time.
            if len(leftover) > 1:
                memory.link(leftover[0].key, leftover[1].key, SUSPECT)
                memory.link(leftover[1].key, leftover[0].key, SUSPECT)
        # Arbitration runs AFTER every verdict is in, so the field is judged
        # whole and not pairwise as the verdicts trickle in.
        for one in field:
            if memory.rivals_of(one, (CONTRA,)):
                arbitrate(memory, one)
    return became, cleared


# The most experiences to keep in a sleep round. The surprising stays, the
# ordinary goes — the round-3 audit measured it: 1000 answers were 1000
# experiences and no maintenance touched them; every "hmm" grew in the file
# forever.
MOST_EXPERIENCES = 2000


def sleep(memory, now=None, verdict=None):
    """The sleep round: judge the suspicions, distill, dampen, arbitrate.

    The counterpart of hippocampal replay, doing three jobs in one round. No
    gradients, no GPU — it is one scan and thresholds.

    Returns: a count dictionary — how many records settled, how many faded,
    how many were arbitrated. The report is NUMBERS, NOT LANGUAGE; the keys
    are this file's own identities.
    """
    now = time.time() if now is None else now
    counted = {"settled": 0, "faded": 0, "judged": 0}
    from v3.memory import CONTRA
    # THE DEFERRED DEBT IS PAID FIRST. `verdict` is the rivalry test the write
    # path could not afford (in LMM: the session's semantic check). Sleep is
    # already a batch pass over the whole memory, which is exactly the place
    # where a per-item cost becomes bearable. If nothing is handed in, the
    # suspicions simply wait one more round — they are not silently dropped.
    if verdict is not None:
        became, cleared = judge_suspects(memory, verdict)
        counted["contradicted"] = became
        counted["cleared"] = cleared
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

    A SUSPECTED contradiction counts too — an unjudged suspicion is precisely
    something to be uneasy about, and it is also the signal that tells sleep
    there is a debt waiting. While the doubt was being dropped at write time,
    this list came back empty on a memory that held paris and berlin in the
    same slot.

    Returns: [(record key, pressure)], high to low.
    """
    from v3.memory import CONTRA, SUSPECT
    found = []
    for record in memory.records.values():
        rivals = memory.rivals_of(record, (CONTRA, SUSPECT))
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
