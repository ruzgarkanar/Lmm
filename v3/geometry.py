"""The continuous layer: geometry for FINDING, not for verifying.

The project's core idea finds half of itself here:

    In a language model everything is continuous — findable but unverifiable.
    In a classic knowledge graph everything is discrete — verifiable but unfindable.
    Here the two are two layers of the same records: CONTINUOUS FINDS, DISCRETE VERIFIES.

It was measured, and this was exactly what was missing: in the old memory 19%
of real questions got "concept not in graph" and the concept WAS IN THE
GRAPH — the string did not match. As long as retrieval rests on
letter-by-letter matching, a known thing looks unknown.

Here retrieval is two steps:

    1. PROXIMITY   the identities nearest to the question's vector (continuous layer)
    2. SPREADING   starting from those identities and walking along LINKS

The second is association: given a part of a pattern, the whole comes back
(the Hopfield intuition). String search cannot do this; a record is found only
if its name is known exactly. Spreading brings the unnamed in from its
neighbor.

Vectors are NOT PRODUCED in this file — they are built by counting and given
from outside. Here they are only used. Nothing belonging to language exists:
no word, no suffix, no rule. Cosine and graph walking, that is all.

THE BOUNDARY — ABSOLUTE. Geometry never WRITES a record. Opposites occur in
the same surroundings and distributional similarity cannot separate them;
this was measured once in this project. Geometry finds candidates, the gate
decides.
"""
import math

# Proximity must be at least this to count as meaningful. Below the threshold
# is noise: in a high-dimensional space everything resembles everything a
# little.
NEAR = 0.35

# How much of the similarity is dampened at each step of spreading. If it is
# close to one, distant records shout as loudly as near ones and association
# loses its meaning.
DECAY = 0.6

# How many steps the spreading runs. Beyond three it spreads over practically
# the whole graph.
STEPS = 3


def cosine(one, other):
    """Directional similarity between two vectors."""
    if not one or not other:
        return 0.0
    total = sum(a * b for a, b in zip(one, other))
    left = math.sqrt(sum(a * a for a in one))
    right = math.sqrt(sum(b * b for b in other))
    if not left or not right:
        return 0.0
    return total / (left * right)


def mean(vectors):
    """The average of vectors — a crude representation of a sentence.

    It loses word order and we know it. What will replace it is the reader
    network; this is only a cheap hand doing the first cull.
    """
    held = [one for one in vectors if one]
    if not held:
        return None
    width = len(held[0])
    return [sum(one[at] for one in held) / len(held) for at in range(width)]


def nearest(memory, vector, count=8, least=NEAR):
    """The identities nearest to the vector — [(identity, proximity)], near to far."""
    if vector is None:
        return []
    scored = []
    for key, held in memory.identities.items():
        if held.vector is None:
            continue
        score = cosine(vector, held.vector)
        if score >= least:
            scored.append((score, key))
    scored.sort(reverse=True)
    return [(key, score) for score, key in scored[:count]]


def resolve(memory, label, vector=None):
    """Picks from context WHICH identity a label is.

    The same spelling can be more than one concept, and memory does not know
    which — nor does it need to. The decision is made here, by picking the
    identity nearest to the context's vector.

    If there is no context, or none is near enough, the most-accessed
    identity is returned: in its absence, picking the most familiar is better
    than picking none, and the weakness of the pick is reflected in the
    trust.
    """
    held = memory.candidates(label)
    if not held:
        return None, 0.0
    if len(held) == 1:
        return held[0], 1.0
    if vector is not None:
        scored = []
        for key in held:
            other = memory.identities[key].vector
            if other is not None:
                scored.append((cosine(vector, other), key))
        if scored:
            scored.sort(reverse=True)
            score, key = scored[0]
            if score >= NEAR:
                return key, score
    best = max(held, key=lambda key: memory.identities[key].seen)
    return best, 0.0


def spread(memory, seeds, steps=STEPS, decay=DECAY, most=40):
    """Association: starts from seed identities and walks along links.

    Returns: {record key: weight}. The weight dampens with distance from the
    seed — a record coming from nearby speaks with a louder voice.

    The walk uses both kinds of link: an identity's records (the subject
    link) and the records' links to one another (cause, then, condition).
    Without the second this would only be listing neighbors; with it, it
    becomes following a chain.
    """
    reached = {}
    frontier = {key: weight for key, weight in seeds}
    for step in range(steps):
        following = {}
        for key, weight in frontier.items():
            for record in memory.about(key, touch=False):
                held = reached.get(record.key, 0.0)
                if weight > held:
                    reached[record.key] = weight
                # If the record's value is an identity, continue from there.
                if isinstance(record.value, int) \
                        and record.value in memory.identities:
                    onward = weight * decay
                    if onward > following.get(record.value, 0.0):
                        following[record.value] = onward
                # Record-to-record links — the real association is here.
                for _, other in record.links:
                    onward = weight * decay
                    if onward > reached.get(other, 0.0):
                        reached[other] = onward
        if not following:
            break
        frontier = following
    return dict(sorted(reached.items(), key=lambda pair: -pair[1])[:most])


def recall(memory, vector=None, labels=(), most=40):
    """The records relevant to a question — found by geometry, gathered by spreading.

    The single gate of retrieval: first the nearest identities, then
    spreading from them. String matching is mandatory nowhere; if a label is
    given, it only strengthens the seed.
    """
    seeds = []
    for label in labels:
        key, score = resolve(memory, label, vector)
        if key is not None:
            seeds.append((key, max(score, 0.5)))
    if vector is not None:
        seeds.extend(nearest(memory, vector, count=6))
    if not seeds:
        return {}
    return spread(memory, seeds, most=most)
