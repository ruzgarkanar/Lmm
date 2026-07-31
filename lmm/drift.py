"""Watching a branch of knowledge go bad, and stopping it before it does.

NELL ran unsupervised for six months and stayed above 90% precision on three
quarters of its categories — while the remaining quarter fell to between 25% and
60%. The failure was not uniform and it was not loud: a few branches drifted
while everything else looked fine. A few minutes of human review per category
every few weeks brought the whole knowledge base back to about 87%.

So this counts, per branch, how much of what arrives is accepted and how much is
refused. A branch whose refusal rate climbs is a branch going wrong, and it gets
frozen: machine sources stop writing there and a person is asked. Nothing is
deleted and nothing already learned is doubted — the branch simply stops
growing on its own until someone looks.

A branch is the outermost type a concept belongs to, because that is the level
at which drift shows: not "penguen" going wrong, but everything under "kuş".
"""
from lmm.trust import level, HUMAN

MINIMUM = 6         # below this, a run of refusals is just a run of refusals
THRESHOLD = 0.4     # refusals above this share mean the branch needs a look


def branch_of(reasoning, concept):
    """The outermost type a concept sits under — where drift becomes visible."""
    ancestors = reasoning.ancestors(concept)
    return ancestors[-1] if ancestors else concept


def record(memory, branch, accepted):
    health = memory.health.setdefault(branch, {"accepted": 0, "refused": 0,
                                               "frozen": False})
    health["accepted" if accepted else "refused"] += 1
    if not health["frozen"] and _rate(health) > THRESHOLD and _seen(health) >= MINIMUM:
        health["frozen"] = True
    return health


def _seen(health):
    return health["accepted"] + health["refused"]


def _rate(health):
    seen = _seen(health)
    return health["refused"] / seen if seen else 0.0


def refusal_rate(memory, branch):
    return _rate(memory.health.get(branch, {"accepted": 0, "refused": 0}))


def frozen(memory, branch):
    return memory.health.get(branch, {}).get("frozen", False)


def frozen_branches(memory):
    return sorted(name for name, health in memory.health.items()
                  if health.get("frozen"))


def thaw(memory, branch):
    """A person has looked. The counters start again from what they decided."""
    health = memory.health.get(branch)
    if not health:
        return False
    health.update({"accepted": 0, "refused": 0, "frozen": False})
    return True


def may_write(memory, reasoning, concept, source):
    """A person may always teach. A machine may not, into a frozen branch."""
    if level(source) >= HUMAN:
        return True
    return not frozen(memory, branch_of(reasoning, concept))
