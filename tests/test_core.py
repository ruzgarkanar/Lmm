"""Core regression tests — pure python, NO model, no pytest.

    python3.11 tests/test_core.py

Every test here is a PATHOLOGY THAT WAS MEASURED on this code, written back as
an assertion so it cannot return. They are grouped by the bug they close:

    A  a contradiction deferred in bulk mode was DESTROYED, not deferred
    B  arbitration depended on teaching ORDER, and killed its own winner
    C  a slot with three rivals was linked as a broken star
    D  the derivation engine wrote AROUND the contradiction gate
    E  transitivity was re-derived from scratch per written cell (quadratic)

Nothing here needs a network, a model or a GPU: the whole discrete layer is
counters, links and thresholds, and that is exactly why it can be tested like
this.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v3 import dynamics                                          # noqa: E402
from v3.gate import Gate, SPEAK                                  # noqa: E402
from v3.memory import (CONTRA, DOCUMENT, INFERRED, Memory,       # noqa: E402
                       OPERATOR, STRANGER, SUSPECT)
from v3.session import Session as V3Session                      # noqa: E402
from v3.transitive import Transitivity                           # noqa: E402

PASSED = []


def test(name):
    """Registers a test — and prints the line the moment it passes."""
    def keep(function):
        PASSED.append((name, function))
        return function
    return keep


def slot(memory, subject, predicate):
    """The records standing in one slot — value key -> record."""
    return {r.value: r for r in memory.about(subject, touch=False)
            if r.predicate == predicate}


def capital_case(rival=None):
    """The measured scenario: fransa → başkent → paris | berlin."""
    memory = Memory()
    gate = Gate(memory)
    gate.rival = rival
    where = memory.identify("fransa")
    what = memory.identify("baskent")
    return memory, gate, where, what, memory.identify("paris"), memory.identify("berlin")


def links_of(record, kind):
    return {key for one, key in record.links if one == kind}


# --- A: the deferred contradiction is KEPT, and sleep pays the debt ------

@test("A1 bulk-deferred contradiction is recorded as a suspicion")
def a1():
    # PENDING (None) is the bulk-mode answer: the rivalry test cannot run per
    # cell. Measured before the fix: links=[[], []], pressure=[] — the doubt
    # was gone, both records were speakable.
    memory, gate, where, what, paris, berlin = capital_case(
        rival=lambda old, new: None)
    gate.admit(where, what, paris, "#table", DOCUMENT)
    gate.admit(where, what, berlin, "#table", DOCUMENT)
    held = slot(memory, where, what)
    assert len(held) == 2, "both claims must be WRITTEN — a claim with a source is kept"
    assert links_of(held[paris], SUSPECT) == {held[berlin].key}
    assert links_of(held[berlin], SUSPECT) == {held[paris].key}
    assert dynamics.pressure(memory), "an unjudged suspicion must generate pressure"


@test("A2 sleep judges the deferred suspicions in bulk and pays the debt")
def a2():
    memory, gate, where, what, paris, berlin = capital_case(
        rival=lambda old, new: None)
    gate.admit(where, what, paris, "#table", DOCUMENT)
    gate.admit(where, what, berlin, "#table", OPERATOR)
    # The verdict that could not be afforded at write time arrives at sleep.
    counted = dynamics.sleep(memory, verdict=lambda old, new: True)
    assert counted["contradicted"] == 1, counted
    held = slot(memory, where, what)
    assert links_of(held[paris], CONTRA) == {held[berlin].key}
    assert not links_of(held[paris], SUSPECT), "a judged suspicion is no longer pending"
    assert held[berlin].trust >= SPEAK, "the winner speaks"
    assert held[paris].trust < SPEAK, "the loser does NOT speak — the debt is paid"


@test("A3 a suspicion the verdict CLEARS leaves no scar")
def a3():
    memory, gate, where, what, paris, berlin = capital_case(
        rival=lambda old, new: None)
    gate.admit(where, what, paris, "#table", DOCUMENT)
    gate.admit(where, what, berlin, "#table", DOCUMENT)
    counted = dynamics.sleep(memory, verdict=lambda old, new: False)
    assert counted["cleared"] == 1, counted
    held = slot(memory, where, what)
    for record in held.values():
        assert not record.links, "coexisting values keep no contradiction link"
        assert record.trust >= SPEAK, "neither claim is punished for coexisting"
    assert not dynamics.pressure(memory)


@test("A4 an unjudgeable suspicion WAITS — it is never silently dropped")
def a4():
    memory, gate, where, what, paris, berlin = capital_case(
        rival=lambda old, new: None)
    gate.admit(where, what, paris, "#table", DOCUMENT)
    gate.admit(where, what, berlin, "#table", DOCUMENT)
    dynamics.sleep(memory, verdict=lambda old, new: None)       # still unknown
    held = slot(memory, where, what)
    assert links_of(held[paris], SUSPECT), "the debt survives a round with no verdict"
    dynamics.sleep(memory, verdict=lambda old, new: True)       # the verdict arrives late
    assert links_of(held[paris], CONTRA)
    assert min(r.trust for r in held.values()) < SPEAK <= max(r.trust for r in held.values())


@test("A5 a crowded slot costs one anchor per round, and the debt survives")
def a5():
    # A slot holding k pending values has k² possible questions. A sleep round
    # may not be able to cost that: it settles the strongest claim's field and
    # RE-MARKS the rest, so the debt shrinks every round and is never dropped.
    memory = Memory()
    gate = Gate(memory)
    gate.rival = lambda old, new: None
    where = memory.identify("konu")
    what = memory.identify("ozellik")
    values = [memory.identify("deger%d" % i) for i in range(6)]
    for value in values:
        gate.admit(where, what, value, "#table", DOCUMENT)
    asked = []

    def coexist(old, new):
        asked.append((old, new))
        return False                    # nothing is a rival: the slow case

    dynamics.sleep(memory, verdict=coexist)
    assert len(asked) == 5, "one round asks about one anchor, not every pair"
    assert dynamics.pressure(memory), "the rest of the debt is still marked"
    rounds = 0
    while dynamics.pressure(memory) and rounds < 20:
        dynamics.sleep(memory, verdict=coexist)
        rounds += 1
    assert not dynamics.pressure(memory), "successive rounds settle the slot"
    held = slot(memory, where, what)
    assert all(r.trust >= SPEAK for r in held.values()), \
        "values that all coexist all keep their voice"


# --- B: arbitration is order-independent, idempotent, and keeps a winner --

@test("B1 DOCUMENT then OPERATOR: the contradiction does NOT escape")
def b1():
    # Measured before the fix: 0.6 and 0.75 — BOTH above SPEAK, both spoken.
    memory, gate, where, what, paris, berlin = capital_case()
    gate.admit(where, what, paris, "belge", DOCUMENT)
    gate.admit(where, what, berlin, "#operator", OPERATOR)
    held = slot(memory, where, what)
    speakers = [r for r in held.values() if r.trust >= SPEAK]
    assert len(speakers) == 1, "exactly one side of a contradiction may speak"
    assert speakers[0].value == berlin


@test("B2 OPERATOR then DOCUMENT gives the SAME outcome — order-independent")
def b2():
    memory, gate, where, what, paris, berlin = capital_case()
    gate.admit(where, what, berlin, "#operator", OPERATOR)
    gate.admit(where, what, paris, "belge", DOCUMENT)
    held = slot(memory, where, what)
    speakers = [r for r in held.values() if r.trust >= SPEAK]
    assert len(speakers) == 1 and speakers[0].value == berlin
    # Both orders must agree down to the numbers, not merely in the verdict.
    other = capital_case()
    memory2, gate2, where2, what2, paris2, berlin2 = other
    gate2.admit(where2, what2, paris2, "belge", DOCUMENT)
    gate2.admit(where2, what2, berlin2, "#operator", OPERATOR)
    a, b = slot(memory, where, what), slot(memory2, where2, what2)
    assert round(a[paris].trust, 6) == round(b[paris2].trust, 6)
    assert round(a[berlin].trust, 6) == round(b[berlin2].trust, 6)


@test("B3 30 witnesses vs one sentence: the slot does NOT end up empty")
def b3():
    # Measured before the fix: 0.188 and 0.375 — mutual destruction, both
    # under SPEAK. A contradiction between two claims is not a reason to know
    # nothing.
    memory, gate, where, what, paris, berlin = capital_case()
    for i in range(30):
        gate.admit(where, what, paris, "belge%d" % i, DOCUMENT)
    gate.admit(where, what, berlin, "#operator", OPERATOR)
    dynamics.sleep(memory)
    held = slot(memory, where, what)
    speakers = [r for r in held.values() if r.trust >= SPEAK]
    assert len(speakers) == 1, [(r.value, r.trust) for r in held.values()]
    assert speakers[0].witnesses == 1 and speakers[0].level == OPERATOR


@test("B4 the winner is never punished, and repeated rounds change nothing")
def b4():
    memory, gate, where, what, paris, berlin = capital_case()
    gate.admit(where, what, paris, "belge", DOCUMENT)
    gate.admit(where, what, berlin, "#operator", OPERATOR)
    held = slot(memory, where, what)
    winner = held[berlin].trust
    assert winner == memory.trust_of(OPERATOR), "the winner keeps its own trust"
    before = {k: r.trust for k, r in held.items()}
    for _ in range(5):
        dynamics.sleep(memory)
        dynamics.arbitrate(memory, held[paris])
        dynamics.arbitrate(memory, held[berlin])
    assert {k: r.trust for k, r in held.items()} == before, "arbitration is idempotent"


@test("B5 the loser stays below BOTH the winner and the speaking threshold")
def b5():
    for first, second in ((STRANGER, DOCUMENT), (DOCUMENT, STRANGER),
                          (OPERATOR, DOCUMENT), (DOCUMENT, OPERATOR)):
        memory, gate, where, what, paris, berlin = capital_case()
        gate.admit(where, what, paris, "a", first)
        gate.admit(where, what, berlin, "b", second)
        held = list(slot(memory, where, what).values())
        held.sort(key=lambda r: -r.trust)
        assert held[0].trust >= SPEAK > held[1].trust, (first, second,
                                                        [r.trust for r in held])


# --- C: the rival SET, not the first rival ------------------------------

@test("C1 three rivals in one slot are all linked to each other")
def c1():
    # Measured before the fix: v2 and v3 were never linked — _contradiction
    # returned the FIRST clash and stopped, so arbitration entered a broken
    # star and judged a different sub-field depending on where it started.
    memory = Memory()
    gate = Gate(memory)
    where = memory.identify("fransa")
    what = memory.identify("baskent")
    values = [memory.identify(name) for name in ("paris", "berlin", "roma")]
    for i, value in enumerate(values):
        gate.admit(where, what, value, "belge%d" % i, DOCUMENT)
    held = slot(memory, where, what)
    for value in values:
        others = {held[o].key for o in values if o != value}
        assert links_of(held[value], CONTRA) == others, (value, held[value].links)


@test("C2 the whole field is judged from whichever member is asked")
def c2():
    memory = Memory()
    gate = Gate(memory)
    where = memory.identify("fransa")
    what = memory.identify("baskent")
    paris = memory.identify("paris")
    berlin = memory.identify("berlin")
    roma = memory.identify("roma")
    gate.admit(where, what, paris, "belge", DOCUMENT)
    gate.admit(where, what, berlin, "yabanci", STRANGER)
    gate.admit(where, what, roma, "#operator", OPERATOR)
    held = slot(memory, where, what)
    for entry in held.values():
        assert len(dynamics.field_of(memory, entry)) == 3
        assert dynamics.arbitrate(memory, entry).value == roma
    speakers = [r for r in held.values() if r.trust >= SPEAK]
    assert len(speakers) == 1 and speakers[0].value == roma


@test("C3 an unjudged pair inside a judged field is still judged at sleep")
def c3():
    # A suspicion is a marker on the FIELD, not a relation between two rows:
    # if the field is already connected, no extra link is written — but the
    # pair must still get its verdict when sleep walks that field. This is the
    # part of the debt that a spanning topology could quietly lose.
    memory = Memory()
    gate = Gate(memory)
    where = memory.identify("fransa")
    what = memory.identify("baskent")
    paris, berlin, roma = (memory.identify(n) for n in ("paris", "berlin", "roma"))
    gate.rival = lambda old, new: None                  # bulk: nothing judgeable
    for value in (paris, berlin, roma):
        gate.admit(where, what, value, "#table", DOCUMENT)
    held = slot(memory, where, what)
    assert len(dynamics.field_of(memory, held[roma], (CONTRA, SUSPECT))) == 3, \
        "one field, however few links carry it"
    asked = []

    def verdict(old, new):
        asked.append((old, new))
        return True

    dynamics.sleep(memory, verdict=verdict)
    judged = {frozenset(pair) for pair in asked}
    assert judged >= {frozenset((paris, berlin)), frozenset((paris, roma))} or \
        len(judged) >= 2, asked
    speakers = [r for r in held.values() if r.trust >= SPEAK]
    assert len(speakers) == 1, [(r.value, r.trust) for r in held.values()]
    assert not any(links_of(r, SUSPECT) for r in held.values()), \
        "every suspicion in the field was settled"


@test("C4 a slot with many pending values does not build a clique")
def c4():
    # Measured while fixing this: linking every pending pair in a slot that a
    # repeated table column had filled cost ~2M links for information the
    # field closure already carries.
    memory = Memory()
    gate = Gate(memory)
    gate.rival = lambda old, new: None
    where = memory.identify("satir")
    what = memory.identify("kolon")
    for i in range(200):
        gate.admit(where, what, memory.identify("hucre%d" % i), "#table", DOCUMENT)
    held = list(slot(memory, where, what).values())
    links = sum(len(r.links) for r in held)
    assert links <= 4 * len(held), "the suspicion topology must stay linear (%d)" % links
    assert len(dynamics.field_of(memory, held[0], (CONTRA, SUSPECT))) == 200, \
        "and it must still be ONE field"


# --- D: the derivation engine goes through the gate ----------------------

@test("D1 an inference contradicting an operator fact is NOT written silently")
def d1():
    # Measured before the fix: gate.inferred called memory.write directly —
    # "kartal→balık" landed next to the operator's "kartal→kuş" with links=[]
    # and pressure=[]. The one writer in the system that could contradict
    # without anyone noticing.
    memory = Memory()
    gate = Gate(memory)
    eagle = memory.identify("kartal")
    kind = memory.identify("tur")
    bird, fish = memory.identify("kus"), memory.identify("balik")
    gate.admit(eagle, kind, bird, "#operator", OPERATOR)
    gate.inferred(eagle, kind, fish, [])
    held = slot(memory, eagle, kind)
    assert links_of(held[fish], CONTRA) == {held[bird].key}
    assert links_of(held[bird], CONTRA) == {held[fish].key}
    assert dynamics.pressure(memory), "the derivation's doubt must be on record"
    assert held[fish].level == INFERRED and held[fish].source == "#inference"
    assert held[fish].trust < SPEAK <= held[bird].trust


@test("D2 a derivation that contradicts nothing still passes cleanly")
def d2():
    memory = Memory()
    gate = Gate(memory)
    eagle = memory.identify("kartal")
    kind = memory.identify("tur")
    animal = memory.identify("hayvan")
    bird = memory.identify("kus")
    because = gate.admit(eagle, kind, bird, "#operator", OPERATOR)[0]
    # kuş → hayvan on a predicate the graph proved transitive: the derived
    # "kartal → hayvan" is the CHAIN of the premise, not its rival.
    gate.admit(bird, kind, animal, "#operator", OPERATOR)
    memory.transitive.add(kind)
    held = gate.inferred(eagle, kind, animal, [because.key])
    assert held is not None and not links_of(held, CONTRA)
    assert (3, because.key) in held.links, "the rationale link survives the gate"
    assert not held.episodic, "a derivation is not an episode"


@test("D3 the derivation loop does not feed on itself")
def d3():
    session = V3Session(None)
    memory = session.memory
    kind = session._identity("tur")
    a, b, c = (session._identity(n) for n in ("kartal", "kus", "hayvan"))
    for one, two in ((a, b), (b, c), (a, c)):
        session.gate.admit(one, kind, two, "#operator", OPERATOR)
        session._learn_transitive(kind, one, two)
    memory.transitive.add(kind)
    d = session._identity("canli")
    session.gate.admit(c, kind, d, "#operator", OPERATOR)
    session._derive(c, kind, d)
    grew = len(memory.records)
    for _ in range(3):
        session._derive(c, kind, d)         # deriving again must add nothing
    assert len(memory.records) == grew, "derivation is a fixed point, not a pump"
    for record in memory.records.values():
        if record.source == "#inference":
            assert record.trust < SPEAK, "no derivation may reach speaking trust"


# --- E: transitivity is learned incrementally, not re-derived ------------

@test("E1 the incremental tracker learns exactly what the full scan learned")
def e1():
    # is-a closes triangles; "sever" does not — and one coincidental triangle
    # is still not enough.
    memory = Memory()
    gate = Gate(memory)
    kind = memory.identify("tur")
    names = ["a", "b", "c", "d"]
    keys = {n: memory.identify(n) for n in names}
    tracker = Transitivity(memory)
    edges = [("a", "b"), ("b", "c"), ("a", "c"),        # first triangle
             ("c", "d"), ("b", "d")]                    # closes the second
    for i, (one, two) in enumerate(edges):
        gate.admit(keys[one], kind, keys[two], "belge%d" % i, DOCUMENT)
        tracker.observe(kind, keys[one], keys[two])
    assert kind in memory.transitive, "two witnessed triangles make it transitive"

    loves = Memory()
    gate2 = Gate(loves)
    predicate = loves.identify("sever")
    people = {n: loves.identify(n) for n in names}
    tracker2 = Transitivity(loves)
    for i, (one, two) in enumerate([("a", "b"), ("b", "c"), ("c", "d")]):
        gate2.admit(people[one], predicate, people[two], "belge%d" % i, DOCUMENT)
        tracker2.observe(predicate, people[one], people[two])
    assert predicate not in loves.transitive, "an open chain is not transitivity"


@test("E2 ONE coincidental triangle is not enough")
def e2():
    memory = Memory()
    gate = Gate(memory)
    predicate = memory.identify("sever")
    keys = {n: memory.identify(n) for n in ("a", "b", "c")}
    tracker = Transitivity(memory)
    for i, (one, two) in enumerate([("a", "b"), ("b", "c"), ("a", "c")]):
        gate.admit(keys[one], predicate, keys[two], "belge%d" % i, DOCUMENT)
        tracker.observe(predicate, keys[one], keys[two])
    assert predicate not in memory.transitive


@test("E3 a memory loaded from file still has its triangles found")
def e3():
    # The tracker's first pass over a predicate reads what already stands —
    # otherwise a restart would forget everything the graph had proven.
    memory = Memory()
    gate = Gate(memory)
    kind = memory.identify("tur")
    keys = {n: memory.identify(n) for n in ("a", "b", "c", "d")}
    for i, (one, two) in enumerate([("a", "b"), ("b", "c"), ("a", "c"),
                                    ("c", "d"), ("b", "d")]):
        gate.admit(keys[one], kind, keys[two], "belge%d" % i, DOCUMENT)
    memory.transitive.clear()               # as if freshly loaded from disk
    assert Transitivity(memory).observe(kind)


@test("E4 derived records are not evidence for the rule that derived them")
def e4():
    memory = Memory()
    gate = Gate(memory)
    kind = memory.identify("tur")
    keys = {n: memory.identify(n) for n in ("a", "b", "c")}
    tracker = Transitivity(memory)
    for one, two in [("a", "b"), ("b", "c")]:
        gate.admit(keys[one], kind, keys[two], "belge", DOCUMENT)
        tracker.observe(kind, keys[one], keys[two])
    gate.inferred(keys["a"], kind, keys["c"], [])
    tracker.observe(kind, keys["a"], keys["c"])
    assert kind not in memory.transitive, "a derivation cannot prove its own rule"


@test("E5 ingestion is near-linear, not quadratic")
def e5():
    # Measured before the fix: 2000 cells 0.91s, 10000 cells 25.6s — 5x the
    # data for 28x the time. The check is the SHAPE: five times the cells may
    # not cost far more than five times the work.
    def ingest(cells):
        memory = Memory()
        gate = Gate(memory)
        tracker = Transitivity(memory)
        columns = [memory.identify("kolon%d" % i) for i in range(5)]
        started = time.time()
        for i in range(cells):
            subject = memory.identify("satir%d" % i)
            value = memory.identify("hucre%d" % i)
            predicate = columns[i % len(columns)]
            gate.admit(subject, predicate, value, "#table", DOCUMENT)
            tracker.observe(predicate, subject, value)
        return time.time() - started

    small = ingest(2000)
    large = ingest(10000)
    ratio = large / max(small, 1e-6)
    print("      2000 cells %.3fs · 10000 cells %.3fs · ratio %.1fx"
          % (small, large, ratio))
    assert ratio < 10.0, "5x the data must not cost ~28x the time (was quadratic)"


# --- behaviour: nothing above broke the ordinary paths -------------------

@test("F1 a table ingested without a model still writes, and reloads")
def f1():
    from lmm.session import Session
    session = Session(None)
    rows = [{"ekipman": "yangin tupu %d" % i, "durum": "uygun",
             "yer": "kat %d" % (i % 5)} for i in range(100)]
    wrote = session.learn_rows(rows)
    assert wrote == 200, wrote
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_core_roundtrip.lmm")
    try:
        session.memory.save(path)
        again = Memory.load(path)
        assert len(again.records) == len(session.memory.records)
        assert all(again.records[k].links == session.memory.records[k].links
                   for k in session.memory.records), "links survive the round trip"
    finally:
        for leftover in (path, path + ".sent"):
            if os.path.exists(leftover):
                os.remove(leftover)


@test("F2 an old .lmm file (no SUSPECT links) still loads and arbitrates")
def f2():
    memory = Memory()
    gate = Gate(memory)
    where = memory.identify("fransa")
    what = memory.identify("baskent")
    paris, berlin = memory.identify("paris"), memory.identify("berlin")
    gate.admit(where, what, paris, "belge", DOCUMENT)
    gate.admit(where, what, berlin, "#operator", OPERATOR)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "_core_legacy.lmm")
    try:
        memory.save(path)
        again = Memory.load(path)
        counted = dynamics.sleep(again, verdict=lambda old, new: True)
        assert counted["judged"] >= 1
        held = slot(again, where, what)
        assert len([r for r in held.values() if r.trust >= SPEAK]) == 1
    finally:
        if os.path.exists(path):
            os.remove(path)


@test("F3 the v3 session still learns and answers without any network")
def f3():
    session = V3Session(None)
    kind = session._identity("tur")
    eagle, bird = session._identity("kartal"), session._identity("kus")
    record, why = session.gate.admit(eagle, kind, bird, "#operator", OPERATOR)
    assert record is not None and why == 0
    assert session.gate.behind(eagle, kind, bird) is not None
    assert "kartal" in session._render([record])
    assert isinstance(session.sleep(), dict)
    assert session.curious() is not None and session.tension() == []


@test("F4 the no-fabrication holes stay closed")
def f4():
    from lmm import evidence, link
    memory = Memory()
    memory.identify("şekersiz")
    assert link.resolve(memory, "şeker", create=False) is None
    memory2 = Memory()
    key = memory2.identify("metaldir")
    assert link.resolve(memory2, "metal", create=True) == key
    memory3 = Memory()
    memory3.identify("organ")
    assert link.resolve(memory3, "organizma", create=False) is None
    assert not evidence.covered("Dijital patoloji Tier 1",
                                "[K1] Dijital patoloji Tier 3", "tier 1 mi")
    assert not evidence.covered("test Tier 5",
                                "[K1] test Tier 3 proje 5 yıl", "")


def main():
    failed = 0
    for name, function in PASSED:
        try:
            function()
        except AssertionError as broke:
            failed += 1
            print("FAIL  %s\n      %s" % (name, broke))
        except Exception as broke:                          # noqa: BLE001
            failed += 1
            print("ERROR %s\n      %r" % (name, broke))
        else:
            print("ok    %s" % name)
    print("\n%d/%d passed" % (len(PASSED) - failed, len(PASSED)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
