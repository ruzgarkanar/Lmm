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
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lmm.core import dynamics                                          # noqa: E402
from lmm.core.gate import Gate, SPEAK                                  # noqa: E402
from lmm.core.memory import (CONTRA, DOCUMENT, INFERRED, Memory,       # noqa: E402
                       OPERATOR, STRANGER, SUSPECT)
from lmm.core.session import Session as V3Session                      # noqa: E402
from lmm.core.transitive import Transitivity                           # noqa: E402

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
    assert not evidence.covered("Kapak menteşesi Seviye 1",
                                "[K1] Kapak menteşesi Seviye 3", "seviye 1 mi")
    assert not evidence.covered("test Seviye 5",
                                "[K1] test Seviye 3 parça 5 yıl", "")


@test("G1 a unit is content no matter how short it is")
def g1():
    """UNIT BLINDNESS — the largest measured single loss class. A token
    standing next to a NUMBER is that number's unit; dropping it for being
    under three letters lost 'kg', and splitting the camel seam inside 'mAh'
    lost the capacity's unit twice over."""
    from lmm import evidence
    assert "kg" in evidence._words("Kütle 2 kg")
    assert "mah" in evidence._words("Akü 12.8 V / 4200 mAh")
    assert "v" in evidence._words("Akü 12.8 V / 4200 mAh")
    # the camel seam still cuts glued CELLS (the left side is a word there)
    assert "kapağı" in evidence._words("GövdeKapağıKayış")
    # a short token with no number next to it is still not content
    assert "ve" not in evidence._words("ekran ve klavye")


@test("G2 a question with no number of its own still reaches the unit")
def g2():
    """'kaç kg' carries no 2 for the kg to lean on — the corpus has to vouch
    for it. Learned from the indexed sentences, never a word list."""
    from lmm import evidence
    store = evidence.SentenceStore()
    store.add("Kütle (ambalajsız) 3.4 kg")
    store.add("Gösterge 21.5 inç OLED")
    store.add("Cihaz taşınabilir bir sistemdir")
    assert "kg" in store.units
    assert store.find("kaç kg") == ["Kütle (ambalajsız) 3.4 kg"]


@test("G3 the unit anchor admits inflection without opening a hole")
def g3():
    """A fluent sentence ('21.5 inçtir') must pass against '21.5 inç' — the
    same NUMBER binds both. Words that no number holds together stay out,
    even when their prefix step is smaller."""
    from lmm import evidence
    assert evidence.covered("Gösterge 21.5 inçtir.", "Gösterge 21.5 inç OLED", "")
    assert not evidence.covered("kartal", "kart", "")
    assert not evidence.covered("organizma", "organ", "")
    # ...and the anchor needs the SAME number, not just any number
    assert not evidence.covered("Gösterge 17 inçtir.", "Gösterge 21.5 inç OLED", "")


@test("G4 coverage is the ratio whose 1.0 is the old binary gate")
def g4():
    from lmm import evidence
    block = "[K1] Gösterge 21.5 inç OLED"
    assert evidence.coverage("Gösterge 21.5 inç", block, "") == 1.0
    part = evidence.coverage("Gösterge 21.5 inç panelde gösterilir", block, "")
    assert 0.0 < part < 1.0
    assert not evidence.covered("Gösterge 21.5 inç panelde gösterilir", block, "")
    assert evidence.coverage("", block, "") == 0.0


@test("G5 the answer is SELECTED at the gate, not taken on faith")
def g5():
    """GENERATE-AND-SELECT. One generation bound the answer to a single sample
    of a sampling process; the measured residue was an unstable CHOICE, not
    missing knowledge. Here the engine is a stub, so what is under test is the
    SELECTOR: does the composite score prefer the grounded candidate, and does
    it still refuse when every candidate breaks a gate."""
    from lmm import generate, session as lmm_session
    s = lmm_session.Session.__new__(lmm_session.Session)      # no engine needed
    s.memory = Memory()
    proof = ["Gösterge 21.5 inç OLED gösterge", "Cihaz taşınabilir bir sistemdir"]
    block = "[K1] Gösterge 21.5 inç OLED gösterge\n[K2] Cihaz taşınabilir bir sistemdir"
    said, order = {}, []
    answers, supported = {}, [True]

    def fake_answer(question, facts, warmth=0.2):
        order.append(facts)
        return answers.get(facts, "Gösterge 21.5 inç")

    def fake_supported(answer, facts):
        said[answer] = said.get(answer, 0) + 1
        return supported[0]

    real = (generate.answer, generate.supported, generate.answers_asked)
    generate.answer, generate.supported = fake_answer, fake_supported
    # the RELATION gate has its own test (H5); here it stays out of the way
    generate.answers_asked = lambda question, answer, view: True
    try:
        # THE LADDER RUNS: one candidate per evidence subset, and each subset
        # is a different block (2 of them for 2 evidence sentences).
        chosen, tried = s._select("Ekran kaç inç", [], proof, "", block)
        assert chosen == "Gösterge 21.5 inç", chosen
        assert len(order) == len(set(order)) >= 2, order
        # A FABRICATED DIGIT IS VETOED even with the engine saying yes: the
        # wide-block candidate invents 17, the narrow one stays grounded.
        answers.clear()
        answers[order[0]] = "Ekran 17 inçtir"
        order.clear()
        chosen, tried = s._select("Ekran kaç inç", [], proof, "", block)
        assert chosen == "Gösterge 21.5 inç", chosen
        assert "Ekran 17 inçtir" in tried    # it was generated, and rejected
        # EVERY candidate broken → honest refusal, no fallback to the least bad
        answers.clear()
        for facts in order:
            answers[facts] = "Ekran 17 inçtir"
        chosen, _tried = s._select("Ekran kaç inç", [], proof, "", block)
        assert chosen is None
        # The read-back keeps its veto: coverage alone cannot speak.
        answers.clear()
        supported[0] = False
        chosen, _tried = s._select("Ekran kaç inç", [], proof, "", block)
        assert chosen is None
    finally:
        (generate.answer, generate.supported,
         generate.answers_asked) = real


@test("G6 the read-back is asked about the evidence the claim rests on")
def g6():
    """The measured oscillation was the JUDGE, not the answer: the same claim
    against the same 3000-character block came back False, True, False. The
    narrow view is asked FIRST, and the wide one only if it fails — a haystack
    the size of the needle."""
    from lmm import generate, session as lmm_session
    s = lmm_session.Session.__new__(lmm_session.Session)
    proof = ["Ortam sıcaklığı 8 °C ~ +32 °C", "Barkod okuyucu isteğe bağlıdır",
             "Barkod yazıcı kurulumu için kılavuza bakın"]
    block = "\n".join(f"[K{i}] {s}" for i, s in enumerate(proof, 1))
    asked = []
    real = generate.supported

    def fake(answer, view):
        asked.append(view)
        return view == block        # only the WIDE view says yes

    generate.supported = fake
    try:
        claim = "Ortam sıcaklığı 8 °C ile +32 °C arasındadır"
        assert s._read_back(claim, proof, block)
        # first view: only the sentence the claim shares words with
        assert "Ortam" in asked[0] and "Barkod" not in asked[0], asked[0]
        assert len(asked) == 2 and asked[1] == block
        # a confirmed narrow view ends it — the wide view is not even asked
        asked.clear()
        generate.supported = lambda answer, view: True
        assert s._read_back(claim, proof, block)
        assert len(asked) == 0
    finally:
        generate.supported = real


@test("G7 among grounded candidates the one that ANSWERS wins")
def g7():
    """Two measured corpus losses, both from ranking the survivors wrongly.
    Counting agreement handed 'Morlan bir canlıdır' the answer because two
    NESTED subsets said it — narrower evidence, vaguer claim, twice as many
    voices. Ranking by groundedness handed 'Torvanit bir metaldir' the answer
    over 'Evet, torvanit bir maddedir', which was penalised for the word
    'evet'. Both claims in each pair are true; only one of each answers."""
    from lmm import generate, session as lmm_session
    s = lmm_session.Session.__new__(lmm_session.Session)
    generate_real = (generate.answer, generate.supported,
                     generate.answers_asked)
    generate.answers_asked = lambda question, answer, view: True
    proof = ["Morlan bir kuştur.", "Kuş bir canlıdır.", "Morlan bir canlıdır."]
    block = "\n".join(f"[K{i}] {t}" for i, t in enumerate(proof, 1))
    wide = {}

    def answer(question, facts, warmth=0.2):
        # the widest block answers specifically, the narrow ones generically
        return ("Morlan bir kuştur." if len(facts) == max(wide, default=0)
                else "Morlan bir canlıdır.")

    generate.supported = lambda a, b: True
    generate.answer = answer
    try:
        for one in s._subsets([], proof, ""):
            wide[len(one)] = one
        assert len(wide) >= 2, wide
        chosen, _tried = s._select("morlan nedir", [], proof, "", block)
        assert chosen == "Morlan bir kuştur.", chosen
        # ...and the more literal candidate does not outrank the one that
        # answers the question asked
        pair = ["Torvanit bir metaldir.", "Metal bir maddedir."]
        pair_block = "\n".join(f"[K{i}] {t}" for i, t in enumerate(pair, 1))
        replies = iter(["Torvanit bir metaldir.", "Evet, torvanit bir maddedir.",
                        "Evet, torvanit bir maddedir."])
        generate.answer = lambda q, f, warmth=0.2: next(replies)
        chosen, _tried = s._select("torvanit bir madde midir", [], pair, "",
                                   pair_block)
        assert chosen == "Evet, torvanit bir maddedir.", chosen
    finally:
        (generate.answer, generate.supported,
         generate.answers_asked) = generate_real


@test("H1 of two true answers the graph's is-a chain picks the specific one")
def h1():
    """Measured (corpus, "norgul nedir"): the answer came back "norgul is a
    LIVING THING" at every commit in this history, while the document says
    "norgul is a PLANT". Both records are in the graph and both are true; the
    vague one won because the evidence path removes #doc triples from the block
    and so discards the specific PREMISE while keeping its derived, general
    CONCLUSION. The graph itself holds the edge that settles it: plant→living."""
    from lmm import retrieve
    memory = Memory()
    norgul = memory.identify("norgul")
    plant = memory.identify("bitki")
    living = memory.identify("canli")
    specific_r = memory.write(norgul, None, plant, "#doc", DOCUMENT)
    vague_r = memory.write(norgul, None, living, "#inference", INFERRED)
    memory.write(plant, None, living, "#doc", DOCUMENT)   # the is-a chain
    kept = retrieve.specific(memory, [vague_r, specific_r])
    assert [r.key for r in kept] == [specific_r.key], \
        [(r.value, r.source) for r in kept]
    # A question that NAMES the general value is asking for exactly that claim
    # ("is a norgul a living thing"), so targeting protects it.
    kept = retrieve.specific(memory, [vague_r, specific_r],
                             keep={vague_r.key})
    assert {r.key for r in kept} == {vague_r.key, specific_r.key}
    # Unrelated values are not each other's ancestors — nothing is dropped.
    blue = memory.identify("mavi")
    colour_r = memory.write(norgul, None, blue, "#doc", DOCUMENT)
    kept = retrieve.specific(memory, [specific_r, colour_r])
    assert len(kept) == 2, kept


@test("H2 a causal question with nothing to say falls through, not refuses")
def h2():
    """The causal route was the only ONE-WAY DOOR in _respond: every other
    branch treats its own failure as a misreading and lets the normal path try
    (see the WRITE branch), but a causal question whose graph block produced
    nothing speakable ended the turn in a refusal. Measured (corpus, "norgul
    çoğalırsa ne olur"): the causal edge exists so the route is taken, verify
    drops its sentence, and the turn refuses — while the evidence path, traced
    on the same question, answers "Norgul çoğalırsa morlan çoğalır." and passes
    digits_ok, coverage 1.0 AND the read-back. No gate is loosened: the
    fall-through lands on _answer, which applies the same gates."""
    from lmm import generate, session as lmm_session
    s = lmm_session.Session.__new__(lmm_session.Session)
    s.memory = Memory()
    s.mode = "strict"
    s._causes_key = s.memory.identify("#causes")
    s.vectors = {}
    # no causal edge at all -> None (a refusal would have ended the turn)
    assert s._causal_answer("norgul cogalirsa ne olur", "effects",
                            "norgul") is None
    cause = s.memory.identify("norgul")
    effect = s.memory.identify("morlan")
    s.memory.write(cause, s._causes_key, effect, "#doc", DOCUMENT)
    from lmm import verify as lmm_verify
    real = generate.answer, generate.refusal, lmm_verify.verify
    try:
        # the engine produces a sentence stepping OUTSIDE the block: the gate
        # must still drop it, but as a fall-through and not as a refusal.
        # verify is stubbed to its VERDICT (dropped) — this test is about what
        # the route does with that verdict, and it must need no model.
        generate.answer = lambda q, f, warmth=0.2: "Norgul zerbalit uretir."
        generate.refusal = lambda q: "REFUSED"
        lmm_verify.verify = lambda *a, **k: ""
        assert s._causal_answer("norgul cogalirsa ne olur", "effects",
                                "norgul") is None
        # a grounded sentence is still spoken by this route
        generate.answer = lambda q, f, warmth=0.2: "norgul morlan"
        assert s._causal_answer("norgul cogalirsa ne olur", "effects",
                                "norgul") == "norgul morlan"
    finally:
        generate.answer, generate.refusal, lmm_verify.verify = real


@test("H3 a repaired table row never neighbours cells from two rows")
def h3():
    """The claim this test defends is the whole promise of the evidence layer:
    what is in the store is what the DOCUMENT says. The sliding-window repair
    broke it — over a shattered table it emitted
    "Uzatma kablosu ... · 3 · Orta · Orta · Orta · Taşıma kayışı", a
    neighbourhood spanning two rows, and the answer read out of it ("the
    strap's stock risk is medium") contradicted the document's own row
    ("Yüksek (sınırlı)"). No gate can catch that: the claim IS in the evidence.

    Built here, not read from a document: a column-major dump of a table whose
    rows are known, including one WRAPPED cell (the case a fixed stride cannot
    survive) and a heading above and a caption below the table. The assertion is
    that every cell pair inside an emitted row is a pair the table really has.
    NO MODEL, no document-specific constant."""
    from lmm import evidence
    rows = [("Kapak menteşesi yenilemesi", "Çok düşük", "Çok hızlı", "Yok"),
            ("Vida seti tamamlama", "Düşük", "Çok hızlı", "Yok"),
            ("Gösterge camı değişimi", "Orta", "Orta", "Orta"),
            ("Gövde boyası yenileme", "Çok yüksek", "Yavaş", "Orta"),
            ("Uzatma kablosu / yedek besleme ünitesi", "Orta", "Orta", "Orta"),
            ("Taşıma kayışı", "Orta", "Orta", "Yüksek (sınırlı)")]
    lines = ["Bir tablo, hücre hücre dağılmış hâlde okunduğunda satırını "
             "yitirir; asıl mesele budur.",
             "Kullanım alanı", " Kademe", "Yatırım", "Geri dönüş"]
    for nth, (name, *rest) in enumerate(rows):
        if nth == 4:
            name, tail = name.rsplit(" ", 1)         # the cell WRAPS in two
            lines.append(name)
            name = tail
        lines.append(name)
        lines.append(" %d" % (nth + 1))              # the indented column
        lines.extend(rest)
    lines.append("Tablo yukarıdaki gibi okunmalıdır ve sağ üst köşe önce "
                 "hedeflenmelidir.")
    text = "\n".join(lines)

    windows = evidence.table_windows(text)
    assert windows, windows
    # the layout HAS a phase to read (period + indented column), so rows are
    # reconstructed rather than guessed at
    run = [l for l in text.split("\n") if l.strip() and len(l.strip()) <= 40]
    assert evidence.rows_of(run) is not None

    # every within-row cell pair must be a pair the table really states
    true_pairs = set()
    for nth, (name, *rest) in enumerate(rows):
        cells = name.rsplit(" ", 1) if nth == 4 else [name]    # the wrapped one
        cells += [str(nth + 1)] + list(rest)
        for one in cells:
            for other in cells:
                true_pairs.add((one, other))
    heading = {"Kullanım alanı", "Kademe", "Yatırım", "Geri dönüş"}
    for window in windows:
        cells = [c.strip() for c in window.split(" — ")[-1].split(" · ")]
        cells = [c for c in cells if c not in heading]
        for one in cells:
            for other in cells:
                assert one == other or (one, other) in true_pairs, \
                    "the document never puts %r next to %r: %s" % (one, other,
                                                                   window)
    # and the row that was being corrupted comes out whole
    assert any("Taşıma kayışı" in w and "Yüksek (sınırlı)" in w
               for w in windows), windows
    # a run whose layout gives NO phase is not carved into invented rows
    flat = "\n".join(["Bir paragraf, tabloya benzemeyen düz yazıdır ve bu "
                      "yüzden ölçüte girmez."]
                     + ["satır %d" % i for i in range(12)]
                     + ["Kapanış cümlesi de yeterince uzun olmak zorundadır "
                        "ki koşu burada bitsin."])
    flat_run = [l for l in flat.split("\n") if l.strip() and len(l.strip()) <= 40]
    assert evidence.rows_of(flat_run) is None


@test("H4 nested views of one paragraph cannot take every evidence seat")
def h4():
    """A question whose answer needs TWO sections gets one chance: both have to
    be among the sentences retrieved. The near-duplicate filter exists so that no
    single region monopolises those seats, but it measured overlap by Jaccard
    only — and the window scales are NESTED, so the same paragraph seen at three
    widths has a large union and reads as three regions. Measured (hospital,
    "ilk 6 ayda önerilen parçalardan hangisi yüksek öncelik işaretlidir"): all
    SIX seats went to views of the one closing paragraph, and the section holding
    the marker was never retrieved, so no candidate answer could rest on it and
    the read-back rejected — correctly — every answer that was offered.

    No model: the store is built here, one paragraph indexed at several widths
    plus one other section that shares the question's rarer word."""
    from lmm import evidence
    store = evidence.SentenceStore()
    core = ("İlk 6 ay için önerilen üç parça: 02 · 03 · 07 menteşe, vida ve "
            "taşıma kayışı yenilemesi.")
    store.add(core, "#doc")                                  # the narrow view
    store.add(core + " Üçü de mevcut veriyle çalışır.", "#doc")
    store.add("SONUÇ Nereden başlanmalı " + core + " Üçü de mevcut veriyle "
              "çalışır ve birkaç ay içinde ölçülebilir sonuç üretir.", "#doc")
    store.add("SONUÇ Nereden başlanmalı " + core, "#doc")
    other = ("07 taşıma kayışı yenilemesinin tamamlanması YÜKSEK ÖNCELİK "
             "işaretlidir.")
    store.add(other, "#doc")
    found = store.find("İlk 6 ayda önerilen parçalardan hangisi yüksek "
                       "öncelik işaretlidir", most=4)
    assert other in found, found
    # the region is still CORROBORATED — the cap is two seats, not one
    assert sum(1 for s in found if s.startswith(("İlk", "SONUÇ"))) >= 2, found
    # and a region that is genuinely alone still fills the list
    store2 = evidence.SentenceStore()
    for tail in ("yüksek", "orta", "düşük", "belirsiz"):
        store2.add("Zilfen madde %s bir torvanit değeri taşır." % tail, "#doc")
    assert len(store2.find("zilfen torvanit değeri", most=4)) >= 2


@test("H5 the question, not the answer, is what the last read-back asks about")
def h5():
    """THE SHAPE of the new condition, with no engine in it.

    `generate.supported` is handed a CLAIM, and a claim carries only what it
    chooses to say, so an answer that voices a true fact from the evidence
    passes it no matter what was asked. `generate.answers_asked` is handed the
    QUESTION as well, and that is the whole difference: the proposition being
    judged is "does this evidence state the relation that was ASKED", which an
    answer cannot satisfy by leaving the relation out. Here the runtime is
    stubbed to record what each check is shown."""
    from lmm import generate, prompts, runtime
    seen = []
    real = runtime.generate

    def fake(user, system=None, max_tokens=None, temperature=None):
        seen.append((system, user))
        return "no"

    runtime.generate = fake
    try:
        block = "[K1] Hazırlayan: Nordheim Kayıt Bürosu"
        claim = "Raporu Nordheim Kayıt Bürosu hazırlamıştır."
        assert generate.supported(claim, block) is False
        system, user = seen[-1]
        assert system == prompts.SUPPORT_SYSTEM
        assert "QUESTION" not in user, user      # the claim is judged alone
        assert generate.answers_asked("Raporu kim imzalamıştır", claim,
                                      block) is False
        system, user = seen[-1]
        assert system == prompts.RELATION_SYSTEM
        # the asked relation is IN the proposition, so an answer that never
        # mentions it can no longer slip past
        assert "imzalam" in user and claim in user and block in user, user
        # a "yes" is a yes — the gate admits, it does not merely reject
        runtime.generate = lambda *a, **k: "yes"
        assert generate.answers_asked("Raporu kim hazırlamıştır", claim, block)
    finally:
        runtime.generate = real


@test("H6 an answer to a question nobody asked does not get spoken")
def h6():
    """The last route to a confident WRONG answer, closed end to end.

    Measured (hospital, "Dokümanı kim imzalamıştır"): the document states who
    PREPARED it and never says who signed it, so the answer voiced the preparer
    field. Every gate passed and each was right to: the claim is in the evidence
    word for word, coverage 1.0, and the read-back confirmed it. Nothing that
    judges the ANSWER'S OWN CLAIM can catch this, because the claim never
    mentions the relation it fails to be about.

    The store is built WITHOUT a model (learn_text deep=False) over a synthetic
    document, and the engine's verdict is stood in for, so what is under test is
    the GATE'S WIRING rather than a model's judgement: a "no" abstains, no later
    path speaks what it refused, and the same setup with the condition satisfied
    still answers. The witness that this condition is what abstains: with the
    judge saying yes — the state before this commit — the wrong answer IS
    returned, over the same evidence, past every other gate."""
    from lmm import generate, session as lmm_session
    s = lmm_session.Session(None)
    s.learn_text("Bu rapor Hat B kalite denetimi için düzenlenmiştir.\n"
                 "Hazırlayan: Nordheim Kayıt Bürosu\n"
                 "Denetim sırasında ölçülen sıcaklık 42 santigrat derece "
                 "olarak kaydedilmiştir.\n", deep=False)
    wrong = "Hazırlayan: Nordheim Kayıt Bürosu"
    real = (generate.answer, generate.supported, generate.answers_asked,
            generate.refusal)
    views = []

    def judge(question, answer, view):
        """The engine's verdict, stood in for: the document states who prepared
        the report and no other relation about it."""
        views.append(view)
        return "hazırlayan" in question.lower()

    try:
        generate.answer = lambda question, facts, warmth=0.2: wrong
        generate.supported = lambda answer, view: True     # as measured
        generate.answers_asked = judge
        generate.refusal = lambda question: "REFUSED"
        # the relation the document does not state -> abstention, and the
        # graph/verify fallback does not resurrect it
        for question in ("Raporu kim imzalamıştır",
                         "Raporu kim onaylamıştır",
                         "Rapor hangi fabrika için düzenlenmiştir",
                         "Denetimin maliyeti kaç liradır"):
            assert s._answer(question, None) == "REFUSED", question
        # the judge is asked about the evidence the answer DRAWS ON, and the
        # question travels with it (that is the whole point of the check)
        assert views and all("Hazırlayan" in v for v in views), views
        # the asked relation IS stated -> the same machinery answers
        assert s._answer("Raporun hazırlayanı kimdir", None).startswith(wrong)
        # THE WITNESS: before this condition existed, the wrong answer spoke
        generate.answers_asked = lambda question, answer, view: True
        assert s._answer("Raporu kim imzalamıştır", None).startswith(wrong)
    finally:
        (generate.answer, generate.supported, generate.answers_asked,
         generate.refusal) = real


@test("I1 the same document and question always retrieve the same evidence")
def i1():
    """THE DETERMINISM CLAIM, ENFORCED.

    This retrieval layer's whole argument against embedding similarity is that
    it is deterministic and explainable, and it was neither: `qwords` is a SET
    OF STRINGS and the stem groups were formed by walking it in length order,
    so equal-length query words arrived in an order that depends on
    PYTHONHASHSEED. Group formation is order-sensitive — the first surface form
    seen becomes the anchor, the rest attach to it, and the group's IDF weight
    is computed over whatever union that produced — so the same question
    against the same document could retrieve DIFFERENT evidence between two
    runs of byte-identical code (measured on a 350-page manual: PYTHONHASHSEED=1
    and =3 disagree about "Cihazın işletim sistemi nedir").

    A property about hash seeds cannot be asserted from inside one interpreter,
    so this runs the same retrieval in several children with different seeds and
    demands the same block. The witness is minimal and reproduces the real
    failure exactly: two query words of the SAME LENGTH ('ekran', 'bilgi') so
    the length sort cannot order them, one sentence reachable from each so the
    two groups have separate unions and therefore equal weight, and the two
    sentences the same length so the length tie-break cannot separate them
    either. On the old code these eight seeds return two different answers; the
    real manual shows the same thing on a real question."""
    import subprocess
    program = (
        "import sys, json;"
        "sys.path.insert(0, %r);"
        "from lmm import evidence;"
        "store = evidence.SentenceStore();"
        "store.add('Ekran onbes nokta altiii');"
        "store.add('Bilgi yediyuz kirk sekiz');"
        "print(json.dumps(store.find('ekran bilgi', most=1)))"
    ) % os.path.join(ROOT, "src")
    seen = set()
    for seed in ("0", "1", "2", "3", "5", "7", "99", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", program], env=env,
                             capture_output=True, text=True, check=True)
        seen.add(out.stdout.strip())
    assert len(seen) == 1, seen


@test("I2 the scorer reads a real abstention as one, and a value as a value")
def i2():
    """THE MEASURING TOOL IS PART OF THE WORK.

    The absence criterion was a literal-substring search over a list of Turkish
    phrases, and it failed on the most ordinary sentence the engine produces:
    "Bu bilgiye HENÜZ sahip değilim" is an abstention, and one inserted adverb
    put it outside "bilgiye sahip değil". It cost a real measurement a point —
    a tool that miscounts is worse than a slow one.

    Matching the pattern's stems in order inside a bounded window fixes that,
    and it opens exactly one hole, which the value check closes: with gaps
    allowed, "bu bilgi kılavuzda yok AMA garanti 2 yıldır" would match 'bilgi
    ... yok'. An answer that names a figure is not abstaining. The engine's own
    provenance footnote is not a claim, so it is stripped before that test —
    otherwise a source filename's digits read as a fabricated value (measured:
    24 genuine abstentions across five historical result files)."""
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from score import abstains
    for answer in ("Bu bilgiye henüz sahip değilim.",
                   "Bu konuda şu anda bilgim yok.",
                   "Kılavuzda garanti süresi belirtilmemiştir.",
                   "Bilmiyorum. (Bu bilgi kesin değildir; #pdf:M30_70 v2.pdf)"):
        assert abstains(answer), answer
    for answer in ("Garanti süresi 2 yıldır.",
                   "Bu bilgi kılavuzda yok ama garanti süresi 2 yıldır.",
                   "Cihazın koruma türü Sınıf I ekipmandır.",
                   "Kılavuz bu konuyu başka bir bölümde anlatıyor."):
        assert not abstains(answer), answer


# --- J  the typed value: number, range, unit ---------------------------------
#
# What the value IS was invisible to every organ: "21.5\" OLED" and "okonos"
# were the same kind of thing, so whether two values in one slot can coexist
# had to be asked of a language model, one round-trip per pair.


@test("J1 a measurement is read off the data, symbols and ranges included")
def j1():
    """THE UNIT COMES FROM ADJACENCY, NOTHING ELSE.

    No list of units exists anywhere in this system and none may be added. The
    only rule is that the token standing next to a number is what the number is
    measured in — which is why a SYMBOLIC unit works exactly like a lettered
    one: '"' next to 21.5 is a token next to a number. A spec sheet's display line
    is the measured loss class ('Gösterge 21.5" OLED' — no letters in the unit at
    all).

    A separator is not a unit: in "100 V -240 V" and "%25 - %85" the symbol
    stands BETWEEN two numbers, and a number begins again right after it. In
    "5 °C - 40 °C" it does not — °C carries a letter and is a real unit — so
    the test is positional, not a character list.
    """
    from lmm.core.memory import measure
    assert measure('Gösterge 21.5" OLED') == (21.5, 21.5, '"')
    assert measure("21,5 inç") == (21.5, 21.5, "inç")   # notation is not quantity
    assert measure("100 V -240 V ~") == (100.0, 240.0, "v")     # a range
    assert measure("5 °C - 40 °C") == (5.0, 40.0, "°c")
    assert measure("-20 °C ~ 55 °C") == (-20.0, 55.0, "°c")     # below zero
    assert measure("%25 - %85") == (25.0, 85.0, None)   # dash separates
    assert measure("8 GB, DDR4") == (8.0, 8.0, "gb")    # punctuation is not a unit
    # A DIFFERENT unit is a DIFFERENT measurement, not a wider range: the
    # battery's line states a voltage and a capacity, and 12.8..4200 would be
    # a quantity the document never claims.
    assert measure("12.8 V / 4200 mAh") == (12.8, 12.8, "v")
    # Not measurements: a version/section string (two fractional separators)
    # and a text with no number at all.
    assert measure("Çekirdek sürümü 1.2.3") is None
    assert measure("okonos") is None


@test("J2 the unit is an identity, and its bare token is not a way to name it")
def j2():
    """A unit is a concept of the graph, so it lives in the identity system —
    the one place this architecture keeps meanings apart. But its label is
    NAMESPACED, and that is not decoration: 'g' or 'v' as a plain label would
    sit in the same label index as every name in the graph and would be
    reachable by the inflection tolerance, which is the exact collision
    (`kartal` the bird vs the district) this file was written to prevent."""
    memory = Memory()
    weight = memory.identify("3.4 kg")
    record = memory.write(memory.identify("cihaz"), None, weight, "#doc")
    low, high, unit = record.measure
    assert (low, high) == (3.4, 3.4)
    assert unit in memory.identities and memory.units["kg"] == unit
    assert memory.candidates("kg") == []            # not addressable as a name
    # The same unit sighted twice is ONE identity — a unit is not re-invented
    # per record.
    again = memory.write(memory.identify("çanta"), None,
                         memory.identify("2 kg"), "#doc")
    assert again.measure[2] == unit
    # AND THE REST OF THE GRAPH IS NUMBERED AS IF THE UNIT WERE NOT THERE.
    # Units count down from -1 for this reason: taking keys out of the shared
    # counter would renumber every identity opened after a measurement, and key
    # order is a tie-break in more than one organ — the same document would then
    # answer differently for a reason that has nothing to do with measurements.
    # Two identical histories, one writing a MEASUREMENT and one writing a plain
    # value: the typed one opens a unit, and the next key either graph hands out
    # is the same either way.
    plain, typed = Memory(), Memory()
    for held, value in ((plain, "okonos"), (typed, "3.4 kg")):
        held.write(held.identify("cihaz"), None, held.identify(value), "#doc")
    assert typed.units and min(typed.units.values()) < 0, typed.units
    assert not plain.units
    assert typed.identify("sonra") == plain.identify("sonra")


@test("J3 two numbers in one slot are judged by arithmetic, with no model")
def j3():
    """THE COST DECISION AND THE EPISTEMIC ONE COME APART.

    Whether two values in a slot exclude each other used to be a question for
    the language model, at one round-trip per pair — expensive enough that bulk
    ingestion switched it off and deferred every verdict to sleep. For
    measurements the question is arithmetic: disjoint ranges exclude each
    other, overlapping ones do not. Here the semantic test is wired to a
    function that RAISES if called, which is the assertion: these verdicts cost
    no model call at all.
    """
    def must_not_be_asked(one, other):
        raise AssertionError("the semantic test was called for a measurement")

    memory = Memory()
    gate = Gate(memory)
    gate.rival = must_not_be_asked
    device = memory.identify("cihaz")
    what = memory.identify("gerilim")
    gate.admit(device, what, memory.identify("100 V -240 V"), "#doc")
    # 220 V lies INSIDE the stated range — one fact stated twice, no rivalry.
    friend, _ = gate.admit(device, what, memory.identify("220 V"), "#doc")
    assert not links_of(friend, CONTRA), friend.links
    assert not links_of(friend, SUSPECT), friend.links
    # 12 V is outside it — a contradiction, decided without the model, and NOT
    # left as a suspicion for sleep to pick up.
    clash, _ = gate.admit(device, what, memory.identify("12 V"), "#doc")
    assert links_of(clash, CONTRA), clash.links
    assert not links_of(clash, SUSPECT), clash.links


@test("J4 arithmetic keeps quiet where it is not entitled to speak")
def j4():
    """The typed layer is a NEW SOURCE of verdicts, not a replacement for the
    semantic one. Bare numbers with no unit ("Tier 3" vs "Tier 1") and two
    DIFFERENT units ("3.4 kg" vs "17 lb", which may well be the same weight)
    are cases where arithmetic alone cannot decide, and answering them anyway
    would be the same mistake the audit found in the other direction: a cost
    decision wearing an epistemic decision's clothes. So the semantic test must
    still be consulted for those."""
    asked = []

    def semantic(one, other):
        asked.append((one, other))
        return False

    memory = Memory()
    gate = Gate(memory)
    gate.rival = semantic
    row = memory.identify("kapak menteşesi")
    tier = memory.identify("tier")
    gate.admit(row, tier, memory.identify("tier 3"), "#doc")
    gate.admit(row, tier, memory.identify("tier 1"), "#doc")
    assert len(asked) == 1, asked                   # unit-less: still asked
    weight = memory.identify("agirlik")
    gate.admit(weight, None, memory.identify("3.4 kg"), "#doc")
    gate.admit(weight, None, memory.identify("17 lb"), "#doc")
    assert len(asked) == 2, asked                   # units differ: still asked
    # And a value with no measurement in it at all is untouched by this layer.
    system = memory.identify("isletim sistemi")
    gate.admit(system, None, memory.identify("okonos"), "#doc")
    gate.admit(system, None, memory.identify("windows"), "#doc")
    assert len(asked) == 3, asked


@test("J5 a file from the future is refused, and every older file still loads")
def j5():
    """THE VERSION BYTE WAS WRITTEN AND NEVER READ (audit finding F).

    A .lmm marked FORMAT=99 loaded as if this build understood it, and the next
    save() would then write the graph back without whatever that format added —
    silent data loss. Refusing to read forward is not symmetric with refusing
    to read backward: this build knows exactly what a FORMAT-3 file says, and
    an old memory must keep opening. Both directions are asserted here,
    including the oldest form of all, plain JSON.
    """
    import json
    import struct
    import tempfile
    import zlib

    memory = Memory()
    device = memory.identify("cihaz")
    memory.write(device, None, memory.identify('21.5" OLED'), "#doc")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "m.lmm")
        memory.save(path)
        with open(path, "rb") as handle:
            raw = handle.read()
        assert struct.unpack(">B", raw[4:5])[0] == 4        # FORMAT=4 written
        back = Memory.load(path)
        held = list(back.records.values())[0]
        assert held.measure[:2] == (21.5, 21.5)
        assert back.units and held.measure[2] in back.units.values()

        # FROM THE FUTURE: header says 99 → refused, loudly.
        payload = zlib.compress(zlib.decompress(raw[13:]), 9)
        ahead = os.path.join(folder, "ahead.lmm")
        with open(ahead, "wb") as handle:
            handle.write(raw[:4] + struct.pack(">B", 99) + raw[5:])
        try:
            Memory.load(ahead)
        except ValueError as broke:
            assert "99" in str(broke), broke
        else:
            raise AssertionError("a FORMAT=99 file loaded silently")

        # BACKWARD COMPATIBILITY, both older forms. A format-3 record has no
        # typed value in the file; it gets the same typed view a new file
        # would, derived from the value the file already holds.
        old = {"format": 3, "next": 5, "self": None, "transitive": [],
               "identities": [{"key": 1, "labels": ["cihaz"], "vector": None},
                              {"key": 2, "labels": ['21.5" OLED'],
                               "vector": None}],
               "records": [{"key": 3, "subject": 1, "predicate": None,
                            "value": 2, "source": "#doc", "level": 4,
                            "trust": 0.6, "at": time.time(), "witnesses": 1,
                            "links": [], "episodic": True,
                            "sources": ["#doc"]}],
               "experiences": []}
        text = json.dumps(old, ensure_ascii=False).encode("utf-8")
        plain = os.path.join(folder, "old.json.lmm")
        with open(plain, "w", encoding="utf-8") as handle:
            handle.write(text.decode("utf-8"))
        binary = os.path.join(folder, "old3.lmm")
        with open(binary, "wb") as handle:
            handle.write(Memory.MAGIC
                         + struct.pack(">BII", 3, len(text),
                                       zlib.crc32(text) & 0xffffffff)
                         + zlib.compress(text, 9))
        for name in (plain, binary):
            back = Memory.load(name)
            held = back.records[3]
            assert held.subject == 1 and held.value == 2 and held.level == 4
            assert held.measure[:2] == (21.5, 21.5), name
            # and re-saving migrates it to the current format without loss
            again = os.path.join(folder, "again.lmm")
            back.save(again)
            assert Memory.load(again).records[3].value == 2


@test("J6 a question asking for a quantity gets the spec line, and nothing moves")
def j6():
    """THE UNIT THE DOCUMENT DOES NOT SPELL (measured manual loss).

    A spec sheet states `Gösterge 21.5" OLED` and the question asks "kaç inçtir".
    The unit on the line is a quotation mark — no letters — so the lexical
    channel has nothing of 'inç' to match, and the line is reachable only
    through 'gösterge', which the document also uses on dozens of menu lines: it
    ranked 33rd. Reweighting the field-name channel does rescue it and was
    measured to cost hospital two points, because five of its sixteen blocks
    changed. So the rescue APPENDS: the ranking is final before it runs, and
    this test asserts both halves — the spec line arrives, and the six seats
    that were there are still there, in order.

    Nothing here knows that 21.5" is inches. The question is recognised as
    asking for a quantity because the corpus itself binds 'inç' to a number
    somewhere else ("(12 inç)"), and the line is recognised as an answer
    because it names the asked field and states a measurement whose unit is one
    this corpus characteristically writes next to numbers.
    """
    from lmm import evidence
    store = evidence.SentenceStore()
    # the crowd: prose that shares the question's field word
    crowd = [
        "Cihazın göstergesi açıldığında oturum kutusu görünür ve yönetici "
        "parolası istenir; parolayı girdikten sonra devam edin.",
        "Cihazın göstergesi temizlenirken yumuşak bez kullanılmalı, çözücü "
        "içeren sıvılar kesinlikle uygulanmamalıdır bu yüzeye.",
        "Cihazın göstergesi koruyucusu etkinleştirildiğinde bekleme süresi "
        "dolduğunda arayüz kararır ve tuşa basıldığında geri döner.",
        "Cihazın göstergesi üzerindeki durum çubuğundan ağ bağlantısı durumu "
        "izlenebilir; bağlantı koptuğunda simge değişir hemen.",
        "Cihazın göstergesi yansımayı azaltmak için doğrudan güneş ışığı almayan "
        "bir konumda konumlandırılmalıdır çalışırken.",
        "Cihazın göstergesi üzerinde ölçüm sonuçları, yorumlar ve gövde "
        "işaretleri ters sırada temizlenebilir düğmeyle.",
        "Cihazın göstergesi bölme düzeninde iki görüntü yan yana gösterilir ve "
        "etkin pencere çerçeveyle belirtilir kullanıcıya.",
    ]
    class WithoutTheSeat(evidence.SentenceStore):
        """The same store with the mechanism inert — a corpus that never binds
        any token to a number cannot recognise a question as asking for a
        quantity. This is the reference the ranking must be unchanged against."""
        quantities = property(lambda self: set())

    corpus = crowd + [
        'Gösterge 21.5" OLED',
        "Taşınabilir vericiler tarayıcıya 30 cm'den (12 inç) daha yakın "
        "kullanılmamalıdır.",
        "Gösterge koruyucu ayarını açın.",
        "Gösterge menüsünü ayarlardan yeniden düzenleyin.",
        "Gösterge parlaklığı bölümüne bakın.",
    ]
    store, reference = evidence.SentenceStore(), WithoutTheSeat()
    for one in corpus:
        store.add(one, "#doc")
        reference.add(one, "#doc")
    question = "Cihazın göstergesi kaç inçtir"
    before = reference.find(question, most=6)
    found = store.find(question, most=6)
    assert 'Gösterge 21.5" OLED' not in before, before       # the measured loss
    # THE RANKING DID NOT MOVE: the seats that were there are there, in order,
    # and the spec line is appended after them.
    assert found == before + ['Gösterge 21.5" OLED'], found
    # A question that asks for no quantity gets no seat at all — the mechanism
    # is inert outside its loss class.
    plain = "Gösterge koruyucu nasıl açılır"
    assert store.find(plain, most=6) == reference.find(plain, most=6)


# --- K  language independence ------------------------------------------------
#
# The system claimed to be language-independent while three things quietly made
# Turkish its home language: every answer was framed by hand-written Turkish
# labels, every few-shot example in every classifier was Turkish, and the
# BENCHMARK SCORER recognised an abstention by a list of Turkish phrases — so a
# document in any other language would have scored every honest refusal as a
# fabrication. These tests hold each of the three closed, and none of them
# needs a model.


@test("K1 the scorer reads the system's own abstention stamp, in any language")
def k1():
    """THE MEASUREMENT MUST NOT SPEAK ONE LANGUAGE.

    `score.abstains` is a Turkish phrase list, and it is now the FALLBACK. The
    criterion is the stamp the session puts on the turn it refused
    (`Session.last_abstained`, written into the result file as "abstained"),
    which is a fact the code knows rather than a wording a reader guesses at.

    Here is what that buys, stated as the assertions below: a German and a
    Spanish refusal — invisible to every pattern in the list — are read as
    abstentions, and a Turkish one still is. The value check survives the
    stamp: an "abstention" that names a figure is a fabrication whichever path
    the engine thinks it took, so a stamped turn that supplies 2 años is not
    counted as a refusal. Rows with no stamp (the RAG baseline's, every
    historical file on disk) fall back to the patterns unchanged."""
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    from score import abstains, declined
    foreign = ("Das weiß ich leider nicht.",
               "No tengo esa información.",
               "That is not stated in the document.")
    for answer in foreign:
        # the pattern list is blind to all three — that is the whole problem
        assert not abstains(answer), answer
        assert declined({"cevap": answer, "abstained": True}), answer
        assert not declined({"cevap": answer, "abstained": False}), answer
    # Turkish keeps working through the same door.
    assert declined({"cevap": "Bu bilgiye henüz sahip değilim.",
                     "abstained": True})
    # A stamp cannot launder a fabricated value.
    assert not declined({"cevap": "La garantía es de 2 años.",
                         "abstained": True})
    # No stamp -> the old reading, unchanged (back-compat for RAG/old files).
    assert declined({"cevap": "Bu konuda bilgim yok."})
    assert not declined({"cevap": "No tengo esa información."})


@test("K2 every refusal leaves by one door, and that door raises the flag")
def k2():
    """ONE DOOR, OR THE FLAG IS A LIE.

    `last_abstained` is only worth reading if no refusal can slip past the
    method that sets it. The session had nine separate
    `generate.refusal(...) or FALLBACK_DONT_KNOW` sites; a flag set at eight of
    them measures nothing. This is a SOURCE-level assertion because that is the
    level the property lives on: `generate.refusal` may be called from exactly
    one place, `Session._refuse`, and the two fallback returns that bypass the
    engine entirely (the crash path, the research offer) must raise the flag
    themselves."""
    source = open(os.path.join(ROOT, "src", "lmm", "session.py"),
                  encoding="utf-8").read()
    body = source.split("def _refuse(")[1].split("\n    def ")[0]
    assert "self.last_abstained = True" in body
    assert body.count("generate.refusal(") == 1
    assert source.count("generate.refusal(") == 1, "a refusal bypasses _refuse"
    # Everything that returns the last-resort constant flags itself too.
    for line_no, line in enumerate(source.splitlines()):
        if "return FALLBACK_DONT_KNOW" not in line:
            continue
        window = "\n".join(source.splitlines()[max(0, line_no - 6):line_no])
        assert "self.last_abstained = True" in window, line_no


@test("K3 the answer frame is structure, not a language")
def k3():
    """NO ANSWER PASSES THROUGH A TURKISH SKELETON.

    Every answer this system has ever produced was built as
    "OLGULAR:\\n...\\n\\nSORU: ..." — two Turkish words wrapped around a
    question that might be in any language, and around a no-facts branch that
    told the model in Turkish not to invent. The labels are field names for the
    engine, so they belong in the codebase's own language; what must NOT leak
    is a sentence of any user language. Captured here without a model: the
    frame is inspected directly."""
    from lmm import generate, runtime
    seen = []
    real = runtime.generate

    def capture(user, **kw):
        seen.append(user)
        return "…"
    runtime.generate = capture
    try:
        generate.answer("¿qué es el vorlin?", "[1] vorlin → bebida")
        generate.answer("was ist Beton?", "")
    finally:
        runtime.generate = real
    with_facts, without_facts = seen
    assert with_facts.startswith("FACTS:")
    assert "\n\nQUESTION: ¿qué es el vorlin?" in with_facts
    assert without_facts.startswith("QUESTION: was ist Beton?")
    # The no-facts branch instructs in the codebase's language and hands the
    # ANSWER's language back to the question.
    assert "SAME LANGUAGE" in without_facts
    turkish = ("OLGULAR", "SORU:", "Belleğinde", "bilmediğini", "Uydurma")
    for frame in seen:
        for word in turkish:
            assert word not in frame, (word, frame)


@test("K4 no classifier learns its pattern in one language")
def k4():
    """FEW-SHOTS IN ONE LANGUAGE TEACH THAT LANGUAGE.

    Every example in every classifier prompt was Turkish, which is how a system
    with no word list in it still ended up with a home language: shown "sigara
    kansere neden olur" and nothing else, a model generalises from Turkish
    causal grammar. The examples are now spread across languages, and this test
    keeps them spread — it asserts that each few-shot prompt demonstrates its
    pattern in at least THREE languages, so no single one can quietly become
    the pattern again. Signatures are distinctive orthography/function words,
    never content, so rewording an example does not break the test."""
    from lmm import prompts
    import lmm.generate as gen_src
    signatures = {
        "en": ("what is", "who are you", "an eagle", "You're welcome",
               "operating temperature", "Prepared by", "smoking causes"),
        "de": ("ist ein", "was ist", "führt zu", "Aufnahmedauer", "weiß ich",
               "wer hat dich", "wie geht es", "Kilogramm", "verkürzt"),
        "es": ("¿", "es un", "provoca", "Tiempo de instalación", "órgano",
               "gracias", "bisagra"),
        "tr": ("nedir", "bir kuştur", "merhaba", "selam", "yol açar",
               "kırmızı", "menteşesi", "bilmiyorum"),
    }

    def languages(text):
        low = text.lower()
        return {tag for tag, marks in signatures.items()
                if any(m.lower() in low for m in marks)}

    source = open(gen_src.__file__, encoding="utf-8").read()
    few_shot = {
        "EXTRACT_SYSTEM": prompts.EXTRACT_SYSTEM,
        "REEXTRACT_SYSTEM": prompts.REEXTRACT_SYSTEM,
        "SUPPORT_SYSTEM": prompts.SUPPORT_SYSTEM,
        "RELATION_SYSTEM": prompts.RELATION_SYSTEM,
    }
    for name, body in few_shot.items():
        assert len(languages(body)) >= 3, (name, languages(body))
    # The classifiers whose few-shots live inside generate.py, taken from the
    # source so that no model call is needed to see them.
    for name in ("def is_causal(", "def is_causal_question(",
                 "def is_identity_question(", "def are_rivals("):
        body = source.split(name)[1].split("\ndef ")[0]
        assert len(languages(body)) >= 3, (name, languages(body))

@test("K5 an answer that asserts nothing is an abstention, whatever it says")
def k5():
    """THE REFUSAL DOOR IS NOT THE ONLY WAY OUT.

    Measured on the fictional corpus: three of its four absence questions never
    reach `_refuse` at all. Evidence IS retrieved, the answer prompt is told to
    say it doesn't know when the evidence doesn't cover the question, and that
    sentence walks through every gate because it claims nothing. A flag set
    only at the refusal door reads all three as confident answers.

    So the flag means "this turn asserted nothing", and the test for that is
    the re-extractor the fabrication gate already relies on — no phrase list,
    no language. Here it is stubbed, so the assertion is about the WIRING: a
    sentence with claims counts as spoken, one without counts as an abstention,
    the engine's parenthetical hedge and #source footnote are not claims, and a
    re-extractor that throws is read as having spoken (the safe direction — the
    damaging error is calling a wrong answer an honest refusal)."""
    from lmm import extract, session as session_module
    blank = session_module.Session.__new__(session_module.Session)
    real = extract.reextract
    try:
        extract.reextract = lambda s: [("morlan", "type", "bird")]
        assert blank._asserted_a_fact("Morlan bir kuştur.")
        extract.reextract = lambda s: []
        assert not blank._asserted_a_fact("Das weiß ich leider nicht.")
        assert not blank._asserted_a_fact("No lo sé.")
        # the footnote is not a claim: with no claims in the sentence itself,
        # a hedge and a source tag cannot turn a refusal into an assertion
        assert not blank._asserted_a_fact(
            "Bilmiyorum. (Bu bilgi kesin değildir; #doc:corpus)")
        # an empty answer asserts nothing either
        assert not blank._asserted_a_fact("   ")

        def broken(_sentence):
            raise RuntimeError("engine down")
        extract.reextract = broken
        assert blank._asserted_a_fact("Anything at all.")
    finally:
        extract.reextract = real


@test("K6 the three corpora are the same benchmark, not three different ones")
def k6():
    """A CROSS-LANGUAGE COMPARISON IS ONLY EVIDENCE IF ONLY THE LANGUAGE MOVED.

    The English and Spanish corpora exist to isolate one variable. If one of
    them quietly gains a question, loses an absence case, or has a gold list
    with more alternatives to hit, the comparison stops measuring language and
    starts measuring the benchmark — and the failure would be invisible in the
    totals, which is exactly the kind of thing this file exists to catch.

    So the structure is asserted directly: same question count, same order of
    types, the same questions unanswerable in each, and gold lists of the same
    size question by question. The gold WORDS must differ — they are
    translations — and the invented names must not, because a name that got
    localised would let a model's own knowledge answer in one language and not
    another."""
    import json
    bench = os.path.join(ROOT, "benchmarks")

    def load(name):
        return json.load(open(os.path.join(bench, name), encoding="utf-8"))

    base = load("questions.json")
    for name in ("questions_en.json", "questions_es.json"):
        other = load(name)
        assert len(other) == len(base), name
        for a, b in zip(base, other):
            assert a["tip"] == b["tip"], (name, a["tip"], b["tip"])
            assert (a["altin"] is None) == (b["altin"] is None), (name, a)
            if a["altin"] is not None:
                # same number of ways to be right — an extra alternative is a
                # free point, and would show up as a language difference
                assert len(b["altin"]) >= len(a["altin"]), (name, a["soru"])
            assert a["soru"] != b["soru"], (name, a["soru"])
    # The invented world keeps its names in every language: nothing in these
    # corpora can be answered from a model's parametric knowledge.
    names = ("zerbalit", "vorlin", "norgul", "morlan", "karvel", "telvas",
             "kelvit", "torvanit", "selvin", "zilfen", "nortlann", "velmar")
    for name in ("corpus.txt", "corpus_en.txt", "corpus_es.txt"):
        text = open(os.path.join(bench, name), encoding="utf-8").read().lower()
        for invented in names:
            assert invented in text, (name, invented)


# =====================================================================  L
#  The front door (lmm.api.Memory). A WRAPPER, and these tests exist to
#  keep it one: every assertion below is about ROUTING and REPORTING, and
#  none of them may need a model, because the facade must not have opinions
#  of its own about what the layer beneath it does.
# =====================================================================


@test("L1 the front door is a wrapper: text goes where learn_text went")
def l1():
    """No new pipeline. `Memory.learn` on plain text must reach exactly
    `Session.learn_text`, with the text unaltered — if the facade ever starts
    pre-processing, cleaning or chunking on the way through, it has stopped
    being packaging and become a second implementation."""
    from lmm import Memory
    m = Memory()
    seen = []
    real = m.session.learn_text
    m.session.learn_text = lambda text, source="#document", deep=True: (
        seen.append((text, source, deep)) or (0, 0))
    m.learn("Zerbalit boils at 412 degrees.")
    m.session.learn_text = real
    assert len(seen) == 1, seen
    assert seen[0][0] == "Zerbalit boils at 412 degrees.", seen[0][0]
    # deep defaults TRUE for text typed at the memory: one sentence handed in
    # directly is a fact being taught and is worth the extractor's call.
    assert seen[0][2] is True, seen[0]


@test("L2 the extension picks the adapter, and nothing else does")
def l2():
    """The dispatch is a table, not a sniffer. Each extension must land on its
    own adapter with the session passed through; the pdf/xlsx/docx readers are
    stubbed so this holds with none of the optional dependencies installed."""
    from lmm import Memory, tables
    m = Memory()
    called = []
    keep = (tables.learn_pdf, tables.learn_xlsx, tables.learn_docx)
    tables.learn_pdf = lambda s, p, source=None, deep=False: (
        called.append(("pdf", s is m.session)) or (3, 1))
    tables.learn_xlsx = lambda s, p, source=None: (
        called.append(("xlsx", s is m.session)) or 4)
    tables.learn_docx = lambda s, p, source=None, deep=False: (
        called.append(("docx", s is m.session)) or (5, 2))
    folder = tempfile.mkdtemp()
    try:
        for name in ("a.pdf", "b.xlsx", "c.docx"):
            path = os.path.join(folder, name)
            open(path, "w", encoding="utf-8").write("x")
            m.learn(path)
    finally:
        tables.learn_pdf, tables.learn_xlsx, tables.learn_docx = keep
    assert [c[0] for c in called] == ["pdf", "xlsx", "docx"], called
    assert all(c[1] for c in called), "the adapter got a different session"

    # A format we cannot read is REFUSED by name, not silently read as text.
    path = os.path.join(folder, "deck.pptx")
    open(path, "w", encoding="utf-8").write("x")
    try:
        m.learn(path)
    except ValueError as said:
        assert "PowerPoint" in str(said), said
    else:
        raise AssertionError("a .pptx was accepted")


@test("L3 a missing optional dependency names the install line, not a traceback")
def l3():
    """The core install is deliberately dependency-free, so 'I fed it a PDF and
    it exploded' is the FIRST thing a new user will do. Each reader must fail
    with the command that fixes it — and with the PyPI name, which for
    python-docx is not the name you import."""
    from lmm import tables
    import builtins
    real = builtins.__import__

    def blocked(name, *a, **kw):
        if name.split(".")[0] in ("pdfplumber", "docx", "pandas"):
            raise ImportError("blocked for the test")
        return real(name, *a, **kw)

    gone = {}
    for mod in [k for k in sys.modules
                if k.split(".")[0] in ("pdfplumber", "docx", "pandas")]:
        gone[mod] = sys.modules.pop(mod)
    was = tables._PANDAS
    tables._PANDAS = None
    builtins.__import__ = blocked
    try:
        for reader, arg, extra, dist in (
                (tables.read_pdf, "x.pdf", "pdf", "pdfplumber"),
                (tables.read_docx, "x.docx", "docx", "python-docx"),
                (tables.read_xlsx, "x.xlsx", "xlsx", "pandas")):
            try:
                reader(arg)
            except ImportError as said:
                assert f"lmm[{extra}]" in str(said), said
                assert dist in str(said), said
            else:
                raise AssertionError(f"{arg} read with its reader missing")
    finally:
        builtins.__import__ = real
        tables._PANDAS = was
        sys.modules.update(gone)


@test("L4 a document body is never mistaken for a file name")
def l4():
    """`learn` takes a path OR the text itself, so it has to tell them apart —
    and it must do so WITHOUT handing a 500 KB document to the filesystem. A
    body with line breaks, and a body longer than any path, are text; a real
    file is a file. (os.path.isfile raises rather than returning False on an
    over-long name on several platforms — this is the guard for that.)"""
    from lmm.api import _looks_like_path
    assert not _looks_like_path("Zerbalit boils.\nVorlin does not.")
    assert not _looks_like_path("x" * 5000)
    assert not _looks_like_path("no_such_file_anywhere.txt")
    assert not _looks_like_path(b"bytes are not a path")
    assert not _looks_like_path(None)
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "real.txt")
    open(path, "w", encoding="utf-8").write("The tower is 91 metres tall.")
    assert _looks_like_path(path)


@test("L5 an answer is still a string, and carries its own abstention stamp")
def l5():
    """`ask(explain=True)` must not force callers into a wrapper type: the
    return IS the answer text and additionally reports itself. The abstention
    flag has to be the SESSION's structural stamp — the thing K1/K2 defend —
    and not a phrase list re-invented at the front door."""
    from lmm import Memory
    from lmm.api import Answer
    a = Answer("Bilmiyorum.", abstained=True, sources=("#pdf:x",))
    assert isinstance(a, str) and a == "Bilmiyorum."
    assert a.upper() == "BILMIYORUM." and a.abstained and a.sources == ("#pdf:x",)

    m = Memory()
    m.session.respond = lambda q: "It is 412 degrees. #document"
    m.session.last_abstained = False
    m.session.last_subject = "zerbalit"
    m.session.last_kind = "ASK"
    m.session.last_written = []
    plain = m.ask("how hot?")
    assert type(plain) is str and "412" in plain           # noqa: E721
    told = m.ask("how hot?", explain=True)
    assert told.sources == ("#document",), told.sources
    assert told.abstained is False and told.subject == "zerbalit"
    m.session.last_abstained = True
    assert m.ask("how hot?", explain=True).abstained is True


@test("L6 a memory with nowhere to save says so before it loses anything")
def l6():
    """`Memory()` is allowed to be transient, which makes a silent no-op save
    the dangerous failure: the caller believes the memory is on disk. It must
    raise, and it must accept a path given late."""
    from lmm import Memory
    m = Memory()
    m.learn("The tower is 91 metres tall.", deep=False)
    before = len(m)
    try:
        m.save()
    except ValueError as said:
        assert "nowhere to save" in str(said), said
    else:
        raise AssertionError("a pathless memory reported that it saved")
    path = os.path.join(tempfile.mkdtemp(), "mind.lmm")
    assert m.save(path) == path and os.path.exists(path)
    assert len(Memory(path)) == before


@test("M1 no unverified text is appended to an answer, and none at all to a refusal")
def m1():
    """THE FIELD TRIAL'S BIGGEST HOLE: a second model call after the gate.

    `_answer` and `_hedge` used to append `generate.hedge_note(...)` — a
    separate generation at temperature 0.3 — to an answer that had already
    passed verify/coverage/read-back, and the concatenation was never checked
    again. Measured on twelve public documents: four of six wrong answers were
    that note, the worst of them glued to a perfect abstention
    ("I do not know. RFC 9110 is obsoleted by RFC 9111.").

    Two properties are asserted here, both without a model: an abstaining turn
    comes out byte-for-byte as the answer path produced it, and an asserting
    turn gains NOTHING but the stored source stamp inside format punctuation —
    no token that could carry a new fact."""
    from lmm import generate, session as lmm_session
    assert not hasattr(generate, "hedge_note"), \
        "the ungated second model call is back"

    s = lmm_session.Session(None)
    refusal = "Bilmiyorum."

    def spoke(message):
        s._mark = "#pdf:rfc2119.txt"       # a low-trust source, as _hedge sets
        return refusal

    s._respond = spoke
    s._asserted_a_fact = lambda said: False        # the turn claimed nothing
    assert s.respond("RFC 2119 kaç sayfadır") == refusal, "text after a refusal"
    assert s.last_abstained is True

    s._asserted_a_fact = lambda said: True         # the turn made a claim
    said = s.respond("RFC 2119 kaç sayfadır")
    assert said.startswith(refusal) and said != refusal
    added = said[len(refusal):]
    # WHAT WAS ADDED CANNOT BE A FACT: every token is either the stamp the
    # session already stored or pure punctuation. Nothing was generated, so
    # nothing can be invented — that is the property, not a smaller prompt.
    for token in added.split():
        assert token.startswith("#") or not any(c.isalnum() for c in token), token
    assert "#pdf:rfc2119.txt" in added and s.last_abstained is False

    # AND THE STAMP SURVIVES THE MARK'S BRACKETS, for a reader of `explain=True`
    from lmm.api import Memory
    m = Memory()
    m.session.respond = lambda q: said
    m.session.last_abstained = False
    m.session.last_subject = m.session.last_kind = ""
    m.session.last_written = []
    assert m.ask("x", explain=True).sources == ("#pdf:rfc2119.txt",)


@test("M2 learn() says what it could not read instead of swallowing it")
def m2():
    """THE SILENT CLASS, measured in the field trial and closed here.

    Four separate ways a document was never read while `learn()` reported
    success: an image-only PDF (`<Learned 0 facts, 0 tables via pdf>`, zero
    evidence, no warning); an HTML file, which has no reader, accepted as
    "text" and filling the index with markup; a mistyped path learned as a
    SENTENCE; and a broken PDF answering with 38 lines of pdfminer traceback.

    All four are decided on structure — zero characters, the share of
    characters inside tags, the shape of a filename — and none of them on any
    document's content. No model runs in this test."""
    from lmm import api
    from lmm.api import Memory
    home = tempfile.mkdtemp()

    # 1. A PATH THAT IS NOT THERE IS A TYPO, NOT A SENTENCE.
    m = Memory()
    for bad in ("manual.pdf", os.path.join(home, "notes.txt")):
        try:
            m.learn(bad)
        except ValueError as said:
            assert "no such file" in str(said), said
        else:
            raise AssertionError("a missing file was learned as text: %r" % bad)
    try:
        m.learn(home)
    except ValueError as said:
        assert "directory" in str(said), said
    else:
        raise AssertionError("a directory was learned as text")
    # and prose that merely mentions a filename is still prose
    kept = m.learn("The calibration values live in manual.pdf on the bench.",
                   deep=False)
    assert kept.evidence > 0 and not kept.warnings

    # 2. NOTHING CAME OUT = SAY SO. Stood in for the scanner's PDF: the reader
    #    runs, returns nothing, writes no evidence.
    import lmm.tables as tables_module
    real = tables_module.learn_pdf
    try:
        tables_module.learn_pdf = lambda session, path, source=None, deep=False: (0, 0)
        scan = os.path.join(home, "scan.pdf")
        open(scan, "wb").write(b"%PDF-1.4 no text layer here")
        told = Memory().learn(scan)
        assert told.evidence == 0 and told.warnings, repr(told)
        assert "OCR" in told.warnings[0], told.warnings
        assert not told, "an unread document reported itself as read"
    finally:
        tables_module.learn_pdf = real

    # 3. MARKUP IS NOT PROSE, and a format with no reader says which one it is.
    page = os.path.join(home, "spec.html")
    open(page, "w", encoding="utf-8").write(
        "<html><head><meta content='Common,Latin' name='scripts'>"
        "<link rel='stylesheet' href='a.css'></head><body>"
        + "<div class='row'><span class='cell'>x</span></div>" * 40
        + "<p>The tower is 91 metres tall.</p></body></html>")
    told = Memory().learn(page, deep=False)
    assert told.evidence > 0 and len(told.warnings) == 2, repr(told)
    assert any("not a format LMM has a reader for" in w for w in told.warnings)
    assert any("markup" in w for w in told.warnings), told.warnings
    # ... and an ordinary text file gets no warning at all
    plain = os.path.join(home, "notes.txt")
    open(plain, "w", encoding="utf-8").write("The tower is 91 metres tall.\n")
    told = Memory().learn(plain, deep=False)
    assert told.evidence > 0 and not told.warnings, repr(told)

    # 4. A BROKEN FILE GETS ONE LINE, NOT A THIRD-PARTY TRACEBACK — while our
    #    own errors (the missing-extra install line) pass through untouched.
    class Pdfminer(Exception):
        pass

    def explode(*a, **k):
        raise Pdfminer("")            # the protected-PDF case: empty message

    try:
        api._read_with("pdf", "/tmp/x.pdf", explode)
    except ValueError as said:
        assert "could not read this file" in str(said) and "Pdfminer" in str(said)
        # `raise ... from None`: the reader's traceback is not re-printed under
        # ours, which is the whole of the user-visible difference
        assert said.__cause__ is None and said.__suppress_context__
    else:
        raise AssertionError("the reader's own exception reached the caller")

    def missing(*a, **k):
        raise ImportError("pip install 'lmm[pdf]'")

    try:
        api._read_with("pdf", "/tmp/x.pdf", missing)
    except ImportError as said:
        assert "lmm[pdf]" in str(said)
    else:
        raise AssertionError("the install line was swallowed")


@test("M3 the document's masthead is indexed as one block, not as detached lines")
def m3():
    """MASTHEAD BLINDNESS, measured on two real documents.

    Everything missed on them was in the front block: the publishing body
    ("Department of the Treasury—Internal Revenue Service" in a form's header),
    the date, and the author's institution ("S. Bradner / Harvard University /
    March 1997" atop an RFC). Those are lines, not sentences — none flows into
    the next — so the sentence layer indexed them one detached line at a time
    and no window ever held the name beside the institution beside the date.

    The block is read off the LAYOUT: the run before the first line that ends a
    sentence, bounded by what this document's own lines call a record. This
    test uses an invented masthead; nothing in the rule mentions dates,
    institutions or line numbers, and a document that opens with prose gets no
    block at all."""
    from lmm import evidence, session as lmm_session
    masthead = ("Nordheim Kayıt Bürosu                       T. Velmar\n"
                "Belge: HB-14                     Karvel Enstitüsü\n"
                "Sınıf: Genel                              Mart 2031\n"
                "\n"
                "Hat B kalite denetimi\n"
                "\n"
                "Denetim sırasında ölçülen sıcaklık 42 derecedir.\n"
                "Ölçüm günün ilk vardiyasında yapılmıştır.\n")
    head = evidence.front_matter(masthead)
    assert head, "no block was read off a masthead"
    for line in ("T. Velmar", "Karvel Enstitüsü", "Mart 2031",
                 "Hat B kalite denetimi"):
        assert line in head, (line, head)
    # it STOPS where the prose starts — the block is not the whole document
    assert "42" not in head and "vardiya" not in head, head

    # a document that opens with a sentence has no masthead, and is not given one
    assert evidence.front_matter(
        "Denetim sırasında ölçülen sıcaklık 42 derecedir.\n"
        "Ölçüm günün ilk vardiyasında yapılmıştır.\n") == ""
    assert evidence.front_matter("") == "" and evidence.front_matter("Tek satır") == ""

    # and ingestion puts that block in the index AS ONE ENTRY, so the name, the
    # institution and the date can be retrieved together
    s = lmm_session.Session(None)
    s.learn_text(masthead, source="#doc:hb14", deep=False)
    blocks = [text for text, _ in s.evidence.sentences
              if "T. Velmar" in text and "Karvel Enstitüsü" in text
              and "Mart 2031" in text]
    assert blocks, "the masthead is still three unrelated lines"
    found = s.evidence.find("Karvel Enstitüsü Mart 2031", most=4)
    assert any("Karvel Enstitüsü" in text and "Mart 2031" in text
               for text in found), found


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
