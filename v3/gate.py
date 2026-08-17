"""The epistemic gate: the ONLY way in and out of memory.

It is a two-way gate and asks two questions:

    ON ENTRY   should this claim be written — what is its source, does it
               contradict, does it have a witness
    ON EXIT    should this sentence be spoken — does every claim in it have
               a record

The second is the project's fourth condition: let there be no path to an
unsupported claim. A language model has no such gate and cannot have one —
there, knowledge is buried in the weights, so the question "what does this
sentence rest on" is uncomputable. Here, record keys stand behind every
sentence.

Nothing belonging to language exists here: no word, no pattern, no suffix.
The gate takes claims as RECORDS, not as sentences. Turning a sentence into a
record is the reader network's job; the gate looks only at records.

THE ONE-WAY RULE — absolute: geometry cannot write a record. The continuous
layer produces candidates, the discrete layer decides. Once this rule is
relaxed even once, the system opens itself to confabulation, and then what
remains is a bad copy of an LLM.
"""
from v3.memory import CONTRA, IF, INFERRED, STRANGER, SUSPECT

# The minimum trust a claim needs to be speakable. Those below it stay in
# memory — they are not discarded, because they have a source — but they do
# not count when speaking.
SPEAK = 0.4

# A stranger's word that nobody has verified. It stays in memory, it is
# marked in the answer. Measured in the old memory: unmarked, a single
# stranger could change the graph's voice with a single sentence.
UNVERIFIED = 1


class Gate:
    """The control standing in front of memory. Memory cannot be written directly from outside."""

    def __init__(self, memory):
        self.memory = memory
        # Optional semantic RIVAL check:
        # (old_value_key, new_value_key) -> True | False | None.
        # In LMM the session wires this to Qwen. Two different values in the
        # same slot count as a contradiction only if they are RIVALS (mutually
        # exclusive) — otherwise coexisting facts like "kartal→kuş,
        # kartal→yırtıcı" were mistaken for contradictions and broken by
        # arbitrate.
        #
        # THE VERDICT IS THREE-VALUED, and the third value is the one the
        # audit found missing:
        #
        #     True   rivals            -> CONTRA, arbitrate now
        #     False  they coexist      -> no link
        #     None   CANNOT JUDGE NOW  -> SUSPECT, sleep judges it in bulk
        #
        # PENDING exists because the test is not always affordable at write
        # time (bulk table ingestion: one model call per cell). Before, "not
        # affordable" was returned as False and the doubt was DESTROYED, not
        # deferred — paris and berlin landed in the same slot with no link, no
        # pressure, and both speakable. A cost decision must never look like
        # an epistemic decision.
        self.rival = None

    # --- entry ----------------------------------------------------------

    def admit(self, subject, predicate, value, source, level=None,
              episodic=True):
        """Tries to put a claim into memory.

        Returns: (record | None, reason). The reason is a NUMBER — not
        language:
            0  written or reinforced
            1  written, but it LOST the arbitration in its slot
            2  source unrecognized

        The record is written IN EVERY CASE (a claim with a source is never
        thrown away); what changes is whether it can speak. The trust of the
        loser is set by ARBITRATION, not here — see `dynamics.arbitrate`. It
        used to be set here too, with a rule that read `clash.trust >= trust`,
        and that rule was ORDER DEPENDENT: teaching the same two facts in the
        opposite order gave opposite outcomes, and in one order BOTH sides
        stayed speakable. One decision, one place.
        """
        if source is None:
            return None, 2
        from v3 import dynamics
        from v3.memory import DOCUMENT
        level = DOCUMENT if level is None else level
        rivals, suspects = self._rivals(subject, predicate, value)
        held = self.memory.write(subject, predicate, value, source, level,
                                 episodic=episodic)
        # THE LINKS ARE MADE IN EVERY CASE. The audit caught it: when the new
        # claim was the stronger one no link was made at all, and
        # `arbitrate`/`pressure` never saw that contradiction — the place
        # where the system should have been uneasy stayed unrecorded.
        for other in rivals:
            if other.key == held.key:
                continue
            self.memory.link(held.key, other.key, CONTRA)
            self.memory.link(other.key, held.key, CONTRA)
        # A SUSPICION IS A MARKER, NOT A RELATION. A judged rivalry is a fact
        # about two particular claims, so every such pair is written down; an
        # unjudged one only has to say "THIS SLOT holds something unsettled",
        # and one link says that no matter how many values are waiting.
        # `judge_suspects` reads the marker and then judges the SLOT, not the
        # link topology, so nothing is lost by writing one instead of k.
        #
        # The cost is the reason, and it was measured on a table whose first
        # column repeats — the whole file lands in one slot. Linking every
        # pending pair there cost ~2M links; walking the field per write to
        # link only what was unconnected cost a closure per cell, which is the
        # same quadratic wearing a different coat. One marker is O(1).
        if suspects and not held.links:
            other = suspects[0]
            self.memory.link(held.key, other.key, SUSPECT)
            self.memory.link(other.key, held.key, SUSPECT)
        if not rivals:
            return held, 0
        winner = dynamics.arbitrate(self.memory, held)
        return held, (0 if winner is held else 1)

    def _rivals(self, subject, predicate, value):
        """The COMPLETE rival set of a slot — and the suspicions.

        The slot is (subject, predicate); within it, another value is a rival
        only if the rivalry test says so. Returns two lists:
        (judged rivals, pending suspicions).

        TWO FIXES, both measured:

        THE SET, NOT THE FIRST. This used to return the FIRST clash and stop.
        With three values in one slot the third was linked only to the first,
        the second and third never met, and arbitration saw a broken star
        instead of the field it was supposed to judge.

        PENDING IS NOT "NO". A `None` verdict means the test could not run
        (bulk ingestion, no engine), and that is a separate answer from "they
        coexist": it becomes a SUSPECT link that sleep judges later.

        If `rival` is not wired (v3), a slot match alone counts as a
        contradiction (legacy). The rivalry test applies to ALL predicates —
        it used to live only in the predicate=None branch, so an explicit
        predicate ("tür", is-a) skipped it and counted lion felid/mammal as a
        false contradiction.
        """
        rivals, suspects = [], []
        for held in self.memory.about(subject, touch=False):
            if held.predicate != predicate or held.value == value:
                continue
            if self._chained(held.value, value, predicate):
                continue          # hierarchy, not rivalry — see below
            verdict = self._measured(held.value, value)
            if verdict is None:
                verdict = True if self.rival is None else self.rival(held.value,
                                                                     value)
            if verdict is None:
                suspects.append(held)
            elif verdict:
                rivals.append(held)
        return rivals, suspects

    # How many rivalry questions this gate answered by ARITHMETIC, split by
    # verdict, and how many it had to hand to the semantic test. Kept because
    # the cost of the semantic test is the reason bulk ingestion switched it
    # off, and a claim about that cost has to be measurable.
    decided = {"excluded": 0, "compatible": 0, "asked": 0}

    def _measured(self, one, other, count=True):
        """Rivalry BY ARITHMETIC — True, False, or None for "not my call".

        Two values in one slot that both state a measurement IN THE SAME UNIT
        need no language model and no deferral: disjoint numbers exclude each
        other, overlapping ones do not. 100..240 V and 220 V are one fact
        stated twice; 220 V and 12 V are a contradiction. That verdict is
        available at write time, for free, in bulk, offline — which is exactly
        what the semantic test is not.

        None is returned whenever the arithmetic is not entitled to speak: a
        value with no measurement in it, a bare number with no unit, or two
        different units (see `memory.overlap`). Then the semantic test decides
        as it always did — this is a new source of verdicts, not a replacement.
        """
        from v3.memory import overlap
        get = getattr(self.memory, "measure_of", None)
        if get is None:                                     # pragma: no cover
            return None
        room = overlap(get(one), get(other))
        if count:
            Gate.decided["asked" if room is None
                         else ("compatible" if room else "excluded")] += 1
        return None if room is None else not room

    def _chained(self, one, other, predicate, depth=6):
        """Are these two values links of the SAME CHAIN under this predicate.

        On a predicate the graph has PROVEN transitive (A tür B, B tür C ⊢
        A tür C), a value reachable from another by that same predicate is its
        ancestor or its descendant — kuş and hayvan for kartal. Such values do
        not compete for the slot, they stack in it: "an eagle is a bird" and
        "an eagle is an animal" are both true at once, and calling that a
        contradiction would make the system's own derivations contradict their
        own premises. Rivalry means MUTUAL EXCLUSION, and a chain is the
        structural proof that there is none.

        This is graph structure, not a rule about any particular predicate:
        the transitivity itself was learned from data, and only a predicate
        that earned it is read this way. The LMM session had the same
        reasoning wired for the semantic test (`_connected`); here it holds
        for every caller, including the derivation engine.
        """
        if predicate is None or predicate not in self.memory.transitive:
            return False
        for start, goal in ((one, other), (other, one)):
            seen, frontier = {start}, [start]
            for _ in range(depth):
                nxt = []
                for node in frontier:
                    for key in self.memory.by_subject.get(node, ()):
                        held = self.memory.records[key]
                        if held.predicate != predicate:
                            continue
                        if held.value == goal:
                            return True
                        if held.value not in seen:
                            seen.add(held.value)
                            nxt.append(held.value)
                if not nxt:
                    break
                frontier = nxt
        return False

    def _contradiction(self, subject, predicate, value):
        """The first JUDGED rival in the slot — None if there is none.

        Kept as the single-answer form for the callers that only ask "is there
        a clash to surface to the user"; the arbitration itself works on the
        whole set (`_rivals`).
        """
        rivals, _suspects = self._rivals(subject, predicate, value)
        return rivals[0] if rivals else None

    # --- exit -----------------------------------------------------------

    def supported(self, claims):
        """Which of the claims wanting to be spoken are backed by a record.

        `claims`: [(subject, predicate, value)] — what the generator network
        wants to say.
        Returns: (passed, dropped). The dropped are NOT SPOKEN.

        This is the last wall in front of the generator network. The network
        can be fluent, even confident; if there is no record, the sentence
        falls.
        """
        passed, dropped = [], []
        for claim in claims:
            record = self.behind(*claim)
            if record is not None and record.trust >= SPEAK:
                passed.append((claim, record))
            else:
                dropped.append(claim)
        return passed, dropped

    def behind(self, subject, predicate, value):
        """The record behind this claim — None if there is none.

        This is the answer to "why did you say that", and that it is
        computable is the distinguishing feature of this architecture.
        """
        for held in self.memory.about(subject, touch=False):
            # If the predicate is None it is a WILDCARD: a value match
            # suffices. The reason was measured — the reader extracts the
            # predicate in ~0.5% of sentences; when the predicate match was
            # made mandatory, natural sentences with subject+value but no
            # predicate (their records are also stored predicate-less) were
            # counted unsupported and fell. Subject+value matching is already
            # enough as a confabulation barrier.
            if held.value == value and (predicate is None
                                        or held.predicate == predicate):
                return held
        return None

    def doubtful(self, record):
        """Does this record rest solely on a single low-level word.

        It must be marked when spoken: the knowledge is not discarded but it
        does not speak with the same voice.
        """
        # We look at the source's LEVEL, not the trust VALUE. In the first
        # draft I wrote `trust_of(source) <= STRANGER`: 0.6 <= 1 is always
        # true and document records also looked doubtful. Level and value are
        # not on the same scale; comparing them was itself a mistake.
        return record.witnesses <= UNVERIFIED and record.level == STRANGER

    def inferred(self, subject, predicate, value, because):
        """A record derived by inference — with low trust, bound to its rationale.

        This is the only point where the derivation engine touches memory,
        and every record passing through here carries its source as
        "inference": later, which knowledge came from observation and which
        from reasoning can be separated.

        IT GOES THROUGH `admit` — it used to call `memory.write` directly and
        so walked AROUND the contradiction gate. Measured: with the operator's
        "kartal→kuş" standing, the derived "kartal→balık" was written
        silently, no link, no pressure — the derivation engine was the one
        writer in the system that could contradict without anyone noticing.
        Passing through the gate it still loses (INFERRED is the weakest level
        and its trust sits below SPEAK), but the DOUBT IS RECORDED, which is
        the whole point.
        """
        held, _why = self.admit(subject, predicate, value, "#inference",
                                INFERRED, episodic=False)
        if held is None:
            return None
        for key in because:
            self.memory.link(held.key, key, IF)     # condition link
        return held

    # --- experience -----------------------------------------------------

    def weigh(self, expected, actual):
        """Surprise: the gap between prediction and reality.

        The counterpart of predictive coding. If the prediction held, the
        experience is cheap and does not occupy memory; if it failed, it is
        expensive and permanent. This is why touching the stove once is
        stronger than a thousand repetitions.
        """
        return min(1.0, abs(float(expected) - float(actual)))
