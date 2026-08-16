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
from v3.memory import CONTRA, INFERRED, STRANGER

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
        # Optional semantic RIVAL check: (old_value_key, new_value_key)->bool.
        # In LMM the session wires this to Qwen. When the predicate is unknown
        # (predicate=None), two different values count as a contradiction only
        # if they are RIVALS (same slot, mutually exclusive) — otherwise
        # coexisting facts like "kartal→kuş, kartal→yırtıcı" were mistaken
        # for contradictions and broken by arbitrate.
        self.rival = None

    # --- entry ----------------------------------------------------------

    def admit(self, subject, predicate, value, source, level=None,
              episodic=True):
        """Tries to put a claim into memory.

        Returns: (record | None, reason). The reason is a NUMBER — not
        language:
            0  written or reinforced
            1  an opposing record exists in the same slot with higher trust
            2  source unrecognized
        """
        if source is None:
            return None, 2
        from v3.memory import DOCUMENT
        level = DOCUMENT if level is None else level
        clash = self._contradiction(subject, predicate, value)
        if clash is not None:
            trust = self.memory.trust_of(level)
            # The contradiction link is established IN EVERY CASE. The audit
            # caught it: when the new claim was stronger the link was not
            # made, and `arbitrate`/`pressure` never saw that contradiction —
            # the place where the system should have been uneasy stayed
            # unrecorded.
            weaker = clash.trust >= trust
            held = self.memory.write(subject, predicate, value, source,
                                     level,
                                     trust=trust * 0.5 if weaker else None,
                                     episodic=True)
            self.memory.link(held.key, clash.key, CONTRA)
            self.memory.link(clash.key, held.key, CONTRA)
            return held, (1 if weaker else 0)
        return self.memory.write(subject, predicate, value, source, level,
                                 episodic=episodic), 0

    def _contradiction(self, subject, predicate, value):
        """A RIVAL record carrying ANOTHER value on the same subject + same predicate.

        Are two different values a CONTRADICTION — INDEPENDENT OF THE
        PREDICATE, what matters is whether the values are RIVALS. If `rival`
        is wired (LMM): hierarchically related values (felid/mammal — a felid
        IS a mammal) are NOT rivals → not a contradiction, they coexist;
        unrelated/exclusive values (paris/berlin, bird/fish) → contradiction.
        If `rival` is not wired (v3), a predicate match counts as a
        contradiction (legacy).

        FIX (test-proven): the rival check used to live only in the
        predicate=None branch; an explicit predicate ("tür", is-a) skipped
        that check and counted lion felid/mammal as a false contradiction.
        Now it applies to all predicates."""
        for held in self.memory.about(subject, touch=False):
            if held.predicate != predicate or held.value == value:
                continue
            if self.rival is not None and not self.rival(held.value, value):
                continue          # related/hierarchical values → not a contradiction
            return held
        return None

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
        """
        held = self.memory.write(subject, predicate, value, "#inference",
                                 INFERRED, episodic=False)
        for key in because:
            self.memory.link(held.key, key, 3)      # condition link
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
