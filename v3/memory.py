"""The core of the living memory: identity, record, link, experience.

There is NOT A SINGLE THING belonging to language in this file. No word list,
no suffix, no syllable, no pattern, no template, no vowel rule. Every name here
is the name of a DATA FIELD — like a database column name. Language enters the
system only through the two trained networks.

Where the old memory broke was that a node was a STRING: `kartal` was both the
bird and the district, and the two collided in the same node. It was measured —
across 1,500 polysemous cases the rate of picking the right sense was 21-24%,
random being 14.4%. So sense disambiguation was barely working at all, and the
cause was not the mechanism but the ABSENCE OF IDENTITY.

Here a concept is a NUMERIC identity. A word is merely a label attached to that
identity, and one identity can carry labels in multiple languages — knowledge
is learned once, spoken in every language.

    #1  labels: kartal, eagle      records: tür → #2
    #7  labels: kartal             records: tür → #9
                                   the two are SEPARATE ENTITIES, same spelling

A record is a node too: a thing with a source, a trust, a time, a witness count
and an EXPERIENCE. Records can link to one another (cause, then, condition) —
the thing a flat triple cannot do, and what gives long narrative its skeleton.
"""
import json
import os
import re
import struct
import time
import zlib

from v3.dataset import fold

# Link kinds. These are RELATION IDENTITIES, not language: which sentence
# carries which link is learned by the reader network; no written list here
# does the matching.
CAUSE, THEN, IF, CONTRA = 1, 2, 3, 4

# SUSPECT — a contradiction that is SUSPECTED but NOT YET JUDGED. The rivalry
# test (semantic, model-backed in LMM) is not always available at write time:
# during bulk structural ingestion it costs one model round-trip per cell, so
# it is deferred. Deferring must not mean FORGETTING — the audit measured the
# hole: with the test unavailable the rivalry silently became "not a rival",
# the CONTRA link was never made and the debt disappeared. A PENDING verdict
# is now recorded as this link, `dynamics.sleep` judges those in BULK (sleep
# is already a batch pass) and turns them into CONTRA or clears them.
SUSPECT = 5

# Source levels. A higher number means the word carries more weight. A single
# sentence from a stranger must not crush a verified document — in the old
# memory this was measured and it could crush it.
OPERATOR, DOCUMENT, DISTILLED, INFERRED, STRANGER = 5, 4, 3, 2, 1


# --- TYPED VALUE: a measurement is a number, a range and a unit --------------
#
# Until now a value was either an identity key or a flat string, so
# "15.6\" LCD" and "xubuntu" were the same KIND of thing to every organ that
# read them. Two costs were measured for that:
#
#   (1) whether two values in one slot are COMPATIBLE was outsourced to the
#       language model (`Gate.rival`), one round-trip per pair — expensive
#       enough that bulk ingestion had to switch it off and defer to sleep.
#       Two numbers with the same unit need no model: disjoint ranges exclude
#       each other, overlapping ones do not. That is arithmetic.
#   (2) the unit was invisible to the graph. A question that asks for a
#       measurement ("how many inches") had nothing structural to match
#       against, only the letters of the unit — and a symbolic unit (", %, °C)
#       has no letters at all.
#
# THE UNIT IS READ FROM THE DATA, NEVER FROM A LIST. The only rule is
# ADJACENCY: the token standing next to a number is what that number MEASURES
# IN. It is the same principle the evidence index already uses for kg/mAh, and
# it holds for symbolic units too, because a symbol next to a digit is a token
# like any other. Nothing here knows that 15.6" is inches — it knows only that
# 15.6 is measured in `"`, and that another 15.6 `"` measures the same thing.
# The NAMING is not done here and is not done anywhere; only the binding is.

# A NUMBER: digits with at most one fractional separator. Two separators is a
# version/section string ("4.8.0-36", "1.2.3"), not a quantity — reading it as
# one would put section numbers into arithmetic comparisons.
_NUMBER = re.compile(r"(?<![\w.,])(-?)(\d+(?:[.,]\d+)?)(?![\w.,]*[.,]\d)")
# The unit follows its number, at most one space away, and is the run of
# non-space non-digit characters after it: letters ("kg", "mm", "VA"), symbols
# ('"', "%", "°C"), or a mix. Bounded in length because a unit is a token, not
# a sentence — beyond that the number simply has no unit.
_UNIT = re.compile(r"[ \t]?([^\s\d]{1,4})")


def measure(text):
    """The measurement a text states: (low, high, unit) — or None.

    STRUCTURE ONLY. The first number in the text opens the measurement; a
    later number carrying the SAME unit widens it into a range ("100 V -240 V"
    → 100..240 v; "5 °C ... 40 °C" → 5..40 °c). A later number with a
    different unit belongs to a different measurement and is left alone
    ("14.4 V / 6500 mAh" is the voltage, and the capacity is its own value).

    Both decimal notations parse to the same number (15.6 == 15,6): the
    notation is orthography, the quantity is not.

    `unit` may be None (a bare number). It is still a typed value — it can be
    compared with another bare number — but no organ may treat two DIFFERENT
    units as comparable, so unit-less numbers are the weakest form and the
    gate refuses to judge rivalry on them (see `Gate._measured`).
    """
    if not text:
        return None
    text = str(text)
    found = []
    for m in _NUMBER.finditer(text):
        try:
            number = float(m.group(2).replace(",", "."))
        except ValueError:                                  # pragma: no cover
            continue
        # A MINUS SIGN BELONGS TO THE FIRST NUMBER ONLY. "-20 °C ~ 55 °C"
        # starts below zero; "100 V -240 V" does not hold a minus 240 — there
        # the dash sits BETWEEN two numbers and joins them into a range. What
        # separates the two cases is whether a number came before it.
        if m.group(1) and not found:
            number = -number
        after = _UNIT.match(text, m.end())
        unit = after.group(1) if after else None
        if unit is not None and any(ch.isalnum() for ch in unit):
            # Punctuation that ENDS a unit is punctuation, not part of it:
            # "8 GB," and "8 GB" must state the same unit, or a comma would
            # make two measurements incomparable.
            unit = unit.rstrip("".join(ch for ch in unit
                                       if not ch.isalnum())) or None
        elif unit is not None:
            # A SEPARATOR IS NOT A UNIT. "360 mm * 380 mm", "100 V -240 V" and
            # "%25 - %85" put a symbol between two numbers; a symbol standing
            # BETWEEN numbers separates them, it does not measure them. The
            # test is positional, not a list of characters: a number begins
            # again just past the candidate.
            rest = text[after.end():].lstrip()
            while rest and not rest[0].isalnum():
                rest = rest[1:].lstrip()
            if rest[:1].isdigit():
                unit = None
        found.append((number, fold(unit) if unit else None))
    if not found:
        return None
    number, unit = found[0]
    low = high = number
    for other, other_unit in found[1:]:
        if other_unit != unit:
            continue
        low, high = min(low, other), max(high, other)
    return (low, high, unit)


def overlap(one, other):
    """Do two measurements leave room for each other.

    True  — the same quantity, or one containing the other (100..240 and 220)
    False — the same unit, disjoint numbers: they EXCLUDE each other
    None  — not comparable: a missing unit, or two DIFFERENT units. Different
            units are deliberately NOT called compatible: "7.8 kg" and "17 lb"
            are the same quantity in two notations and might well exclude each
            other, and knowing that would take a conversion table — a
            hand-written list, which this layer is not allowed to have. So the
            honest answer is "no verdict here", and the caller's semantic test
            keeps its say.
    """
    if one is None or other is None:
        return None
    low, high, unit = one
    olow, ohigh, ounit = other
    if unit is None or ounit is None:
        return None
    if unit != ounit:
        return None
    return low <= ohigh and olow <= high


class Identity:
    """A concept. Not a word — words are only its labels."""

    __slots__ = ("key", "labels", "vector", "seen")

    def __init__(self, key, labels=(), vector=None):
        self.key = key
        self.labels = list(labels)      # which spelling in which language
        self.vector = vector            # continuous layer: for finding
        self.seen = 0                   # how many times accessed (for fading)

    def to_dict(self):
        return {"key": self.key, "labels": self.labels,
                "vector": self.vector, "seen": self.seen}

    @classmethod
    def from_dict(cls, held):
        found = cls(held["key"], held.get("labels", ()), held.get("vector"))
        found.seen = held.get("seen", 0)
        return found


class Record:
    """A fact — and the fact itself is a node.

    In a flat triple a record was an edge, and nothing could attach to an
    edge. Here the record has an identity, so it can link to other records:
    the record "it rained" links to the record "the soil got wet" with a
    CAUSE link.
    """

    __slots__ = ("key", "subject", "predicate", "value", "source", "level",
                 "trust", "at", "witnesses", "last_seen", "links", "episodic",
                 "sources", "measure")

    def __init__(self, key, subject, predicate, value, source, level=None,
                 trust=0.5, at=None, episodic=True, measure=None):
        self.key = key
        self.subject = subject          # identity key
        self.predicate = predicate      # predicate identity — vocabulary OPEN
        self.value = value              # identity key or plain value
        # NAME and LEVEL are separate: `source` is who said it
        # ("hayvanlar.txt", "ali"), `level` is how much it counts
        # (OPERATOR..STRANGER). In the first draft they were one field, and
        # the smoke test gave it away — the output printed the source as ‹4›:
        # the number had eaten the name's place. The "a source in every
        # answer" claim demands the name; arbitration demands the level; the
        # two must be separate fields.
        self.source = str(source)
        self.level = level if level is not None else DOCUMENT
        self.trust = trust
        self.at = at if at is not None else time.time()
        self.witnesses = 1
        # Witness SET: the third repetition of the same source is not a third
        # witness. The audit measured it — without the set, two sources had
        # been counted as nine witnesses.
        self.sources = {self.source}
        self.last_seen = self.at
        self.links = []                 # [(link kind, record key)]
        # Episodic: came from a single event ("Ali said so"). In the
        # distillation round it rises to semantic through repetition. The
        # memory-side counterpart of the brain's hippocampus/cortex split.
        self.episodic = episodic
        # TYPED VALUE: (low, high, unit_key) when the value states a
        # measurement, else None. The unit is an IDENTITY KEY, not a string —
        # a unit is a concept of the graph like any other, whose label is
        # whatever token the document put next to the number. Filled by
        # `Memory.write`, which is the only place that can see both the value's
        # labels and the unit table.
        self.measure = tuple(measure) if measure else None

    def to_dict(self):
        return {"key": self.key, "subject": self.subject,
                "predicate": self.predicate, "value": self.value,
                "source": self.source, "level": self.level,
                "trust": self.trust, "at": self.at,
                "witnesses": self.witnesses, "last_seen": self.last_seen,
                "links": self.links, "episodic": self.episodic,
                "sources": sorted(self.sources),
                "measure": list(self.measure) if self.measure else None}

    def strengthen(self, source, level=None):
        """A NEW independent witness: a fixed share of the remaining doubt closes.

        The calculation lives ONLY here and carries its own protection: the
        same source is not counted twice. The audit measured it — without the
        protection a single teaching became two witnesses, and three
        repetitions by the same man became nine witnesses.
        """
        source = str(source)
        if source in self.sources:
            self.last_seen = time.time()
            return self
        self.sources.add(source)
        self.witnesses += 1
        # PROMOTION: a higher level arriving later raises the record's level.
        # Round 3 measured it — if the stranger spoke first, the record opened
        # as STRANGER and the operator's confirmation could never raise it:
        # the stranger, by speaking first, could lock in the fact slot on the
        # cheap.
        if level is not None and level > self.level:
            self.level = level
            self.trust = max(self.trust, Memory.trust_of(level))
        self.trust += (1.0 - self.trust) * 0.15
        self.trust = min(self.trust, 0.98)
        self.last_seen = time.time()
        return self

    @classmethod
    def from_dict(cls, held):
        found = cls(held["key"], held["subject"], held["predicate"],
                    held["value"], held["source"], held.get("level"),
                    held.get("trust", 0.5), held.get("at"),
                    held.get("episodic", True),
                    # A FORMAT-3 record has no typed value; `Memory.load`
                    # derives it after the identities are in (the unit table
                    # needs them), so an old file ends up with exactly the
                    # typed view a new one would have. Nothing ASSERTED is
                    # invented by that: the measurement is a reading of the
                    # value the file already holds.
                    held.get("measure"))
        found.witnesses = held.get("witnesses", 1)
        found.last_seen = held.get("last_seen", found.at)
        found.links = [tuple(one) for one in held.get("links", ())]
        found.sources = set(held.get("sources", (found.source,)))
        return found


class Experience:
    """Experience — the engineering counterpart of emotion.

    A child touches the stove, gets burned, learns. Nobody hands it a "fire
    is dangerous" list; the experience becomes a record and that record
    brakes the next behavior.

    The same here: the system says something, receives a reaction, the
    reaction is recorded. In the next similar situation the candidate answers
    are struck against these records.

    `surprise` — the counterpart of predictive coding. The brain predicts
    constantly and updates only when SURPRISED. If the prediction held, the
    record is cheap; if it failed, it is expensive and permanent. One-shot
    learning (touching the stove once) comes from here — not a thousand
    repetitions, one surprise.
    """

    __slots__ = ("key", "said", "outcome", "surprise", "at", "about")

    def __init__(self, key, said, outcome, surprise=0.0, at=None, about=()):
        self.key = key
        self.said = said                # what was said (raw)
        self.outcome = outcome          # what happened — numeric signal
        self.surprise = surprise        # gap between prediction and reality
        self.at = at if at is not None else time.time()
        self.about = list(about)        # related identity keys

    def to_dict(self):
        return {"key": self.key, "said": self.said, "outcome": self.outcome,
                "surprise": self.surprise, "at": self.at, "about": self.about}

    @classmethod
    def from_dict(cls, held):
        return cls(held["key"], held["said"], held["outcome"],
                   held.get("surprise", 0.0), held.get("at"),
                   held.get("about", ()))


class Memory:
    """Identities, records, experiences — and its own history.

    Cannot be written to directly from outside: the only way to write is
    `write`, and there the source level and the trust calculation are
    mandatory. In the company scenario the graph's immutability rests on
    this gate.
    """

    # 4 — records carry a TYPED VALUE (number, range, unit) and the file
    # carries the unit table. 3 — untyped values only; still loads (see
    # `load`), its records get their typed view derived on the way in.
    FORMAT = 4

    def __init__(self):
        self.identities = {}            # key -> Identity
        self.records = {}               # key -> Record
        self.experiences = {}           # key -> Experience
        self.by_label = {}              # label -> [identity key]
        self.by_subject = {}            # identity key -> [record key]
        # REVERSE INDEXES (optimization): so derivation/inference does not
        # scan the WHOLE graph on every write. by_value: value node -> the
        # records that take it as VALUE (_derive incoming); by_predicate:
        # predicate -> records with that predicate (_learn_transitive).
        # O(N)/write → O(degree)/write.
        self.by_value = {}              # value key -> [record key]
        self.by_predicate = {}          # predicate key -> [record key]
        self.self_key = None            # #SELF — the node of the autobiography
        # Predicates whose transitivity is proven BY DATA. Not a hand-written
        # list: when the graph sees a closed triangle (A→B, B→C, A→C all
        # WITNESSED), that predicate enters here. "tür" learns from one
        # example, "sever" never does.
        self.transitive = set()
        # UNIT TABLE: unit token -> its identity key. A unit is a concept, so
        # it lives in the identity system; its label is NAMESPACED (`#unit:kg`)
        # because a unit's bare token must not become a way to address it by
        # name. Two identities sharing a spelling is precisely what this file
        # exists to keep apart, and 'g' or 'v' as a plain label would collide
        # with every short name in the graph and hijack the label index's
        # inflection tolerance. The token itself is not lost — it is this
        # table's key, and it is what the document said.
        self.units = {}
        self._next = 1
        # Derived-view cache: value -> measurement. A value is read many times
        # (every rivalry test in its slot) and the reading is pure.
        self._measured = {}

    # --- identity -------------------------------------------------------

    def _key(self):
        found = self._next
        self._next += 1
        return found

    def identify(self, label, vector=None, same_as=None):
        """Binds a label to an identity; opens a NEW identity if needed.

        That the same spelling can be two concepts is this place's whole
        point: if `same_as` is not given and the label is already bound to
        another identity, the caller must say which identity it means. The
        decision is not made here — this place only keeps the record; finding
        which sense was meant is the job of geometry and context.
        """
        if same_as is not None:
            found = self.identities[same_as]
            if label not in found.labels:
                found.labels.append(label)
                self.by_label.setdefault(label, []).append(same_as)
            return same_as
        key = self._key()
        self.identities[key] = Identity(key, [label], vector)
        self.by_label.setdefault(label, []).append(key)
        return key

    def unit(self, token):
        """The identity of a unit token — opened on first sighting.

        No list decides what a unit is: `measure` found this token standing
        next to a number in the data, and that is the whole qualification.
        """
        token = fold(token)
        key = self.units.get(token)
        if key is None:
            key = self.identify("#unit:" + token)
            self.units[token] = key
        return key

    def measure_of(self, value):
        """The measurement this VALUE states — (low, high, unit_key) or None.

        The value may be an identity key (then its labels are read — the label
        is what the document wrote) or a plain value. This is a READING, never
        a write: it cannot change what the value is, only say what quantity it
        states.
        """
        if value in self._measured:
            return self._measured[value]
        text = None
        if isinstance(value, int):
            held = self.identities.get(value)
            if held is not None and held.labels:
                text = held.labels[0]
        elif value is not None:
            text = str(value)
        found = measure(text) if text else None
        if found is not None:
            low, high, unit = found
            found = (low, high, self.unit(unit) if unit else None)
        self._measured[value] = found
        return found

    def candidates(self, label):
        """All identities carrying this label — which one it is, is a separate job."""
        return list(self.by_label.get(label, ()))

    # --- record ---------------------------------------------------------

    def write(self, subject, predicate, value, source, level=None,
              trust=None, episodic=True):
        """Puts a fact into memory; if the same one exists, REINFORCES it.

        Reinforcement is the second independent source closing a share of the
        remaining doubt — not additive, because when it was additive a few
        documents pierced the ceiling (measured in the old memory).
        """
        level = DOCUMENT if level is None else level
        trust = self.trust_of(level) if trust is None else trust
        for key in self.by_subject.get(subject, ()):
            held = self.records[key]
            if held.predicate == predicate and held.value == value:
                return held.strengthen(source, level)
        key = self._key()
        found = Record(key, subject, predicate, value, source, level,
                       trust, episodic=episodic,
                       measure=self.measure_of(value))
        self.records[key] = found
        self.by_subject.setdefault(subject, []).append(key)
        if isinstance(value, int):
            self.by_value.setdefault(value, []).append(key)
        if predicate is not None:
            self.by_predicate.setdefault(predicate, []).append(key)
        return found

    def link(self, one, other, kind):
        """Links two records — cause, then, condition, contradiction."""
        held = self.records[one]
        if (kind, other) not in held.links:
            held.links.append((kind, other))
        return held

    def unlink(self, one, other, kind):
        """Removes a link. Used when a SUSPECT is judged: the suspicion either
        becomes CONTRA or is cleared — an unjudged suspicion must not linger
        as pressure forever."""
        held = self.records.get(one)
        if held is not None and (kind, other) in held.links:
            held.links.remove((kind, other))
        return held

    def rivals_of(self, record, kinds=(CONTRA,)):
        """The records this one is bound to by the given link kinds.

        One place for the walk, because three organs (arbitration, pressure,
        the gate) used to each write their own comprehension and they drifted
        apart.
        """
        return [self.records[key] for kind, key in record.links
                if kind in kinds and key in self.records]

    def about(self, subject, touch=True):
        """The records about an identity.

        `touch=False`: for graph walks and internal audits. The audit
        measured it — every spreading pass incremented the `seen` counter and
        the "access" counter had turned into a "graph adjacency" counter; in
        ten queries three identities showed 11 accesses each, none of them
        coming from the user.
        """
        found = self.identities.get(subject)
        records = [self.records[key] for key in self.by_subject.get(subject, ())]
        if touch and found is not None:
            found.seen += 1
            # C2: a REAL access also refreshes the record's last-seen.
            # Otherwise even a fact asked 1000 times faded after the `fade`
            # window (access only incremented identity.seen, never touching
            # record.last_seen) — contradicting "permanent knowledge without
            # retraining".
            now = time.time()
            for r in records:
                r.last_seen = now
        return records

    @staticmethod
    def trust_of(level):
        """Initial trust FROM the level. The bands do not overlap."""
        # INFERRED is BELOW the SPEAK(0.4) threshold: an inferred record
        # (possibly from a fragile chain) stays in the graph and is shown
        # marked to the "what did you infer" question, but is NOT SPOKEN as
        # a fact. The audit caught it — at 0.45 it passed the gate and was
        # spoken like a certain fact (its #inference source vanishing). If it
        # is independently reinforced by observation, trust rises and it is
        # spoken.
        return {OPERATOR: 0.75, DOCUMENT: 0.6, DISTILLED: 0.5,
                INFERRED: 0.35, STRANGER: 0.3}.get(level, 0.3)

    # --- experience -----------------------------------------------------

    def lived(self, said, outcome, surprise=0.0, about=()):
        """Records an experience. A surprising experience carries more weight."""
        key = self._key()
        found = Experience(key, said, outcome, surprise, about=about)
        self.experiences[key] = found
        return found

    def recall(self, about, most=8):
        """Experiences about these identities, highest surprise first."""
        wanted = set(about)
        held = [one for one in self.experiences.values()
                if wanted & set(one.about)]
        held.sort(key=lambda one: -one.surprise)
        return held[:most]

    # --- file -----------------------------------------------------------

    MAGIC = b"LMM1"       # closed binary .lmm header

    def save(self, path):
        """Write as CLOSED binary .lmm: magic + version + raw-length + crc32 +
        zlib(json). It does NOT LOOK like JSON, cannot be EDITED by hand; NOT
        pickle/.pt (loading runs only json.loads — no code-execution risk).
        The write is ATOMIC (.tmp + os.replace): a half-write does not corrupt
        the permanent file."""
        held = {"format": self.FORMAT, "next": self._next,
                "self": self.self_key,
                "transitive": sorted(self.transitive),
                "units": {token: key for token, key in self.units.items()},
                "identities": [one.to_dict()
                               for one in self.identities.values()],
                "records": [one.to_dict() for one in self.records.values()],
                "experiences": [one.to_dict()
                                for one in self.experiences.values()]}
        payload = json.dumps(held, ensure_ascii=False).encode("utf-8")
        frame = (self.MAGIC
                 + struct.pack(">BII", self.FORMAT, len(payload),
                               zlib.crc32(payload) & 0xffffffff)
                 + zlib.compress(payload, 9))
        tmp = path + ".tmp"
        with open(tmp, "wb") as handle:
            handle.write(frame)
        os.replace(tmp, path)               # atomic swap

    @classmethod
    def load(cls, path):
        found = cls()
        if not os.path.exists(path):
            return found            # new file: an empty memory is legitimate
        # A CORRUPT file is NOT an empty memory. The round-3 audit measured
        # it: a corrupt file silently returned empty and the next save()
        # crushed the recoverable file — total data loss without an error.
        # If corrupt, stop loudly.
        with open(path, "rb") as handle:
            raw = handle.read()
        if raw[:4] == cls.MAGIC:
            version, rawlen, crc = struct.unpack(">BII", raw[4:13])
            payload = zlib.decompress(raw[13:])
            if len(payload) != rawlen or (zlib.crc32(payload) & 0xffffffff) != crc:
                raise ValueError(f"corrupt .lmm (crc/length mismatch): {path}")
            cls._version_gate(version, path)
            held = json.loads(payload.decode("utf-8"))
        else:
            # OLD plain-JSON file (backward compat) — the next save() migrates
            # it to binary.
            text = raw.decode("utf-8")
            if not text.lstrip().startswith("{"):
                raise ValueError(f"unrecognized .lmm file (neither binary nor JSON): {path}")
            held = json.loads(text)
        # The header's version and the payload's must agree, and the payload is
        # gated too — a plain-JSON file has no header to gate.
        cls._version_gate(held.get("format", cls.FORMAT), path)
        found._next = held.get("next", 1)
        found.self_key = held.get("self")
        found.transitive = set(held.get("transitive", ()))
        found.units = {token: key
                       for token, key in held.get("units", {}).items()}
        for one in held.get("identities", ()):
            made = Identity.from_dict(one)
            found.identities[made.key] = made
            for label in made.labels:
                found.by_label.setdefault(label, []).append(made.key)
        for one in held.get("records", ()):
            made = Record.from_dict(one)
            found.records[made.key] = made
            found.by_subject.setdefault(made.subject, []).append(made.key)
            if isinstance(made.value, int):     # rebuild reverse indexes too
                found.by_value.setdefault(made.value, []).append(made.key)
            if made.predicate is not None:
                found.by_predicate.setdefault(made.predicate, []).append(made.key)
        for one in held.get("experiences", ()):
            made = Experience.from_dict(one)
            found.experiences[made.key] = made
        # A FILE WRITTEN BEFORE TYPED VALUES EXISTED gets the same typed view a
        # file written now would have. The measurement is DERIVED from the
        # value the file already holds, so nothing asserted is added; leaving it
        # out would give the same graph two behaviours depending on which
        # version happened to write it, and the gate's determinism would then
        # depend on file provenance.
        for made in found.records.values():
            if made.measure is None:
                made.measure = found.measure_of(made.value)
            else:
                made.measure = tuple(made.measure)
        return found

    @classmethod
    def _version_gate(cls, version, path):
        """A file from the FUTURE is not readable, and silence about that is
        data loss.

        The audit found the hole: the version byte was written and never read,
        so a file marked FORMAT=99 loaded as if it were understood. Whatever a
        later format adds — a field this code drops, a meaning it reads
        differently — the next `save` would then write the graph back WITHOUT
        it, quietly. An older format is a different matter: this code knows
        what those files say, and keeps reading them.
        """
        if not isinstance(version, int) or version > cls.FORMAT:
            raise ValueError(f"unreadable .lmm format {version} "
                             f"(this build understands up to {cls.FORMAT}): "
                             f"{path}")
