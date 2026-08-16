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
import struct
import time
import zlib

# Link kinds. These are RELATION IDENTITIES, not language: which sentence
# carries which link is learned by the reader network; no written list here
# does the matching.
CAUSE, THEN, IF, CONTRA = 1, 2, 3, 4

# Source levels. A higher number means the word carries more weight. A single
# sentence from a stranger must not crush a verified document — in the old
# memory this was measured and it could crush it.
OPERATOR, DOCUMENT, DISTILLED, INFERRED, STRANGER = 5, 4, 3, 2, 1


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
                 "sources")

    def __init__(self, key, subject, predicate, value, source, level=None,
                 trust=0.5, at=None, episodic=True):
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

    def to_dict(self):
        return {"key": self.key, "subject": self.subject,
                "predicate": self.predicate, "value": self.value,
                "source": self.source, "level": self.level,
                "trust": self.trust, "at": self.at,
                "witnesses": self.witnesses, "last_seen": self.last_seen,
                "links": self.links, "episodic": self.episodic,
                "sources": sorted(self.sources)}

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
                    held.get("episodic", True))
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

    FORMAT = 3

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
        self._next = 1

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
                       trust, episodic=episodic)
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
            held = json.loads(payload.decode("utf-8"))
        else:
            # OLD plain-JSON file (backward compat) — the next save() migrates
            # it to binary.
            text = raw.decode("utf-8")
            if not text.lstrip().startswith("{"):
                raise ValueError(f"unrecognized .lmm file (neither binary nor JSON): {path}")
            held = json.loads(text)
        found._next = held.get("next", 1)
        found.self_key = held.get("self")
        found.transitive = set(held.get("transitive", ()))
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
        return found
