"""The session: the place that joins all the organs into one flow.

Up to here six pieces were built separately and each was tested on its own.
This file wires them together and the whole architecture runs end to end for
the first time:

    sentence
      -> READER NETWORK    produces an operation (PASS · ASK · WRITE)
      -> GEOMETRY          turns labels into identities, gathers records by association
      -> GATE              audits if writing · seeks supports if speaking
      -> WEIGHING          runs N candidates through three scales before speaking
      -> SPEAKER NETWORK   builds sentences from records
      -> GATE (exit)       a sentence that does not read back FALLS
      -> EXPERIENCE        what was said, what happened — weighted by surprise

The inner weighing is the counterpart of the "should I say it this way or
that" step, and it has three scales:

    SUPPORT     does each claim have a record (the gate)
    EXPERIENCE  what happened when I said something like this (past outcomes)
    COVERAGE    how much of the requested records it speaks

If the networks are absent the system does not go mute, it dumps raw records:
it cannot speak, but it does not lie either. The fourth condition holds in
every case.

Nothing belonging to language exists.
"""
from v3 import dynamics, geometry
from v3.gate import Gate, SPEAK
from v3.memory import Memory, OPERATOR, STRANGER
from v3.reader import ASK, PASS, Reader, WRITE
from v3.speaker import Speaker, prompt_of

# The minimum confidence the reader's operation must carry before it is
# applied. A reading below it does not count as a reading — writing what you
# are not sure of is the door to confabulation.
CERTAIN = 0.5

# The most records GATHERED in one answer (wide, for context).
MOST = 8

# The most records actually HANDED to the speaker. MOST gathers wide, but
# handing all of it to the speaker turned the answer into a DUMP — measured:
# 8 loosely associated records → a scattered sentence that cannot stay on
# topic. The few most relevant (retrieval already ranked by relevance) is
# both focused and fast on CPU.
TELL = 3


class Session:
    """One conversation. Carries the memory, wires the organs, records its experience."""

    def __init__(self, path=None, speaker_name=None):
        self.path = path
        self.memory = Memory.load(path) if path else Memory()
        self.gate = Gate(self.memory)
        self.reader = Reader()
        self.speaker = Speaker()
        # Who is speaking: the operator, or someone unrecognized. The level of
        # every written record comes from here, and a stranger cannot crush
        # verified knowledge. NAME and LEVEL are separate: the name is written
        # on the record ("ali", "#operator"), the level enters arbitration.
        # They were one field until the smoke test printed ‹4›.
        self.level = STRANGER if speaker_name else OPERATOR
        self.who = speaker_name or "#operator"
        self.focus = []             # the last identities spoken about — context
        self.turns = 0              # auto-sleep counter
        # The geometry table: if present, each identity receives its vector at
        # birth. Without it the system works but sense disambiguation weakens
        # — a convenience, not a dependency.
        self.vectors = {}
        root = __import__("os").path.dirname(
            __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
        table = __import__("os").path.join(root, "models/v3/vectors.json")
        try:
            import json, os
            if os.path.exists(table):
                self.vectors = json.load(open(table, encoding="utf-8"))
                from v3 import embed
                embed.attach(self.memory, self.vectors)
        except Exception:                                   # noqa: BLE001
            self.vectors = {}
        if self.memory.self_key is None:
            self.memory.self_key = self.memory.identify("#self")

    # --- main flow ------------------------------------------------------

    def respond(self, line):
        """An answer to a sentence. What returns is always text, never an unsupported claim."""
        # AUTO-SLEEP: doc §117 wires the sleep round to live operation. The
        # audit caught it — fading/distillation/arbitration were only called
        # by hand, the system never "slept". Every 50 turns it distills,
        # dampens, arbitrates.
        self.turns += 1
        if self.turns % 50 == 0:
            dynamics.sleep(self.memory)
        operation = self.reader.read(line)
        # The WRITE path opens only if there IS something to write. The audit
        # caught it: when the reader mistook a question for WRITE (it had
        # been trained without ASK data), `_write` returned an empty string
        # with an empty subject/value and the user never got an answer to the
        # question. A WRITE with missing pieces now falls through to the
        # gathering path — the price of a misreading becomes an attempt, not
        # silence.
        # For writing, subject + value are REQUIRED (both in the text,
        # alignable). The PREDICATE is NOT required: measured, the predicate
        # occurs in only 0.5% of sentences ("tür", "özellik" are inferences,
        # not in the text) — requiring the predicate dropped 99.5% of the
        # teaching. The predicate will be determined on the graph side from
        # the value's type; demanding from the reader something not in the
        # text was wrong.
        #
        # So that "penguen nedir" is not written as a fact, the real gate is
        # the READ-BACK: the record to be built is laid out and read back,
        # and if the same WRITE does not come out, it is not written. This
        # knows no language — it looks at the reader's own consistency.
        if (operation.kind == WRITE and operation.confidence >= CERTAIN
                and operation.subject and operation.value):
            return self._write(operation, line)
        # PASS + no subject = an utterance keeping the conversation going
        # ("selam", "naber"). The SPEAKER produces the answer — it takes the
        # input as dialogue context and builds the reply. The audit caught
        # it: there was NO path from here to the speaker, a fixed "[·]" was
        # returned; even if the speaker learned dialogue it was never called.
        #
        # The gate still stands in front: a chat reply must not carry a FACT
        # claim. If it contains a memory-writable triple (fabrication) it
        # falls; if it carries none (greeting, reaction) it passes — because
        # it cannot be true or false.
        if operation.kind == PASS and not operation.subject:
            said = self._chat(line)
            self._live(line, said, [])
            return said or "[·]"
        records = self._gather(operation, line)
        said = self._speak(records, line)
        # IF THERE IS NOTHING TO FETCH, FALL TO CHAT: the reader can mistake
        # a short utterance ("selam", "naber") for ASK (short-sentence bias)
        # — then no answer is found in memory and returning a flat [?] makes
        # the greeting look rejected. When empty, the speaker is asked for a
        # chat reply; _chat's gate prevents the candidate from FABRICATING a
        # fact, so a fact claim still cannot leak toward the unknown.
        if not said:
            said = self._chat(line)
        self._live(line, said, records)
        # If chat is empty too: an explicit REFUSAL.
        return said or "[?]"

    # --- writing --------------------------------------------------------

    def _write(self, operation, line):
        """Puts something taught into memory — passing through the gate."""
        if not (operation.subject and operation.value):
            return ""
        subject = self._identity(operation.subject)
        predicate = self._identity(operation.predicate or "")
        value = self._identity(operation.value)
        # Know what is in memory BEFORE writing — a prediction. If the new
        # fact contradicts it, SURPRISE. Doc §110: if the prediction failed
        # the experience is expensive. Without waiting for the user's
        # reaction, the contradiction itself is a prediction error and enters
        # automatically (audit: reacted was never called, surprise was always
        # 0).
        expected = self.gate._contradiction(subject, predicate, value)
        record, why = self.gate.admit(subject, predicate, value,
                                      self.who, self.level)
        if record is None:
            return ""
        surprise = 1.0 if (why == 1 or expected is not None) else 0.0
        # NO reinforcement HERE: `memory.write` already meets the repeated
        # fact with `strengthen(source)` and that protection looks at the
        # source set. The second call here counted a single teaching as two
        # witnesses — the audit measured it and it was removed.
        self.focus = [subject]
        # DERIVATION: the new fact is CHAINED with what is known — not flat
        # memorization, inference. "kartal→kuş" + "kuş→hayvan" ⊢
        # "kartal→hayvan". The derived record is written with the
        # `#inference` source, at LOW trust, and bound to its rationale; when
        # spoken it is clearly an inference, not a fabrication. This is item
        # 5 of the architecture — geometry/the chain proposes, the gate
        # verifies. Its adding of its own interpretation comes from here.
        self._learn_transitive(predicate)
        if predicate in self.memory.transitive:
            self._derive(subject, predicate, value)
        # #SELF: every learned fact is also bound to #self — so the system
        # can answer "what did I learn" from its own history (doc §98).
        self.last = self.memory.lived(line, outcome=1.0, surprise=surprise,
                                      about=[subject, self.memory.self_key])
        return self._render([record])

    def _learn_transitive(self, predicate):
        """Is this predicate transitive — did the graph see a closed TRIANGLE.

        Transitivity is not declared by hand, it is learned from data: if
        with the same predicate A→B, B→C AND A→C are all WITNESSED
        (taught/read, not inferred), that predicate is transitive. "tür"
        learns from one is-a example; "sever" never — because the chain of
        love does not close in the graph. The rule comes from the data
        itself.

        ONE triangle is NOT ENOUGH: a non-transitive relation like
        "sever/yakın/tanır" can close a triangle once in the data by
        coincidence (A loves B, B loves C, A loves C each taught
        separately). With a single accident that predicate was counted
        transitive for life and produced a wrong #inference from EVERY new
        edge (the audit caught it, the "no fabrication" hole). At least TWO
        independently witnessed triangles are required — in a truly
        transitive predicate these are plentiful, in coincidence rare.
        """
        WITNESSED_TRIANGLES = 2
        if predicate in self.memory.transitive:
            return
        edges = [r for records in self.memory.by_subject.values()
                 for r in (self.memory.records[k] for k in records)
                 if r.predicate == predicate and r.source != "#inference"]
        forward = {}
        for r in edges:
            forward.setdefault(r.subject, set()).add(r.value)
        triangles = 0
        for a, bs in forward.items():
            for b in bs:
                for c in forward.get(b, ()):
                    if c in bs:                 # A→B, B→C and A→C all witnessed
                        triangles += 1
                        if triangles >= WITNESSED_TRIANGLES:
                            self.memory.transitive.add(predicate)
                            return

    def _derive(self, subject, predicate, value):
        """TWO-WAY chained inference around the new fact (subject→value).

        On a transitive predicate (A tür B, B tür C ⊢ A tür C) a new edge
        opens a chain in two directions:

            FORWARD    if value→W exists   ⊢ subject→W   (if kuş→hayvan is known,
                                                          writing kartal→kuş gives kartal→hayvan)
            BACKWARD   if X→subject exists ⊢ X→value     (if kartal→kuş is known,
                                                          writing kuş→hayvan gives kartal→hayvan)

        One direction was not enough — the chain broke depending on which
        fact was taught first. The derived record is `#inference`, low trust,
        bound to its rationale; a wrong chain is visible and is not spoken as
        a fact.
        """
        forward = [(subject, r.value, [r.key])
                   for r in self.memory.about(value, touch=False)
                   if r.predicate == predicate]
        # BACKWARD: records whose value is the SUBJECT (X → subject) ⊢
        # X → value. In the first draft I had written `r.value == value` — it
        # was looking for what points at the value, whereas what points at
        # the SUBJECT is what must be sought.
        incoming = [(r.subject, value, [r.key])
                    for records in self.memory.by_subject.values()
                    for r in (self.memory.records[k] for k in records)
                    if r.predicate == predicate and r.value == subject
                    and r.subject != value]
        for who, what, because in forward + incoming:
            if who != what and not self.gate.behind(who, predicate, what):
                self.gate.inferred(who, predicate, what, because)

    # --- reading --------------------------------------------------------

    def _gather(self, operation, line):
        """Gathers the records relevant to the question — geometry finds, spreading fetches."""
        labels = [one for one in (operation.subject, operation.value) if one]
        # THE READER DOES NOT MARK THE SUBJECT IN A QUESTION: question
        # examples were trained without roles (questions teach the operation
        # kind, fact sentences the roles). So "kartal nedir" reads as ASK but
        # subject=None arrives and the question goes unanswered. The subject
        # is resolved not by the READER but by the GRAPH: the FIRST word in
        # the question that resolves to a known identity counts as the
        # subject. This is not a word list, it is the already-existing
        # `candidates` — it looks at what the graph itself recognizes.
        if not labels:
            from v3.dataset import fold
            for word in line.split():
                piece = fold("".join(ch for ch in word if ch.isalnum()))
                if piece and self.memory.candidates(piece):
                    labels = [piece]
                    break
        if not labels and self.focus:
            # Falling back to context only on a FAMILIAR sentence: if it
            # contains a word that is in neither the corpus nor memory
            # ("zzzq nedir"), falling to the last topic is answering the
            # unknown with the known — closed off.
            from v3.dataset import fold
            for word in line.split():
                piece = fold("".join(ch for ch in word if ch.isalnum()))
                if (piece and piece not in self.vectors
                        and not self.memory.candidates(piece)):
                    return []
            return self._focused()          # no subject: continue from the last topic
        # If the subject IS STATED but resolves to no identity, the answer
        # must be a REFUSAL. Measured: "zzzq nedir" fetched the kartal record
        # through pure vector association — answering the unknown with the
        # known is fabrication.
        if labels and not any(self.memory.candidates(one) for one in labels):
            return []
        weights = geometry.recall(self.memory, vector=self._vector_of(line),
                                  labels=labels, most=MOST * 3)
        if not weights:
            return []
        records = [self.memory.records[key] for key in weights
                   if key in self.memory.records]
        records.sort(key=lambda one: (-weights.get(one.key, 0.0), -one.trust))
        self.focus = [one.subject for one in records[:1]] or self.focus
        return records[:MOST]

    def _focused(self):
        """A question with no subject stated: continue from the last identity spoken about."""
        records = []
        for key in self.focus:
            records.extend(self.memory.about(key))
        records.sort(key=lambda one: -one.trust)
        return records[:MOST]

    # --- speaking -------------------------------------------------------

    def _chat(self, line):
        """A reply to a chat sentence — from the speaker, passing through the gate.

        The input is given as dialogue context ("previous utterance\n\n");
        in the speaker's training the dialogue pairs are in exactly this
        form. If the produced candidate carries a fact claim it is rejected:
        a chat reply does not PRODUCE knowledge, it only keeps the
        conversation going.
        """
        if not self.speaker.ready:
            return ""
        for candidate in self.speaker.say(line + "\n\n", count=3):
            claim = self.reader.read(candidate)
            # If the candidate sets out to write a FACT (subject+value) the
            # gate rejects it: a chat reply cannot be a graph claim, a claim
            # with no record is not spoken.
            # HOLE CLOSED: the condition used to carry
            # `and self._known(claim.value)` — if the value did not resolve
            # to a known identity the check was SKIPPED and the candidate
            # accepted. So a FABRICATION like "X bir Y'dir" (Y entirely
            # unknown) leaked through the chat path. Now every fact-shaped
            # candidate falls if no record stands behind it (EVEN IF the
            # value is unknown).
            if claim.subject and claim.value:
                subject = self._known(claim.subject)
                value = self._known(claim.value)
                predicate = (self._known(claim.predicate)
                             if claim.predicate else None)
                if not (subject and value
                        and self.gate.behind(subject, predicate, value)):
                    continue
            return candidate
        return ""

    def _speak(self, records, line):
        """Produces the candidates, weighs them, passes them through the gate. If none pass, raw."""
        # The most relevant TELL of them: retrieval ranked the records by
        # relevance, hand the speaker the first few — a focused answer, not a
        # dump.
        supported = [one for one in records if one.trust >= SPEAK][:TELL]
        if not supported:
            return ""
        if not self.speaker.ready:
            return self._render(supported)
        candidates = self.speaker.say(prompt_of(supported, self.memory))
        best, score = None, -1.0
        for candidate in candidates:
            weighed = self._weigh(candidate, supported)
            if weighed > score:
                best, score = candidate, weighed
        if best is None or score <= 0.0:
            return self._render(supported)
        return best

    def _weigh(self, candidate, records):
        """The inner weighing: support · experience · coverage. If negative the candidate falls.

        Two fixes, both from the line-by-line comparison audit:

        MANY CLAIMS — the candidate is split sentence by sentence and EVERY
        piece is read. A single reading audited only the first claim of a
        multi-sentence candidate; the gate's exit was weaker than promised.

        COVERAGE — the old measure was `len(candidate) / (40 · records)`: 40
        was an unmeasured constant and it rewarded LENGTH — a long rambling
        candidate could beat a short correct one. The new measure is the
        ratio of supported claims to the requested records: coverage is
        measured in CLAIMS, not letters.
        """
        claims = []
        for piece in candidate.replace("!", ".").replace("?", ".").split("."):
            piece = piece.strip()
            if not piece:
                continue
            operation = self.reader.read(piece)
            if operation.subject and operation.value:
                subject = self._known(operation.subject)
                value = self._known(operation.value)
                # SUBJECT + VALUE must resolve to known identities — the
                # fabrication barrier is here. THE PREDICATE IS NOT REQUIRED:
                # the write path also stores the predicate as None (it occurs
                # in ~0.5% of text). Requiring the predicate counted a
                # natural sentence like "kartal bir kuştur" as fabrication
                # and eliminated it with -1.0, and _speak fell to the raw
                # dump — measured.
                if subject is None or value is None:
                    return -1.0     # a claim about an unknown name: suspected fabrication
                predicate = (self._known(operation.predicate)
                             if operation.predicate else None)
                claims.append((subject, predicate, value))
        passed, dropped = self.gate.supported(claims)
        if dropped:
            return -1.0             # an unsupported claim: the sentence falls
        support = len(passed) / max(len(claims), 1) if claims else 0.0
        history = self._history(records)
        covered = len(passed) / max(len(records), 1)
        return support + history + covered

    def _history(self, records):
        """What happened when I said something like this — the brake coming from experience."""
        about = [one.subject for one in records]
        lived = self.memory.recall(about, most=6)
        if not lived:
            return 0.0
        return sum(one.outcome * (0.5 + one.surprise)
                   for one in lived) / len(lived)

    # --- experience -----------------------------------------------------

    def _live(self, line, said, records):
        """Records what it said. The reaction comes later, bound via `reacted`."""
        about = [one.subject for one in records[:3]]
        about.append(self.memory.self_key)
        self.last = self.memory.lived(said or line, outcome=0.0,
                                      surprise=0.0, about=about)

    def reacted(self, outcome, expected=0.0):
        """The user's reaction arrived: weight the experience by SURPRISE.

        If the prediction held it is cheap, if it failed it is expensive and
        permanent — the reason touching the stove once is stronger than a
        thousand repetitions.
        """
        held = getattr(self, "last", None)
        if held is None:
            return None
        held.outcome = float(outcome)
        held.surprise = self.gate.weigh(expected, outcome)
        return held

    # --- helpers --------------------------------------------------------

    def _vector_of(self, text):
        """The sentence's crude vector: the average of its words.

        The audit caught it: `resolve` and `recall` were being called from
        the session entirely without vectors — the continuous layer was
        unreachable code at runtime and polysemy fell back to popularity
        again. The context vector comes from here; it loses order, it
        suffices for the first cull (the reader network's representation
        will take its place).
        """
        from v3.dataset import fold
        held = []
        for word in text.split():
            start, stop = 0, len(word)
            while start < stop and not word[start].isalnum():
                start += 1
            while stop > start and not word[stop - 1].isalnum():
                stop -= 1
            piece = fold(word[start:stop]) if stop > start else ""
            if piece and piece in self.vectors:
                held.append(self.vectors[piece])
        return geometry.mean(held)

    def _identity(self, label):
        """Turns a label into an identity; opens one if absent. Context picks which one."""
        if not label:
            return None
        key, score = geometry.resolve(self.memory, label,
                                      self.vectors.get(label))
        if key is not None:
            return key
        return self.memory.identify(label, vector=self.vectors.get(label))

    def _known(self, label):
        """Turns a label into an EXISTING identity — does NOT open a new one.

        Used in the weighing: if the generator invented a new name, no
        counterpart is found and the claim counts as unsupported.
        """
        if not label:
            return None
        key, _ = geometry.resolve(self.memory, label,
                                   self.vectors.get(label))
        return key

    def _render(self, records):
        """The raw record dump when there is no network — it cannot speak, but it does not lie."""
        lines = []
        for record in records[:MOST]:
            subject = self._label(record.subject)
            predicate = self._label(record.predicate)
            value = self._label(record.value)
            mark = " [~]" if self.gate.doubtful(record) else ""
            lines.append(f"{subject} → {predicate} → {value}"
                         f" ‹{record.source}›{mark}")
        return " · ".join(lines)

    def _label(self, key):
        # A None/empty predicate must not be printed as the string "None":
        # the write path stores the predicate as None (it occurs in ~0.5% of
        # text) and `str(None)` polluted both the raw dump and the speaker's
        # input with "None" garbage — measured, "none" leaked into the
        # output. None -> empty.
        if key is None:
            return ""
        held = self.memory.identities.get(key) if isinstance(key, int) else None
        if held is not None and held.labels:
            return held.labels[0]
        return str(key)

    # --- maintenance ----------------------------------------------------

    def sleep(self):
        """The sleep round: distill, dampen, arbitrate. Returns the count dictionary."""
        return dynamics.sleep(self.memory)

    def curious(self, most=10):
        """The memory's gaps — where to look."""
        return dynamics.gaps(self.memory, most)

    def tension(self):
        """The contradiction pressure — what to be uneasy about."""
        return dynamics.pressure(self.memory)

    def save(self):
        if self.path:
            self.memory.save(self.path)
