"""Orchestrator — binds all of LMM's organs into a single flow.

    message
      → EXTRACT (Qwen)     kind + triples  (CANDIDATE — cannot write)
      → WRITE  → gate.admit                 into the graph, through the gate
      → ASK    → retrieve → generate (Qwen) → verify (GATE) → answer
      → CHAT   → generate (Qwen, fact-free) → verify (GATE) → answer

Carries the memory (`Memory.load/save`), records lived experience. Non-fabrication
is protected on EVERY path via `verify`: whatever Qwen says, a factual claim with
no support in the graph is dropped at the output. "Qwen cannot write records"
(input) + "Qwen cannot state unsupported facts" (output) — the backbone rule at
both ends.

Mode: STRICT (fabrication 0 — unsupported fact is dropped) · ASSIST (flagged).
"""
import os
import re

from lmm.core import dynamics
from lmm.core.dataset import bare, fold
from lmm.core.gate import Gate
from lmm.core.memory import Memory, OPERATOR, STRANGER, DOCUMENT
from lmm.core.transitive import Transitivity
from lmm import (evidence, extract, generate, link, lookup, research, retrieve,
                 verify)

SLEEP_EVERY = 50   # sleep every N turns (distill/fade/adjudicate) — v3 §117

# SOURCE-TRUST threshold: a fact spoken with trust BELOW this gets TAGGED in the
# answer (hedge + source). Operator teaching is certain → untagged; document,
# distillation and the web → "I'm not sure, according to ...". "Strictest" would
# be misinformation.
#
# It is not a number of its own: it is the LINE BETWEEN THE TWO BANDS the core
# defines, so it moves with them instead of having to be kept in step by hand.
CERTAIN = (Memory.trust_of(OPERATOR) + Memory.trust_of(DOCUMENT)) / 2

FALLBACK_DONT_KNOW = "I don't know."   # last-resort only: the normal
# refusal is engine-generated in the USER'S language; this string is hit
# only when generation itself fails.

# THE UNCERTAINTY MARK. A verified answer resting on a below-CERTAIN fact is
# spoken with its provenance attached, and that attachment used to be a SECOND
# MODEL CALL (`generate.hedge_note`) whose output was concatenated onto the
# answer AFTER every gate had run. A field trial measured what that costs: four
# of six wrong answers in twelve documents entered through it, including
# "I do not know. RFC 9110 is obsoleted by RFC 9111." — a fabricated sentence
# glued to a perfect abstention.
#
# The hedge carries no information the session does not already hold, so it is
# built from what is stored: a tilde for "not certain" and the source stamp the
# record already wears. It is FORMAT, not language — the same standing as the
# colon separator and the '#' prefix CONTRIBUTING permits — so it needs no
# engine, cannot fabricate, and reads the same in every language.
UNCERTAIN = "~"
UNKNOWN_SOURCE = "#source"     # a low-trust record with no stamp of its own


def _direct_lookup():
    """Is the graph-first answer path on? Read per call, not at import, so a
    measurement can flip it between two runs in one process."""
    return os.environ.get("LMM_DIRECT_LOOKUP", "1") != "0"


def _grounded_in(value, message):
    """Does the VALUE to be taught actually appear in the user's MESSAGE — did
    Qwen not fabricate it? "You cannot teach what you didn't say". Fold + stem
    matching (a stem against its inflected form). This prevents extract from
    mistaking a question ("what is an atom") for a WRITE and inventing a
    nonexistent value to write into the graph — language-agnostic."""
    want = {fold(w) for w in re.findall(r"\w+", value) if len(w) >= 3}
    if not want:
        return True        # short/token-less value — don't block (rare)
    have = {fold(w) for w in re.findall(r"\w+", message) if len(w) >= 3}
    return any(a == b or a.startswith(b) or b.startswith(a)
               for a in want for b in have)


class Session:
    """One conversation. Qwen (language) + graph/gate (truth & growth)."""

    def __init__(self, path=None, who="#operator", mode="STRICT"):
        self.path = path
        self.memory = (Memory.load(path) if path and os.path.exists(path)
                       else Memory())
        self.gate = Gate(self.memory)
        # Transitivity tracker: learns from data, incrementally (see
        # lmm/core/transitive.py — the full re-scan per cell was quadratic).
        self._transitive = Transitivity(self.memory)
        # Wire the semantic RIVAL check for contradictions to Qwen (cached) —
        # in the predicate-less case bird/predator coexist, bird/fish is a
        # contradiction. See _are_rivals.
        self._rival_cache = {}
        self.gate.rival = self._are_rivals
        self._pending = None    # subject offered for research (ask-first flow)
        self._mark = ""         # this turn's provenance mark (see UNCERTAIN):
        #                         recorded while answering, attached in
        #                         `respond` and ONLY to a turn that asserted
        self.last_written = []  # triples the gate admitted this turn (harvest)
        # WHAT THIS TURN DID, in the session's own words. These are not
        # features of the conversation; they are the session REPORTING ITSELF,
        # so that a reader (a benchmark, a harvest) does not have to infer from
        # the answer's WORDING what the code already knows. `last_abstained` in
        # particular is the structural replacement for guessing at refusal
        # phrases in one language — see `_refuse`.
        self.last_abstained = False
        self.last_kind = ""     # extract's classification: WRITE / ASK / CHAT
        self.last_subject = ""  # the subject label the turn was about, if any
        self.last_from_graph = False   # the turn was answered by `lookup` —
        # straight out of the graph, with no model call anywhere in it
        self.history = []       # SHORT-TERM conversation context (last N turns) —
        # the graph is long-term memory; this is the "what did we just talk
        # about" context (conversation continuity)
        self.who = who
        self.level = OPERATOR if who == "#operator" else STRANGER
        self.mode = mode
        self.vectors = {}
        self.turns = 0
        if self.memory.self_key is None:
            self.memory.self_key = self.memory.identify("#self")
        self._identity = self._seed_identity()
        # CAUSALITY predicate — a causal fact (cause→effect) is stored as a
        # NORMAL Record under this reserved predicate; it inherits the entire
        # gate/verify/inverse-index/derivation machinery for free. PHYSICALLY
        # separate from is-a (type) → verify/contradiction won't confuse the
        # relation type. Label is language-neutral (#causes).
        self._causes_key = link.resolve(self.memory, "#causes", self.vectors,
                                        create=True)
        # EVIDENCE STORE — document sentences (so numbers/ranges/nuance that
        # don't fit a triple aren't lost; representation-narrowness fix). The
        # graph is structure, the sentence is evidence.
        self.evidence = evidence.SentenceStore.load(path)

    def _seed_identity(self):
        """IDENTITY as a fact in the graph: (lmm → creator → rüzgar). This way
        identity too is 'known knowledge' — it passes through the gate and is
        spoken language-independently. The persona (prompts.CHAT_SYSTEM) gives
        Qwen its name; this seed keeps it in the graph.

        The RELATION label is English, like every other identifier here — it
        used to be Turkish, and a Turkish relation node in the seed graph meant
        the one fact the system holds about itself was stored in a language a
        German or Spanish user never uses. The graph's labels are read by the
        engine, which answers in the QUESTION'S language whatever language the
        fact is stored in (that is exactly what the mixed-language corpora
        measure). The owner's NAME stays as it is spelled: a name is not a
        word to be translated."""
        lmm = link.resolve(self.memory, "lmm", self.vectors, create=True)
        owner = link.resolve(self.memory, "rüzgar", self.vectors, create=True)
        maker = link.resolve(self.memory, "creator", self.vectors, create=True)
        if self.gate.behind(lmm, maker, owner) is None:
            self.gate.admit(lmm, maker, owner, "#operator", OPERATOR)
        self._lmm_key = lmm    # identity subject — _chat gather uses this
        return {lmm, owner, maker, self.memory.self_key}

    def _are_rivals(self, old_key, new_key):
        """Are two values semantic RIVALS (same slot, mutually exclusive)? Qwen
        judges (language-agnostic), the verdict is cached — the same pair never
        calls the model again. The gate uses this for the contradiction
        decision.

        THREE-VALUED: True (rivals) · False (they coexist) · None (cannot
        judge right now → the gate records a SUSPECT link and sleep judges it
        in bulk)."""
        ck = frozenset((old_key, new_key))
        if ck in self._rival_cache:
            return self._rival_cache[ck]
        # BULK STRUCTURAL INGESTION: no engine calls per cell — tables
        # legitimately attach many values to one subject (a repeated 'Note:'
        # key made EVERY cell an engine round-trip: minutes for one manual).
        # The answer here is PENDING, not "they coexist". It used to be False,
        # and the audit measured what that cost: paris and berlin written into
        # one slot in bulk ended with no link, no pressure and both speakable —
        # the deferral had silently become an acquittal. The gate now keeps the
        # suspicion and `dynamics.sleep(verdict=...)` settles the batch.
        if getattr(self, "_bulk", False):
            return None
        # HIERARCHICAL LINK (GRAPH — LMM's strength): if the two values are
        # connected by an is-a chain (kedigil→memeli: a felid is a mammal) they
        # are NOT rivals, they coexist — "a lion is both felid and mammal" is
        # hierarchy, not contradiction. Resolve with the graph without asking
        # Qwen (are_rivals was over-firing). Ask Qwen only for UNCONNECTED
        # values (like bird/fish).
        if self._connected(old_key, new_key):
            self._rival_cache[ck] = False
            return False
        old = link.label_of(self.memory, old_key)
        new = link.label_of(self.memory, new_key)
        try:
            verdict = bool(old and new and generate.are_rivals(old, new))
        except Exception:                                   # noqa: BLE001
            verdict = False        # if unsure, don't count as contradiction (don't break)
        self._rival_cache[ck] = verdict
        return verdict

    def _connected(self, a, b, depth=4):
        """Are a and b connected in the graph via an is-a chain (both
        directions)? If connected they are hierarchical → not a contradiction.
        Bounded BFS (cycle-protected)."""
        for start, goal in ((a, b), (b, a)):
            seen, frontier = {start}, [start]
            for _ in range(depth):
                nxt = []
                for node in frontier:
                    for k in self.memory.by_subject.get(node, ()):
                        v = self.memory.records[k].value
                        if v == goal:
                            return True
                        if v not in seen:
                            seen.add(v)
                            nxt.append(v)
                if not nxt:
                    break
                frontier = nxt
        return False

    # --- CAUSALITY (ms, symbolic — not neural) -------------------------
    def learn_cause(self, cause_label, effect_label, source=None):
        """Teach a causal fact: cause→effect, passing THROUGH THE GATE (Qwen
        cannot write). A normal Record under the #causes predicate → separate
        from is-a. Direct for now; the next step will wire extract's causal
        sentence into here."""
        ck = link.resolve(self.memory, cause_label, self.vectors, create=True)
        ek = link.resolve(self.memory, effect_label, self.vectors, create=True)
        if ck is None or ek is None or ck == ek:
            return False
        record, _ = self.gate.admit(ck, self._causes_key, ek,
                                    source or self.who, self.level)
        return record is not None

    def _learn_causal(self, message):
        """Catch a causal sentence → learn_cause. is_causal (Qwen, DIRECTED
        few-shot) + _grounded_in (are both entities in the message —
        fabricated-entity block). Otherwise None → falls back to the normal
        is-a write. HONEST LIMIT: _grounded_in validates the entities but NOT
        the DIRECTION (a 3B can flip it — the design's biggest risk)."""
        try:
            pair = generate.is_causal(message)
        except Exception:                                   # noqa: BLE001
            return None
        if not pair:
            return None
        cause, effect = pair
        if not (_grounded_in(cause, message) and _grounded_in(effect, message)):
            return None                    # entity not in the message → Qwen fabricated it
        if not self.learn_cause(cause, effect):
            return None
        self.memory.lived(f"cause:{cause}->{effect}", outcome=1.0,
                          about=[self.memory.self_key])
        return generate.confirm_cause(cause, effect, message) or "OK"

    def _causal_answer(self, message, direction, subject):
        """Answer a causal question: ms-traversal (causes/effects) → Qwen turns
        it into a sentence → verify audits. If empty, refuse (no fabrication).
        HONEST LIMIT: verify._has_edge is predicate-blind (doesn't distinguish
        is-a/causes) — a causality-aware anchor is future work; for now the
        edge EXISTS so it doesn't fall as unsupported."""
        if direction == "causes":
            edges = [(c, subject) for c in self.causes_of(subject)]
        else:
            edges = [(subject, e) for e in self.effects_of(subject)]
        if not edges:
            return None                 # nothing causal to say — fall through
        # CAUSE/EFFECT-labeled block (internal scaffold) — so on a REVERSE-
        # direction question like "kanserin sebebi ne" the model can flip the
        # arrow and find the cause (it stumbled on unlabeled arrows). The label
        # is internal prompt structure, not output language.
        block = "\n".join(f"[{i}] CAUSE: {c}  EFFECT: {e}"
                          for i, (c, e) in enumerate(edges, 1))
        raw = generate.answer(message, block)
        # WORD-COVERAGE gate (benchmark finding): a causal sentence doesn't fit
        # the is-a pattern — reextract mistook "yağarsa" for a value and dropped
        # a CORRECT answer. Principle: if ALL of the answer's content-words come
        # from the given block+question there is NO new claim → fabrication is
        # structurally impossible, pass. If a word steps outside, the old strict
        # verify applies. Not a language rule — set coverage.
        if evidence.covered(raw or "", block, message):
            return raw
        allowed = set()
        for c, e in edges:
            allowed.add(link.resolve(self.memory, c))
            allowed.add(link.resolve(self.memory, e))
        safe = verify.verify(self.memory, raw, allowed, self.mode, anchor="edge")
        # NOT A REFUSAL — None, so the caller falls through to the evidence
        # path. This route was the only ONE-WAY DOOR in `_respond`: every other
        # branch treats its own failure as "the reading was wrong, try the
        # normal path" (see the WRITE branch's fall-through), but a causal
        # question whose graph block produced nothing speakable ended the turn
        # in a refusal, even when the DOCUMENT said the answer plainly.
        #
        # Measured (corpus, "norgul çoğalırsa ne olur"): the causal edge exists,
        # so this route is taken; its verify drops the sentence and the turn
        # refuses — while the evidence path, traced on the same question,
        # produces "Norgul çoğalırsa morlan çoğalır." and passes EVERY gate
        # (digits_ok, coverage 1.0, read-back). The answer was reachable and
        # the door was shut. Nothing is loosened: the fall-through lands on
        # `_answer`, which applies the same gates.
        return safe or None

    def causes_of(self, effect_label):
        """The effect's CAUSES (labels). ms: by_value inverse index,
        O(in-degree). Pure graph walk — no neural generation."""
        ek = link.resolve(self.memory, effect_label, self.vectors)
        if ek is None:
            return []
        return [link.label_of(self.memory, self.memory.records[k].subject)
                for k in self.memory.by_value.get(ek, ())
                if self.memory.records[k].predicate == self._causes_key]

    def effects_of(self, cause_label):
        """The cause's EFFECTS (labels). ms: by_subject, O(out-degree)."""
        ck = link.resolve(self.memory, cause_label, self.vectors)
        if ck is None:
            return []
        return [link.label_of(self.memory, self.memory.records[k].value)
                for k in self.memory.by_subject.get(ck, ())
                if self.memory.records[k].predicate == self._causes_key]

    def root_causes(self, effect_label, depth=4):
        """The causal CHAIN leading to effect — bounded-depth BFS
        (cycle-protected). Walks back to the root causes. Returns the
        path/intermediate steps; does NOT AUTOMATICALLY write X→Z (causality is
        not always transitive — condition-4 safe). ms."""
        ek = link.resolve(self.memory, effect_label, self.vectors)
        if ek is None:
            return []
        seen, frontier, chain = {ek}, [ek], []
        for _ in range(depth):
            nxt = []
            for node in frontier:
                for k in self.memory.by_value.get(node, ()):
                    r = self.memory.records[k]
                    if r.predicate != self._causes_key or r.subject in seen:
                        continue
                    seen.add(r.subject)
                    chain.append((link.label_of(self.memory, r.subject),
                                  link.label_of(self.memory, node)))
                    nxt.append(r.subject)
            if not nxt:
                break
            frontier = nxt
        return chain            # [(cause, effect)] edges — root→leaf chain

    def respond(self, message, fluent=False):
        """Answer one message + update the CONVERSATION CONTEXT. The real logic
        is in _respond; this wrapper keeps the last N turns in `history`
        (conversation continuity). `last_written`: the triples the GATE
        admitted this turn — for the consolidation harvest (gate-approved =
        trustworthy training target; NOT the model's own raw output).

        `fluent`: ask for a SENTENCE rather than the record. The graph-first
        path (`lookup`) answers with the stored value and its field name, which
        costs nothing and reads the same in every language; a fluent sentence
        is the engine's work and the engine's bill, so it is offered rather
        than assumed. Everything the graph path cannot decide is generated
        either way — this flag only chooses which of the two answers a
        graph-decidable question gets."""
        self.last_written = []
        self.last_abstained = False
        self.last_kind = ""
        self.last_subject = ""
        self.last_from_graph = False
        self._mark = ""
        said = self._respond(message, fluent=fluent)
        if said and not self.last_abstained and not self.last_from_graph:
            # A GRAPH ANSWER NEEDS NO READING BACK. The re-extractor below is
            # asked "did this turn assert anything", and it is a MODEL CALL —
            # on the graph path the answer is a record the graph holds, so the
            # question is already settled and asking it would put a model call
            # back into the one path built to have none.
            # A REFUSAL DOES NOT ONLY COME OUT OF THE REFUSAL DOOR. The engine
            # can decline inside a perfectly normal answer — handed evidence
            # that does not cover the question, the answer prompt is instructed
            # to say it doesn't know, and that sentence passes every gate
            # because it claims nothing. Measured: three of the corpus's four
            # absence questions abstain exactly this way, and `_refuse` never
            # sees them.
            #
            # So the flag's real definition is not "which branch ran" but "did
            # this turn ASSERT anything", and the system already owns the organ
            # that answers it: the same re-extractor the fabrication gate
            # trusts to find the claims in a generated sentence. No claims in
            # what was spoken = nothing was asserted = an abstention, in
            # whatever language it was written. One short model call per
            # answering turn, and only on turns that did not already refuse.
            self.last_abstained = not self._asserted_a_fact(said)
        # THE MARK GOES ON LAST, AND ONLY ON A STATEMENT. Provenance qualifies a
        # claim; there is nothing to qualify in "I don't know", and the field
        # trial's four fabrications were all text appended to exactly that. So
        # the attachment waits until the turn has been read for whether it
        # asserted anything — the same organ the fabrication gate trusts — and
        # an abstention leaves this method byte-for-byte as the answer path
        # produced it.
        if said and self._mark and not self.last_abstained:
            said = f"{said} ({UNCERTAIN} {self._mark})"
        if message and message.strip():
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": said or ""})
            self.history = self.history[-12:]     # last ~6 turns (sliding window)
        return said

    def _asserted_a_fact(self, said):
        """Did the spoken answer CLAIM anything — the language-free half of the
        abstention reading.

        The engine's own annotations are not claims: a parenthetical hedge and
        a '#source' tag are the answer's footnote, and reading them as
        assertions would make every sourced refusal look like a statement. That
        is FORMAT, not language — parentheses and a '#' prefix — and it is the
        same distinction the benchmark scorer draws.

        Errs towards ASSERTED: if the re-extractor cannot be reached, the turn
        is treated as having spoken, because claiming an abstention that did
        not happen is the damaging direction (it would let a wrong answer be
        scored as an honest refusal)."""
        text = re.sub(r"\([^)]*\)", " ", said)
        text = " ".join(w for w in text.split() if not w.startswith("#"))
        if not text.strip():
            return False
        try:
            for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
                if sentence.strip() and extract.reextract(sentence):
                    return True
        except Exception:                                   # noqa: BLE001
            return True
        return False

    def _refuse(self, message):
        """DECLINE to answer — the single door every refusal leaves by.

        The refusal SENTENCE is engine-generated in the user's language, which
        is right for the user and useless for a measurement: to score whether
        the system abstained, a reader had to recognise "I don't know" in
        whatever language it came out in, and the benchmark scorer did that
        with a list of Turkish phrases. That made the MEASUREMENT
        language-dependent even where the system was not — an English document
        would have scored every honest abstention as a fabrication.

        The session already knows. Routing every refusal through one method
        turns that knowledge into a fact on the object (`last_abstained`), and
        the scorer reads the fact instead of guessing at the words. No gate
        moves: this method only records what the code was already doing.
        """
        self.last_abstained = True
        return generate.refusal(message) or FALLBACK_DONT_KNOW

    def _graph_answer(self, record, message):
        """Speak a record the graph already holds — the whole answer, with no
        model call in it.

        Everything the paid path spends calls on is either already decided or
        does not apply. There is no generation to gate: the sentence IS the
        record, so it cannot step outside the facts and there is nothing for
        `verify` to catch. There is no read-back: the claim is not a reading of
        the evidence, it is the stored triple. There is no relation check: the
        question named the field or the value, which is the very condition
        `lookup.find` returned on.

        What is NOT skipped is provenance. A record below CERTAIN is spoken
        with its own source stamp exactly as the generated path speaks it —
        same `_hedge`, same '~' mark attached in `respond` — because a cheaper
        answer is not a less accountable one.
        """
        self.last_from_graph = True
        self.last_abstained = False
        self.last_kind = extract.ASK
        # SEVERAL RECORDS ARE ONE ANSWER (`lookup`'s shape 3). They share the
        # subject by construction — the group is built inside one subject's
        # records — and the LEAST trusted of them decides whether the answer is
        # hedged, because the answer is only as sure as its least sure line.
        held = record if isinstance(record, tuple) else (record,)
        record = min(held, key=lambda r: r.trust)
        self.last_subject = link.label_of(self.memory, record.subject)
        said = lookup.render(self.memory, held if len(held) > 1 else record)
        if record.trust < CERTAIN:
            said = self._hedge(said, record, message) or said
        return said

    def _respond(self, message, fluent=False):
        """Answer one message. The return is always text; never an unsupported
        fact.

        Robustness (code-audit): empty message is guarded, the whole flow is
        inside try/except (one broken generation must not crash the
        conversation — falls back to a safe answer)."""
        if not message or not message.strip():
            return ""
        # RESEARCH APPROVAL (ask-first): if last turn offered "shall I
        # research?", check whether this message is the APPROVAL. If yes,
        # fetch+learn; otherwise drop the offer and process the message
        # normally (it may be a new question).
        # AUTO-SLEEP: every SLEEP_EVERY turns distill/decay/adjudicate once.
        # Measured in v3 — these mechanisms were only ever called manually, the
        # system never "slept"; now wrapped. If it errors, it must not crash
        # the conversation.
        self.turns += 1
        if self.turns % SLEEP_EVERY == 0:
            try:
                self.sleep()
            except Exception:                               # noqa: BLE001
                pass
        # GRAPH FIRST — before the extractor, because the extractor is the
        # first of the 6.6 model calls a question used to cost and skipping it
        # afterwards would still have paid for it. `lookup.find` is a graph
        # walk over an O(1) label index: microseconds, no engine, no network.
        # It returns a record only for the two shapes it can settle safely (see
        # lookup's docstring) and None for everything else, so the semantic
        # path below keeps every question it used to answer.
        #
        # It is skipped while a research offer is OUTSTANDING: that turn's
        # meaning is "yes"/"no" to a question this session asked, not a lookup,
        # and the pending offer has to be consumed by the flow that made it.
        #
        # `LMM_DIRECT_LOOKUP=0` takes the whole graph-first path out, which is
        # the A/B CONTROL: same binary, same graph, same questions, one
        # variable. It exists so a claim about what this path costs or breaks
        # can be measured rather than argued, and it reads the environment for
        # the same reason `LMM_BACKEND` does — a measurement's variable is not
        # a deployment's setting.
        if not fluent and self._pending is None and _direct_lookup():
            try:
                held = lookup.find(self.memory, message)
            except Exception:                               # noqa: BLE001
                held = None
            if held is not None:
                return self._graph_answer(held, message)
        try:
            op = extract.extract(message)
            self.last_kind = op.get("kind") or ""
            # RESEARCH APPROVAL (architecture — content decides): if last turn
            # offered "shall I research?", is this message NEW CONTENT
            # (teaching/question) or a pure affirmation — EXTRACT tells. New
            # content = new turn (pending is dropped, processed normally
            # below); only a content-free affirmation triggers the research.
            # Previously is_affirmative was called blindly; "biliyor musun
            # balina da memelidir" was mistaken for "yes" and the turn was lost.
            if self._pending is not None:
                (subj, orig_q), self._pending = self._pending, None
                new_content = bool(op["triples"] and op["triples"][0][0])
                if not new_content:
                    try:
                        if generate.is_affirmative(message):
                            return self._research(subj, orig_q)
                    except Exception:                       # noqa: BLE001
                        pass
                # new content or not an approval → processed normally below
            if op["kind"] == extract.WRITE and op["triples"]:
                # Is it a CAUSAL sentence ("X causes Y") — store SEPARATE from
                # is-a (#causes predicate). If so learn_cause; else the normal
                # is-a write.
                caused = self._learn_causal(message)
                if caused:
                    return caused
                said = self._write(op["triples"], message)
                if said:
                    return said
                # Empty write (wrong WRITE classification / valueless triple —
                # e.g. "X nedir" wrongly came as WRITE): instead of
                # fabricating, treat it like a QUESTION → retrieve/refuse.
                # Falls through.
            subject = op["triples"][0][0] if op["triples"] else None
            self.last_subject = subject or ""
            subject_key = (link.resolve(self.memory, subject, self.vectors)
                           if subject else None)
            # IDENTITY ROUTE (architectural bridge): if the subject CANNOT BE
            # RESOLVED (CHAT, or "who made you" where extract can't resolve
            # "sen"), classify deterministically: is this an IDENTITY question?
            # If so SKIP extract's gamble and route into the path that
            # GUARANTEES fetching the identity fact from _lmm_key. Called only
            # when the subject is unresolved (no extra call on a clear "kartal
            # nedir").
            if subject_key is None:
                try:
                    if generate.is_identity_question(message):
                        return self._identity_reply(message)
                except Exception:                           # noqa: BLE001
                    pass
            # Is it a CAUSAL QUESTION — ask Qwen ONLY if the subject carries a
            # causal edge (ms pre-check: causes_of/effects_of non-empty). No
            # wasted model call; never asked for a subject with no causal
            # knowledge.
            if subject and (self.causes_of(subject) or self.effects_of(subject)):
                try:
                    cq = generate.is_causal_question(message)
                except Exception:                           # noqa: BLE001
                    cq = None
                if cq:
                    said = self._causal_answer(message, cq[0], cq[1])
                    if said:
                        return said
                    # the causal route had nothing speakable → the normal
                    # retrieval path still gets its turn (see _causal_answer)
            if op["kind"] in (extract.WRITE, extract.ASK):
                return self._answer(message, subject)
            return self._chat(message)
        except Exception:                                   # noqa: BLE001
            # A crashed turn says nothing, which is an abstention like any
            # other — and the flag has to say so, or a benchmark would read
            # the fallback sentence as an answer.
            self.last_abstained = True
            return FALLBACK_DONT_KNOW

    # --- identity (deterministic route) --------------------------------
    def _identity_reply(self, message):
        """Answer an IDENTITY question deterministically. Root cause (audit):
        an identity question became ASK in extract, "sen" couldn't be
        resolved, and the fact was NEVER fetched. Here, GUARANTEE fetching the
        fact from _lmm_key; give the name + the facts to identity_answer;
        verify anchor='value' (the subject is a self-referential pronoun →
        None; the edge path would drop identity, the object 'rüzgar' anchors
        in allowed)."""
        id_records = retrieve.gather(self.memory, self._lmm_key)
        id_block = "\n".join(
            f"{link.label_of(self.memory, r.subject)} "
            f"{link.label_of(self.memory, r.predicate)} → "
            f"{link.label_of(self.memory, r.value)}" for r in id_records)
        name = link.label_of(self.memory, self._lmm_key)
        raw = generate.identity_answer(message, name, id_block)
        safe = verify.verify(self.memory, raw, self._identity, self.mode,
                             anchor="value")
        return safe or self._refuse(message)

    # --- writing -------------------------------------------------------
    def _write(self, triples, message):
        """Puts the taught material into the graph — through the gate (Qwen
        doesn't write, the gate writes).

        Code-audit: a valueless half-triple is skipped (no None value must
        reach the gate); if the same triple arrives twice it is deduplicated
        (no repeated 'learned it').
        """
        wrote = []
        conflicts = []
        seen = set()
        for subject, predicate, value in triples:
            if not value:
                continue                       # half triple — don't write
            if not _grounded_in(value, message):
                continue     # value not in message → Qwen fabricated it ("atom
                             # is X" mistaken for a statement): the mistake-a-question-for-
                             # WRITE trap, don't write
            sk = link.resolve(self.memory, subject, self.vectors, create=True)
            vk = link.resolve(self.memory, value, self.vectors, create=True)
            # sk == vk: SELF-LOOP ("almanya → almanya") — a thing cannot be
            # itself, meaningless record. extract produced it occasionally;
            # don't let it through the gate.
            if sk is None or vk is None or sk == vk or (sk, vk) in seen:
                continue
            seen.add((sk, vk))
            pk = (link.resolve(self.memory, predicate, self.vectors, create=True)
                  if predicate else None)
            # CONTRADICTION: does the new fact clash with a record carrying a
            # DIFFERENT value under the same subject+predicate — check before
            # writing (the CONTRA link is created in admit; we SURFACE it to
            # the user: no silent overwrite).
            clash = self.gate._contradiction(sk, pk, vk)
            if clash is not None:
                conflicts.append((link.label_of(self.memory, sk),
                                  link.label_of(self.memory, clash.value),
                                  link.label_of(self.memory, vk)))
            record, _ = self.gate.admit(sk, pk, vk, self.who, self.level)
            if record is not None:
                wrote.append((link.label_of(self.memory, sk),
                              link.label_of(self.memory, vk)))
                # consolidation harvest: the CANDIDATE triple itself that
                # passed the gate (NOT the node label — the node might have
                # been wrongly merged and the extractor would be taught to
                # produce text absent from its input; code-review finding #3).
                # These are extract._clean output: folded, grounded in the
                # message.
                self.last_written.append((subject, predicate or "", value))
                # DERIVATION (transitive reasoning): "eagle→bird, bird→animal ⊢
                # eagle→animal". Transitivity is learned from data (triangle
                # with ≥2 witnesses), the inference is written with the
                # #inference source at low trust. This is the "adds its own
                # interpretation" mechanism — wrapped from v3.
                self._learn_transitive(pk, sk, vk)
                if pk in self.memory.transitive:
                    self._derive(sk, pk, vk)
        if not wrote:
            return ""
        # LIVED EXPERIENCE (design note — the audit said "recall/weigh
        # unused"): in lmm, lived experience is DELIBERATELY write-only. In v3
        # experience WEIGHED candidate answers by surprise; in lmm the answer
        # comes from facts+gate, there is NO candidate-weighing — so
        # recall/weigh are meaningless here. lived() is a "what did I do"
        # journal: sleep/distillation prunes it, it stays for provenance.
        # Forcing a connection adds unused complexity (dilutes), doesn't
        # harden.
        self.memory.lived(str(wrote), outcome=1.0,
                          about=[self.memory.self_key])
        # DYNAMIC LANGUAGE: the confirmation sentence is not HAND-WRITTEN
        # Turkish — Qwen produces it in the user's language (the learned fact +
        # any contradiction). Since the facts are written to the graph, the
        # confirmation is safe.
        said = generate.confirm(wrote, conflicts, message)
        return said or "OK"

    # --- document ingestion (RAG-less learning) -------------------------
    def learn_text(self, text, source="#document", deep=True):
        """Ingest a TEXT (document/paragraph) into the graph — in place of
        embed/chunk-RAG.

        RAG stores the text and searches for CHUNKS by similarity at query
        time; LMM READS the text once, turns it into facts, writes them
        through the gate — answering is an ms graph-walk at query time, and
        multi-step derivation comes for free (RAG cannot combine chunks).
        Sentence by sentence: reextract (factual claims; greetings/comments →
        []) → the same protections as _write (grounding, self-loop) → DOCUMENT
        trust (below CERTAIN → source-tagged in answers, cannot override an
        operator fact). Returns: (written, skipped)."""
        wrote, skipped = 0, 0
        seen = set()          # same as _write: same triple across two sentences
        self.unread = []      # READING GUARANTEE: sentences that couldn't be
        #                       learned — NO silent skipping, the caller sees
        #                       them (honesty)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?;])\s+|\n+", text)
                     if s.strip()]
        # EVIDENCE: every sentence enters the store — numbers/ranges/nuance
        # that don't fit a triple come back as evidence at answer time
        # (representation-narrowness fix).
        for sent in sentences:
            self.evidence.add(sent, source)
        # THE MASTHEAD IS A BLOCK, NOT A SENTENCE (`evidence.front_matter`).
        # Publisher, date and the author's institution are written once, at the
        # top, as lines that do not flow into each other — and the sentence
        # windows above index them one detached line at a time, which is how
        # two real documents lost every question about who published them and
        # when. Read off the layout: the run before the first line that ends a
        # sentence.
        head = evidence.front_matter(text)
        if head:
            self.evidence.add(head, source)
        # CONTEXT WINDOWS (`evidence.SCALES` — a geometric ladder, defined with
        # the store): neighboring sentences are indexed together too —
        # the sentence "Uyarı: Eşik ayarı..." doesn't carry the word 'sepsis'
        # but carries the section; single-sentence granularity was losing the
        # section context (hospital finding). Two scales: narrow (3) for
        # precise matching, wide (6) for the heading↔content bridge (so the
        # word in a section heading and the Warning line at the section's end
        # meet in the same window — the "kamera ... KVKK" class).
        # A third, RECORD scale (12): a document record is a heading plus its
        # labelled lines ("Ne:/Neden:/Kâr:/Uyarı:") and runs longer than 6
        # sentences. Measured (hospital trace, "sepsis ... uyarısı"): the
        # heading sits 8 sentences above its "Uyarı:" line, so NO window at
        # either narrower scale held both the subject and its answer — the
        # answer was unreachable at every commit in this history.
        #
        # THE WIDEST SCALE IS FOR RECORDS, AND THE DOCUMENT SAYS WHAT ONE IS.
        # A record is made of SHORT lines; the same number of LONG sentences is
        # prose, which the middle scale already covers, and indexing it only
        # dilutes — a diluted wide window displaced a precise spec line and cost
        # the manual a point (measured: manual 27->26 with the scale ungated,
        # hospital 12->13 with it).
        #
        # That guard used to be `len(window) > 700`, and 700 was this document's
        # measurement: twelve sentences of about fifty-eight characters, counted
        # off the PDF in front of whoever wrote it. The document supplies the
        # bound instead — the MEDIAN sentence is the middle of what this text
        # writes, so a window of `size` sentences is a record when it is no
        # longer than `size` median sentences, and prose, whose sentences are the
        # long half, is above it. Nothing about any document is known here.
        typical = sorted(len(s) for s in sentences)
        median = typical[len(typical) // 2] if typical else 0
        widest = evidence.SCALES[-1]
        for size in evidence.SCALES:
            step = max(2, size // 3)        # a constant overlap, not a per-scale
            #                                 choice: each window starts a third
            #                                 of a width after the last
            for i in range(0, max(1, len(sentences) - size + 1), step):
                window = " ".join(sentences[i:i + size])
                if size >= widest and len(window) > size * median:
                    continue
                if len(window) > len(sentences[i]):
                    self.evidence.add(window, source)
        # TABLE REPAIR (evidence-only): PDF tables shatter row by row —
        # "Dijital patoloji" and the tier cell "3" land apart, the model grabs
        # the wrong number (hospital test finding; RAG made the same mistake
        # on the same table). `evidence.table_windows` puts the ROWS back —
        # reading the row length off the cell-length period and the row's start
        # off the layout's own column indentation — and where the layout says
        # nothing it falls back to windows of consecutive lines. The reason it
        # is not simply a sliding window any more: a window that spanned two
        # rows put cells side by side that the DOCUMENT NEVER PUT TOGETHER, and
        # an answer read out of one of those ("the camera's regulatory risk is
        # medium", against the document's "Yüksek (KVKK)") is past every gate
        # by construction, because the claim really is in the evidence.
        for window in evidence.table_windows(text):
            self.evidence.add(window, source)

        # INSTANT-READY mode (deep=False): evidence layer only — ZERO model
        # calls, even 500 pages become QUERYABLE within seconds (answers via
        # the evidence path, with coverage+support gates). Graph extraction is
        # completed later with deep=True / during sleep-consolidation:
        # "read-and-be-ready first, digest in the background" — same as the
        # brain's fast/slow learning.
        if not deep:
            return 0, 0

        def _read(sent):
            """The PURE-READ stage of one sentence (model calls; doesn't touch
            the graph → parallel-safe). Returns: (sentence, causal|None,
            triples)."""
            causal = generate.is_causal(sent)
            if causal and _grounded_in(causal[0], sent) \
                    and _grounded_in(causal[1], sent):
                return sent, causal, []
            triples = extract.reextract(sent)
            if not triples:
                # SECOND READ (reading guarantee): if reextract returned empty
                # try the other prompt — one prompt was one chance, the
                # "karvel kayboldu" class.
                second = extract.extract(sent)
                if second["kind"] == extract.WRITE:
                    triples = second["triples"]
            return sent, None, triples

        # PARALLEL READING: sentences are independent — on the API backend
        # they're read concurrently (20 min → minutes). The local engine is
        # not parallel-safe → 1 worker.
        backend = os.environ.get("LMM_BACKEND", "")
        workers = int(os.environ.get("LMM_INGEST_WORKERS",
                                     "8" if backend == "azure" else "1"))
        if backend != "azure":
            workers = 1     # the local torch model is NOT thread-safe (review
            #                 #3): even an env override doesn't permit parallel
            #                 local generation
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                readings = list(pool.map(_read, sentences))
        else:
            readings = [_read(s) for s in sentences]

        # SERIAL WRITING: the graph/gate is single-threaded — order is kept.
        for sent, causal, triples in readings:
            wrote_before = wrote
            if causal:
                self.learn_cause(causal[0], causal[1], source=source)
                wrote += 1
                continue
            for subject, predicate, value in triples:
                if not value or not _grounded_in(value, sent):
                    skipped += 1
                    continue
                sk = link.resolve(self.memory, subject, self.vectors,
                                  create=True)
                vk = link.resolve(self.memory, value, self.vectors,
                                  create=True)
                if sk is None or vk is None or sk == vk or (sk, vk) in seen:
                    skipped += 1
                    continue
                seen.add((sk, vk))
                pk = (link.resolve(self.memory, predicate, self.vectors,
                                   create=True) if predicate else None)
                record, _ = self.gate.admit(sk, pk, vk, source, DOCUMENT)
                if record is None:
                    skipped += 1
                    continue
                wrote += 1
                self._learn_transitive(pk, sk, vk)
                if pk in self.memory.transitive:
                    self._derive(sk, pk, vk)
            if wrote == wrote_before:
                self.unread.append(sent)    # NO fact came out of this sentence
        if wrote:
            self.memory.lived(f"document:{source}:{wrote}", outcome=1.0,
                              about=[self.memory.self_key])
        return wrote, skipped

    # --- structured ingestion (table — extractor-LESS) ------------------
    def learn_cell(self, subject_label, predicate_label, value_label, source):
        """A SINGLE structural fact — NO model call, still THROUGH THE GATE.
        The only entry path for already-structured data such as a table cell /
        a 'Key: Value' header. Returns: number of records written (0|1)."""
        sk = link.resolve(self.memory, subject_label, self.vectors, create=True)
        vk = link.resolve(self.memory, value_label, self.vectors, create=True)
        if sk is None or vk is None or sk == vk:
            return 0
        pk = (link.resolve(self.memory, predicate_label, self.vectors,
                           create=True) if predicate_label else None)
        record, _ = self.gate.admit(sk, pk, vk, source, DOCUMENT)
        if record is None:
            return 0
        self._learn_transitive(pk, sk, vk)
        if pk in self.memory.transitive:
            self._derive(sk, pk, vk)
        return 1

    def learn_rows(self, rows, source="#table"):
        """Table rows → graph, extractor-LESS (the permanent form of the Excel
        lesson): row=entity, column=predicate, cell=value. NO model call →
        deterministic and ms; a 500-page table takes seconds. Every triple
        still goes through the gate (DOCUMENT trust). The row sentence is also
        written to evidence; the row's most information-dense cell becomes an
        ALIAS of the row node so a question can find it by its natural name
        ("yangın ekipmanı tespiti..."). rows: [{column: value}]."""
        wrote = 0
        prev_bulk = getattr(self, "_bulk", False)
        self._bulk = True       # no per-cell engine calls (see _are_rivals)
        for row in rows:
            cells = [(str(k).strip(), str(v).strip()) for k, v in row.items()
                     if str(v).strip()]
            if not cells:
                continue
            self.evidence.add(" · ".join(f"{k}: {v}" if k else v
                                         for k, v in cells), source)
            # A ROW'S INDENTATION IS NOT PART OF ITS NAME. A spreadsheet cell
            # has no margin, so a sheet that nests rows draws the nesting with
            # a character — the Census file writes `.Alabama`, `.Puerto Rico` —
            # and that dot made every nested row unreachable by the name a
            # question uses for it (`benchmarks/field/REPORT.md` §4: `about(
            # "puerto rico")` came back empty while `about(".puerto rico")`
            # held the whole row). What counts as layout is decided by
            # Unicode's category (`core/dataset.bare`), not by this sheet's
            # convention, so no document is being read here.
            #
            # THE CELL IS THE LABEL AND THE PLAIN NAME IS THE ALIAS, in that
            # direction: the node goes on carrying exactly what the document
            # wrote, and the reachable name is added beside it. The other
            # direction would have the reader silently editing a cell it does
            # not understand — `-5` would lose its sign to the same rule.
            anchor = cells[0][1]
            sk = link.resolve(self.memory, anchor, self.vectors, create=True)
            if sk is None:
                continue
            plain = bare(anchor)
            if plain and plain != anchor:
                self.memory.identify(fold(plain), same_as=sk)
            rich = max((kv for kv in cells[1:]), default=None,
                       key=lambda kv: len(kv[1]))
            # AN ALIAS HAS TO BE A NAME, and a name is a PHRASE — more than one
            # word. The test used to be "at least 8 characters", a length picked
            # off a spreadsheet; what it was reaching for is that a code, a date
            # or a status word ("NEW", "2026-03-01") is not what anyone would
            # call the row by, while "yangın ekipmanı tespiti" is. Word count
            # says that without measuring any document.
            if rich and len(rich[1].split()) >= 2:
                self.memory.identify(fold(rich[1]), same_as=sk)
            for col, val in cells[1:]:
                wrote += self.learn_cell(anchor, col, val, source)
        self._bulk = prev_bulk
        if wrote:
            self.memory.lived(f"table:{source}:{wrote}", outcome=1.0,
                              about=[self.memory.self_key])
        return wrote

    # --- derivation (transitive reasoning — from v3) -------------------
    def _learn_transitive(self, predicate, subject=None, value=None):
        """Is this predicate transitive — has the graph seen closed triangles
        with ≥2 witnesses. Transitivity is learned from DATA, not BY HAND:
        an is-a relation learns from is-a examples; "loves" never does (love
        chains don't
        close in the graph). One coincidental triangle isn't enough (wrong-
        inference hole) — at least two independent triangles.

        The rule lives in `v3.transitive`; this is the seam. Even walking only
        the inverse index, this was a RE-SCAN of the whole predicate per
        written cell, and a table's columns are a handful of predicates
        carrying every row: measured 2000 cells 0.91s vs 10000 cells 25.6s —
        quadratic. The tracker counts only what the NEW EDGE closes and keeps
        the running answer, negative included."""
        self._transitive.observe(predicate, subject, value)

    def _derive(self, subject, predicate, value):
        """TWO-DIRECTIONAL chain inference around the new edge. The derived
        record is written with the #inference source, at LOW trust, linked to
        its justification — it stays separable as inference rather than
        observation, and the gate won't let it be spoken like a fact."""
        forward = [(subject, r.value, [r.key])
                   for r in self.memory.about(value, touch=False)
                   if r.predicate == predicate]
        # INVERSE INDEX: take the records with subject as VALUE from by_value
        # (O(N)→O(degree)).
        incoming = [(r.subject, value, [r.key])
                    for r in (self.memory.records[k]
                              for k in self.memory.by_value.get(subject, ()))
                    if r.predicate == predicate and r.subject != value]
        for who, what, because in forward + incoming:
            if who != what and not self.gate.behind(who, predicate, what):
                self.gate.inferred(who, predicate, what, because)

    # --- answering -----------------------------------------------------
    def _answer(self, question, subject_label):
        """Answer the question by fetching from the graph + Qwen; the gate
        drops any claim stepping outside the INJECTED facts."""
        subject = (link.resolve(self.memory, subject_label, self.vectors)
                   if subject_label else None)
        fallback_keys = []
        if subject is None or not self.memory.by_subject.get(subject):
            # SUBJECT FALLBACK (manual-benchmark finding): extract may hand us
            # a subject ('cihaz') that resolves to nothing, while ANOTHER
            # question word ('ekranı'→'ekran') IS a graph node carrying the
            # answer. Try the remaining content words against the O(1) label
            # index — pure graph, longest (most specific) words first. UNION,
            # not first-match: with only the first hit, a generic word
            # ('cihazın'→cihaz) shadowed the field node ('ekranı'→ekran) and
            # the answer-carrying fact never entered the block (measured).
            # No CAP on how many of the question's words may contribute: it used
            # to stop at three, which is a number with no argument behind it —
            # every one of these is a word the QUESTION itself used and a node
            # the graph really holds, so there is nothing to ration. The block
            # is bounded downstream (evidence outranks document triples, and
            # `retrieve.specific` prunes), and a question has only so many words.
            #
            # THE ORDER IS A TOTAL ONE, and that is a determinism fix of the same
            # family as the one in `evidence.find`: this sorted a SET OF STRINGS
            # by length alone, so equal-length question words came out in
            # PYTHONHASHSEED order — and the first of them becomes the SUBJECT of
            # the whole answer. Byte-identical code could pick a different
            # subject between two runs.
            for w in sorted(set(evidence._words(question)),
                            key=lambda t: (-len(t), t)):
                cand = link.resolve(self.memory, w)
                if cand is not None and cand not in fallback_keys \
                        and self.memory.by_subject.get(cand):
                    fallback_keys.append(cand)
            if fallback_keys:
                subject = fallback_keys[0]
        # associative=False: the answer is built ONLY from direct facts
        # (consistent with the edge check — see retrieve.gather /
        # verify._has_edge).
        records = retrieve.gather(self.memory, subject, associative=False)
        for extra_key in fallback_keys[1:]:
            for r in retrieve.gather(self.memory, extra_key,
                                     associative=False):
                if r not in records:
                    records.append(r)
        # EVIDENCE (representation-narrowness fix): document sentences
        # intersecting the question — numbers/ranges/nuance that don't fit a
        # triple come from here. The graph is structure, the sentence is
        # evidence.
        seats = evidence.WINDOW
        proof = self.evidence.find(f"{subject_label or ''} {question}",
                                   most=seats)
        if proof and len(proof) < seats:
            # TWO-HOP (only when the first search did not FILL the block): a
            # second search with the best evidence's words — composition
            # questions want the union of two separate sections. When the first
            # search was already full, the second hop added table-crumbs and
            # derailed the answer (hospital trace) — don't touch when there's
            # plenty. "Scarce" used to be the constant 4 against a block of 6;
            # what it means is that retrieval came back UNDER-SUBSCRIBED, which
            # is a comparison with the number of seats asked for, not a number.
            for extra in self.evidence.find(f"{proof[0]} {question}",
                                            most=seats - len(proof)):
                if extra not in proof and len(proof) < seats:
                    proof.append(extra)
        # TRUE GAP (gaps signal) — if the graph holds NO fact at all about
        # this subject, offer research BEFORE GENERATION. That way the model's
        # polite "I don't know" (which passes verify and fills safe) does NOT
        # BLOCK the offer (Bug #2). Gap = curiosity = research. With facts
        # present this branch is never entered → it doesn't flee into
        # researching what it knows. If EVIDENCE exists it doesn't count as a
        # gap — answered from the sentence.
        if not records and not proof:
            if subject_label:
                self._pending = (subject_label, question)   # ask-first
                # The research OFFER is an abstention that ends in a question
                # mark: it states no fact and declines the one that was asked.
                # It is flagged as such, so the "did it abstain" reading does
                # not depend on the offer's wording either.
                self.last_abstained = True
                offer = generate.offer_research(subject_label, question)
                return offer or self._refuse(question)
            return self._refuse(question)
        # TARGETED EDGE (multi-hop — benchmark finding): if the question
        # mentions a second KNOWN concept ("zilfen bir canlı mıdır" → 'canlı')
        # and the graph knows that edge (derived included), move that record
        # to the FRONT — the graph had derived it but the answer selector was
        # voicing another fact. Pure graph, ms, no language.
        qwords = {fold(w) for w in re.findall(r"\w+", question) if len(w) >= 3}
        qwords.discard(fold(subject_label or ""))
        targeted = []
        for r in records:
            vlab = fold(link.label_of(self.memory, r.value))
            if vlab and any(w == vlab or vlab.startswith(w) or w.startswith(vlab)
                            for w in qwords):
                targeted.append(r)
        if targeted:
            records = targeted + [r for r in records if r not in targeted]
        # SPECIFICITY: with two true answers about one subject, the graph's own
        # is-a chain says which one answers (see retrieve.specific). It runs
        # AFTER targeting, so a question that NAMES the general value keeps it.
        records = retrieve.specific(self.memory, records,
                                    keep={r.key for r in targeted})
        # Facts and/or EVIDENCE exist → grounded answer + output gate.
        # EVIDENCE FIRST: it carries the full sentence; triples extracted from
        # prose can be crumbs ("projeler → en kolay") and, sitting at the
        # front of the block, derailed the answer (hospital trace). Graph
        # records step down to support.
        if proof:
            # With evidence present, DOCUMENT-sourced triples do NOT enter the
            # block: prose extraction can produce crumbs ("projeler → en
            # kolay") and, with ingestion variance, changed the answer from
            # run to run. The evidence is the FULL sentence of the same
            # document — no information loss, stability gained. Operator/
            # inference records (taught in conversation, derived) stay in the
            # block.
            #
            # WHAT THIS RULE ACTUALLY REACHES, checked rather than assumed
            # (COST.md §8.4): the stamp it tests is the one `Memory.learn(text)`
            # writes, `#doc:…`. A table adapter stamps `#xlsx:…` / `#pdf:…`, so
            # a cell written by `learn_rows` — `(row, column, value)`, with no
            # engine in the loop — was never excluded here at all. The rule only
            # ever reached prose extraction, which is exactly what it was built
            # for, and lifting it entirely was measured over the three corpora
            # (the only sets that carry the stamp): every column identical.
            records = [r for r in records
                       if not str(r.source).startswith("#doc")]
        fact_block = retrieve.facts_block(self.memory, records) if records else ""
        proof_block = "\n".join(f"[K{i}] {s}"
                                for i, s in enumerate(proof, 1)) if proof else ""
        block = ((proof_block + "\n" + fact_block) if proof_block and fact_block
                 else (proof_block or fact_block))
        # GENERATE AND SELECT AT THE GATE (see `_select`). One generation
        # bound the answer to a single sample of a sampling process; the
        # measured residue was not missing knowledge but an unstable CHOICE.
        safe, tried = self._select(question, records, proof, fact_block, block)
        if safe is None:
            # THE FALLBACK CANNOT UNDO THE RELATION GATE. This path re-verifies
            # the first candidate against the GRAPH, and the graph holds the
            # unasked relation as a perfectly good edge ("document → preparer →
            # X"), so the answer the relation gate just refused would be spoken
            # anyway — and so would a freshly generated one, since it answers
            # the same unasked question. Everything else keeps its old second
            # chance; only what THIS gate refused is out.
            spare = [t for t in tried if fold(t) not in self._refused]
            if tried and not spare:
                return self._refuse(question)
            allowed = verify.allowed_of(self.memory, records)
            safe = verify.verify(self.memory,
                                 spare[0] if spare
                                 else generate.answer(question, block),
                                 allowed, self.mode, anchor="edge")
            if not safe:
                return self._refuse(question)
        # SOURCE-TRUST (strictest): if the weakest fact is below CERTAIN,
        # source+hedge.
        if records:
            weakest = min(records, key=lambda r: r.trust)
            if weakest.trust < CERTAIN:
                safe = self._hedge(safe, weakest, question) or safe
        elif proof:
            # An EVIDENCE-ONLY answer carries a hedge too (review #5): the
            # source is DOCUMENT level — same source-transparency principle as
            # the graph path.
            src = next((s for s in self.evidence.last_sources if s), "#document")
            self._mark = src
        return safe

    # How many answers the answer step may generate. A BUDGET (model calls),
    # not a threshold.
    CANDIDATES = 3

    def _subsets(self, records, proof, fact_block):
        """The distinct EVIDENCE SUBSETS to answer from — at most CANDIDATES.

        Candidates have to differ in something that MATTERS. Sampling the same
        prompt again differs only in noise; what changes an answer's content is
        WHICH EVIDENCE it rests on. Retrieval hands back a ranked list, so the
        meaningful variation is a focus ladder: everything retrieved → the
        better-ranked half → the single best sentence. Distraction and context
        move in opposite directions along that ladder (the wide block carries
        the composition questions, the narrow one stops a neighbouring table
        row from attaching to the wrong attribute), and the gate score below
        decides which of the two the question needed — instead of one fixed
        width having to be right for every question.
        """
        sizes, subsets = [], []
        for size in (len(proof), max(1, len(proof) // 2), 1):
            if proof and size not in sizes:
                sizes.append(size)
                subsets.append(proof[:size])
        blocks = []
        for sub in subsets:
            pb = "\n".join(f"[K{i}] {s}" for i, s in enumerate(sub, 1))
            one = (pb + "\n" + fact_block) if fact_block else pb
            if one not in blocks:
                blocks.append(one)
        if not proof:
            # Graph-only path: the same ladder over the RECORDS, which are
            # ordered by targeting and trust.
            blocks.append(fact_block)
            if len(records) > 1:
                half = retrieve.facts_block(self.memory,
                                            records[:len(records) // 2])
                if half and half not in blocks:
                    blocks.append(half)
        return [b for b in blocks if b][:self.CANDIDATES]

    def _select(self, question, records, proof, fact_block, block):
        """Generate one answer per evidence subset, choose by GATE SCORE.

        Returns `(chosen | None, [raw answers])`.

            score = digit × coverage_ratio × evidence_use × supported

        Every factor is a gate that already existed; what is new is that they
        RANK instead of merely admitting or rejecting, so several candidates
        can be compared on the same scale:

        digit          VETO. `digits_ok` (digit present in the block AND in a
                       same-neighbour bigram) → 1. If only the looser
                       `digits_present` holds it scores 0.5 — the second tier
                       the old code had, kept as a DISCOUNT so a
                       strict-digit candidate always outranks it. A digit in
                       neither → 0, the candidate is gone. Untouchable.
        coverage_ratio how much of the answer comes from the block+question
                       (`evidence.coverage`). The old binary `covered` is the
                       special case ratio == 1.0.
        evidence_use   how much of the answer comes from the EVIDENCE ITSELF
                       rather than from echoing the question — this is what
                       separates a real answer from a fluent restatement of
                       the question, and it is the factor that pushes the
                       narrow-block candidate down when it had nothing to say.
        supported      the engine's read-back (`generate.supported`): does the
                       evidence actually SAY this. 0 kills the candidate.

        THE UNINVENTED-0 GUARANTEE IS UNCHANGED. A candidate with ratio < 1.0
        speaks only if `supported` confirms it — exactly the old second tier.
        On the graph-only path (no evidence sentences, so no read-back to
        appeal to) full coverage is still REQUIRED. Everything eliminated →
        `None`, and the caller refuses honestly.

        Candidates are scored on the FULL block, not on the subset they were
        given: the subset is a generation choice, the evidence we hold is what
        the answer must be true against.
        """
        # the evidence WITHOUT the [K..] labels: the labels are scaffolding,
        # and counting them as evidence words would credit an answer for
        # quoting the scaffold.
        evidence_text = "\n".join(proof) if proof else fact_block
        graded, tried, seen = [], [], {}
        for one in self._subsets(records, proof, fact_block):
            # warmth 0: each candidate is the DETERMINISTIC answer to its own
            # evidence, so the differences between candidates carry
            # information (which evidence) instead of sampling noise.
            raw = (generate.answer(question, one, warmth=0.0) or "").strip()
            if not raw:
                continue
            # TRIM BEFORE JUDGING, so that what every gate below reads is what
            # the user will read. A candidate arrives as one text and may be
            # two things — an honest refusal with a fabrication stapled to it
            # was three of the field trial's six wrong answers and both of the
            # corpus's remaining stable losses. `grounded_sentences` removes
            # the stapled half deterministically, and it can only ever remove
            # (see its docstring); trimming here rather than at the return
            # means coverage, the digit veto, the read-back and the relation
            # check all judge the surviving text instead of judging a text the
            # caller never sees.
            raw = evidence.grounded_sentences(raw, block, question).strip()
            if not raw:
                continue
            tried.append(raw)
            key = fold(raw)
            if key in seen:
                continue                # already scored — a subset re-derived it
                #
                # AGREEMENT IS NOT EVIDENCE HERE, and counting it cost two
                # corpus questions. The subsets are NESTED, so the candidates
                # are not independent witnesses, and narrow evidence
                # systematically produces the vaguer answer: with the wide
                # block saying "Morlan bir kuştur" and both narrow ones saying
                # "Morlan bir canlıdır", a vote handed the answer to the
                # generic one — twice as many voices, all reading less of the
                # document. Both claims are true; only one of them answers.
            if evidence.digits_ok(raw, block):
                digit = 1.0
            elif proof and evidence.digits_present(raw, block):
                digit = 0.5
            else:
                continue                # digit veto — not negotiable
            ratio = evidence.coverage(raw, block, question)
            if not proof and ratio < 1.0:
                continue                # no read-back to appeal to → strict
            use = evidence.coverage(raw, evidence_text) if evidence_text else 1.0
            # THE TIE-BREAK: does the answer NAME WHAT WAS ASKED. `coverage`
            # with the roles swapped measures how much of the QUESTION the
            # answer accounts for, and that is what separates two claims the
            # score cannot: asked "torvanit bir madde midir", "Torvanit bir
            # maddedir" answers and "Torvanit bir metaldir" changes the
            # subject, though both are true and both fully grounded. It is
            # only a tie-break — it can never rescue a candidate a gate
            # rejected, so a false-premise echo gains nothing by it (the
            # question's digits are excluded from coverage, and the read-back
            # still has to confirm). The same targeting the graph path already
            # does with `targeted` records, applied to the choice of answer.
            target = evidence.coverage(question, raw)
            # last resort: the candidate that read the MOST evidence, which is
            # the order the subsets were built in.
            entry = [digit * ratio * use, target, len(graded), raw]
            seen[key] = entry
            graded.append(entry)
        # GROUNDEDNESS ELIMINATES, RELEVANCE RANKS.
        #
        # Ranking the survivors by the groundedness score itself was measured
        # to be wrong: it prefers the candidate that COPIES the block, and the
        # answer prompt asks for a fluent sentence. Asked "torvanit bir madde
        # midir", "Evet, torvanit bir maddedir" scores below "Torvanit bir
        # metaldir" — for the word 'evet', which claims nothing — and the
        # question goes unanswered though both sentences are true. Corpus
        # median fell 15 -> 13 that way.
        #
        # Groundedness is a GATE property, not a quality ranking: once two
        # candidates are both admissible, being more literal does not make one
        # a better answer. So the score's job is elimination (a digit veto, and
        # under full coverage the read-back's confirmation), and among what
        # survives the order is: answers the question · more grounded · read
        # more of the evidence.
        graded = [e for e in graded if e[0] > 0]
        # `supported` is a model call, so ask in that order: it can only
        # multiply by 1 or 0, so the first candidate it confirms is the choice
        # and the rest need not be asked.
        graded.sort(key=lambda e: (-e[1], -e[0], e[2]))
        self._refused = set()
        for _score, _target, _rank, raw in graded:
            if proof and not self._read_back(raw, proof, block):
                continue
            # THE RELATION MUST BE THE ASKED ONE. Every gate up to here judges
            # the answer's own claim, and a claim built out of true material can
            # answer a question nobody asked — the last route to a confident
            # wrong answer. See `_relation_held`. Refusal is remembered so the
            # fallback below cannot speak it after all.
            if proof and not self._relation_held(question, raw, proof, block):
                self._refused.add(fold(raw))
                continue
            return raw, tried
        return None, tried

    def _relation_held(self, question, raw, proof, block):
        """Does the evidence carry THE RELATION THE QUESTION ASKS FOR.

        The remaining failure class had nothing wrong with its evidence: asked
        who SIGNED the document, the answer gave the document's PREPARER field.
        The claim is in the evidence word for word, so coverage is 1.0 and the
        read-back rightly says yes — a true fact answering a question nobody
        asked. Nothing that judges the ANSWER'S OWN CLAIM can see this, because
        the claim never mentions the relation it fails to be about; the
        proposition being judged has to come from the QUESTION
        (`generate.answers_asked`).

        WHY THIS IS STRUCTURAL, not a list of relation pairs. What the document
        asserts is a SLOT and its value — a field name in a record line, a
        predicate on a graph edge — and the question names a slot too. The
        condition is that those two slots be the same one. The structure says
        where the slots are (`evidence.field_names`, format only) and whether
        the answer is voicing an unasked one (`evidence.voices_other_slot`);
        deciding whether two NAMES are one slot needs meaning, since a question
        speaks plainly where a table abbreviates, and that is the engine's part.
        No pair of relations is written down anywhere, and nothing here knows a
        word of any language.

        The views mirror the read-back's, for the same measured reason (a judge
        searching 3000 characters wobbles): the evidence the answer DRAWS ON
        first, everything retrieved only if that fails.
        """
        views = [v for v in (self._focus_view(raw, proof), block) if v]
        return any(generate.answers_asked(question, raw, view)
                   for view in dict.fromkeys(views))

    def _focus_view(self, raw, proof, first=()):
        """The evidence a claim DRAWS ON: the retrieved sentences it shares
        content words with, most-shared first. A judge asked to find one line
        inside three thousand characters wobbles; this is the haystack the size
        of the needle. It can only ever be a SUBSET of what the gate already
        admitted.

        `first`: sentences to place at the head of the view whatever their
        overlap count — see `_relation_held`."""
        words = set(evidence._words(raw))
        # ORDER BY OVERLAP COUNT, NOT DENSITY. Density (overlap / length, and
        # the Jaccard and sqrt-damped forms of it) was the obvious suspicion —
        # a long prose window sharing five common words looks less relevant
        # than a three-token spec line sharing two — and it was MEASURED WRONG
        # here, twice over. On real recorded answers replayed against rebuilt
        # evidence (81 answer/proof pairs across all three benchmarks), count
        # ordering puts the proving line in the FIRST slot 81/81 times; density
        # scored 77/81, Jaccard 78/81, sqrt-damped 81/81 but with fewer proving
        # slots overall (159 -> 156 of 162). On synthetic correct answers (47
        # cases, the pairs the replay cannot reach because the answer was
        # refused) count ordering is again 47/47 and every density form 44/47.
        # The reason is that a claim's proof shares MORE with the claim, full
        # stop; dividing by length turns the measure into "how little else does
        # this line say", which rewards crumbs — the same way it did in the
        # retrieval score. Views do get 25% shorter under density, so if the
        # judge ever needs a smaller haystack, take it from the SEAT COUNT, not
        # from the ordering.
        overlap = sorted(proof, key=lambda s: -len(words
                                                   & set(evidence._words(s))))
        focus = list(first)
        focus += [s for s in overlap[:2]
                  if s not in focus and words & set(evidence._words(s))]
        return "\n".join(f"[K{i}] {s}" for i, s in enumerate(focus, 1))

    def _read_back(self, raw, proof, block):
        """Does the evidence SAY this — asked of the evidence it RESTS ON.

        This is where the measured oscillation actually lives. Traced on the
        manual with every candidate identical and temperature 0, the same
        claim against the same block came back False, then True, then False:
        the answers were stable and the JUDGE was not. A judge is being asked
        to find one line inside three thousand characters of flattened PDF, and
        that search — not the answer — is what wobbles. Two questions from the
        manual were losing that coin flip.

        So the claim is judged first against the evidence sentences it actually
        DRAWS ON (the ones sharing content words with it), and only if that
        fails against everything retrieved. Same judge, same veto, a haystack
        the size of the needle. It also fixes the opposite failure — a claim
        that IS in the block verbatim ('İŞLEMCİ : Intel Celeron G 3902') was
        being denied because it sat buried in unrelated spec noise.

        The narrow view can only be a SUBSET of the retrieved evidence, so
        nothing outside what the gate already admitted can be confirmed here.
        """
        views = []
        for view in (self._focus_view(raw, proof), block):
            if view and view not in views:
                views.append(view)
        return any(generate.supported(raw, view) for view in views)

    def _hedge(self, answer, record, message):
        """RECORDS the provenance mark for a VERIFIED answer resting on a
        low-trust fact. It returns the answer UNCHANGED and appends nothing:
        the mark is attached in `respond`, after the turn has been read for
        whether it asserted anything (see UNCERTAIN).

        The label is the record's own stamp — what the writer stored, not a
        sentence about it. The previous version asked the engine to compose a
        caveat and pasted the result onto an already-verified answer, outside
        every gate; that is where two thirds of this system's measured false
        statements came from."""
        self._mark = record.source or UNKNOWN_SOURCE
        return answer

    # --- agentic research (approve, learn from the web) ----------------
    def _research(self, subject_label, question):
        """The user approved → fetch from Wikipedia, extract triples, write to
        the graph with #web+low trust, then answer the ORIGINAL question
        normally (source-tagged). The web is UNTRUSTED: the fact enters not as
        'I know it' but source-stamped+low trust (condition-4)."""
        text, url = research.wiki_summary(subject_label)
        if not text:
            return self._refuse(question)
        # Extract ONE plain category (the essence, not taxonomic clutter) →
        # clean fact, clean answer. Many values/Latin terms were drowning the
        # small model.
        value = generate.category_from(subject_label, text)
        sk = link.resolve(self.memory, subject_label, self.vectors, create=True)
        vk = (link.resolve(self.memory, value, self.vectors, create=True)
              if value else None)
        if not value or sk is None or vk is None:
            return self._refuse(question)
        # #web stamp + DOCUMENT (0.6 < CERTAIN) → source-tagged in the answer.
        self.gate.admit(sk, None, vk, f"#web:{url}", DOCUMENT)
        self.memory.lived(f"web:{subject_label}", outcome=1.0,
                          about=[self.memory.self_key])
        return self._answer(question, subject_label)     # now in the graph → answer+hedge

    # --- chat ----------------------------------------------------------
    def _chat(self, message):
        """Chat answer — the gate still filters any leaking factual claim.

        HOLE CLOSED (F2): this used to be `safe or raw` — if verify found all
        sentences unsupported and dropped them, the RAW (unaudited) output was
        returned, bypassing the gate entirely. Now on empty it falls to the
        safe side, no raw fabrication is returned.
        """
        id_records = retrieve.gather(self.memory, self._lmm_key)
        # The IDENTITY block is PREDICATED (the maker must not be hidden) — so
        # "who made you" can be answered.
        id_block = "\n".join(
            f"{link.label_of(self.memory, r.subject)} "
            f"{link.label_of(self.memory, r.predicate)} → "
            f"{link.label_of(self.memory, r.value)}" for r in id_records)
        # CONVERSATION CONTEXT: give the recent turns too → conversation
        # continuity (what "look it up" refers to, the context of "what are you
        # doing"
        # is kept). Fabrication is still filtered in verify.
        raw = generate.chat(message, id_block, history=self.history)
        # In chat, allowed = ONLY the identity facts. anchor="value": the
        # subject is a self-referential pronoun (ben/beni) that can't be
        # resolved; it suffices that the OBJECT (rüzgar) is allowed; external
        # fabrication (Google) still falls. See verify.verify.
        safe = verify.verify(self.memory, raw, self._identity, self.mode,
                             anchor="value")
        return safe or self._refuse(message)

    # --- LMM strength: self-awareness (curiosity + contradiction pressure)
    def curiosity(self, most=5):
        """The system knows what it does NOT know: labels of concepts that
        appeared in conversation but have no / only-episodic records. Wraps
        `dynamics.gaps` (that organ existed in v3, unconnected to the lmm
        flow). The input of proactive research: 'I don't know this, shall I
        look it up'. Self/identity nodes excluded."""
        # Don't count predicate nodes (relation labels like type/property) as
        # curiosity — they aren't concepts. Self/identity nodes excluded too.
        predicates = {r.predicate for r in self.memory.records.values()
                      if r.predicate is not None}
        out = []
        for key, _gap in dynamics.gaps(self.memory, most=most * 4):
            if key in self._identity or key in predicates:
                continue
            label = link.label_of(self.memory, key)
            if label and label not in out:
                out.append(label)
            if len(out) >= most:
                break
        return out

    def tension(self, most=5):
        """The system is BOTHERED by contradiction: the highest-pressure
        contradictions (subject, [rival values]). Wraps `dynamics.pressure`.
        Can be surfaced to the user as 'I hold contradictory knowledge about
        this'."""
        from lmm.core.memory import CONTRA, SUSPECT
        out, seen = [], set()
        for rkey, _p in dynamics.pressure(self.memory):
            r = self.memory.records.get(rkey)
            if r is None or rkey in seen:
                continue
            # SUSPECTED rivals count too: a contradiction whose verdict is
            # still pending is exactly something to be uneasy about, and
            # bulk-ingested data holds nothing else until the next sleep.
            rivals = self.memory.rivals_of(r, (CONTRA, SUSPECT))
            seen.add(rkey)
            seen.update(x.key for x in rivals)
            subject = link.label_of(self.memory, r.subject)
            values = [link.label_of(self.memory, r.value)] + [
                link.label_of(self.memory, x.value) for x in rivals]
            out.append((subject, values))
            if len(out) >= most:
                break
        return out

    # --- maintenance ---------------------------------------------------
    def sleep(self):
        """The sleep round — distill, fade, arbitrate, AND settle the deferred
        contradiction suspicions.

        Bulk ingestion cannot afford the rivalry test per cell, so it defers
        the verdict (a SUSPECT link). Sleep is the batch pass and pays that
        debt: the test runs here, once per suspected PAIR, and the field is
        arbitrated afterwards. Without this the deferral was a leak — the
        suspicion was recorded and then nobody ever came back for it."""
        return dynamics.sleep(self.memory, verdict=self.verdict)

    def verdict(self, old_value, new_value):
        """The rivalry test as sleep asks it: OUTSIDE bulk mode, so the answer
        is a real verdict rather than another deferral."""
        prev = getattr(self, "_bulk", False)
        self._bulk = False
        try:
            return self._are_rivals(old_value, new_value)
        finally:
            self._bulk = prev

    def save(self):
        if self.path:
            self.memory.save(self.path)
            self.evidence.save(self.path)
