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
from lmm import (evidence, extract, generate, inflect, link, lookup,
                 research, retrieve, runtime, verify)

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

    # class-level default, so a Session built without __init__ (tests use
    # __new__ to skip the heavy constructor) still has a voice — an empty one
    persona = ""
    _no_teach = False
    _brief = ()
    voice_warmth = None
    voice_tokens = None
    style = ""
    _composed = False

    def __init__(self, path=None, who="#operator", mode="STRICT",
                 persona="", warmth=None, reply_tokens=None, style=""):
        # THE OPERATOR'S VOICE — tone, greeting style, when to ask a
        # clarifying question. It rides in front of the PHRASING prompts only
        # (generate._voiced): the gates read the output, never the prompt, so
        # a persona is structurally unable to loosen what may be spoken —
        # the invariant test (W11) orders a fabrication through it and
        # watches the turn refuse anyway.
        self.persona = persona or ""
        # THE VOICE KNOBS — they travel exactly where the persona travels
        # (chat, answer, refusal, compose) and nowhere else: a judge whose
        # thermostat the caller can turn is not a judge. Unset = the
        # measured defaults, byte for byte.
        self.voice_warmth = warmth
        self.voice_tokens = reply_tokens
        # THE OPERATOR'S FORMAT SHEET — the persona's mirror twin. The
        # persona is the voice and stops at the document's edge; the style
        # is the document's shape (sections, headings, ordering) and
        # travels ONLY to the composer's request. Its words are
        # operator-attested the way the brief's are; the material still
        # owns every fact beneath them.
        self.style = style or ""
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
        # THE CONSULTATION'S LEDGER — the topic terms of a no-teach
        # conversation, harvested from the extractor's triples turn by turn.
        # Raw sentences were tried first and measured wrong: conversational
        # words ("you", "actually", "for me") are RARE in formal documents,
        # so the seat weighting read them as discriminative and handed the
        # catalogue to whichever document happens to quote its reader — the
        # third member of the IDF-inversion family. The extractor is the
        # organ whose one job is "what does this sentence talk about", and
        # it is already called on every turn; its terms are the brief.
        self._brief = []
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
        raw = generate.answer(message, block,
                              **self._voice(persona=self.persona))
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

    def _voice(self, **base):
        """The caller's voice settings, laid over a surface's defaults —
        only the knobs the caller actually set."""
        if self.voice_warmth is not None:
            base["warmth"] = self.voice_warmth
        if self.voice_tokens is not None:
            base["max_tokens"] = self.voice_tokens
        return base

    def respond(self, message, fluent=False, teach=True, on_line=None):
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
        # which CONTRACT this turn runs under, readable by the organs below:
        # a conversation that may not teach is a CONSULTATION, and two of
        # its behaviours (the search reading prior turns, the composer
        # bridge) exist only there — the teaching surface and the
        # benchmarks keep their single-turn semantics byte for byte.
        self._no_teach = not teach
        # what the CONVERSATION was about a turn ago — kept before the reset,
        # for the follow-up whose own words name no subject (see the
        # retrieval call)
        self._prior_subject = self.last_subject
        self.last_subject = ""
        self.last_from_graph = False
        self._mark = ""
        self._last_proof = []
        # the tally too — it is written by retrieval, and a turn that never
        # retrieves (chat, context statements) must not inherit the previous
        # turn's census, or the informed refusal would answer small talk
        # with a document count.
        self._census_line = ""
        self._spec_wants = None
        self._spec_chat = None
        self._composed = False
        self._on_line = on_line
        said = self._respond(message, fluent=fluent, teach=teach)
        if said and not self.last_abstained and not self.last_from_graph \
                and not self._composed:
            # (a composed delivery is exempt the way a graph answer is: every
            # line of it was admitted by the composer's gate, so it ASSERTS by
            # construction — re-reading it would put a model call back into a
            # path that already proved itself)
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
        # THE INFORMED REFUSAL. A turn that declines while the census is
        # rich is refusing with its hands full: the memory could not answer
        # WHAT WAS ASKED, but it holds an attested tally of documents that
        # speak to the topic, and "I don't know" hides it. So the engine is
        # asked once more, over a block that is ONLY the tally, and the
        # replacement speaks solely if it survives the same reading as any
        # other turn — no invented number, no words beyond tally and
        # question, and it must actually assert. The trigger is a STATE
        # (refused + tally in hand), not any reading of the question's
        # wording — which is what lets "what would you recommend" get a
        # useful, sourced reply without anyone classifying intent.
        # THE COMPOSER BRIDGE. The store can HOLD the very catalogue the
        # user is asking for while the refusal speaks — measured, live:
        # "you decide everything, give me the material" ended in a shrug
        # with compose() sitting unreachable behind a command prefix. The
        # bridge is a STATE in the informed-refusal family: a CONSULTATION
        # turn (teach=False) that ended in abstention, with prior turns in
        # hand, asks the engine one cheap structured question — is this
        # turn asking me to PRODUCE something? — and on yes the composer
        # runs over the whole consultation as its brief. The draft passes
        # the composer's own gate as always; when the composer refuses,
        # this turn falls back to the paths below unchanged. A teaching
        # turn never asks the question, so the benchmarks cannot pay a
        # call for a bridge they never cross.
        if (said and self.last_abstained and self._no_teach
                and not self.last_from_graph
                and any(h.get("role") == "user" for h in self.history)):
            wants = False
            try:
                spec_wants = getattr(self, "_spec_wants", None)
                wants = (bool(spec_wants.result()) if spec_wants is not None
                         else generate.wants_material(message))
            except Exception:                           # noqa: BLE001
                pass
            if wants:
                brief = " ".join([h["content"] for h in self.history
                                  if h.get("role") == "user"] + [message])
                try:
                    text, sources = self.compose(
                        brief, topics=list(self._brief) or None,
                        on_line=getattr(self, "_on_line", None))
                except Exception:                       # noqa: BLE001
                    text, sources = "", []
                if text and sources:
                    said = text
                    self.last_abstained = False
                    self._composed = True
        census_line = getattr(self, "_census_line", "")
        if (said and self.last_abstained and census_line
                and not self.last_from_graph):
            # v2: THE OFFER GETS REASONS, NOT JUST NAMES. With only the tally
            # to read, the best rescue the engine could write was a list; the
            # named documents are sitting in the store with the very
            # sentences that justify naming them. So the rescue block is
            # rebuilt the way the composer gathers: floor off (weak material
            # is harmless — the gate below still rules the OUTPUT), source
            # cap giving the spread, every line datelined, tally at the end.
            # The gate is v1's, unchanged: names and numbers only from what
            # the block attests.
            gathered = self.evidence.find(message, most=8, floor_share=0.0)
            origins2 = list(self.evidence.last_sources[:len(gathered)])
            named_in_tally = set()
            for src, _hits in self.evidence.last_census:
                named_in_tally.add(src)
            lines = [f"{evidence._source_name(src)} — {text}"
                     for text, src in zip(gathered, origins2)
                     if src in named_in_tally]
            rescue_block = "\n".join(
                f"[K{i}] {line}"
                for i, line in enumerate(lines + [census_line], 1))
            offered = (generate.answer(message, rescue_block, warmth=0.0,
                                       persona=self.persona) or "").strip()
            # WHAT THE OFFER MUST NOT DO IS INVENT A MEMBER OR A NUMBER.
            # Full word-coverage was tried first and refused every honest
            # phrasing over its connectives ("and", "both" are in no tally).
            # The dangerous tokens have a SHAPE: a designation is written
            # with capitals off the sentence start, and a quantity with
            # digits — the same format cue the benchmark scorer reads. Every
            # such token must be in the tally or the question; the plain
            # words in between are the engine doing its one job, phrasing.
            offered = self._strip_marks(offered)
            if offered and not self._substance_ok(
                    offered, lines, message):
                offered = ""
            if offered and evidence.digits_ok(offered, census_line):
                # ACCENT-BLIND ON BOTH SIDES, locally. The engine writes
                # "Zeka" where the document writes "Zekâ", and the core fold
                # normalises NFC before stripping combining marks — which
                # strips nothing, the same defect the benchmark scorer's
                # fold carried until this morning. Repairing core fold
                # re-keys every identity in every saved graph and is a
                # measured migration, not a rider on this gate (ledgered);
                # here both sides pass through one NFD-stripping normaliser,
                # so they cannot disagree.
                def _plain(text):
                    import unicodedata                     # noqa: PLC0415
                    return "".join(
                        ch for ch in unicodedata.normalize("NFD", text)
                        if not unicodedata.combining(ch))
                tally_words = {_plain(w)
                               for w in evidence._words(census_line)}
                allowed = (tally_words
                           | {_plain(w) for w in evidence._words(message)}
                           | {_plain(w) for w in evidence._words(
                                 "\n".join(lines))})
                sound, names_spoken = True, 0
                for m in re.finditer(r"\w+", offered, re.UNICODE):
                    token = m.group()
                    folded = _plain(fold(token))
                    # CREDIT reads every position: a tally name counts
                    # wherever it stands — offers habitually OPEN with the
                    # name, and the first cut's sentence-start exemption
                    # zeroed exactly those (measured: a correct reasoned
                    # offer refused as "naming no member")
                    if any(inflect.same_stem(folded, w)
                           for w in tally_words):
                        names_spoken += 1
                        continue
                    # SUSPICION keeps the exemption: an unknown capitalised
                    # word at a sentence start is ordinary orthography, not a
                    # designation
                    starts = m.start() == 0 or offered[
                        :m.start()].rstrip()[-1:] in ".!?:•-–—\n"
                    named = (any(ch.isupper() for ch in token[1:])
                             or (token[:1].isupper() and not starts)
                             or any(ch.isdigit() for ch in token))
                    if named and not any(inflect.same_stem(folded, w)
                                         for w in allowed):
                        sound = False
                        break
                # an offer that names no member offers nothing — the honest
                # refusal stands (an engine can decline twice, and a second
                # refusal must not be spoken as if it were a tally)
                if sound and names_spoken:
                    said = offered
                    self.last_abstained = False
        # THE MARK GOES ON LAST, AND ONLY ON A STATEMENT. Provenance qualifies a
        # claim; there is nothing to qualify in "I don't know", and the field
        # trial's four fabrications were all text appended to exactly that. So
        # the attachment waits until the turn has been read for whether it
        # asserted anything — the same organ the fabrication gate trusts — and
        # an abstention leaves this method byte-for-byte as the answer path
        # produced it.
        # THE CONVERSATION IS THE FALLBACK, NOT THE SHRUG. Measured with
        # the wager's receipts in hand: the answering chain abstained (its
        # candidates died at the gate wearing an unattested persona
        # introduction) and the turn ended in a bare refusal — while the
        # wagered chat reply, already written and already through its own
        # reading, said exactly the right thing and had been discarded on
        # a classifier's verdict. In a consultation, when delivery and the
        # informed refusal have both declined, that reply speaks; the bare
        # refusal remains for surfaces that have no conversation.
        if (said and self.last_abstained and self._no_teach
                and getattr(self, "_spec_chat", None) is not None):
            try:
                voiced = self._chat(message, self._spec_chat)
            except Exception:                           # noqa: BLE001
                voiced = ""
            if voiced:
                said = voiced
        # A TURN THAT SAYS NOTHING CARRIES NO STAMP. Measured: gate-trimmed
        # candidates could leave an empty text with the provenance mark
        # still set, and the caller printed a bare citation — a stamp on
        # silence. Silence is a refusal, and is flagged as one.
        if not (said or "").strip():
            self.last_abstained = True
            self._mark = ""
            said = FALLBACK_DONT_KNOW
        bare = said                      # what the voice may later re-read
        if said and self._mark and not self.last_abstained:
            said = f"{said} ({UNCERTAIN} {self._mark})"
        if message and message.strip():
            self.history.append({"role": "user", "content": message})
            # the history carries the UNSTAMPED text: the mark is the
            # system's, and a voice that reads it learns to imitate it
            self.history.append({"role": "assistant", "content": bare or ""})
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
        # THE EVIDENCE ALREADY ANSWERS THIS, WITHOUT A MODEL. A spoken claim
        # on the evidence path got here by sharing content words with the
        # block — the coverage gates admitted it on exactly that ground — and
        # a refusal shares none, which is `grounded_sentences`' own "claims
        # nothing" class. So when the turn had evidence, the reading is a word
        # intersection: any sentence that touches the proof asserted.
        #
        # Measured, and this is why the model call had to go: asked for a
        # programme's outcomes, the engine wrote one long list-sentence, the
        # re-extractor found no tidy triple in it, and the turn was stamped an
        # ABSTENTION — a full correct answer, recorded as a refusal, by the
        # organ every benchmark of this system trusts for that stamp. The
        # word intersection cannot make that mistake, costs nothing, and
        # removes one model call from every answering turn (the profiler had
        # it at 14% of a turn's wall clock).
        proof = getattr(self, "_last_proof", None)
        if proof:
            # THE MAJORITY, NOT A TOUCH. The first cut of this read "any
            # shared word = asserted", and one incidental function word sank
            # it: "I do not know" shares "not" with a standard full of
            # SHALL-NOTs, and an honest refusal was stamped an assertion — a
            # benchmark point lost to a stopword. A claim admitted by the
            # gates carries MOST of its words from the evidence; a refusal
            # carries almost none; the mixture in between asserted something
            # either way. More-than-half is the same non-dial boundary every
            # other mixture reading in this codebase uses.
            held = "\n".join(proof)
            return any(evidence.coverage(sentence, held) > 0.5
                       for sentence in re.split(r"(?<=[.!?])\s+",
                                                text.strip())
                       if sentence.strip())
        try:
            sentences = [x for x in re.split(r"(?<=[.!?])\s+", text.strip())
                         if x.strip()]
            # independent reads go side by side where the backend allows —
            # and the memo on reextract means sentences the gate already
            # read this turn cost nothing here
            return any(runtime.parallel_map(extract.reextract, sentences))
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
        return (generate.refusal(message,
                                 **self._voice(persona=self.persona))
                or FALLBACK_DONT_KNOW)

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

    def _respond(self, message, fluent=False, teach=True):
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
            # THE WAGER: on the consultation surface most turns end at the
            # chat voice, so the voice starts speaking WHILE the router
            # classifies — both calls leave together where the backend can
            # carry them, and the router's verdict decides whether the
            # prepared reply is used (CHAT, and the context statements) or
            # quietly dropped (a question takes the answer path as always).
            # One small discarded call on consultation question turns; a
            # teaching turn — the benchmarks — never places the bet.
            spec = None
            spec_queue = None
            spec_wants = None
            # THE WAGER AND THE STREAM ARE ONE MECHANISM. The first cut of
            # streaming stood the wager down and paid for it: the voice no
            # longer overlapped the router, and the FIRST sentence arrived
            # later than the old whole-reply had. So when a caller listens
            # (on_line), the wager fills a QUEUE of chunks from the turn's
            # first instant; routed to chat, the main flow drains it
            # sentence by sentence — routed elsewhere, the queue is
            # discarded like any lost wager.
            if (self._no_teach and runtime.parallel_ok()
                    and self._on_line is not None):
                import queue as _q
                from concurrent.futures import ThreadPoolExecutor
                spec_queue = _q.Queue()
                def _pour(q=spec_queue, msg=message):
                    try:
                        for chunk in generate.chat_stream(
                                msg, self._chat_id_block(),
                                history=self.history,
                                **self._voice(persona=self.persona)):
                            q.put(chunk)
                    except Exception:                   # noqa: BLE001
                        pass
                    q.put(None)                         # the stream's end
                pool = ThreadPoolExecutor(max_workers=2)
                pool.submit(_pour)
                spec_wants = pool.submit(generate.wants_material, message)
                pool.shutdown(wait=False)
            elif (self._no_teach and runtime.parallel_ok()
                    and self._on_line is None):
                from concurrent.futures import ThreadPoolExecutor
                pool = ThreadPoolExecutor(max_workers=2)
                spec = pool.submit(self._chat_raw, message)
                self._spec_chat = spec
                # the delivery question rides out beside the router too —
                # measured with the stream clock, asking it LAST put the
                # catalogue's first line at second 13 of an 18-second turn
                spec_wants = pool.submit(generate.wants_material, message)
                pool.shutdown(wait=False)
            elif self._no_teach and runtime.parallel_ok():
                from concurrent.futures import ThreadPoolExecutor
                pool = ThreadPoolExecutor(max_workers=1)
                spec_wants = pool.submit(generate.wants_material, message)
                pool.shutdown(wait=False)
            self._spec_wants = spec_wants
            op = extract.extract(message)
            self.last_kind = op.get("kind") or ""
            if self._no_teach and op.get("triples"):
                if not isinstance(self._brief, list):   # __new__-built
                    self._brief = list(self._brief)
                for t in op["triples"]:
                    for term in (t[0], t[2] if len(t) > 2 else ""):
                        term = (term or "").strip()
                        if term and term.lower() not in (
                                x.lower() for x in self._brief):
                            self._brief.append(term)
                self._brief = self._brief[-12:]         # recency window
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
            # A QUESTION API CANNOT WRITE MEMORY. `ask` passes teach=False:
            # a single-shot question, however imperative its grammar, is not
            # an act of teaching — measured: "draft a programme for the sales
            # team" was classified WRITE, written into the graph as operator
            # fact, and answered with a confirmation of having learned it.
            # The conversational surface keeps teach=True, because teaching
            # IS one of its jobs; what changes is that the caller now states
            # which contract it wants, instead of the classifier deciding
            # both.
            if teach and op["kind"] == extract.WRITE and op["triples"]:
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
            if not teach and op["kind"] == extract.WRITE:
                # AND THE MIRROR RULE, measured in the field: in a
                # conversation that may NOT teach, a statement is neither a
                # lesson nor a query — it is CONTEXT. "I am in banking and my
                # team is ten people" was classified WRITE (correctly), the
                # write was skipped (correctly), and the turn then fell into
                # the QUESTION treatment: retrieval found nothing to a
                # sentence that asked nothing, and the refusal spoke — with
                # the persona's greeting leaking into it. Three correct
                # mechanisms, one uncovered seam. The chat surface carries
                # the history and the persona; the sentence lands there and
                # the conversation simply continues. Nothing written, nothing
                # refused.
                return self._chat(message, spec, spec_queue)
            # A DELIVERY REQUEST GOES STRAIGHT TO THE COMPOSER. When the
            # wagered verdict is already in hand and says YES, the turn
            # routes to the organ built for deliverables before the
            # answering chain ever starts; the composer's own refusal is
            # the fallback that keeps the old path whole. Without the
            # wager (a backend that cannot parallelise) the tail bridge
            # below still catches the same state, as before.
            if spec_wants is not None and self._brief:
                try:
                    early_wants = bool(spec_wants.result())
                except Exception:                       # noqa: BLE001
                    early_wants = False
                if early_wants:
                    brief = " ".join(
                        [h["content"] for h in self.history
                         if h.get("role") == "user"] + [message])
                    try:
                        text, sources = self.compose(
                            brief, topics=list(self._brief) or None,
                            on_line=getattr(self, "_on_line", None))
                    except Exception:                   # noqa: BLE001
                        text, sources = "", []
                    if text and sources:
                        self.last_abstained = False
                        self._composed = True
                        return text
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
            return self._chat(message, spec, spec_queue)
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
                    # DERIVED: a window is these same sentences again, at a
                    # different width. It is evidence like any other, and it is
                    # marked only so that the offline expansion does not pay to
                    # expand the same words once per scale (`evidence.
                    # SentenceStore.pending_expansion`).
                    self.evidence.add(window, source, derived=True)
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
        workers = self._workers()
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

    @staticmethod
    def _workers():
        """How many sentences may be read at once. One definition, because the
        expansion pass reads the same way the extractor does and had no business
        deciding this a second time."""
        # A HOSTED ENDPOINT IS WAITED ON, A LOCAL MODEL IS RUN. The two cloud
        # backends are httpx calls whose cost is latency, so asking eight at
        # once costs nothing and saves seven waits; the local torch model is a
        # single object that is NOT thread-safe (review #3), and no env
        # override may permit parallel generation on it.
        backend = os.environ.get("LMM_BACKEND", "")
        hosted = backend in ("azure", "openai")
        workers = int(os.environ.get("LMM_INGEST_WORKERS",
                                     "8" if hosted else "1"))
        if not hosted:
            workers = 1
        return workers

    def expand(self):
        """OFFLINE DOCUMENT EXPANSION (doc2query--) — pay ONCE, at ingestion,
        for the questions each line of the document answers, and index them
        SEPARATELY so that a question asked in other words can still find the
        line that answers it.

        The measured weakness this is aimed at is the one a purely lexical
        index has by construction: the document writes `15.6"` and the reader
        asks "how many inches". No amount of inflection tolerance closes that —
        the words are not forms of one another — and the alternative on offer is
        an embedding index, which is the dependency stack and the similarity
        gamble this layer exists to avoid.

        WHAT IS PAID, PLAINLY: one model call per unit the document wrote,
        once. That is why it is not on by default — `Memory.learn(...,
        expand=True)` asks for it — and why the measurement of what it costs is
        reported beside what it buys (`benchmarks/COST.md` §9).

        WHAT IS NOT AT RISK: nothing generated here can be spoken. It goes to
        `evidence.SentenceStore.expansions`, never to `.sentences`, so the
        answer block — and therefore everything the gate audits and everything a
        reader sees — is still the document. `LMM_EXPAND=0` skips the pass
        entirely.

        Returns (asked, kept): units expanded, generated queries indexed.
        """
        if not evidence._expansion_on():
            return 0, 0
        pending = self.evidence.pending_expansion()
        if not pending:
            return 0, 0
        workers = self._workers()
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                made = list(pool.map(lambda pair: generate.expansions(pair[1]),
                                     pending))
        else:
            made = [generate.expansions(text) for _sid, text in pending]
        generated = {sid: queries
                     for (sid, _text), queries in zip(pending, made)}
        kept = self.evidence.learn_expansions(generated)
        return len(pending), kept

    # --- long-form composition (evidence -> draft, gate per line) -------
    def compose(self, brief, seats=24, topics=None, on_line=None):
        """A structured draft built from the evidence — the long-form answer.

        The short path proved the rule; this is the same rule at document
        length. The engine is handed MATERIAL — evidence lines, each opening
        with its document's name — and asked to organise, not to know.
        Structure (sections, order, titles) is the engine's; content is the
        material's; and the draft is not trusted for having been asked
        nicely: it is RE-READ line by line on the way out.

        Two vetoes, both language-free, both already proven on the short path:
        `grounded_sentences` drops every MIXTURE line — one that wears some
        material words around content the material never supplied — and
        `digits_ok` drops any line whose numbers the material does not carry,
        which is what an invented agenda time or price looks like from
        outside. A line sharing nothing with the material claims nothing
        about it (a title, a [no material] marker) and stands.

        What survives is returned with the documents it drew on. When the
        store has nothing on the brief at all, that is said instead — the
        composition refuses the way every other path refuses.

        Returns (text, sources): the draft and the source stamps it rests on.
        """
        # PER-TOPIC GATHERING, when the caller names the topics. One find()
        # over a four-topic brief makes the topics race each other for
        # seats, and the fourth loses (measured: a catalogue's fourth
        # section came back [no material] while its document sat in the
        # store). Topics are the CALLER's — an application knows what its
        # user asked for; the library guesses no wording.
        found, origins = [], []
        # THE TOPIC IS THE WHOLE QUERY. The first cut appended the brief
        # to every per-topic search — and measured on a 61-document store,
        # every such query converged on the same generic winners, because
        # the brief's words re-ran the seat race the per-topic split exists
        # to prevent. "risk management" alone finds the risk document;
        # "risk management" plus the brief finds whatever the brief's
        # commonest words find. Dedicated seats need dedicated queries.
        queries = (list(topics) if topics else [brief])
        share = max(4, seats // len(queries))
        for query in queries:
            for text, src in zip(
                    self.evidence.find(query, most=share, floor_share=0.0),
                    self.evidence.last_sources):
                if text not in found:
                    found.append(text)
                    origins.append(src)
        if not found:
            return self._refuse(brief), []
        many = len(self.evidence.by_source) > 1
        lines = [(f"{evidence._source_name(src)} — {text}"
                  if many and src else text)
                 for text, src in zip(found, origins)]
        # THE FACT-SHEET RIDES WITH THE PROSE. A logistics line ("DURATION 2
        # days · SEATS 16-18") shares no word with any topic — it is a
        # RECORD, and records lose lexical seat races by construction, which
        # is how a catalogue kept arriving without the durations its brief
        # asked for. So every source that contributed material also
        # contributes its record-shaped lines, recognised by FORMAT alone
        # (the same key_index reading the whole store uses), capped so a
        # table-heavy document cannot flood the block.
        bounds = self.evidence._record_bounds()
        brief_words = set(evidence._words(brief))
        many = len(self.evidence.by_source) > 1
        for src in dict.fromkeys(origins):
            sheet = []
            for sid in sorted(self.evidence.by_source.get(src, ())):
                if sid in self.evidence._derived:
                    continue
                sentence = self.evidence.sentences[sid][0]
                keys = self.evidence._key_words(sentence, *bounds)
                if keys and sentence not in found:
                    # the records the BRIEF asked about ride first — a
                    # duration line answers "state the duration", and the
                    # first cut's first-four-by-position cap was filling the
                    # quota with whatever records came earliest in the file
                    asked = sum(1 for k in keys
                                if any(inflect.same_stem(k, w)
                                       for w in brief_words))
                    sheet.append((-asked, sid, sentence))
            for _rank, sid, sentence in sorted(sheet)[:4]:
                found.append(sentence)
                origins.append(src)
                lines.append(f"{evidence._source_name(src)} — {sentence}"
                             if many else sentence)
        material = "\n".join(lines)
        # THE PERSONA STOPS AT THE DOCUMENT'S EDGE. A composed draft is a
        # document, not a conversational turn: the operator's voice has no
        # work to do in it, and measured, it did damage instead — a
        # consultative persona made the engine REPHRASE every material line,
        # the verbatim gate read each rephrasing as a mixture, and a
        # nine-turn conversation ended in a title over thirteen sources and
        # no content. The voice colours the turns; the draft speaks in the
        # material's words.
        # THE GATE'S TABLE IS SET BEFORE THE ENGINE SPEAKS — allowed
        # tokens depend only on material and brief, so a streamed line can
        # be judged the moment its newline arrives (see below).
        def _plain(text):
            import unicodedata                             # noqa: PLC0415
            return "".join(
                ch for ch in unicodedata.normalize("NFD", text)
                if not unicodedata.combining(ch))
        allowed = ({_plain(fold(w)) for w in evidence._words(material)}
                   | {_plain(fold(w)) for w in evidence._words(brief)}
                   | {_plain(fold(w)) for w in evidence._words(self.style)})

        def _admit(line):
            """One draft line through the gate, SECOND FORM: names and
            numbers may not be invented; everything else may be phrased
            freely. Returns the kept text, "" for a blank line, or None
            for a refused one. (History: the first form was the mixture
            test — coverage 0 or 1 per line — and against a real catalogue
            request it returned raw evidence dumps. The offer gate had
            already solved this: the dangerous tokens have a SHAPE —
            capitals off the sentence start, digits — and each must be
            attested by the material or the brief. Stated residue: an
            invented lowercase quality can pass; names, designations and
            every number cannot, and the digit discipline keeps its
            bigram-neighbour form. Both sides of the token comparison pass
            one NFD-stripping normaliser, for the accent defect the core
            fold still carries — ledgered.)"""
            text = line.strip()
            if not text:
                return ""
            # the digit veto first — non-negotiable on the short path, and
            # a composed agenda is where invented numbers would try to live
            if not evidence.digits_ok(text, material):
                return None
            # A TITLE IS STRUCTURE, AND TITLE CASE IS ITS FORMAT. In a
            # heading every word is capitalised, so capitals there carry no
            # designation signal — but a heading can still smuggle an
            # invented programme name, so it must EARN its capitals: at
            # least one capitalised word the material or brief attests.
            # Prose keeps the strict rule: every shaped token attested.
            tokens = re.findall(r"\w+", text, re.UNICODE)
            alpha = [t for t in tokens if t[:1].isalpha()]
            titled = bool(alpha) and all(t[:1].isupper() for t in alpha)
            sound, earned = True, False
            for m in re.finditer(r"\w+", text, re.UNICODE):
                token = m.group()
                starts = m.start() == 0 or text[
                    :m.start()].rstrip()[-1:] in ".!?:\u2022-\u2013\u2014([\n"
                named = (any(ch.isupper() for ch in token[1:])
                         or (token[:1].isupper() and not starts)
                         or any(ch.isdigit() for ch in token))
                attested = any(inflect.same_stem(_plain(fold(token)), w)
                               for w in allowed)
                if token[:1].isupper() and attested:
                    earned = True
                if named and not attested and not titled:
                    sound = False
                    break
            if titled:
                sound = earned
            return text if sound else None

        kept_lines = []
        if on_line is not None:
            # THE DRAFT STREAMS THROUGH THE GATE. The composer writes ~900
            # tokens; waiting for the last before showing the first is the
            # whole of the catalogue turn's perceived delay. The gate above
            # is pure string arithmetic, so each line is judged the moment
            # its newline arrives and handed to the caller while the engine
            # still writes — same gate, same verdicts, same returned draft;
            # only the waiting moves. On a backend that cannot stream, the
            # generator yields once and this loop IS the batch path.
            buffer = ""
            for chunk in generate.compose_stream(brief, material,
                                                 style=self.style,
                                                 **self._voice()):
                buffer += chunk or ""
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    kept = _admit(line)
                    if kept is None:
                        continue
                    kept_lines.append(kept)
                    if kept:
                        on_line(kept)
            if buffer.strip():
                kept = _admit(buffer)
                if kept is not None:
                    kept_lines.append(kept)
                    if kept:
                        on_line(kept)
        else:
            draft = (generate.compose(brief, material, style=self.style,
                                      **self._voice()) or "").strip()
            if not draft:
                return self._refuse(brief), []
            for line in draft.splitlines():
                kept = _admit(line)
                if kept is not None:
                    kept_lines.append(kept)
        text = "\n".join(kept_lines).strip()
        if not text:
            return self._refuse(brief), []
        self.last_abstained = False
        self._composed = True
        used = sorted({src for src in origins if src})
        return text, used

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
        rows = list(rows)
        # HOW OFTEN THE TABLE SAYS EACH VALUE. A name belongs to one row; a
        # value that the sheet repeats belongs to none of them. This count is
        # what separates the two, and it is the table counting itself — no
        # column is named here and no document is read.
        seen = {}
        for row in rows:
            for value in row.values():
                key = fold(str(value).strip())
                if key:
                    seen[key] = seen.get(key, 0) + 1
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
            #
            # AND IT HAS TO BE THIS ROW'S ALONE. Word count is not enough on a
            # sheet whose columns hold other entities: with a `supplier` column
            # reading "Delta AS" the rule aliased that supplier onto the FIRST
            # row that named it, so `about("delta as")` came back with Acme's
            # facts — and the row whose own company was Delta AS could no
            # longer be reached at all. Fusing two identities is the one
            # mistake the memory cannot undo later, so the alias is only taken
            # when the whole table says it once.
            # ONLY A ROW THAT HAS NO NAME OF ITS OWN BORROWS ONE. Measured, and
            # this is the whole condition: on `{part: "Valve V2", supplier:
            # "Delta AS"}` the rule aliased the supplier onto the part, and
            # because the two were now one concept `learn_cell`'s `sk == vk`
            # guard refused the record — so the alias DESTROYED the very fact
            # it was drawn from. `about("Valve V2")` came back empty while the
            # same sheet with a one-word supplier answered normally.
            #
            # A row whose anchor is already a phrase is already reachable by
            # the name a question uses. The rule exists for the sheet that
            # names its rows `R-1`, where the description is the only human
            # name in the row and its own record says nothing anyway. One word
            # in the anchor is what tells those two apart, and it is the row
            # counting its own words.
            #
            # THE ALIAS STILL RUNS BEFORE THE CELLS, and so the aliased cell's
            # own record is still not written — the two are one concept by
            # then and `learn_cell` declines. That loss is the point rather
            # than a leak: "R-1 is called the fire-equipment inspection" is
            # exactly what the alias records, and it is the reading a question
            # can actually use. What was wrong before was firing this on rows
            # where the cell was a SEPARATE FACT about a row that already had
            # a name; the two conditions above are what keep it off those.
            if (rich and len(rich[1].split()) >= 2
                    and len(anchor.split()) == 1
                    and seen.get(fold(rich[1]), 0) == 1):
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
        # A FOLLOW-UP NAMES ITS SUBJECT BY POINTING, and retrieval cannot
        # see a pointer. Measured, five-turn conversation: "what does the
        # Alpha programme teach?" answered correctly, then "and who
        # is it for?" answered from a DIFFERENT document — the engine gets
        # the history, the search got only this turn's words. The first cut
        # keyed on the subject's ABSENCE and never fired: the extractor
        # dutifully returns the pointer itself as a subject, so the signal is
        # not that the turn has no subject but that its subject RESOLVES TO
        # NOTHING THE MEMORY KNOWS — the exact condition the identity route
        # already reads one screen up. An unresolvable subject in a turn that
        # follows a resolvable one is the conversation pointing backwards,
        # and the previous turn's subject rides along as query words. No
        # wording is read anywhere.
        # DOES THIS TURN'S SUBJECT NAME ANYTHING HERE — two oracles, because
        # the memories come in two builds. A deep memory answers through the
        # graph: the label resolves, or it does not. An evidence-mode memory
        # holds almost no graph (measured: 16 identities over 62 documents),
        # and against it EVERY subject looked unresolved — the first cut of
        # this fell back on explicit topic switches too, and the write-back
        # then dragged the stale subject across the switch ("let's move to
        # Delegation" was answered right and REMEMBERED wrong). There the
        # oracle is the evidence itself: a subject stands when it is a NAME,
        # and a name is a phrase — more than one word, each known to the
        # index — the same reading the table alias rule earned. A pointer
        # ("this one") and a bare generic never pass both; a programme's
        # name always does.
        def _names_something(label):
            if not label:
                return False
            if link.resolve(self.memory, label, self.vectors) is not None:
                return True
            words = evidence._words(label)
            return len(words) >= 2 and all(
                any(inflect.same_stem(w, key) for key in self.evidence.index)
                for w in words)
        if _names_something(subject_label):
            anchor_label = subject_label
        else:
            anchor_label = (getattr(self, "_prior_subject", "")
                            or subject_label)
        if anchor_label and anchor_label != subject_label:
            # the pointer resolved BACKWARDS, so the conversation's subject
            # is still the resolved one — write it back, or the next
            # follow-up inherits the useless pointer instead of the chain
            self.last_subject = anchor_label
        # A CONSULTATION'S SEARCH READS THE WHOLE CONSULTATION. Measured,
        # live: "in banking, ten people" · "risk management is our side" ·
        # "the best three-hour material, please" — and the third turn's
        # search saw three words of nothing, because the query has always
        # been this turn's sentence while the topic lived two turns back.
        # W14's seam, seen from the other side: context turns were dropped
        # from memory (correctly) but thereby also from RETRIEVAL, where
        # they were needed. So in a no-teach conversation the user's prior
        # turns ride along as query words — and only there: a teaching
        # turn and the benchmarks keep the single-turn query. No wording is
        # read, no cap is ours: the seat weighting (log S/s zeroes a word
        # every source speaks) already decides which of the ridden words
        # matter.
        consult = ""
        if self._no_teach and self._brief:
            consult = " ".join(self._brief)
        if anchor_label and anchor_label != subject_label:
            # THE RIDE IS A CHALLENGER, NOT A PREEMPTION. Measured on the
            # NIST sequence: two password questions, then "the publication
            # date of this document" — the pointer resolves to nothing, the
            # prior subject rides by design, and a hundred password lines
            # push the date line out of the seats; the candidates then say
            # the true date and the reading rightly refuses what the seats
            # do not attest. The ride exists for the real follow-up, where
            # the turn's own words seat junk — so both queries run (local,
            # engine-free) and the ridden proof stands only if it covers
            # the question at least as well as the turn's own. Ties keep
            # the ride; the follow-up keeps its fix.
            own_query = f"{question} {consult}".strip()
            own = self.evidence.find(own_query, most=seats)
            cov_own = (evidence.coverage(question, "\n".join(own))
                       if own else 0.0)
            ridden_query = f"{anchor_label} {question} {consult}".strip()
            ridden = self.evidence.find(ridden_query, most=seats)
            cov_ridden = (evidence.coverage(question, "\n".join(ridden))
                          if ridden else 0.0)
            if cov_own > cov_ridden:
                # re-run the winner LAST: last_sources / last_census must
                # describe the proof everything downstream reads
                proof = self.evidence.find(own_query, most=seats)
            else:
                proof = ridden
        else:
            proof = self.evidence.find(
                f"{anchor_label or ''} {question} {consult}".strip(),
                most=seats)
        # which document each seat came from, aligned with `proof` — read off
        # the store's `last_sources`, which `find` leaves beside its result
        proof_origins = list(self.evidence.last_sources[:len(proof)])
        # THE COMPARISON READING. "Do A and B run the same length?" is
        # answered by NO single document — the verdict is born when the two
        # lines lie side by side, and the single race cannot promise that:
        # each field line enters on its own luck. The trigger is a STATE
        # the name channel computes (the question NAMES two or more
        # sources — W38's pointer-and-bigram reading), and the assembly is
        # organs off the shelf: each named source gathers its own seats
        # (dedicated seats need dedicated queries), its record-shaped
        # lines ride along (the fact-sheet rider), and the named-source
        # jury already holds every claim to the source it names. No new
        # model call; no wording read.
        laid_out = False
        named_two = set(getattr(self.evidence, "last_named", ()) or ())
        # A FIELD QUESTION THAT NAMES NO SOURCE READS THAT FIELD ACROSS
        # THE CORPUS. Six seats cannot hold sixty documents' values, and
        # the source cap rightly keeps any one document from taking
        # them — so "which course runs the longest" met an arbitrary
        # handful of lines and abstained. The census answered this shape
        # of question long ago ("who all speaks of this" is a COUNT, not
        # a retrieval); this is that reading applied to VALUES: when the
        # question names no source and three or more sources answer the
        # same record head, the block is one row per source, datelined
        # and capped. The engine then ranks attested values instead of
        # guessing, and an honest tie stays visible. Local work only —
        # no model call, no wording read.
        if not named_two and len(self.evidence.by_source) > 2:
            qw = evidence._words(question)
            heads = {}                      # head words -> {source: row}
            for src in sorted(self.evidence.by_source):
                for _sid, text in self._record_rows(src, question, cap=12):
                    head, _sep, val = text.partition(":")
                    if not val.strip(" ."):
                        continue
                    hw = tuple(evidence._words(head))
                    if hw:
                        heads.setdefault(hw, {}).setdefault(
                            src, (head.strip(), text))
            # THE FIELD IS THE ONE THE QUESTION COVERS BEST, and A TIE IS
            # NOT A GUESS. Measured on the corpus: "which training takes
            # the most people" covers KATILIMCI SAYISI and EĞİTİM SÜRESİ
            # equally — one word each — and picking either by file order
            # is how the wrong field got harvested. Every field at the
            # top coverage comes to the table, at most three, and the
            # reader takes the one the question means.
            scored_heads = []
            for hw, per_src in heads.items():
                if len(per_src) < 3:
                    continue
                matched = sum(1 for h in hw
                              if any(inflect.same_stem(h, w)
                                     or inflect.kin(h, w) for w in qw))
                if matched:
                    scored_heads.append((matched / len(hw), hw, per_src))
            if scored_heads:
                best = max(r[0] for r in scored_heads)
                chosen = [r for r in scored_heads if r[0] == best][:3]
                field, field_origins = [], []
                for _ratio, _hw, per_src in chosen:
                    ranked, units = [], {}
                    for src in sorted(per_src):
                        head_text, text = per_src[src]
                        if len(field) >= 60:
                            break
                        field.append(text)
                        field_origins.append(src)
                        # A NUMBER BELONGS TO THE UNIT BESIDE IT: a value
                        # reading "4 modules x half a day — about 2 days"
                        # ranked as FOUR days when the largest digit in
                        # the line was taken for the value; the four
                        # counts modules. Digits pair with the word they
                        # stand next to, and a range keeps both ends. A
                        # qualifier in between ("2 full days") makes its
                        # own group and loses the ranking to the plain
                        # one — out of the comparison is the honest place
                        # for a value whose unit cannot be read.
                        _h, _sep, val = text.partition(":")
                        pairs, buf = {}, []
                        for tok in re.findall(r"\w+", val, re.UNICODE):
                            if tok.isdigit():
                                buf.append(int(tok))
                                continue
                            if buf:
                                pairs.setdefault(evidence.fold(tok),
                                                 []).extend(buf)
                            buf = []
                        for unit_name, nums in pairs.items():
                            units[unit_name] = units.get(unit_name, 0) + 1
                            ranked.append((max(nums), min(nums), src,
                                           head_text, val.strip(" ."),
                                           unit_name))
                    # THE EXTREMES OF A FIELD ARE WRITTEN INTO THE
                    # EVIDENCE — W44's rule at corpus scale. A superlative
                    # is a conclusion no single line states, so the
                    # read-back rightly refused it and the turn abstained
                    # with the whole field on the table. One row states
                    # both ends, each with its source and its value, and
                    # only within the unit the field mostly speaks. THE
                    # VERDICT SPEAKS FIRST: a block is read from the top
                    # and cut at the seats, so a row appended after sixty
                    # field lines is a row nobody reads.
                    if ranked:
                        main = max(units, key=lambda u: (units[u], u != ""))
                        same = [r for r in ranked if r[5] == main]
                        if len(same) >= 3:
                            hi = max(same, key=lambda r: r[0])
                            lo = min(same, key=lambda r: r[1])
                            if hi[2] != lo[2]:
                                field.insert(0,
                                    f"{hi[3]} \u2014 largest: "
                                    f"{evidence._source_name(hi[2])}: "
                                    f"{hi[4]} \u00b7 smallest: "
                                    f"{evidence._source_name(lo[2])}: "
                                    f"{lo[4]}")
                                field_origins.insert(0, "")
                if len(field) >= 3:
                    proof, proof_origins = field, field_origins
                    self.evidence.last_sources = list(field_origins)
                    laid_out = True
                    self._laid_out = True
        if len(named_two) >= 2:
            census_keep = list(self.evidence.last_census)
            share = max(2, seats // len(named_two))
            comp, comp_origins = [], []
            for src in sorted(named_two, key=evidence._source_name):
                lines, origins2 = self._gather_source(src, question, share)
                for line, origin in zip(lines, origins2):
                    # IN A COMPARISON, TWO SOURCES SAYING THE SAME THING
                    # SAY IT TWICE — W39's rule, one layer up. Sibling
                    # documents share a template, so two courses of equal
                    # length write the same record line word for word;
                    # deduping by TEXT dropped the second, the block held
                    # one value where the question asked about two, and
                    # the verdict row could never be written. Measured:
                    # seven of fifteen field comparisons ask about EQUAL
                    # values — exactly the case this silenced. Where
                    # sources are contrasted, sameness is the answer, and
                    # identity is per (line, source).
                    if (line, origin) not in zip(comp, comp_origins):
                        comp.append(line)
                        comp_origins.append(origin)
            # THE COMPARISON VERDICT IS WRITTEN INTO THE EVIDENCE. Laying
            # two records side by side and leaving "same or different?"
            # to the engine gave a verdict that inverted between runs
            # (the referee's one comparison flip). Equality of records is
            # arithmetic the system already trusts — the digit sets, the
            # digits_ok tradition. When two named sources answer the same
            # FIELD HEAD, one row states it: head, each source's value,
            # joined by = when the digit sets agree, by \u2260 when they
            # differ, and by \u00b7 when a value carries no digits (there
            # the voice still judges). Every word is the documents'; the
            # marker is notation; the engine reads a verdict instead of
            # building one.
            fields = {}
            for text, origin in zip(comp, comp_origins):
                if ":" not in text:
                    continue
                head, _sep, val = text.partition(":")
                hw = tuple(evidence._words(head))
                if hw and len(evidence._words(head)) <= 3:
                    fields.setdefault(hw, {})[origin] = (
                        head.strip(), val.strip().rstrip("."))
            for hw, per_src in fields.items():
                if len(per_src) < 2:
                    continue
                items = sorted(per_src.items(),
                               key=lambda kv: evidence._source_name(kv[0]))
                digitsets = [frozenset(re.findall(r"\d+", v))
                             for _o, (_h, v) in items]
                if all(digitsets) and len(set(digitsets)) == 1:
                    sep = " = "
                elif all(digitsets):
                    sep = " \u2260 "
                else:
                    sep = " \u00b7 "
                head_text = items[0][1][0]
                row = f"{head_text} \u2014 " + sep.join(
                    f"{evidence._source_name(o)}: {v}"
                    for o, (_h, v) in items)
                if row not in comp:
                    comp.append(row)
                    comp_origins.append("")
            self.evidence.last_census = census_keep
            if comp:
                proof, proof_origins = comp, comp_origins
                laid_out = True
        elif len(named_two) == 1:
            # A QUESTION THAT NAMES ONE SOURCE GETS THAT SOURCE'S BLOCK —
            # the single-name completion of the comparison reading.
            # Measured: the named course was called correctly, its two
            # capped seats went to its densest windows, and the line that
            # answered waited outside while a sibling's window supplied
            # the names and the voice blended a faculty that does not
            # exist. Naming a document is asking to read THAT document:
            # it fills the block, and two seats remain for the rest of
            # the corpus to interject. Questions naming nothing keep the
            # open race untouched.
            src = next(iter(named_two))
            census_keep = list(self.evidence.last_census)
            sources_keep = list(self.evidence.last_sources)
            lines, origins2 = self._gather_source(
                src, question, max(2, seats - 2))
            self.evidence.last_census = census_keep
            if len(lines) >= 2:
                for line, origin in zip(proof, sources_keep[:len(proof)]):
                    if (origin != src and line not in lines
                            and len(lines) < seats + 3):
                        lines.append(line)
                        origins2.append(origin)
                proof, proof_origins = lines, origins2
                laid_out = True
            else:
                self.evidence.last_sources = sources_keep
        if not laid_out and proof and len(proof) < seats:
            # TWO-HOP (only when the first search did not FILL the block): a
            # second search with the best evidence's words — composition
            # questions want the union of two separate sections. When the first
            # search was already full, the second hop added table-crumbs and
            # derailed the answer (hospital trace) — don't touch when there's
            # plenty. "Scarce" used to be the constant 4 against a block of 6;
            # what it means is that retrieval came back UNDER-SUBSCRIBED, which
            # is a comparison with the number of seats asked for, not a number.
            second = self.evidence.find(f"{proof[0]} {question}",
                                        most=seats - len(proof))
            for extra, origin in zip(second, self.evidence.last_sources):
                if extra not in proof and len(proof) < seats:
                    proof.append(extra)
                    proof_origins.append(origin)
        # IN A MULTI-DOCUMENT STORE, EVERY EVIDENCE LINE OPENS WITH ITS
        # DOCUMENT'S NAME — a dateline, the way a wire story opens with its
        # city. Measured (62 sibling training outlines): asked WHICH programme
        # uses a certain exercise, the engine answered with a section heading,
        # the only name-shaped thing it could see — the true answer sat in the
        # source stamp the evidence never showed. And the dateline has to be
        # IN THE LINE, not in some block header, because everything downstream
        # re-reads the evidence: the candidate blocks are rebuilt from these
        # lines (`_subsets`), the read-back verifies the answer against them,
        # coverage counts their words. A name the verifier cannot see is a
        # name the answer is forbidden to say — the first version of this
        # labelled only one of the three block builders, and the gate kept
        # dropping exactly the phrasing the fix existed to allow.
        #
        # The name is provenance the store already attests, so an answer that
        # quotes it is grounded by construction. A single-document store adds
        # nothing, and keeps its benchmarks byte for byte.
        if len(self.evidence.by_source) > 1:
            proof_origins += [""] * (len(proof) - len(proof_origins))
            proof = [f"{evidence._source_name(origin)} — {line}"
                     if origin else line
                     for line, origin in zip(proof, proof_origins)]
            # ...AND THE CENSUS AS ONE MORE LINE OF EVIDENCE. The seats hold
            # the best evidence; the census says who ALL speaks of this, and
            # a conversational answer to "where does X appear" needs the
            # second, not the first. Counts are facts the store attests —
            # names and numbers only, no sentence is written for the engine.
            census = [(evidence._source_name(src), hits)
                      for src, hits in self.evidence.last_census]
            self._census_line = ""
            if len(census) > 1:
                # six entries, not twelve: the line is read by ~5 model calls
                # per question, and the measured cost of the longer form on a
                # local engine was a 217-second turn. Six names still answer
                # "who all speaks of this"; the tail was paying for itself in
                # neither accuracy nor phrasing.
                self._census_line = " · ".join(
                    f"{name} ×{hits}" for name, hits in census[:6])
                proof.append(self._census_line)
        # kept for the abstention reading — see _asserted_a_fact
        self._last_proof = list(proof)
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
                                 else generate.answer(question, block,
                                                      persona=self.persona),
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
            # source is DOCUMENT level — and it names the origin of the
            # line the answer actually rests on, not whichever seat came
            # first (see _load_bearing).
            origins = [""] * len(proof)
            for i, orig in enumerate(proof_origins[:len(proof)]):
                origins[i] = orig
            src = (self._load_bearing(safe, proof, origins)
                   or next((s for s in self.evidence.last_sources if s),
                           "#document"))
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
        # THE TALLY RIDES IN EVERY SUBSET. The census line sits last in the
        # proof, and prefix subsets were cutting exactly it — so the narrow
        # candidates, which usually win the grading, never saw the one line
        # that answers "which documents". Metadata the store attests belongs
        # in every view of the evidence, the same rule the dateline follows.
        tail = getattr(self, "_census_line", "")
        for size in (len(proof), max(1, len(proof) // 2), 1):
            if proof and size not in sizes:
                sizes.append(size)
                sub = list(proof[:size])
                if tail and tail not in sub:
                    sub.append(tail)
                subsets.append(sub)
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
        # THE CANDIDATES DO NOT DEPEND ON EACH OTHER, so on a backend that
        # permits it they are asked at once. Measured on six NIST questions,
        # this call is 43% of the time a question takes — the largest single
        # share, and the whole of it was being spent in sequence for no reason
        # but the shape of the loop.
        #
        # NOTHING ELSE MOVES. The subsets are built in the same order and the
        # grading below still walks them in that order, so `len(graded)` — the
        # rank a tie falls back on — and the `seen` de-duplication are what
        # they were. Every call is warmth 0, so the answers are the same
        # answers, and `_workers()` is the ONE definition of how much
        # concurrency the engine tolerates: it pins the local torch model to a
        # single worker because that model is not thread-safe, which means this
        # is a cloud-backend speedup and honestly nothing at all on a laptop.
        subsets = list(self._subsets(records, proof, fact_block))
        workers = min(self._workers(), len(subsets)) if subsets else 1
        # warmth 0: each candidate is the DETERMINISTIC answer to its own
        # evidence, so the differences between candidates carry information
        # (which evidence) instead of sampling noise.
        def _say(one):
            return generate.answer(question, one, warmth=0.0,
                                   persona=self.persona)
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
            with ThreadPoolExecutor(max_workers=workers) as pool:
                spoken = list(pool.map(_say, subsets))
        else:
            spoken = [_say(one) for one in subsets]
        for said in spoken:
            raw = self._strip_marks(said)
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
            # THE TWO JUDGMENTS OF A CANDIDATE ARE ASKED SIDE BY SIDE.
            # The read-back asks whether the evidence SAYS this; the
            # relation check asks whether it answers what was ASKED (see
            # `_relation_held`: a claim built out of true material can
            # answer a question nobody asked — the last route to a
            # confident wrong answer). Neither reads the other's verdict,
            # so they travel together and the candidate is admitted only
            # if BOTH hold. Refusal is remembered so the fallback below
            # cannot speak it after all.
            if proof and not self._judge(question, raw, proof, block):
                self._refused.add(fold(raw))
                continue
            return raw, tried
        return None, tried

    def _judge(self, question, raw, proof, block):
        """Both judgments of one candidate, asked together where the
        backend allows: does the evidence SAY this (`_read_back`), and
        does it answer what was ASKED (`_relation_held`). Independent
        readings, one verdict — admitted only if both hold. The
        occasional cost is one relation call for a candidate the
        read-back would have refused; the gain is an engine round-trip
        off every answered turn."""
        jobs = (lambda: self._read_back(raw, proof, block,
                                        question=question),
                lambda: self._relation_held(question, raw, proof, block))
        return all(runtime.parallel_map(lambda f: f(), jobs))

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

    def _gather_source(self, src, question, share):
        """One named source's contribution to a laid-out block: its best
        `share` lines for this question, gathered by name, plus its
        record rows — the same hand the comparison reading deals, now
        dealt for one player too."""
        got = self.evidence.find(
            f"{evidence._source_name(src)} {question}",
            most=share * 2, floor_share=0.0)
        lines, origins = [], []
        for line, origin in zip(got, self.evidence.last_sources):
            if origin == src and line not in lines and len(lines) < share:
                lines.append(line)
                origins.append(origin)
        for _sid, sentence in self._record_rows(src, question):
            if sentence not in lines:
                lines.append(sentence)
                origins.append(src)
        return lines, origins

    def _record_rows(self, src, question, cap=4):
        """The fact-sheet rider, on the answering path: the record-shaped
        lines of ONE source, read by the record's own format — a SHORT line
        whose first colon is preceded by its field name. (_key_words over
        arbitrary sentences read junk keys out of glued window segments,
        and the derived flag proved an unreliable window test — measured:
        the rider kept riding kazanim prose while "SEATS: 16-20" sat
        unread, which is also where the composer's "[no material]" rows
        were born.) The rows the question asks about ride first."""
        limit_chars, _limit_toks = self.evidence._record_bounds()
        cap_chars = max(80, limit_chars // 2)

        def _head(text):
            if ":" not in text or len(text) > cap_chars:
                return None
            words = evidence._words(text.partition(":")[0])
            if not words or len(words) > 3:
                return None                 # a colon mid-prose is no record
            return tuple(words)

        # A FIELD IS A HEAD THE CORPUS REPEATS. "DURATION:" opens a line in
        # sixty documents; a titled bullet ("The source of worry: ...")
        # opens one — same colon, same shape, and only one of them is a
        # record. Field-ness is read off the store, not off any word: a
        # head that appears in two or more SOURCES is a field.
        head_sources = {}
        for text, origin in self.evidence.sentences:
            h = _head(text)
            if h:
                head_sources.setdefault(h, set()).add(origin)
        qw = set(evidence._words(question))
        sheet = []
        for sid in sorted(self.evidence.by_source.get(src, ())):
            text = self.evidence.sentences[sid][0]
            h = _head(text)
            if h is None:
                continue
            fieldish = len(head_sources.get(h, ())) >= 2
            # RANKING READS KINSHIP, GATES READ same_stem. The rider's
            # "did the question ask about this field" is an ORDER, not a
            # licence: with the strict rule a four-letter root and its
            # inflection are strangers ("sürede" vs "SÜRESİ"), and in a
            # short-rooted language the rider ranked by nothing at all.
            asked = sum(1 for k in h
                        if any(inflect.same_stem(k, w) or inflect.kin(k, w)
                               for w in qw))
            sheet.append((-int(fieldish), -asked, sid, text))
        return [(sid, text) for _f, _a, sid, text in sorted(sheet)[:cap]]

    def _named_lines(self, claim, proof, question):
        """The full proof lines of every source the CLAIM names — a source
        is named when all its name words (question words aside) appear in
        the claim. Empty when the claim names nothing."""
        claim_words = set(evidence._words(claim))
        qw = set(evidence._words(question))
        named = []
        for line in proof:
            if " \u2014 " not in line:
                continue
            name, _body = line.split(" \u2014 ", 1)
            name_words = [w for w in evidence._words(name) if w not in qw]
            if name_words and all(w in claim_words for w in name_words):
                named.append(line)
        return named

    def _strip_marks(self, text):
        """PROVENANCE NOTATION BELONGS TO THE SYSTEM. The engine sees the
        notation in its instructions and can imitate it — measured: an
        answer arrived wearing a stamp for a file that does not exist.
        Anything stamp-shaped in a generated candidate is stripped before
        any gate reads it; the system attaches its own mark afterwards,
        from the proof's real origins (see _load_bearing)."""
        text = re.sub(r"\(~[^)]*\)", "", text or "")
        text = re.sub(r"#\S+", "", text)
        return re.sub(r"  +", " ", text).strip()

    def _load_bearing(self, said, proof, origins):
        """The origin of the proof line the spoken answer covers best — the
        LOAD-BEARING line. The mark used to name the FIRST seat's origin,
        whatever line the answer actually rested on: a correct sentence
        about one course, stamped with another, question after question.
        Deterministic: word-coverage argmax, first on ties."""
        # A CLAIM THAT NAMES A SOURCE IS STAMPED WITH THAT SOURCE. The
        # system's own rows — a field's extremes, a comparison's verdict —
        # belong to no single document and carry no origin, so an answer
        # resting on one of them fell back to whichever seat came first
        # and stamped a true sentence about one document with another
        # document's name. A stamp is a promise about where a claim comes
        # from, so it follows the claim: when the answer names a source
        # the store holds, that source is the mark. The jury already
        # reads claims this way (W32); the mark now reads them the same.
        spoken = set(evidence._words(said))
        for origin in dict.fromkeys(o for o in origins if o):
            name_words = evidence._words(evidence._source_name(origin))
            if name_words and all(
                    any(inflect.same_stem(w, t) or inflect.kin(w, t)
                        for t in spoken) for w in name_words):
                return origin
        for origin in sorted(self.evidence.by_source):
            name_words = evidence._words(evidence._source_name(origin))
            if len(name_words) >= 2 and all(
                    any(inflect.same_stem(w, t) or inflect.kin(w, t)
                        for t in spoken) for w in name_words):
                return origin
        best, best_cov = "", -1.0
        for line, origin in zip(proof, origins):
            if not origin:
                continue
            cov = evidence.coverage(said, line)
            if cov > best_cov:
                best, best_cov = origin, cov
        return best

    def _substance_ok(self, claim, proof, question):
        """AN ANSWER'S SUBSTANCE CANNOT BE BORROWED FROM THE QUESTION — the
        one rule, held by one organ, read at BOTH doors (the read-back and
        the informed refusal's offer). Question words are excused from
        coverage for fluency; source names are attested by the datelines;
        a sentence built of nothing else asserts nothing the evidence
        says. When the substance beyond question and names is EMPTY, the
        borrowed words must stand in the gathered lines' BODIES — and in
        the bodies OF THE SOURCE THE CLAIM NAMES, when it names one:
        measured, "in person" attested in one course's box let the OTHER
        course's name pass, every word covered somewhere and the pairing
        nowhere. Returns True when the rule stands aside (free substance
        exists — the other gates own that case)."""
        qw = set(evidence._words(question))
        nw = set()
        names_bodies = []
        for line in proof:
            if " \u2014 " in line:
                name, body = line.split(" \u2014 ", 1)
                nw |= set(evidence._words(name))
                names_bodies.append((name, body))
            else:
                names_bodies.append(("", line))
        aw = evidence._words(claim)
        substance = [w for w in aw if w not in nw and not any(
            inflect.same_stem(w, q) for q in qw)]
        if substance:
            return True
        claim_words = set(aw)
        named_bodies = []
        for name, body in names_bodies:
            name_words = [w for w in evidence._words(name) if w not in qw]
            if name_words and all(w in claim_words for w in name_words):
                named_bodies.append(body)
        pool = named_bodies if named_bodies else [b for _, b in names_bodies]
        bw = set()
        for body in pool:
            bw |= set(evidence._words(body))
        borrowed = [w for w in aw if w not in nw and any(
            inflect.same_stem(w, q) for q in qw)]
        return not borrowed or all(
            any(inflect.same_stem(w, b) for b in bw) for w in borrowed)

    def _read_back(self, raw, proof, block, question=""):
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
        # A RESTATED EVIDENCE LINE IS NOT A CASE FOR THE JURY. The judge's
        # one measured failure class is denying a claim the evidence carries
        # essentially verbatim, for sitting in spec noise — and its verdict
        # on that class also wobbles across identical runs. When some single
        # evidence line and the claim cover EACH OTHER's content words
        # (footnote stripped, inflection tolerated), the claim is that line,
        # re-inflected; there is nothing left to judge. One-directional
        # coverage is NOT enough and the first cut of this proved why before
        # it could ship: a subset of a line's words can invert it — "optional"
        # lifted out of "is not optional" is fully covered and false — which
        # is exactly the veto the selector test (G5) pins down. The jury
        # keeps everything short of mutual coverage.
        claim = re.sub(r"\([^)]*\)", " ", raw)
        claim = " ".join(w for w in claim.split() if not w.startswith("#"))
        # the borrowed-substance rule — see _substance_ok, the one organ
        # both doors read
        if question and claim.strip() and not self._substance_ok(
                claim, proof, question):
            return False
        if claim.strip():
            for line in proof:
                if (evidence.coverage(claim, line) == 1.0
                        and evidence.coverage(line, claim) == 1.0):
                    return True
        # A CLAIM THAT NAMES ITS SOURCE IS JUDGED BY THAT SOURCE ALONE.
        # Measured: "the Delegation course does not state its seat count" —
        # real substance, borrowed from ANOTHER course's box, stapled to
        # the named course — and the jury confirmed it against the full
        # block, where those words genuinely stand. The fallback view of
        # everything retrieved is exactly the door the stapled sentence
        # walks through; when the claim names a source, that door closes.
        named = self._named_lines(claim, proof, question) if question else []
        if named:
            views = ["\n".join(named)]
        else:
            views = []
            for view in (self._focus_view(raw, proof), block):
                if view and view not in views:
                    views.append(view)
        # the two views are independent readings of one claim — they go to
        # the engine side by side where the backend allows (the occasional
        # cost: one small extra call when the first view would have
        # sufficed; the gain: the second view no longer queues behind a
        # refusal). Verdict unchanged: ANY confirming view confirms.
        return any(runtime.parallel_map(
            lambda view: generate.supported(raw, view), views))

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
    def _chat_id_block(self):
        """The identity rows the chat voice may speak from — one builder,
        read by the blocking voice, the wagered voice and the stream."""
        id_records = retrieve.gather(self.memory, self._lmm_key)
        return "\n".join(
            f"{link.label_of(self.memory, r.subject)} "
            f"{link.label_of(self.memory, r.predicate)} \u2192 "
            f"{link.label_of(self.memory, r.value)}" for r in id_records)

    def _chat_raw(self, message):
        """The chat voice's ENGINE CALL alone — factored out so the wager
        above and the chat path below speak with byte-identical arguments;
        the gate below never moves."""
        id_records = retrieve.gather(self.memory, self._lmm_key)
        id_block = "\n".join(
            f"{link.label_of(self.memory, r.subject)} "
            f"{link.label_of(self.memory, r.predicate)} \u2192 "
            f"{link.label_of(self.memory, r.value)}" for r in id_records)
        return generate.chat(message, id_block, history=self.history,
                             **self._voice(persona=self.persona))

    def _chat(self, message, spec=None, spec_queue=None):
        """Chat answer — the gate still filters any leaking factual claim.

        HOLE CLOSED (F2): this used to be `safe or raw` — if verify found all
        sentences unsupported and dropped them, the RAW (unaudited) output was
        returned, bypassing the gate entirely. Now on empty it falls to the
        safe side, no raw fabrication is returned.
        """
        # THE CHAT VOICE STREAMS WHEN SOMEONE LISTENS. Sentences are the
        # gate's own unit — verify has always judged them one at a time —
        # so each completed sentence goes through its reading the moment
        # its full stop arrives: a kept sentence is on screen while the
        # engine writes the next, a struck sentence is never seen at all.
        # Without a listener the wagered path stands, byte for byte.
        on_line = getattr(self, "_on_line", None)
        if on_line is not None:
            self._spec_chat = None
            prior_user = [h["content"] for h in self.history
                          if h.get("role") == "user"]
            echo_pool = (prior_user if self.last_kind == extract.ASK
                         else prior_user + [message])
            echo_text = " ".join(echo_pool)
            id_records = retrieve.gather(self.memory, self._lmm_key)
            id_block = "\n".join(
                f"{link.label_of(self.memory, r.subject)} "
                f"{link.label_of(self.memory, r.predicate)} \u2192 "
                f"{link.label_of(self.memory, r.value)}" for r in id_records)
            def _chunks():
                if spec_queue is not None:
                    while True:
                        chunk = spec_queue.get()
                        if chunk is None:
                            return
                        yield chunk
                else:
                    yield from generate.chat_stream(
                        message, id_block, history=self.history,
                        **self._voice(persona=self.persona))
            kept, buffer = [], ""
            for chunk in _chunks():
                buffer += chunk or ""
                while True:
                    m2 = re.search(r"[.!?]\s", buffer)
                    if not m2:
                        break
                    sentence, buffer = (buffer[:m2.end()].strip(),
                                        buffer[m2.end():])
                    sentence = self._strip_marks(sentence)
                    if not sentence:
                        continue
                    ok = verify.verify(self.memory, sentence,
                                       self._identity, self.mode,
                                       anchor="value", echo=echo_text)
                    if ok:
                        kept.append(ok)
                        on_line(ok)
            tail = self._strip_marks(buffer)
            if tail:
                ok = verify.verify(self.memory, tail, self._identity,
                                   self.mode, anchor="value",
                                   echo=echo_text)
                if ok:
                    kept.append(ok)
                    on_line(ok)
            return " ".join(kept).strip() or self._refuse(message)
        # The IDENTITY block (the maker must not be hidden) and the recent
        # turns both ride inside _chat_raw — the wager and this path speak
        # with byte-identical arguments. Fabrication is still filtered below.
        raw = spec.result() if spec is not None \
            else self._chat_raw(message)
        # the voice may have LEARNED the stamp notation from the history it
        # reads (prior turns carry system marks) — measured: with every
        # claiming sentence gate-dropped, the sole survivor of a reply was
        # an imitated citation, and the turn printed a stamp on silence.
        # System notation is not the voice's to write.
        raw = self._strip_marks(raw)
        # In chat, allowed = ONLY the identity facts. anchor="value": the
        # subject is a self-referential pronoun (ben/beni) that can't be
        # resolved; it suffices that the OBJECT (rüzgar) is allowed; external
        # fabrication (Google) still falls. And the ECHO: the user's own
        # words in this conversation are things the reply may repeat back —
        # listening is not asserting. See verify.verify.
        self._spec_chat = None          # the wager is spent
        # AN ECHO CANNOT TURN A QUESTION INTO ITS OWN ASSERTION. Caught by
        # our own trap set: "do participants receive a certificate?" —
        # unanswerable, honestly abstained, and the fallback said "Yes,
        # participants generally receive a certificate", every word of
        # the claim covered by the QUESTION and licensed by the echo. But
        # a question asserts nothing: echo exists so a consultant may
        # repeat what the user STATED, and a turn the router classified
        # as ASK stated nothing at all. On an ASK turn, the echo pool is
        # the prior turns alone; a context statement keeps its same-turn
        # echo.
        prior_user = [h["content"] for h in self.history
                      if h.get("role") == "user"]
        if self.last_kind == extract.ASK:
            echo = " ".join(prior_user)
        else:
            echo = " ".join(prior_user + [message])
        safe = verify.verify(self.memory, raw, self._identity, self.mode,
                             anchor="value", echo=echo)
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
