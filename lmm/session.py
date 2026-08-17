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

from v3 import dynamics
from v3.dataset import fold
from v3.gate import Gate
from v3.memory import Memory, OPERATOR, STRANGER, DOCUMENT
from v3.transitive import Transitivity
from lmm import evidence, extract, generate, link, research, retrieve, verify

SLEEP_EVERY = 50   # sleep every N turns (distill/fade/adjudicate) — v3 §117

# SOURCE-TRUST threshold: a fact spoken with trust BELOW this gets TAGGED in the
# answer (hedge + source). Operator teaching (0.75) is certain → untagged;
# document (0.6), distillation (0.5), web (low) → "I'm not sure, according to
# ...". "Strictest" would be misinformation.
CERTAIN = 0.7

FALLBACK_DONT_KNOW = "I don't know."   # last-resort only: the normal
# refusal is engine-generated in the USER'S language; this string is hit
# only when generation itself fails.


def _grounded_in(value, message):
    """Does the VALUE to be taught actually appear in the user's MESSAGE — did
    Qwen not fabricate it? "You cannot teach what you didn't say". Fold + stem
    matching (kuş~kuştur). This prevents extract from mistaking a question
    ("atom nedir") for a WRITE and inventing a nonexistent value ("birleşik")
    to write into the graph — language-agnostic."""
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
        # v3/transitive.py — the full re-scan per cell was quadratic).
        self._transitive = Transitivity(self.memory)
        # Wire the semantic RIVAL check for contradictions to Qwen (cached) —
        # in the predicate-less case "kuş/yırtıcı" coexist, "kuş/balık" is a
        # contradiction. See _are_rivals.
        self._rival_cache = {}
        self.gate.rival = self._are_rivals
        self._pending = None    # subject offered for research (ask-first flow)
        self.last_written = []  # triples the gate admitted this turn (harvest)
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
        """IDENTITY as a fact in the graph: (lmm → üretici → rüzgar). This way
        identity too is 'known knowledge' — it passes through the gate and is
        spoken language-independently. The persona (prompts.CHAT_SYSTEM) gives
        Qwen its name; this seed keeps it in the graph."""
        lmm = link.resolve(self.memory, "lmm", self.vectors, create=True)
        owner = link.resolve(self.memory, "rüzgar", self.vectors, create=True)
        maker = link.resolve(self.memory, "üretici", self.vectors, create=True)
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
        # values (like kuş/balık).
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
            return generate.refusal(message) or FALLBACK_DONT_KNOW
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
        return safe or generate.refusal(message) or FALLBACK_DONT_KNOW

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

    def respond(self, message):
        """Answer one message + update the CONVERSATION CONTEXT. The real logic
        is in _respond; this wrapper keeps the last N turns in `history`
        (conversation continuity). `last_written`: the triples the GATE
        admitted this turn — for the consolidation harvest (gate-approved =
        trustworthy training target; NOT the model's own raw output)."""
        self.last_written = []
        said = self._respond(message)
        if message and message.strip():
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": said or ""})
            self.history = self.history[-12:]     # last ~6 turns (sliding window)
        return said

    def _respond(self, message):
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
        try:
            op = extract.extract(message)
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
            subject_key = (link.resolve(self.memory, subject, self.vectors)
                           if subject else None)
            # IDENTITY ROUTE (architectural bridge): if the subject CANNOT BE
            # RESOLVED (CHAT, or "seni kim yaptı" where extract can't resolve
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
                    return self._causal_answer(message, cq[0], cq[1])
            if op["kind"] in (extract.WRITE, extract.ASK):
                return self._answer(message, subject)
            return self._chat(message)
        except Exception:                                   # noqa: BLE001
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
        return safe or generate.refusal(message) or FALLBACK_DONT_KNOW

    # --- writing -------------------------------------------------------
    def _write(self, triples, message):
        """Puts the taught material into the graph — through the gate (Qwen
        doesn't write, the gate writes).

        Code-audit: a valueless half-triple is skipped (no None value must
        reach the gate); if the same triple arrives twice it is deduplicated
        (no repeated 'Öğrendim').
        """
        wrote = []
        conflicts = []
        seen = set()
        for subject, predicate, value in triples:
            if not value:
                continue                       # half triple — don't write
            if not _grounded_in(value, message):
                continue     # value not in message → Qwen fabricated it ("atom
                             # nedir"→"birleşik"): the mistake-a-question-for-
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
                # DERIVATION (transitive reasoning): "kartal→kuş, kuş→hayvan ⊢
                # kartal→hayvan". Transitivity is learned from data (triangle
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
        # CONTEXT WINDOWS: neighboring sentences are indexed together too —
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
        # RECORD_CHARS guards it. A record is made of SHORT lines; 12 LONG
        # sentences are prose, which the 6-window already covers, and indexing
        # them only dilutes — a diluted wide window displaced a precise spec
        # line and cost the manual a point (measured: manual 27->26 with the
        # scale ungated, hospital 12->13 with it). The bound is what "short
        # lines" means: 12 sentences averaging under ~58 characters. Length
        # only — no language rule, no document-specific rule.
        RECORD_CHARS = 700
        for size, step in ((3, 2), (6, 2), (12, 4)):
            for i in range(0, max(1, len(sentences) - size + 1), step):
                window = " ".join(sentences[i:i + size])
                if size >= 12 and len(window) > RECORD_CHARS:
                    continue
                if len(window) > len(sentences[i]):
                    self.evidence.add(window, source)
        # TABLE REPAIR (evidence-only): PDF tables shatter row by row —
        # "Dijital patoloji" and the tier cell "3" land apart, the model grabs
        # the wrong number (hospital test finding; RAG made the same mistake
        # on the same table). Consecutive SHORT lines (cells) are joined with
        # a sliding window so row-adjacency stays preserved in the evidence.
        # No language rule — length.
        lines = [l.strip() for l in text.split("\n")]
        run = []
        for line in lines + [""]:
            if line and len(line) <= 40:
                run.append(line)
            else:
                if len(run) >= 3:
                    # The table header row (the run's first cells) is prefixed
                    # to every window — when column names detach from the row,
                    # it was unknowable which attribute "Orta" was the value
                    # of. If it isn't a header, it's harmless extra context.
                    header = " · ".join(run[:6])
                    for i in range(0, len(run), 2):
                        window = run[max(0, i - 1):i + 6]
                        if len(window) >= 2:
                            row = " · ".join(window)
                            self.evidence.add(f"{header} — {row}" if i else row,
                                              source)
                run = []

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
            anchor = cells[0][1]
            sk = link.resolve(self.memory, anchor, self.vectors, create=True)
            if sk is None:
                continue
            rich = max((kv for kv in cells[1:]), default=None,
                       key=lambda kv: len(kv[1]))
            if rich and len(rich[1]) >= 8:
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
        "tür" learns from is-a examples; "sever" never does (love chains don't
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
            for w in sorted(set(evidence._words(question)),
                            key=len, reverse=True):
                cand = link.resolve(self.memory, w)
                if cand is not None and cand not in fallback_keys \
                        and self.memory.by_subject.get(cand):
                    fallback_keys.append(cand)
                    if len(fallback_keys) >= 3:
                        break
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
        proof = self.evidence.find(f"{subject_label or ''} {question}", most=6)
        if proof and len(proof) < 4:
            # TWO-HOP (only when evidence is SCARCE): a second search with the
            # best evidence's words — composition questions want the union of
            # two separate sections. When the first search was already full,
            # the second hop added table-crumbs and derailed the answer
            # (hospital trace) — don't touch when there's plenty.
            for extra in self.evidence.find(f"{proof[0]} {question}", most=4):
                if extra not in proof and len(proof) < 6:
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
                offer = generate.offer_research(subject_label, question)
                return offer or generate.refusal(question) or FALLBACK_DONT_KNOW
            return generate.refusal(question) or FALLBACK_DONT_KNOW
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
            records = [r for r in records
                       if not str(r.source).startswith("#doc")]
        block = retrieve.facts_block(self.memory, records) if records else ""
        if proof:
            proof_block = "\n".join(f"[K{i}] {s}"
                                    for i, s in enumerate(proof, 1))
            block = (proof_block + "\n" + block) if block else proof_block
        raw = generate.answer(question, block)
        # WORD-COVERAGE gate (same principle as on the causal path): if all of
        # the answer's content-words come from the given block+question there
        # is NO new claim — fabrication is structurally impossible → pass.
        # (Number/range answers from evidence sentences could never pass the
        # old triple-verify; coverage lets them through.)
        if evidence.covered(raw or "", block, question) and (
                not proof or generate.supported(raw, block)):
            # EVERY evidence-based answer goes through the support check (not
            # just numeric ones): with table cells wandering column-less,
            # "Orta" could stick to the wrong attribute — coverage sees words,
            # the auditor sees the binding.
            safe = raw
        elif raw and proof and evidence.digits_present(raw, block) \
                and generate.supported(raw, block):
            # SECOND TIER: word-coverage tripped on innocent narration words
            # ("olarak", "belirtilmiştir") but the digits are sound AND the
            # engine's support check said "the evidence says this" → pass. A
            # digit violation cannot be rescued by this tier (digits_ok first,
            # non-negotiable).
            safe = raw
        else:
            allowed = verify.allowed_of(self.memory, records)
            safe = verify.verify(self.memory, raw, allowed, self.mode,
                                 anchor="edge")
            if not safe:
                # generation stumbled → retry once; if it falls again, safe refusal.
                safe = verify.verify(self.memory,
                                     generate.answer(question, block),
                                     allowed, self.mode, anchor="edge")
                if not safe:
                    return generate.refusal(question) or FALLBACK_DONT_KNOW
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
            note = generate.hedge_note(src, question)
            if note:
                safe = f"{safe} {note}"
        return safe

    def _hedge(self, answer, record, message):
        """Appends a hedging NOTE to a VERIFIED answer resting on a low-trust
        fact. The answer text doesn't change (no fabrication can be added),
        only source+hedge at the end."""
        source = record.source or ""
        label = source[5:] if source.startswith("#web:") else "a stored source"
        note = generate.hedge_note(label, message)
        return f"{answer} {note}".strip() if note else answer

    # --- agentic research (approve, learn from the web) ----------------
    def _research(self, subject_label, question):
        """The user approved → fetch from Wikipedia, extract triples, write to
        the graph with #web+low trust, then answer the ORIGINAL question
        normally (source-tagged). The web is UNTRUSTED: the fact enters not as
        'I know it' but source-stamped+low trust (condition-4)."""
        text, url = research.wiki_summary(subject_label)
        if not text:
            return generate.refusal(question) or FALLBACK_DONT_KNOW
        # Extract ONE plain category (the essence, not taxonomic clutter) →
        # clean fact, clean answer. Many values/Latin terms were drowning the
        # small model.
        value = generate.category_from(subject_label, text)
        sk = link.resolve(self.memory, subject_label, self.vectors, create=True)
        vk = (link.resolve(self.memory, value, self.vectors, create=True)
              if value else None)
        if not value or sk is None or vk is None:
            return generate.refusal(question) or FALLBACK_DONT_KNOW
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
        # "kim yaptı" can be answered.
        id_block = "\n".join(
            f"{link.label_of(self.memory, r.subject)} "
            f"{link.label_of(self.memory, r.predicate)} → "
            f"{link.label_of(self.memory, r.value)}" for r in id_records)
        # CONVERSATION CONTEXT: give the recent turns too → conversation
        # continuity (what "araştır" refers to, the context of "ne yapıyorsun"
        # is kept). Fabrication is still filtered in verify.
        raw = generate.chat(message, id_block, history=self.history)
        # In chat, allowed = ONLY the identity facts. anchor="value": the
        # subject is a self-referential pronoun (ben/beni) that can't be
        # resolved; it suffices that the OBJECT (rüzgar) is allowed; external
        # fabrication (Google) still falls. See verify.verify.
        safe = verify.verify(self.memory, raw, self._identity, self.mode,
                             anchor="value")
        return safe or generate.refusal(message) or FALLBACK_DONT_KNOW

    # --- LMM strength: self-awareness (curiosity + contradiction pressure)
    def curiosity(self, most=5):
        """The system knows what it does NOT know: labels of concepts that
        appeared in conversation but have no / only-episodic records. Wraps
        `dynamics.gaps` (that organ existed in v3, unconnected to the lmm
        flow). The input of proactive research: 'I don't know this, shall I
        look it up'. Self/identity nodes excluded."""
        # Don't count predicate nodes (relation labels like tür/özellik) as
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
        from v3.memory import CONTRA, SUSPECT
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
