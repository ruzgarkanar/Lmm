"""Mind — respond()'s message-less, signal-driven TWIN. The autonomous growth loop.

respond() is REACTIVE: a message arrives → an answer. Mind is PROACTIVE: even
without a message it produces work from internal signals (contradiction pressure,
knowledge gaps) — resolves its contradictions, derives, distills, (later)
researches. "A mind that thinks on its own and grows on its own."

TWO SPEEDS (a sharp split):
  ms-internal : derive / resolve / distill — PURE GRAPH, NO model/web, microseconds
  slow-external: research — Qwen+web, seconds, budgeted+approved (Step 4, not yet)

This is NOT LIFE: scheduled, bounded, resumable computation over the graph.
When the signals drain it RESTS — not substrate-free. Numbers ≠ understanding.

For now this file is only the PURE-MS core (Steps 0-2): zero model/web/thread risk.
Slow-external research + guardrails (budget, #auto stamp, frontier freeze) will be
discussed and added separately.
"""
from lmm.core import dynamics
from lmm.core.gate import SPEAK
from lmm.core.memory import DOCUMENT
from lmm import generate, link
from lmm import research as web


class Mind:
    """Wraps a Session; produces proactive work from internal signals. The
    Session (conversation) is untouched; the two share the same `memory`."""

    def __init__(self, session):
        self.session = session
        self.memory = session.memory

    # --- one tick (ms, pure graph) -------------------------------------
    def step(self):
        """One thinking tick — a free (ms) structural move. Priority: resolve
        CONTRADICTION > DERIVE (transitive closure) > DISTILL (sleep). Returns:
        a NUMERIC report (dict) of the action taken. NO model/web —
        deterministic, offline.

        (In Step 3 the if-order will be upgraded to a drive/cost argmax; but the
        behavior is already 'prefer the cheap structural move' — no
        model-judgment actions.)"""
        # 1) if there is a CONTRADICTION, resolve it — arbitrate the record
        #    under highest pressure
        pres = dynamics.pressure(self.memory)
        if pres:
            record = self.memory.records.get(pres[0][0])
            if record is not None:
                dynamics.arbitrate(self.memory, record)
                return {"action": "resolve", "remaining_conflicts": len(pres) - 1}
        # 2) DERIVE — fill in missing #inference edges on transitive predicates
        derived = self._derive_closure()
        if derived:
            return {"action": "derive", "derived": derived}
        # 3) DISTILL — sleep (settle/fade/prune). If nothing changes, saturated.
        report = self.session.sleep()   # sleep also settles SUSPECT verdicts
        return {"action": "distill", **report}

    def _derive_closure(self):
        """Close missing derivations on transitive predicates. First learn
        transitivity from the data (_learn_transitive), then _derive around each
        edge. ms, pure graph. Returns: number of new #inference edges derived
        this round."""
        for pk in list(self.memory.by_predicate.keys()):
            self.session._learn_transitive(pk)
        new = 0
        for pk in list(self.memory.transitive):
            for key in list(self.memory.by_predicate.get(pk, ())):
                record = self.memory.records.get(key)
                if record is None or record.source == "#inference":
                    continue
                before = len(self.memory.records)
                self.session._derive(record.subject, pk, record.value)
                new += len(self.memory.records) - before
        return new

    # --- loop (stops at saturation) ------------------------------------
    def run(self, max_steps=200):
        """Think until the signals or the budget run out. Stops (rests) at
        SATURATION: no contradiction + nothing to derive + distilling changes
        nothing. The scheduler state is transient; it is recomputed from the
        graph → stopping and resuming is free."""
        log = []
        for _ in range(max_steps):
            report = self.step()
            log.append(report)
            if report["action"] == "distill" and not any(
                    report.get(k) for k in
                    ("settled", "faded", "judged", "pruned", "distilled")):
                break                       # saturated — rest
        return log

    # --- semi-autonomous: SURFACE its curiosity, fill it with approval --
    def wonder(self, most=3):
        """The most VALUABLE curiosity: concepts referenced OFTEN but UNDEFINED
        — "I use this constantly but don't know the thing itself" (e.g. animal:
        it says lion→animal, tiger→animal but doesn't know what an animal IS).
        Priority = reference count (by_value in-degree) × undefinedness. So a
        gap that is ACTUALLY used comes first, not a leaf. Does NOT touch the
        web — only a suggestion; a human approves (semi-autonomous)."""
        scored = []
        for key in self.memory.identities:
            if key in self.session._identity:
                continue
            # defined?: is there a SPEAK-or-above fact ABOUT this concept (as subject)
            if any(self.memory.records[k].trust >= SPEAK
                   for k in self.memory.by_subject.get(key, ())):
                continue                    # defined → not a curiosity
            refs = len(self.memory.by_value.get(key, ()))   # how often it's used
            label = link.label_of(self.memory, key)
            if label and refs > 0:          # at least one reference → meaningful gap
                scored.append((refs, label))
        scored.sort(reverse=True)
        return [label for _refs, label in scored[:most]]

    def top_curiosity(self, min_refs=2):
        """The most DOMINANT curiosity — the ONE concept referenced at least
        `min_refs` times but undefined (label, refs). For proactive surfacing:
        the loop says 'I want to learn this' when it has used something often
        enough without knowing it (a drive threshold). The threshold separates
        a frequently used gap from noise."""
        best = None
        for key in self.memory.identities:
            if key in self.session._identity:
                continue
            if any(self.memory.records[k].trust >= SPEAK
                   for k in self.memory.by_subject.get(key, ())):
                continue
            refs = len(self.memory.by_value.get(key, ()))
            if refs >= min_refs and (best is None or refs > best[1]):
                best = (link.label_of(self.memory, key), refs)
        return best

    def research(self, subject_label):
        """WITH HUMAN APPROVAL, fill a gap from the web — INGEST-ONLY,
        autonomy-safe: wiki_summary(title) → category_from → gate.admit(#web,
        LOW trust). NO synthetic question (the title is already the label →
        condition-5), NO speech generation — only sourced+low-trust ingestion.
        Provenance (#web:url) remains for audit; since the web is untrusted it
        is never CERTAIN, it never overrides an operator fact."""
        text, url = web.wiki_summary(subject_label)
        if not text:
            return False
        value = generate.category_from(subject_label, text)
        sk = link.resolve(self.memory, subject_label, self.session.vectors,
                          create=True)
        vk = (link.resolve(self.memory, value, self.session.vectors, create=True)
              if value else None)
        if not value or sk is None or vk is None or sk == vk:
            return False
        self.session.gate.admit(sk, None, vk, f"#web:{url}", DOCUMENT)
        self.memory.lived(f"auto-research:{subject_label}", outcome=1.0,
                          about=[self.memory.self_key])
        return True

    def status(self):
        """The growth pulse — numbers, not language (the project's ethic). How
        much the baby has 'grown'."""
        return {
            "facts": len(self.memory.records),
            "concepts": len(self.memory.identities),
            "open_gaps": len(dynamics.gaps(self.memory)),
            "open_conflicts": len(dynamics.pressure(self.memory)),
            "derived": sum(1 for r in self.memory.records.values()
                           if r.source == "#inference"),
        }
