"""THE REAL GATE — the answer's FACT claims are checked against the facts
INJECTED for this question.

Backbone rule: "Qwen cannot say what is not in the graph" → even stricter:
"Qwen cannot say what was not given for THIS question". Both the subject and
the value of every fact claim in the answer must be inside `allowed` (the
injected identities); otherwise it drops.

This catches Qwen's parametric leakage like "heart → blood flow": "blood flow"
was not injected, resolves to no identity → not in allowed → the sentence
drops. It also catches value substitution (Berlin instead of Paris: Berlin is
not in allowed).

Two holes of the old version are closed:
  F2  in chat allowed=∅ → a chat answer carrying a fact claim drops (never
      returned raw)
  F5  no loose prefix tolerance — EXACT match (a false positive = fabrication
      passing, dangerous; a false negative = "I don't know", safe)

HONEST LIMIT (F3 residue): claim EXTRACTION still relies on Qwen (reextract).
If Qwen hides a fabrication in reextract, the gate can't see it. The
allowed-check NARROWS F3 substantially (the decision lives in the allowed set,
not in Qwen), but full independence requires a non-Qwen entity extractor —
future work.

Mode: STRICT (an unsupported sentence drops) · ASSIST (marked ‹doğrulanmamış›).
"""
import os
import re

from lmm import extract, link
from v3.dataset import fold


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def allowed_of(memory, records):
    """Subject+value identities of the injected records — the 'allowed' set."""
    keys = set()
    for r in records:
        if isinstance(r.subject, int):
            keys.add(r.subject)
        if isinstance(r.value, int):
            keys.add(r.value)
    return keys


def _has_edge(memory, sk, vk, v_label=None):
    """Is there a REAL edge between sk—vk in the graph (UNDIRECTED). Also sees
    derived (#inference) edges — `about` does not filter trust — so on
    TRANSITIVE predicates like is-a the derived "eagle→animal" passes, but the
    un-derived "france→eiffel" (capital is not transitive) drops.

    If v_label is given (reextract's value TEXT): when the key doesn't match, it
    tries a WORD-level match against THE SUBJECT'S OWN value labels
    (inflection/multi-word tolerance: "bird" ~ "sea bird"). SAFE: it only looks
    at the subject's own values — it cannot FABRICATE a relation to a different
    node, doesn't open the recombination hole (so that multi-word concepts
    learned from the web can be answered)."""
    if sk is None:
        return False
    if vk is not None:
        if any(r.value == vk for r in memory.about(sk, touch=False)):
            return True
        if any(r.value == sk for r in memory.about(vk, touch=False)):
            return True
    if v_label:
        want = {fold(w) for w in v_label.split() if len(w) >= 3}
        if want:
            for r in memory.about(sk, touch=False):
                have = {fold(w) for w in link.label_of(memory, r.value).split()
                        if len(w) >= 3}
                for a in want:
                    for b in have:
                        if a == b or a.startswith(b) or b.startswith(a):
                            return True
                        # SINGLE-LETTER GARBLING TOLERANCE: the engine can
                        # write "meyvedir" as "meyvedır" (a LoRA vowel-harmony
                        # error) and the CORRECT answer was dropping ("what is
                        # an olive"→"I don't know"). The criterion is as strict
                        # as link.resolve: common prefix >=5 AND both remainders
                        # <=3. SAFE: still only looks at the subject's OWN
                        # values — cannot fabricate a relation to another node
                        # (blast radius = the same subject's values).
                        common = os.path.commonprefix((a, b))
                        if len(common) >= 5 and len(a) - len(common) <= 3 \
                                and len(b) - len(common) <= 3:
                            return True
    return False


def verify(memory, answer, allowed, mode="STRICT", anchor="edge"):
    """Check the answer sentence by sentence. A sentence carrying no fact
    (greeting/opinion) passes. The edge check looks at the whole graph via
    `_has_edge` (derived edges included), not at the injected set.

    `anchor`:
      "edge"  (DEFAULT, ASK path) — the claim's (subject,value) must be a REAL
              edge in the graph. Node membership is not enough → recombination
              fabrication is blocked ("France→Eiffel").
      "value" (chat-IDENTITY path) — the object must be in allowed; since a
              self-referential pronoun subject (ben/beni/I) cannot be resolved
              it is NOT an anchor, BUT if the subject resolves to a real node it
              must either be allowed or an edge must be found — "Rüzgar wrote
              Python" (python≠allowed, no {python,rüzgar} edge) thus drops;
              "Rüzgar made me" (me→None) passes."""
    kept = []
    for sentence in _sentences(answer):
        claims = extract.reextract(sentence)
        if not claims:
            kept.append(sentence)          # no fact claim → passes
            continue
        grounded = True
        for subject, _predicate, value in claims:
            sk = link.resolve(memory, subject)
            vk = link.resolve(memory, value)
            if anchor == "value":
                ok = (vk in allowed) and (sk is None or sk in allowed
                                          or _has_edge(memory, sk, vk, value))
            else:                          # "edge": must be a real edge in the graph
                ok = _has_edge(memory, sk, vk, value)
            if not ok:
                grounded = False
                break
        if grounded:
            kept.append(sentence)
        elif mode == "ASSIST":
            kept.append(sentence + " ‹doğrulanmamış›")
        # STRICT: the unsupported sentence is discarded
    return " ".join(kept).strip()
