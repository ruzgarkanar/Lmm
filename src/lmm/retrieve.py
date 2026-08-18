"""Pulls facts from the graph and builds the FACTS block handed to Qwen.

Three differences from RAG: (1) what is pulled is a structurally VERIFIED
triple, not a free-text chunk; (2) every fact carries a SOURCE; (3) AFTER the
answer is generated it is re-checked against the same facts (verify.py) —
classic RAG never audits the output.
"""
from lmm.core.gate import SPEAK
from lmm import link


def gather(memory, subject_key, most=4, associative=True):
    """The relevant records around the subject.

    `associative=True` (default): direct facts (eagle→bird) + one-hop facts
    about the subject's VALUES (bird→animal) — associative context.
    `associative=False` (answer path): ONLY direct facts. For consistency with
    the EDGE check (verify): with the one-hop "bird→animal" injected, the model
    said "an eagle is an animal" but {eagle,animal} is not a direct edge (not
    derived if is-a isn't transitive yet) → the answer was dropping. Given the
    direct fact, the model says the grounded "an eagle is a bird" and passes the
    edge check. trust>=SPEAK.
    """
    if subject_key is None:
        return []
    # #inference exception: a derived fact (INFERRED 0.35) is below SPEAK(0.4)
    # but STAYING SILENT is wrong — the graph derived it via two-witness
    # transitivity ("zilfen→living") and in the benchmark the answer was
    # escaping to "is a bird". A derived fact IS SPOKEN, but since its trust is
    # below CERTAIN, _hedge appends a caveat note: "according to my
    # inference..." — behavior that is honest yet doesn't hide the capability.
    direct = [r for r in memory.about(subject_key)
              if r.trust >= SPEAK or r.source == "#inference"]
    records = list(direct)
    if associative:
        for r in direct:                    # one-hop: around the values
            if isinstance(r.value, int):
                records += [x for x in memory.about(r.value) if x.trust >= SPEAK]
    seen, uniq = set(), []
    for r in records:
        if r.key not in seen:
            seen.add(r.key)
            uniq.append(r)
    uniq.sort(key=lambda r: -r.trust)
    return uniq[:most]


def specific(memory, records, keep=()):
    """Drop the VAGUER of two true answers about the same subject.

    The graph derives its is-a chain, so both "norgul is a plant" and "norgul
    is a living thing" end up as records about norgul — both true, and only one
    of them ANSWERS "what is a norgul". Which is which is not a matter of
    trust or of wording: the graph itself knows, because it holds the edge
    plant→living-thing. A value that is an ANCESTOR of another value present
    under the same subject and predicate is the more general claim, and the
    descendant is what the question wanted.

    Measured (corpus, "norgul nedir"): the answer came back "norgul is a
    living thing" at every commit in this history. The cause is structural
    rather than a ranking accident — the evidence path removes #doc triples
    from the block, so the SPECIFIC observation (norgul→plant, read from the
    document) leaves and its VAGUE derived conclusion (norgul→living, source
    #inference) stays. The filter kept the conclusion and discarded the
    premise, and the conclusion is always the more general one.

    `keep`: record keys that must survive regardless — a question that NAMES
    the general value ("is a zilfen a living thing") is asking for exactly the
    general claim, and the caller's targeting already knows which those are.
    Pure graph: one edge lookup per pair, no model call, no language.
    """
    drop = set()
    for one in records:
        if not isinstance(one.value, int):
            continue
        # the values the graph places ABOVE this one, on the same relation
        above = {r.value for r in memory.about(one.value, touch=False)
                 if r.predicate == one.predicate}
        if not above:
            continue
        for other in records:
            if (other.key not in keep and other.key != one.key
                    and other.subject == one.subject
                    and other.predicate == one.predicate
                    and other.value in above):
                drop.add(other.key)
    return [r for r in records if r.key not in drop]


def facts_block(memory, records):
    """Converts records into the numbered text handed to Qwen. The source label
    (#operator etc.) stays INSIDE, it is not put in the injected text —
    otherwise Qwen copies it into the answer (measured). Provenance sits on the
    record key, shown separately if needed."""
    lines = []
    for i, r in enumerate(records, 1):
        subject = link.label_of(memory, r.subject)
        value = link.label_of(memory, r.value)
        lines.append(f"[{i}] {subject} → {value}")
    return "\n".join(lines)
