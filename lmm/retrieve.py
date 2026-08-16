"""Pulls facts from the graph and builds the FACTS block handed to Qwen.

Three differences from RAG: (1) what is pulled is a structurally VERIFIED
triple, not a free-text chunk; (2) every fact carries a SOURCE; (3) AFTER the
answer is generated it is re-checked against the same facts (verify.py) —
classic RAG never audits the output.
"""
from v3.gate import SPEAK
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
