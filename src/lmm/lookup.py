"""THE GRAPH-FIRST ANSWER PATH — an answer with no model call in it.

The architecture always permitted this and the measurement said it never
happened: `benchmarks/COST.md` §3, **0 of 17 questions** answered with zero
model calls, on a corpus whose answers sit in the graph as triples. Every
question went to the engine, was re-read by the engine, and had its relation
checked by the engine — 6.6 calls and 4,790 prompt tokens each. That is the
thesis ("the symbolic part needs no model") being true in the code and false
in the bill.

This module is the missing step. Given a message and a memory it returns THE
RECORD THE MESSAGE ASKS FOR, or `None`. It makes no model call, holds no word
of any language, and never invents: what it returns is a record the graph
already holds, and the answer the session speaks is that record's own value.

WHEN IT ANSWERS — the two safe shapes
-------------------------------------
Both require the question to NAME the thing it asks about, because naming is
the only structural evidence that the graph's record is the record wanted.

1. **A named value under a transitive predicate.** "is vorlin a liquid" names
   `liquid`, and the graph holds `vorlin -[type]-> liquid` as a DERIVED record
   (drink→liquid, vorlin→drink). The restriction to a transitive predicate is
   the load-bearing part: a transitive predicate is a taxonomy the graph has
   CLOSED under derivation (`memory.transitive`, learned from witnessed
   triangles in `core/transitive.py`), so a hit is a settled question rather
   than whatever one sentence happened to yield. Open fields are not closed —
   the graph holding `norgul -[relation]-> multiplies` says nothing about what
   else norgul does, and answering "what happens if norgul multiplies" out of
   it would be a true fact answering a question nobody asked, which is exactly
   the failure class `Session._relation_held` exists to stop. Measured: with
   the transitive restriction removed, that question is answered "norgul →
   multiplies" and the EN corpus drops 17/17 → 16/17.

2. **A named field.** The question names a PREDICATE the subject carries
   ("what is X's colour" against `X -[colour]-> blue`), and that predicate has
   exactly one value left after `retrieve.specific` has dropped the vaguer of
   two true ones. Here there is no unasked-relation risk at all: the asked slot
   and the answered slot are the same slot by construction.

WHEN IT DECLINES — and it declines far more often than it answers
-----------------------------------------------------------------
Anything else falls through to the existing semantic path, unchanged. In
particular THE FIELDLESS QUESTION IS OUT ON PURPOSE. "norgul nedir" names no
field, and so does "Nortlann'ın başkenti neresidir"; nothing structural
separates an interrogative the graph has never seen from a field name the
graph has never seen, and the graph holds `nortlann -[type]-> ada`. Answering
the first from the subject's single taxonomy value means answering the second
with it too — a confident wrong answer to a question the corpus deliberately
cannot answer. That is a real cost (the eight "what is X" questions per
language stay on the paid path) and it is the honest price of the guarantee.
It is written down rather than tuned away.

Everything the decision uses is already in the system: `link.resolve` for
label→identity (the O(1) index, no model), `inflect.same_stem` for "are these
two surface forms one word" — the ONE inflection criterion the read paths
share — and `retrieve.specific` for "which of two true values answers". No
list of words, no stop-word set, no per-document constant.
"""
from lmm.core.gate import SPEAK
from lmm import evidence, inflect, link, retrieve


def _label_words(memory, key):
    """The content words of an identity's label, by the same reading the
    evidence gate uses. A label of two words ("ser vivo", "green flame") is two
    words here, and that is what makes NAMING it mean naming all of it."""
    return evidence._words(link.label_of(memory, key))


def _named_by(words, memory, key):
    """Do the question's words NAME this identity — all of its label, not a
    word of it.

    The whole label, because a shared word is not a name: the English corpus
    holds `island -[action]-> write the guide`, and "what is the population of
    the island" shares the word 'the' with it. Requiring every word of the
    label to be present is what keeps a function word from nominating a record
    (measured: that question was answered "island → write the guide" while a
    single shared word sufficed). An identity whose label carries no content
    word cannot be named at all, and returns False rather than vacuously True.
    """
    parts = _label_words(memory, key)
    if not parts:
        return False
    return all(any(inflect.same_stem(part, word) for word in words)
               for part in parts)


def _spans(words):
    """Every contiguous run of the question's content words, longest first.

    A graph node is not always spelled with one word. A spreadsheet's row
    label, a PDF spec table's field, a country — `united states`, `ser vivo`,
    `screen diagonal` — is one identity whose label is several words, and a
    scan of the question's words ONE AT A TIME can never name it: neither
    `united` nor `states` resolves, so a question that says exactly what the
    graph stores was falling through to the paid path. The runs are the
    question's own word ORDER, so nothing is assembled that the question did
    not say in that order.

    There is no length bound, and deliberately no constant: a run is a
    candidate only if the index resolves it, which is a dict lookup, and a
    question has a handful of content words. Bounding the run length would be
    a guess about how long a document's labels are — exactly the per-document
    constant this repository refuses.
    """
    return [tuple(words[i:j])
            for size in range(len(words), 0, -1)
            for i in range(len(words) - size + 1)
            for j in (i + size,)]


def _subjects(memory, words):
    """The graph nodes this message could be ABOUT: the question's own word
    RUNS that resolve to an identity CARRYING RECORDS.

    Longest run first, ties broken by the run itself, for the same reason
    `Session._answer`'s subject fallback sorts that way — a set of strings
    iterated raw comes out in PYTHONHASHSEED order, and here that order would
    decide which node is asked about.

    A run CONTAINED IN a longer run that also resolved is dropped. Both name
    something the graph holds, and the longer one is the more specific naming
    — the same judgement `retrieve.specific` makes between two true values,
    made here between two true readings of the same words. Without it a
    question naming `ser vivo` would nominate `ser vivo` AND `vivo` as two
    separate subjects, and `find`'s uniqueness test would then decline a
    question the graph can settle.

    `touch=False` is not used at this stage because nothing is read yet; the
    candidates are drawn from the index alone."""
    found, spoken = [], []
    for span in sorted(_spans(words), key=lambda s: (-len(s), s)):
        if any(_inside(span, wider) for wider in spoken):
            continue
        key = link.resolve(memory, " ".join(span))
        if key is None or not memory.by_subject.get(key):
            continue
        spoken.append(span)
        if key not in found:
            found.append(key)
    return found


def _inside(span, wider):
    """Is this run a contiguous part of that longer one?"""
    return len(span) < len(wider) and any(
        wider[i:i + len(span)] == span
        for i in range(len(wider) - len(span) + 1))


def _records(memory, subject):
    """The subject's speakable records — the same admission rule as
    `retrieve.gather`, including its #inference exception, and UNCAPPED: gather
    keeps the best four for a prompt, and a prompt is not what this is for.

    `touch=False`: this is a probe. Several candidate subjects are examined per
    message and only one of them (at most) is answered from, so counting all of
    them as accesses would turn the identity counter into a "was mentioned in a
    question" counter — the same confusion `Memory.about` documents. The record
    that is actually spoken is touched by the caller.
    """
    return [r for r in memory.about(subject, touch=False)
            if r.trust >= SPEAK or r.source == "#inference"]


def find(memory, message):
    """The single record this message asks for, or None.

    None means "not safely decidable here", never "unknown" — the caller falls
    through to the semantic path and loses nothing.

    The uniqueness test is over ALL candidate subjects at once, not per
    subject. A message naming two things the graph can answer about is a
    message this path has no business resolving, and taking the first would
    make the answer depend on the candidate ordering.
    """
    words = evidence._words(message)
    if not words:
        return None
    hits = []
    for subject in _subjects(memory, words):
        # The subject's own name is not a field name. Without this, a message
        # is forever naming the record whose value happens to repeat its
        # subject's word.
        own = _label_words(memory, subject)
        asked = [w for w in words
                 if not any(inflect.same_stem(part, w) for part in own)]
        if not asked:
            continue
        by_value, by_field = [], []
        for record in _records(memory, subject):
            if (record.predicate in memory.transitive
                    and _named_by(asked, memory, record.value)):
                by_value.append(record)
            elif record.predicate is not None \
                    and not link.label_of(memory, record.predicate).startswith("#") \
                    and _named_by(asked, memory, record.predicate):
                # A '#'-prefixed predicate is a RESERVED slot (#causes), not a
                # field of the document. Format, not language — the same '#'
                # convention the sources and the hedge already use. It is
                # excluded because its direction is the whole content of the
                # relation, and a question that happens to share a word with
                # the reserved label says nothing about which way it is being
                # asked; the causal route reads that direction properly.
                by_field.append(record)
        # A NAMED VALUE IS ALREADY THE SPECIFIC ONE — the question chose it, so
        # it is not for `specific` to prefer another. A named FIELD may hold
        # both "metal" and "substance", and there `specific` is exactly the
        # organ that says which of two true values answers.
        hits += by_value + retrieve.specific(memory, by_field)
    unique = []
    for record in hits:
        if record.key not in {r.key for r in unique}:
            unique.append(record)
    if len(unique) != 1:
        return None
    memory.about(unique[0].subject)      # a real access — refresh the record
    return unique[0]


def render(memory, record):
    """The answer, as the graph holds it: the subject, the field it was asked
    under, and the value.

    NOT A SENTENCE, and that is the point. A sentence needs the language model,
    which is the call this path exists to avoid; a field name and its value is
    a complete answer to "what is X's Y" in the sense that matters — it states
    the fact and nothing else, so there is nothing in it to verify, nothing to
    read back and nothing to fabricate. Whoever wants prose can ask for it
    (`ask(..., fluent=True)`) and pay for it.

    The arrow and the dash are FORMAT, the same standing as the '#' source
    prefix and the '~' uncertainty mark — they read identically in every
    language, and `retrieve.facts_block` already renders a record this way for
    the engine's eyes."""
    subject = link.label_of(memory, record.subject)
    value = link.label_of(memory, record.value)
    field = link.label_of(memory, record.predicate)
    return f"{subject} — {field} → {value}" if field else f"{subject} → {value}"
