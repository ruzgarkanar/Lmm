"""What the conversation is about — one object, read by every organ.

WHY THIS FILE EXISTS. Over two days of live audit, five separate
readings grew for the same thing: the sentences the memory last spoke,
the documents those sentences named, the scope a follow-up inherits,
the scope one retrieval honoured, the topic a turn belongs to. Each was
right on its own and none of them agreed, and the last one exposed the
flaw in all: the topic was REBUILT every turn from the names in the last
sentence, so an answer whose words named no document — "EĞİTİM SÜRESİ: 1
gün", with the document in its stamp — reset the conversation to the
whole corpus, and the next question was answered from anywhere.

THE RULES, and there are only three.

  * WHAT WAS SAID IS EVIDENCE FOR WHAT IS ASKED NEXT. A conversational
    turn that asserted something keeps its own sentences; the next
    question can be answered from them ("what was it called?", "how
    many did you recommend?"). They are stored like anything else and
    gated like anything else — what keeps them in their place is the
    source they are filed under, which is never a document name.

  * THE TOPIC HOLDS UNTIL ANOTHER IS NAMED. A question that names
    documents takes the topic. An answer that names documents takes it.
    An answer that names none but RESTS on one takes it from that
    stamp. An answer that names nothing and rests on nothing leaves the
    topic where it was — silence is not a change of subject.

  * A QUESTION THAT NAMES ITS OWN SOURCE OWNS THE TURN. The carried
    topic is for the turn that names nothing: "the first one", "how
    many participants?", "and the difference?".
"""
from lmm import evidence

# WHERE THE MEMORY'S OWN SPEECH IS FILED. Not a document name, so every
# source-scoped reading — the census, `named_in`, the scope rule —
# passes it by, and a citation can never confuse the two.
SAID = evidence.SAID_SOURCE

# How many of the memory's own sentences stay readable. A conversation
# is a window, not an archive: the lines live in the store like any
# other evidence, and what this holds is the last answer.
WINDOW = 24


class Topic:
    """The conversation's subject, and the words it was last given in."""

    def __init__(self, store):
        self.store = store
        self.said = []                  # the last answer, sentence by sentence
        self.sources = set()            # the documents the topic is about

    # ------------------------------------------------------------ write
    def remember(self, said, mark=""):
        """Keep this turn's answer, and move the topic if it named one.

        `mark` is the provenance stamp the turn rested on — the topic's
        last resort, and the one that matters most in practice: a record
        answer speaks the row and names the document only in its stamp.
        """
        text = (said or "").strip()
        if not text:
            return 0
        kept = []
        for line in evidence.sentences_of(text):
            if len(line) >= 12:
                self.store.add(line, SAID)
                kept.append(line)
        if kept:
            self.said = kept[-WINDOW:]
        named = {src for src, _name in self._named_in(kept)}
        if named:
            self.sources = named
        elif mark and mark in self.store.by_source:
            self.sources = {mark}
        return len(kept)

    # ------------------------------------------------------------- read
    def _named_in(self, lines):
        """The store's own source names that stand in these lines —
        longest first, so a name is not swallowed by a shorter one it
        contains."""
        if not lines:
            return []
        folded = " ".join(evidence._words(" ".join(lines)))
        pairs = sorted(
            ((src, evidence._source_name(src))
             for src in self.store.by_source if src != SAID),
            key=lambda pair: -len(pair[1]))
        found, taken = [], ""
        for src, name in pairs:
            key = " ".join(evidence._words(name))
            if key and key in folded and key not in taken:
                found.append((src, name))
                taken += " " + key
        return found

    def names(self):
        """The topic's documents, by name — what a recap answers with."""
        return sorted(evidence._source_name(src) for src in self.sources)

    def scope_for(self, question, shape=""):
        """The documents this turn inherits, or nothing when the
        question opens a subject of its own.

        Two ways a turn refuses its inheritance. It NAMES a document, in
        which case that document is the subject; or it ASKS FOR SOMETHING
        TO BE PRODUCED, which is a new request rather than a follow-up.
        The second was learned from a live consultation: asked how long
        one course ran and then, next turn, what to give a sales team
        that stalls on objections, the recommendation was answered from
        inside the course just discussed — the need statement names no
        document, so the topic held and the reader got the wrong subject
        with the right grammar. A recap ("which ones, briefly") is a
        follow-up and still inherits; a request to produce is not.
        """
        if shape == "material":
            return set()
        named = self.store.named_in(question) if question else set()
        if named:
            # A QUESTION THAT NAMES A DOCUMENT IS ANSWERED FROM IT. This
            # used to return nothing — which cleared the inheritance and
            # left the turn reading the WHOLE store, so "X eğitiminin
            # katılımcı sayısı nedir" was seated with that field's line
            # from four other documents and the right one never arrived.
            # Measured on a 103-document catalogue: the named document
            # ranked first and its own answering line was not in the top
            # forty. The gate then refused, correctly, and the reader got
            # "I don't have that" about something the document states.
            #
            # Naming is the strongest thing a question can do, so it
            # binds: the turn reads that document. Asking it about
            # something the document does not hold now abstains, which is
            # the honest answer to that question.
            return set(named)
        return set(self.sources)

    def lines(self):
        """The memory's own last sentences — what a follow-up reads."""
        return list(self.said)
