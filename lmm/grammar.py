"""Grammar as data, not as code.

Sentence patterns used to be a chain of `if len(tokens) == 3 and ...` in the
parser, which meant the system could learn any fact but not a single new way of
saying one — and a second language would have meant a second parser.

A pattern is knowledge like any other, so it lives in a list that can be added
to, shipped in a pack, or worked out from an example. Nothing here is Turkish
except the entries themselves; the machinery matches slots against tokens and
knows nothing about which language it is looking at.

A slot is one of:
    KAVRAM   a thing being spoken about, plural stripped
    TUR      a type name, its copula stripped if it carries one
    NITELIK  a property, which must carry a copula
    SOZ      any word at all
    FIIL     a verb the lexicon knows, in either polarity
    SORU     a question particle
    KIM      an interrogative
Anything else in a pattern is a literal that must appear exactly.
"""
from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                           LACKS_PROPERTY, OBJECT, ALL)

KAVRAM = "{kavram}"
TUR = "{tür}"
NITELIK = "{nitelik}"
SOZ = "{söz}"
FIIL = "{fiil}"
SORU = "{soru}"
KIM = "{kim}"
ROL = "{rol}"        # a second concept wearing a case ending
NICEL = "{nicel}"    # how much of a kind: bütün / çoğu / bazı / hiçbir
SAHIP = "{sahip}"    # a possessor, marked as one by the language

SLOTS = (KAVRAM, TUR, NITELIK, SOZ, FIIL, SORU, KIM, ROL, NICEL, SAHIP)

FROM_VERB = "fiilden"       # the relation follows the verb's own polarity

# Concepts, types and properties are routinely more than one word — "müşteri
# bakiyesi", "bilgi türü", "çok hızlı". Verbs and particles never are.
PHRASE_SLOTS = (KAVRAM, TUR, NITELIK, SOZ)
MAX_PHRASE = 3


def _widths(slot):
    """How many tokens to try for a slot, in the order worth trying.

    A subject absorbs its modifiers while a predicate stays compact: "müşteri
    bakiyesi gizlidir" is one thing being called one word, not one thing being
    called two. So a concept reaches for the longest span it can and everything
    else takes the shortest that works.
    """
    if slot == KAVRAM:
        return range(MAX_PHRASE, 0, -1)
    if slot in PHRASE_SLOTS:
        return range(1, MAX_PHRASE + 1)
    return (1,)


class Pattern:
    def __init__(self, tokens, kind, relation=None, concept=None, target=None,
                 name="", object=None, quantifier=None):
        self.tokens = tokens
        self.kind = kind
        self.relation = relation
        self.concept = concept      # slot index, or None if the sentence omits it
        self.target = target
        self.object = object        # where the object sits — a matter of language
        self.quantifier = quantifier    # slot holding "how much of the kind"
        self.name = name or " ".join(tokens)

    def to_dict(self):
        return {"tokens": self.tokens, "kind": self.kind,
                "relation": self.relation, "concept": self.concept,
                "target": self.target, "object": self.object,
                "quantifier": self.quantifier, "name": self.name}

    @staticmethod
    def from_dict(data):
        return Pattern(**data)


class Grammar:
    """An ordered list of patterns and the machinery to match tokens against it.

    Order carries meaning: "kim uçar" has the same shape as "kuşlar uçar", so
    the question has to be tried before the lesson.
    """

    def __init__(self, patterns, morphology, known=()):
        self.patterns = list(patterns)
        self.morphology = morphology    # supplies the language's suffix rules
        # Live, not a snapshot: a concept taught mid-conversation must be
        # recognisable in the very next sentence.
        self._known = known

    def known(self):
        return set(self._known() if callable(self._known) else self._known)

    def add(self, pattern, first=False):
        self.patterns.insert(0, pattern) if first else self.patterns.append(pattern)
        return pattern

    def match(self, tokens, lexicon):
        """First pattern that fits, with the slots it captured."""
        for pattern in self.patterns:
            captured = self._fit(pattern, tokens, lexicon)
            if captured is not None:
                return pattern, captured
        return None, None

    def _fit(self, pattern, tokens, lexicon):
        """Match slots against tokens, allowing a concept to span several words.

        "müşteri bakiyesi" and "kredi tahsisi" are one thing each, and a domain
        is mostly made of terms like them. Shorter spans are tried first, so a
        sentence that fit before fits the same way now.
        """
        return self._fit_from(pattern.tokens, 0, tokens, 0, lexicon, [])

    def _fit_from(self, slots, slot_at, tokens, token_at, lexicon, captured):
        if slot_at == len(slots):
            return list(captured) if token_at == len(tokens) else None
        slot = slots[slot_at]
        widths = _widths(slot)
        for width in widths:
            if token_at + width > len(tokens):
                continue        # widths are not always ascending — never break
            value = self._capture_span(slot, tokens[token_at:token_at + width],
                                       lexicon)
            if value is None:
                continue
            captured.append(value)
            found = self._fit_from(slots, slot_at + 1, tokens, token_at + width,
                                   lexicon, captured)
            captured.pop()
            if found is not None:
                return found
        return None

    def _capture_span(self, slot, span, lexicon):
        """A slot's value over one or more tokens; suffixes ride the last word."""
        if len(span) == 1:
            return self._capture(slot, span[0], lexicon)
        if slot not in PHRASE_SLOTS:
            return None
        # A case-marked word is doing its own job in the sentence. Letting one
        # into a phrase turned "kartal serçeden büyüktür" into the concept
        # "kartal serçeden" being "büyük", written to memory without a murmur.
        if any(self.morphology.is_oblique(token) for token in span):
            return None
        tail = self._capture(slot, span[-1], lexicon)
        if tail is None:
            return None
        return " ".join(list(span[:-1]) + [tail])

    def _capture(self, slot, token, lexicon):
        """What this token means in this slot, or None if it does not fit."""
        morphology = self.morphology
        if slot not in SLOTS:
            return token if token == slot else None
        if slot == KAVRAM:
            # A closed-class word is never the thing being talked about.
            # "bazı kuşlar uçmaz" was read as the concept "bazı" doing something.
            if token in getattr(morphology, "quantifiers", ()):
                return None
            return morphology.strip_plural(token)
        if slot == SOZ:
            return token
        if slot == TUR:
            return morphology.strip_copula(token)
        if slot == NITELIK:
            if not morphology.has_copula(token):
                return None
            return morphology.strip_copula(token)
        if slot == FIIL:
            reading = lexicon.reading(token)
            return reading if reading is not None else None
        if slot == SORU:
            return token if token in morphology.question_particles else None
        if slot == KIM:
            return token if token in morphology.interrogatives else None
        if slot == SAHIP:
            stem = morphology.strip_genitive(token, self.known())
            return stem if stem != token else None
        if slot == NICEL:
            return morphology.quantifiers.get(token)
        if slot == ROL:
            stem, role = morphology.role_of(token)
            return (stem, role) if role else None
        return None

    def read(self, pattern, captured):
        """Turn a match into (kind, relation, concept, target, object)."""
        concept = self._slot_value(pattern, captured, pattern.concept)
        target = self._slot_value(pattern, captured, pattern.target)
        obj, role = self._object_and_role(pattern, captured)
        relation = pattern.relation
        if relation == FROM_VERB:
            relation = CAN if self._polarity(pattern, captured) else CANNOT
        return (pattern.kind, relation, concept, target, obj, role,
                self._slot_value(pattern, captured, pattern.quantifier) or ALL)

    def _object_and_role(self, pattern, captured):
        """The second concept and what it is doing, when a pattern captured one."""
        if pattern.object is None:
            return None, None
        value = captured[pattern.object]
        if isinstance(value, tuple):
            return value[0], value[1]
        return value, OBJECT

    def _slot_value(self, pattern, captured, index):
        if index is None:
            return None
        value = captured[index]
        return value[0] if isinstance(value, tuple) else value

    @staticmethod
    def _polarity(pattern, captured):
        for slot, value in zip(pattern.tokens, captured):
            if slot == FIIL:
                return value[1]
        return True


def learn_pattern(tokens, kind, relation, concept_index, target_index,
                  lexicon, morphology):
    """Work out a reusable pattern from one labelled sentence.

    Given "kediler zıplar" labelled as a lesson about ability, this decides
    which words were the example and which were the shape, and returns a pattern
    that will match the next sentence built the same way.
    """
    slots = []
    for index, token in enumerate(tokens):
        if index in (concept_index, target_index):
            slots.append(_slot_for(token, index, target_index, relation,
                                   lexicon, morphology))
        elif token in morphology.question_particles:
            slots.append(SORU)
        elif token in morphology.interrogatives:
            slots.append(KIM)
        elif lexicon.knows(token):
            slots.append(FIIL)
        else:
            slots.append(token)         # a function word: part of the shape
    return Pattern(slots, kind, relation, concept_index, target_index)


def induce(pairs, parser, morphology, lexicon, max_length=6, evidence=2):
    """Work patterns out of prose, with nobody labelling anything.

    Each pair holds a passage a person wrote and the plain sentences a model
    restated it as. We already know what the plain sentences mean, because our
    own parser reads them — so every pair is a labelled example that no human
    had to label.

    Alignment is by content: the prose sentence that mentions both the concept
    and the target is the one that said it. Sentences longer than a handful of
    words are left alone, since a pattern taken from a twenty-word sentence
    would only ever match that sentence again.

    A shape has to turn up more than once before it is believed. One occurrence
    is a coincidence; the same shape twice is a rule — the same standard we hold
    facts to.
    """
    seen = {}
    for pair in pairs:
        prose = _sentences(pair.get("düzyazı", ""))
        for plain in pair.get("sade", "").splitlines():
            intent = parser(plain)
            if intent.concept is None or intent.target is None:
                continue
            for sentence in prose:
                tokens = _tokens(sentence, morphology)
                if not 2 < len(tokens) <= max_length:
                    continue
                concept_at = _where(tokens, intent.concept, morphology)
                target_at = _where(tokens, intent.target, morphology)
                if concept_at is None or target_at is None or concept_at == target_at:
                    continue
                pattern = learn_pattern(tokens, intent.kind, intent.relation,
                                        concept_at, target_at, lexicon, morphology)
                key = (tuple(pattern.tokens), pattern.kind, pattern.relation,
                       pattern.concept, pattern.target)
                seen[key] = seen.get(key, 0) + 1
                break
    return [Pattern(list(tokens), kind, relation, concept, target)
            for (tokens, kind, relation, concept, target), count in seen.items()
            if count >= evidence]


def _sentences(text):
    found, current = [], ""
    for character in text:
        if character in ".!?\n;":
            if current.strip():
                found.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        found.append(current.strip())
    return found


def _tokens(sentence, morphology):
    cleaned = "".join(c for c in sentence if c not in ".,!?;:\"'")
    return cleaned.replace("İ", "i").replace("I", "ı").lower().split()


def _where(tokens, word, morphology):
    """Which token carried this concept, allowing for the suffixes it wears."""
    for index, token in enumerate(tokens):
        stem = morphology.strip_copula(morphology.strip_plural(token))
        if stem == word or token == word or token.startswith(word):
            return index
    return None


def _slot_for(token, index, target_index, relation, lexicon, morphology):
    if lexicon.knows(token):
        return FIIL
    if index == target_index and relation in (IS_A, NOT_A):
        return TUR
    if index == target_index and relation in (HAS_PROPERTY, LACKS_PROPERTY):
        return NITELIK if morphology.has_copula(token) else SOZ
    return KAVRAM
