"""The training set: from raw text to the form the reader will learn.

There are two things the reader will learn, and both come out of the same
sentence:

    what it wants to do        PASS · ASK · WRITE
    which letter in what role  0 outside · 1 subject · 2 predicate · 3 value

The labels are NOT HAND-WRITTEN, they come from ALIGNMENT: we hold a
(sentence, fact) pair; where the fact's parts occur in the sentence is found,
and the letters are marked with that role. No suffix list, pattern or word
class is used.

    "An eagle is a kind of bird of prey."   +   (eagle, type, bird)
     1111110000000000000033300000000         WRITE

The labels for question sentences come by the same road: a question is the
INCOMPLETE form of a fact — its subject stated, its value asked.

Nothing belonging to language exists in this file. Alignment amounts to
searching for where a string occurs in a string, and it does not know what
language it is in.
"""
import json
import os
import random
import unicodedata


def layout_char(ch):
    """Is this character LAYOUT rather than part of a word?

    Unicode's own general category answers it (P*, the punctuation classes),
    so the question is asked once, in the place `fold` already lives, and the
    three organs that need it — the span grammar's word forms
    (`lmm/grammar.py`), a table's row label (`lmm/tables.py`, where a leading
    dot is the sheet's indentation and not part of `.Puerto Rico`) — ask the
    same one. No character is named anywhere.
    """
    return unicodedata.category(ch).startswith("P")


def bare(text):
    """`text` without the layout characters it begins or ends with.

    A US Census sheet writes its sub-rows as `.Alabama`, `.California`,
    `.Puerto Rico` — the leading dot is INDENTATION, drawn with a character
    because a spreadsheet cell has no margin, and it was measured making every
    one of those rows unreachable by its own name (`benchmarks/field/REPORT.md`
    §4). Reading it as layout is not a convention this file knows about; it is
    what the character's Unicode category says it is.
    """
    text = str(text)
    while text and layout_char(text[0]):
        text = text[1:]
    while text and layout_char(text[-1]):
        text = text[:-1]
    return text


def fold(text):
    """Length-PRESERVING letter folding — for matching.

    `lower()` is language-dependent and changes length in Turkish:
    'İ'.lower() produces two code points ('i̇'). Measured — 'İstanbul büyük
    bir şehirdir' went from 27 letters to 28, the indexes shifted and
    `span_of` returned None: EVERY example with an İ-bearing subject was
    silently dropping out of the training set. We were broken in our own
    language, and what noticed it was the language-independence sweep.

    The folding is letter by letter: the FIRST code point of each letter's
    Unicode casefold. The length never changes, the indexes stay valid in the
    original text. This is not a language rule, it is Unicode's own table —
    it treats every script the same, German ß and Greek Σ included.
    """
    return "".join(ch.casefold()[0] for ch in text)

OUT, SUBJECT, PREDICATE, VALUE = 0, 1, 2, 3
PASS, ASK, WRITE = 0, 1, 2


def span_of(text, name):
    """The name's place in the text — even in its inflected form.

    The root is found and extended to the end of the word: the suffix counts
    as part of the piece itself, because the network will see the suffixed
    form and must learn to extract the root ITSELF. Which suffix it is, is
    never asked — had it been asked, this file would need a suffix list.
    """
    if not name:
        return None
    folded, name = fold(text), fold(name)
    at = folded.find(name)
    if at < 0:
        return None
    end = at + len(name)
    # Extension to the end of the word only BY LETTER: the apostrophe special
    # case (`== "'"`) was a character rule put in for the Turkish proper-noun
    # suffix, and it was torn out. In the subject "ankara'da" only "ankara"
    # is marked now — the root already matches, marking the suffix demanded a
    # rule.
    while end < len(text) and text[end].isalpha():
        end += 1
    return at, min(end, len(text))


def label(text, subject, predicate, value):
    """A role for each letter of the sentence. None if it cannot be aligned.

    The subject is mandatory: there is nothing to learn from a sentence whose
    subject cannot be found. If the predicate and value cannot be found their
    places stay empty — in question sentences that is already how it is.
    """
    roles = [OUT] * len(text)
    place = span_of(text, subject)
    if place is None:
        return None
    for at in range(*place):
        roles[at] = SUBJECT
    for name, role in ((predicate, PREDICATE), (value, VALUE)):
        if not name:
            continue
        where = span_of(text, name)
        if where is None:
            continue
        if any(roles[at] != OUT for at in range(*where)):
            continue        # clash: the same letter cannot be given two roles
        for at in range(*where):
            roles[at] = role
    return roles


def from_facts(paths, longest=256, most=None):
    """WRITE examples from (sentence, fact) files.

    ONE example per sentence: when the same sentence yielded several facts,
    making each one a separate example set the data against itself — measured
    in the old architecture, joint concept+value accuracy stalled at 0.9%. A
    sentence has one subject; the most-repeated subject is taken, the others
    drop.
    """
    held = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            text = row.get("cümle")
            subject = (row.get("kavram") or "").strip()
            predicate = (row.get("ilişki") or "").strip()
            value = (row.get("hedef") or "").strip()
            if not (text and subject and predicate) or len(text) > longest:
                continue
            held.setdefault(text, []).append((subject, predicate, value))

    found = []
    for text, facts in held.items():
        counted = {}
        for subject, _, _ in facts:
            counted[subject] = counted.get(subject, 0) + 1
        subject = max(counted, key=counted.get)
        chosen = next((one for one in facts if one[0] == subject), None)
        if chosen is None:
            continue
        roles = label(text, *chosen)
        if roles is None:
            continue
        # CLEANLINESS FILTER: a label is taught only if the subject AND value
        # align properly in the sentence. Measured — 66% of the facts were
        # noisy (the old reading pipeline had extracted them at 57% accuracy):
        # labels like "beşiktaş/sempatik" that never occur in the sentence. A
        # noisy label teaches the model WRONG and caps VAL at ~42%. Few but
        # correct > many but noisy.
        from lmm.core.reader import spans as _spans
        gs, _, gv = _spans(text, roles)
        if not (gs and gv):
            continue
        fs = fold(chosen[0])
        fv = fold(chosen[2]) if chosen[2] else ""
        if not (gs == fs[:len(gs)] and fv and fv.startswith(gv[:min(4, len(gv))])):
            continue        # alignment is noisy — skip
        found.append((text, WRITE, roles))
        if most and len(found) >= most:
            return found
    return found


def from_dialogue(paths, longest=256, most=None, keep=0.25):
    """PASS examples from conversation lines.

    Most lines of a chat carry no knowledge: greeting, reaction, transition.
    These are PASS, and the reader recognizing them matters as much as
    recognizing the fact-bearers — if it does not, it tries to invent a fact
    for every utterance.

    A `keep` share is sampled because the conversation corpus is a much
    larger and unbalanced set than the fact corpus, and it teaches collapsing
    onto the majority class.
    """
    found = []
    chance = random.Random(7)
    for path in paths:
        if not os.path.exists(path):
            continue
        opener = _open(path)
        for line in opener:
            text = line.strip()
            if not (4 < len(text) <= longest):
                continue
            if chance.random() > keep:
                continue
            found.append((text, PASS, [OUT] * len(text)))
            if most and len(found) >= most:
                return found
    return found


def _open(path):
    if path.endswith(".gz"):
        import gzip
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, encoding="utf-8", errors="ignore")


def from_questions(paths, longest=256, most=None):
    """ASK examples from real questions.

    The audit's most critical finding: there was NO ASK producer. The reader
    was three-class but was being trained with two; when a question came it
    said WRITE, the session found nothing to write and returned empty — the
    question-answer path was severed end to end.

    Input: one real question per line (the intent files in the archive: if
    tab-separated, the second column is taken). Roles are not marked — None
    is returned and training writes no loss to the role head on that example:
    questions teach the operation kind; fact sentences teach the roles.
    """
    found = []
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in _open(path):
            text = line.rstrip("\n")
            if "\t" in text:
                text = text.split("\t", 1)[1]
            text = text.strip()
            if not (4 < len(text) <= longest):
                continue
            found.append((text, ASK, None))
            if most and len(found) >= most:
                return found
    return found


def alphabet(rows):
    """The corpus's own letters — declared nowhere, counted."""
    seen = set()
    for text, _, _ in rows:
        seen.update(text)
    return {letter: at + 1 for at, letter in enumerate(sorted(seen))}


def build(fact_paths, dialogue_paths=(), question_paths=(), most_facts=None,
          most_dialogue=None, most_questions=None, seed=7):
    """A mixed, balanced, shuffled training set — all THREE classes at once."""
    rows = from_facts(fact_paths, most=most_facts)
    if dialogue_paths:
        share = most_dialogue or max(1, len(rows) // 3)
        rows += from_dialogue(dialogue_paths, most=share)
    if question_paths:
        share = most_questions or max(1, len(rows) // 4)
        rows += from_questions(question_paths, most=share)
    random.Random(seed).shuffle(rows)
    return rows
