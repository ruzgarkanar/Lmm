"""Two readings the whole library shares: `bare` and `fold`.

WHAT THIS FILE USED TO BE. It built the training set for a fine-tuned
reader — span labelling, dialogue and question corpora, an alphabet — and
that reader was retired when the extractor moved to the engine. The
builders went with it on 2026-09-12; what stayed is the pair of text
readings every other module imports, and they stayed HERE so that the
fold a saved graph's keys were written with cannot drift from the fold
that reads them back.
"""

import json
import os
import random
import unicodedata


def layout_char(ch):
    """Is this character LAYOUT rather than part of a word?

    Unicode's own general category answers it (P*, the punctuation classes),
    so the question is asked once, in the place `fold` already lives, rather
    than each caller deciding for itself what a dot is. No character is named
    anywhere. Its reader today is `bare`, below.
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
