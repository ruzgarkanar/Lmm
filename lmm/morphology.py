"""Discovering a language's suffixes from its words, with nobody describing them.

The suffix lists in turkish.py are the last thing in this system a person wrote
about a particular language, and they are exactly the thing that would have to be
written eighty more times. They do not have to be.

Given enough words, a suffix announces itself: the same ending turns up on many
stems, and the stems it leaves behind are themselves words. Vowel harmony
announces itself too — the variants of one suffix differ only in their vowels,
and which variant appears is decided by the last vowel of the stem. That is a
correlation in a table, not a rule anyone has to state.

What this cannot discover is what a suffix *means*. That a plural marks number
and a copula marks predication is knowledge about the world, and knowledge about
the world belongs in memory where a person can correct it.
"""
VOWELS = "aeıioöuü"


class SuffixFamily:
    """One suffix and the shapes it takes, with the rule that picks between them."""

    def __init__(self, skeleton, harmony, examples, endings=None):
        self.skeleton = skeleton        # the consonants, e.g. "lr" for -lar/-ler
        self.harmony = harmony          # {stem's last vowel: the suffix's tail}
        self.endings = endings or {}    # {stem's last letter: the suffix's head}
        self.examples = examples

    def _head(self, stem):
        if not self.endings:
            return ""
        return self.endings.get(stem[-1], max(set(self.endings.values()),
                                              key=list(self.endings.values()).count))

    @property
    def variants(self):
        heads = set(self.endings.values()) or {""}
        return sorted({head + tail for head in heads
                       for tail in set(self.harmony.values())})

    def applies_to(self, word):
        for suffix in self.variants:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return suffix
        return None

    def strip(self, word):
        suffix = self.applies_to(word)
        return word[: -len(suffix)] if suffix else word

    def attach(self, stem):
        """Build the variant this stem calls for, vowel and consonant alike."""
        tail = self.harmony.get(last_vowel(stem))
        if tail is None:
            tail = max(set(self.harmony.values()),
                       key=list(self.harmony.values()).count)
        return stem + self._head(stem) + tail

    def __repr__(self):
        return f"<ek {'/'.join(self.variants)}>"


def last_vowel(word):
    for character in reversed(word):
        if character in VOWELS:
            return character
    return ""


def _vote(table, key, value):
    table.setdefault(key, {})
    table[key][value] = table[key].get(value, 0) + 1


def _skeleton(suffix):
    return "".join(c for c in suffix if c not in VOWELS)


def discover(words, minimum=3, max_length=4):
    """Suffix families the word list supports, commonest first.

    A candidate is any ending that leaves behind a stem which is itself a word we
    have seen — that is what separates a suffix from a coincidence of spelling.
    """
    known = set(words)
    sightings = {}
    for word in known:
        for length in range(1, max_length + 1):
            if len(word) <= length + 1:
                continue
            stem, suffix = word[:-length], word[-length:]
            if stem not in known:
                continue
            sightings.setdefault(suffix, []).append(stem)

    # The threshold belongs to the family, not to each face of it: "-tür" turns
    # up on one word and "-tır" on two, and neither survives a per-variant count
    # even though the ending they belong to is everywhere.
    families = {}
    for suffix, stems in sightings.items():
        families.setdefault(_skeleton(suffix), []).append((suffix, stems))
    families = _merge_allomorphs(families)
    families = {skeleton: variants for skeleton, variants in families.items()
                if sum(len(stems) for _, stems in variants) >= minimum}

    found = []
    for skeleton, variants in families.items():
        if not skeleton:
            continue
        # Which variant a stem vowel takes is decided by a count, not by
        # whichever was seen first: a language has loanwords that break its own
        # harmony — "kalp" takes the front variant — and one of those was enough
        # to teach the wrong rule for every word with an "a" in it.
        vowel_votes, ending_votes, examples = {}, {}, []
        for suffix, stems in variants:
            # A one-letter suffix has no head to alternate: counting its single
            # character as both head and tail once produced "teachss".
            head, tail = (suffix[0], suffix[1:]) if len(suffix) > 1 else ("", suffix)
            for stem in stems:
                _vote(vowel_votes, last_vowel(stem), tail)
                if head:
                    _vote(ending_votes, stem[-1], head)
                examples.append(stem + suffix)
        harmony = {vowel: max(counts, key=counts.get)
                   for vowel, counts in vowel_votes.items()}
        endings = {letter: max(counts, key=counts.get)
                   for letter, counts in ending_votes.items()}
        found.append(SuffixFamily(skeleton, harmony, examples[:5], endings))
    found.sort(key=lambda family: -len(family.examples))
    return found


def _merge_allomorphs(families, overlap=0.15):
    """Join families that are one suffix wearing two faces.

    "-dır" and "-tır" are the same ending: which one appears is decided by the
    stem, so they are never seen on the same word. "-lar" and "-dır" are not,
    and "kuşlar" and "kuştur" both exist. That is the whole test — forms in
    complementary distribution are variants; forms that share stems are
    different suffixes. Nobody has to say which is which.
    """
    keys = list(families)
    merged, taken = {}, set()
    for key in keys:
        if key in taken:
            continue
        group = list(families[key])
        stems = {stem for _, found in families[key] for stem in found}
        for other in keys:
            if other in taken or other == key or len(other) != len(key):
                continue
            if other[1:] != key[1:]:            # same shape but the first letter
                continue
            theirs = {stem for _, found in families[other] for stem in found}
            shared = len(stems & theirs) / max(1, min(len(stems), len(theirs)))
            if shared <= overlap:
                group.extend(families[other])
                stems |= theirs
                taken.add(other)
        taken.add(key)
        merged[key] = group
    return merged


def harmony_is_consistent(family):
    """Does one stem vowel always pick the same variant? That is harmony."""
    return len(set(family.harmony.values())) > 1 and \
        len(family.harmony) >= len(set(family.harmony.values()))
