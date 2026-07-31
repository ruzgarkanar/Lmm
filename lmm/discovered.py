"""Morphology the system worked out, standing in for the lists we wrote.

Discovery already finds a language's suffix families from a word list, with
their harmony and their consonant alternation. What it cannot find is what a
suffix is *for*: nothing in the shape of "-lar" says it marks number. That is
knowledge about the world, and it takes one example to convey.

So instead of eight copula endings written out by hand, this is told that
"kuştur" is the copula form of "kuş" — and it finds all eight, in a language it
was never described.

What stays declared is the closed class: question particles, interrogatives,
quantifiers, pronouns. Those are words rather than rules, a language has a few
dozen of them, and no amount of staring at a word list reveals that "bazı" means
some. Discovery is for the machinery of a language, not its vocabulary.
"""
from lmm.morphology import discover

PLURAL = "çoğul"
COPULA = "koşaç"


class DiscoveredMorphology:
    """A morphology whose suffix rules came from the words, not from a list.

    Anything not discovered falls through to the declared one, so a thin word
    list degrades to the hand-written behaviour instead of breaking.
    """

    def __init__(self, declared, words=(), anchors=None, minimum=3):
        self.declared = declared
        self.families = {}
        self.found = discover(list(words), minimum=minimum)
        for role, (stem, inflected) in (anchors or {}).items():
            family = self._family_producing(stem, inflected)
            if family is not None:
                self.families[role] = family

    def _family_producing(self, stem, inflected):
        """The family that turns this stem into this word — that is the anchor."""
        for family in self.found:
            if family.attach(stem) == inflected or family.strip(inflected) == stem:
                return family
        return None

    # --- the morphology interface, discovered where possible ---------------

    def strip_plural(self, word):
        family = self.families.get(PLURAL)
        return family.strip(word) if family else self.declared.strip_plural(word)

    def has_copula(self, word):
        family = self.families.get(COPULA)
        if family is None:
            return self.declared.has_copula(word)
        return family.applies_to(word) is not None

    def strip_copula(self, word):
        family = self.families.get(COPULA)
        return family.strip(word) if family else self.declared.strip_copula(word)

    def attach_copula(self, stem):
        family = self.families.get(COPULA)
        if family is None:
            return stem + self.declared.copula_suffixes[0]
        return family.attach(stem)

    def is_oblique(self, word):
        return self.declared.is_oblique(word)

    def role_of(self, word):
        return self.declared.role_of(word)

    def __getattr__(self, name):
        # Closed-class words and anything else still declared.
        return getattr(self.declared, name)

    @property
    def learned(self):
        """Which roles were worked out rather than taken from the list."""
        return sorted(self.families)


def words_of(memory):
    """Every surface form a memory has seen — the raw material for discovery."""
    seen = []
    for edge in memory.edges:
        for part in (edge.concept, edge.target, edge.object):
            if part:
                seen.extend(part.split())
    for entry in memory.vocabulary:
        seen.extend((entry["positive"], entry["negative"], entry["infinitive"]))
    return seen
