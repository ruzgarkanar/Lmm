"""The words the system knows.

Vocabulary used to be a constant in the source, which meant LMM could only ever
discuss what we had hardcoded. It is knowledge like any other: learned, stored
with the memory, and carried by packs. A domain extends the language by shipping
its verbs alongside its facts — no change to this codebase.

One running LMM has one vocabulary, kept in ACTIVE. Words are only ever added,
never removed, so growth is safe. A process that hosts several memories at once
should give each its own Lexicon instance rather than sharing ACTIVE.
"""

# The starting vocabulary of the controlled world: surface -> (infinitive, is_positive)
CORE_VERBS = {
    "uçar": ("uçmak", True), "uçamaz": ("uçmak", False),
    "yüzer": ("yüzmek", True), "yüzemez": ("yüzmek", False),
    "koşar": ("koşmak", True), "koşamaz": ("koşmak", False),
    "okur": ("okumak", True), "okuyamaz": ("okumak", False),
    "içer": ("içmek", True), "içemez": ("içmek", False),
    "konuşur": ("konuşmak", True), "konuşamaz": ("konuşmak", False),
}


class Lexicon:
    def __init__(self, verbs=None):
        self.verbs = dict(CORE_VERBS if verbs is None else verbs)
        self._forms = {value: surface for surface, value in self.verbs.items()}

    def knows(self, surface):
        return surface in self.verbs

    def reading(self, surface):
        """"uçamaz" -> ("uçmak", False); None when the word is unknown."""
        return self.verbs.get(surface)

    def surface(self, infinitive, positive):
        """"uçmak", False -> "uçamaz"; falls back to the infinitive itself."""
        return self._forms.get((infinitive, positive), infinitive)

    def learn_verb(self, infinitive, positive, negative):
        """Teach one verb in both polarities. Idempotent."""
        for surface, polarity in ((positive, True), (negative, False)):
            self.verbs[surface] = (infinitive, polarity)
            self._forms[(infinitive, polarity)] = surface

    def signature(self):
        """Identity of this vocabulary, for caching things derived from it."""
        return frozenset(self.verbs)


ACTIVE = Lexicon()
