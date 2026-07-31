"""The words the system knows.

Vocabulary used to be a constant in the source, which meant LMM could only ever
discuss what we had hardcoded. It is knowledge like any other: learned, stored
with the memory, and carried by packs. A domain extends the language by shipping
its verbs alongside its facts — no change to this codebase.

Each memory carries its own vocabulary, because a service holds more than one at
a time and two tenants must never teach each other words. The module keeps a
default for the common case of a single memory, and `use()` swaps in another for
the duration of a request.
"""
import contextvars

# The starting vocabulary of the controlled world: surface -> (infinitive, is_positive)
CORE_VERBS = {
    "uçar": ("uçmak", True), "uçamaz": ("uçmak", False),
    "yüzer": ("yüzmek", True), "yüzemez": ("yüzmek", False),
    "koşar": ("koşmak", True), "koşamaz": ("koşmak", False),
    "okur": ("okumak", True), "okuyamaz": ("okumak", False),
    "içer": ("içmek", True), "içemez": ("içmek", False),
    "konuşur": ("konuşmak", True), "konuşamaz": ("konuşmak", False),
}


ABILITY_SUFFIXES = ("ebilir", "abilir")


class Lexicon:
    def __init__(self, verbs=None):
        self.verbs = dict(CORE_VERBS if verbs is None else verbs)
        self._forms = {value: surface for surface, value in self.verbs.items()}

    def knows(self, surface):
        return self.reading(surface) is not None

    def reading(self, surface):
        """"uçamaz" -> ("uçmak", False); None when the word is unknown.

        Turkish also marks ability with -ebilir/-abilir, and "penguen yüzebilir"
        is how people actually say it. Rather than storing a third form for every
        verb, the suffix is peeled off and matched to a stem we already know.
        """
        direct = self.verbs.get(surface)
        if direct is not None:
            return direct
        for suffix in ABILITY_SUFFIXES:
            if surface.endswith(suffix) and len(surface) > len(suffix) + 1:
                stem = surface[: -len(suffix)].rstrip("y")
                for infinitive, positive in self.verbs.values():
                    if positive and infinitive.startswith(stem):
                        return infinitive, True
        return None

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


CORE = Lexicon()        # the default, for a process holding a single memory
_current = contextvars.ContextVar("lexicon", default=None)


def current():
    """The vocabulary in force right now."""
    return _current.get() or CORE


def use(lexicon):
    """Make a vocabulary current. Returns a token for restoring the previous."""
    return _current.set(lexicon)


def restore(token):
    _current.reset(token)


class _Active:
    """`ACTIVE.verbs` keeps working while meaning "whichever is current"."""

    def __getattr__(self, name):
        return getattr(current(), name)


ACTIVE = _Active()
