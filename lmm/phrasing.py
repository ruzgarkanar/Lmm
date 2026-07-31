"""Turkish surface language: the lexicon and every user-facing wording.

All Turkish lives here. The other organs speak in concepts and relations, and
this module is the only place that turns them into sentences — so the internal
labels never leak into what the system says, and a second language would mean
swapping this file alone.
"""
from lmm.memory import IS_A, CAN, CANNOT

# Verb lexicon of the controlled world: surface form -> (infinitive, is_positive)
VERBS = {
    "uçar": ("uçmak", True), "uçamaz": ("uçmak", False),
    "yüzer": ("yüzmek", True), "yüzemez": ("yüzmek", False),
    "koşar": ("koşmak", True), "koşamaz": ("koşmak", False),
    "okur": ("okumak", True), "okuyamaz": ("okumak", False),
    "içer": ("içmek", True), "içemez": ("içmek", False),
    "konuşur": ("konuşmak", True), "konuşamaz": ("konuşmak", False),
}
_SURFACE_FORMS = {value: surface for surface, value in VERBS.items()}

_BACK_UNROUNDED = "aı"
_FRONT_UNROUNDED = "ei"
_BACK_ROUNDED = "ou"
_VOICELESS = "fstkçşhp"


def _copula(word):
    """The -DIr suffix, obeying vowel harmony and consonant assimilation."""
    vowels = [c for c in word if c in "aeıioöuü"]
    last_vowel = vowels[-1] if vowels else "a"
    if last_vowel in _BACK_UNROUNDED:
        harmony = "ı"
    elif last_vowel in _FRONT_UNROUNDED:
        harmony = "i"
    elif last_vowel in _BACK_ROUNDED:
        harmony = "u"
    else:
        harmony = "ü"
    consonant = "t" if word and word[-1] in _VOICELESS else "d"
    return consonant + harmony + "r"


def verb_form(infinitive, positive):
    """"uçmak", True -> "uçar"; "uçmak", False -> "uçamaz"."""
    return _SURFACE_FORMS.get((infinitive, positive), infinitive)


def is_a_clause(concept, target):
    """penguen, kuş -> "penguen bir kuştur"."""
    return f"{concept} bir {target}{_copula(target)}"


def ability_clause(concept, action, positive):
    """penguen, uçmak, False -> "penguen uçamaz"."""
    return f"{concept} {verb_form(action, positive)}"


def describe(concept, relation, target):
    """A fact stated as a Turkish sentence, whatever its relation."""
    if relation == IS_A:
        return is_a_clause(concept, target)
    return ability_clause(concept, target, relation == CAN)
