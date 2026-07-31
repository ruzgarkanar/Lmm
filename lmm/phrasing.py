"""Turkish surface language: the lexicon and every user-facing wording.

All Turkish lives here. The other organs speak in concepts and relations, and
this module is the only place that turns them into sentences — so the internal
labels never leak into what the system says, and a second language would mean
swapping this file alone.
"""
from lmm.relations import IS_A, NOT_A, CAN, HAS_PROPERTY, LACKS_PROPERTY

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


def _harmony_vowel(word):
    """The vowel four-way harmony picks for a suffix following this word."""
    vowels = [c for c in word if c in "aeıioöuü"]
    last_vowel = vowels[-1] if vowels else "a"
    if last_vowel in _BACK_UNROUNDED:
        return "ı"
    if last_vowel in _FRONT_UNROUNDED:
        return "i"
    if last_vowel in _BACK_ROUNDED:
        return "u"
    return "ü"


def copula(word):
    """The -DIr suffix, obeying vowel harmony and consonant assimilation."""
    consonant = "t" if word and word[-1] in _VOICELESS else "d"
    return consonant + _harmony_vowel(word) + "r"


def question_particle(word):
    """The mı/mi/mu/mü that follows this word."""
    return "m" + _harmony_vowel(word)


def verb_form(infinitive, positive):
    """"uçmak", True -> "uçar"; "uçmak", False -> "uçamaz"."""
    return _SURFACE_FORMS.get((infinitive, positive), infinitive)


def is_a_clause(concept, target):
    """penguen, kuş -> "penguen bir kuştur"."""
    return f"{concept} bir {target}{copula(target)}"


def is_not_a_clause(concept, target):
    """penguen, memeli -> "penguen bir memeli değildir"."""
    return f"{concept} bir {target} değildir"


def property_clause(concept, prop, positive=True):
    """kar, beyaz -> "kar beyazdır" / "kar beyaz değildir"."""
    if positive:
        return f"{concept} {prop}{copula(prop)}"
    return f"{concept} {prop} değildir"


def property_question(concept, prop):
    """kar, beyaz -> "kar beyaz mı?"."""
    return f"{concept} {prop} {question_particle(prop)}?"


def property_summary(concept, properties):
    """kar, [(beyaz, True), (sıcak, False)] -> "kar beyazdır ama sıcak değildir"."""
    positives = [p for p, is_so in properties if is_so]
    negatives = [p for p, is_so in properties if not is_so]
    parts = []
    if positives:
        parts.append(f"{listing(positives)}{copula(positives[-1])}")
    if negatives:
        parts.append(f"{listing(negatives)} değildir")
    return f"{concept} " + " ama ".join(parts)


def listing(items):
    """["kuş", "serçe", "kartal"] -> "kuş, serçe ve kartal"."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " ve " + items[-1]


def who_clause(concepts, action, positive=True):
    """["kuş", "serçe"], uçmak -> "kuş ve serçe uçar"."""
    return f"{listing(concepts)} {verb_form(action, positive)}"


def ability_summary(concept, abilities):
    """penguen, [(yüzmek, True), (uçmak, False)] -> "penguen yüzer ve uçamaz"."""
    clauses = [verb_form(action, positive) for action, positive in abilities]
    return f"{concept} {listing(clauses)}"


def ability_clause(concept, action, positive):
    """penguen, uçmak, False -> "penguen uçamaz"."""
    return f"{concept} {verb_form(action, positive)}"


def definition_question(concept):
    """penguen -> "penguen nedir?"."""
    return f"{concept} nedir?"


def ability_question(concept, action):
    """penguen, yüzmek -> "penguen yüzer mi?"."""
    verb = verb_form(action, True)
    return f"{concept} {verb} {question_particle(verb)}?"


def wondering(question):
    """How the system voices a gap it noticed in itself."""
    return f"bu arada, bunu hiç öğrenmedim: {question}"


def describe(concept, relation, target):
    """A fact stated as a Turkish sentence, whatever its relation."""
    if relation == IS_A:
        return is_a_clause(concept, target)
    if relation == NOT_A:
        return is_not_a_clause(concept, target)
    if relation in (HAS_PROPERTY, LACKS_PROPERTY):
        return property_clause(concept, target, relation == HAS_PROPERTY)
    return ability_clause(concept, target, relation == CAN)
