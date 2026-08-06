# -*- coding: utf-8 -*-
"""Çekim mekaniği: ünlü uyumu, ünsüz benzeşmesi ve derlem sayımından çekim.

Buradaki hiçbir şey şablon değil: her işlev ya dilin ses kurallarını
(uyum, yumuşama) ya da derlemdeki sayımı uygular. Hazır cümle yok.
"""
from lmm.relations import PLACE, SOURCE
from lmm.lexicon import ACTIVE

_BACK_UNROUNDED = "aı"
_FRONT_UNROUNDED = "ei"
_BACK_ROUNDED = "ou"
_VOICELESS = "fstkçşhp"

_SOFTENS = {"k": "ğ", "p": "b", "ç": "c", "t": "d"}


def _harmony_vowel(word):
    """Bu kelimeden sonra gelen ekin dört yönlü uyum ünlüsü."""
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
    """-DIr eki: ünlü uyumu ve ünsüz benzeşmesiyle."""
    consonant = "t" if word and word[-1] in _VOICELESS else "d"
    return consonant + _harmony_vowel(word) + "r"


def case_form(word, role):
    """Durum ekini geri takar: kutup + yer -> "kutupta"."""
    if role not in (PLACE, SOURCE):
        return word
    vowels = [c for c in word if c in "aeıioöuü"]
    last = vowels[-1] if vowels else "a"
    vowel = "a" if last in "aıou" else "e"
    consonant = "t" if word and word[-1] in _VOICELESS else "d"
    suffix = consonant + vowel + ("n" if role == SOURCE else "")
    return word + suffix


def clitic_da(word):
    """Ayrı yazılan "da"/"de" — iki yönlü uyum."""
    vowels = [c for c in word if c in "aeıioöuü"]
    last = vowels[-1] if vowels else "a"
    return "da" if last in "aıou" else "de"


def question_particle(word):
    """Bu kelimeyi izleyen mı/mi/mu/mü."""
    return "m" + _harmony_vowel(word)


def verb_form(infinitive, positive, lexicon=None):
    """"uçmak", True -> "uçar"; "uçmak", False -> "uçamaz"."""
    return (lexicon or ACTIVE).surface(infinitive, positive)


def genitive(word):
    """İyelik ekleyenin eki: kuş -> kuşun, kedi -> kedinin, balık -> balığın."""
    if not word:
        return word
    stem = word
    if stem[-1] in _SOFTENS and len(stem) > 2:
        stem = stem[:-1] + _SOFTENS[stem[-1]]
    buffer = "n" if word[-1] in "aeıioöuü" else ""
    return stem + buffer + _harmony_vowel(word) + "n"


def possessed(word):
    """Sahip olunanın eki: kanat -> kanadı, kedi -> kedisi, göz -> gözü."""
    if not word:
        return word
    stem = word
    if stem[-1] in _SOFTENS and len(stem) > 2:
        stem = stem[:-1] + _SOFTENS[stem[-1]]
    buffer = "s" if word[-1] in "aeıioöuü" else ""
    return stem + buffer + _harmony_vowel(word)


def aorist(infinitive, positive=True):
    """Mastardan geniş zaman — çekimi DERLEM söyler, liste değil."""
    if " " in infinitive:
        head, _, last = infinitive.rpartition(" ")
        return f"{head} {aorist(last, positive)}"
    stem = infinitive
    for ending in _of("infinitive_suffixes"):
        if stem.endswith(ending) and len(stem) > len(ending) + 1:
            stem = stem[: -len(ending)]
            break
    best = _attested(stem, positive, infinitive)
    if best is not None:
        return best
    if not positive:
        return stem + ("maz" if _harmony_vowel(stem) in "aıou" else "mez")
    if stem and stem[-1] in _letters()[0]:
        return stem + "r"
    return stem + _harmony_vowel(stem) + "r"


_ALPHABET = {}


def _letters():
    """(ünlüler, ünsüzler) — dilin kendi bildirdiği alfabeden. Saklanır."""
    if _ALPHABET:
        return _ALPHABET["held"]
    from lmm.turkish import TurkishMorphology
    vowels = getattr(TurkishMorphology, "vowels", "aeıioöuü")
    alphabet = getattr(TurkishMorphology, "alphabet", None)
    if alphabet is None:
        from lmm import frequency
        seen = set()
        for word in frequency.counts():
            seen.update(letter for letter in word if letter.isalpha())
        alphabet = "".join(sorted(seen))
    _ALPHABET["held"] = (vowels, "".join(c for c in alphabet if c not in vowels))
    return _ALPHABET["held"]


def _attested(stem, positive, infinitive_wanted=None):
    """Derlemin FİİL olarak tanıdığı çekim. Yoksa None."""
    from lmm import frequency
    verbs = frequency.verbs()
    counts = frequency.counts()
    if not verbs or len(stem) < 2:
        return None
    vowels, _ = _letters()

    def tail_fits(surface, root):
        if not surface.startswith(root):
            return False
        tail = surface[len(root):]
        return tail == "r" or (len(tail) == 2 and tail[0] in vowels
                               and tail[1] == "r")

    exact, loose = [], []
    for surface, (name, polarity) in verbs.items():
        if polarity is not positive:
            continue
        if name != infinitive_wanted:
            continue
        if tail_fits(surface, stem):
            exact.append((counts.get(surface, 0), surface))
        elif len(stem) > 2 and tail_fits(surface, stem[:-1] + surface[len(stem) - 1:len(stem)]) \
                and surface[:len(stem) - 1] == stem[:-1]:
            loose.append((counts.get(surface, 0), surface))
    held = exact or loose
    if not held and positive:
        held = [(counts.get(stem + vowel + "r", 0), stem + vowel + "r")
                for vowel in vowels]
        held.append((counts.get(stem + "r", 0), stem + "r"))
    if not held:
        return None
    count, best = max(held)
    return best if count > 0 else None


def _of(name):
    from lmm.turkish import TurkishMorphology
    return getattr(TurkishMorphology, name, ())


def capitalize(text):
    """Türkçe büyütme: "içer" cümle başında "İçer" olur, "Icer" değil."""
    if not text:
        return text
    first = "İ" if text[0] == "i" else text[0].upper()
    return first + text[1:]


def listing(items):
    """["kuş", "serçe", "kartal"] -> "kuş, serçe ve kartal"."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " ve " + items[-1]
