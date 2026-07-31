"""Turkish surface language: the lexicon and every user-facing wording.

All Turkish lives here. The other organs speak in concepts and relations, and
this module is the only place that turns them into sentences — so the internal
labels never leak into what the system says, and a second language would mean
swapping this file alone.
"""
from lmm.relations import IS_A, NOT_A, CAN, HAS_PROPERTY, LACKS_PROPERTY
from lmm.lexicon import ACTIVE

from lmm.trust import INFERENCE, DISTILLED_PREFIX

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


def clitic_da(word):
    """The separate "da"/"de", which harmonises two ways, not four."""
    vowels = [c for c in word if c in "aeıioöuü"]
    last = vowels[-1] if vowels else "a"
    return "da" if last in "aıou" else "de"


def question_particle(word):
    """The mı/mi/mu/mü that follows this word."""
    return "m" + _harmony_vowel(word)


def verb_form(infinitive, positive, lexicon=None):
    """"uçmak", True -> "uçar"; "uçmak", False -> "uçamaz"."""
    return (lexicon or ACTIVE).surface(infinitive, positive)


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


# One sample sentence per shape, used to guess at what a teacher meant.
SHAPE_EXAMPLES = {
    "TEACH_TYPE": "penguen bir kuştur",
    "TEACH_NOT_TYPE": "penguen bir memeli değildir",
    "TEACH_ABILITY": "kuşlar uçar",
    "TEACH_PROPERTY": "kuşlar tüylüdür",
    "TEACH_NOT_PROPERTY": "penguen tüylü değildir",
    "ASK_DEFINITION": "penguen nedir",
    "ASK_ABILITY": "penguen uçar mı",
    "ASK_PROPERTY": "penguen tüylü mü",
    "ASK_WHO": "kimler uçar",
    "ASK_ABILITIES": "penguen ne yapabilir",
    "ASK_PROPERTIES": "penguen nasıldır",
    "ASK_WHY": "penguen neden uçamaz",
}

KNOWN_SHAPES = ("'penguen bir kuştur', 'kuşlar uçar', 'kuşlar tüylüdür', "
                "'penguen nedir', 'penguen uçar mı', 'kimler uçar', "
                "'penguen ne yapabilir', 'penguen neden uçamaz'")


def not_understood(resembles=None):
    """What to say when nothing parsed — with a guess, if there is one."""
    example = SHAPE_EXAMPLES.get(resembles)
    if example:
        return (f"bunu anlamadım. '{example}' gibi bir şey mi demek istedin? "
                f"kelimelerinden birini bilmiyor olabilirim.")
    return f"bunu anlamadım. şu kalıpları biliyorum: {KNOWN_SHAPES}."


def teach_me_the_word(word):
    """Asking for vocabulary the way it asks for anything else it lacks."""
    return (f"'{word}' kelimesini bilmiyorum. şöyle öğretebilirsin: "
            f"kelime: <mastar> = {word} / <olumsuzu>")


def learned_word(infinitive, positive, negative):
    return f"kelimeyi öğrendim: {positive} / {negative} ({infinitive})."


def wondering(question):
    """How the system voices a gap it noticed in itself."""
    return f"bu arada, bunu hiç öğrenmedim: {question}"


def attribution(source):
    """Where a fact came from, said plainly."""
    if source == INFERENCE:
        return "kendi çıkarımım"
    if source.startswith(DISTILLED_PREFIX):
        return f"bir dil modelinden, {source[len(DISTILLED_PREFIX):]}"
    return f"doğrudan bilgi, kaynak: {source}"


def _origin(source):
    if source == INFERENCE:
        return "çıkarımla varsaymıştım"
    if source.startswith(DISTILLED_PREFIX):
        return "bir dil modelinden almıştım"
    return f"{source} kaynağından öğrenmiştim"


def generalising(examples, statement):
    """Announcing a pattern it spotted on its own."""
    return f"şunu fark ettim: {listing(examples)} — sanırım {statement}."


def corrected(statement, source):
    """Yielding a weaker source to a stronger one, naming what it gave up."""
    return f"bunu {_origin(source)}, senin sözünü üstün tutuyorum: {statement}."


def computed(expression, value):
    """A result the system produced, with the working as its justification."""
    return f"{expression} = {value} (hesapladım)."


def cannot_compute(reason):
    return f"bunu hesaplayamadım: {reason}."


def did_you_mean(suggestions):
    """The particle harmonises with the last word, like every other suffix."""
    words = list(suggestions)
    return f"yoksa {listing(words)} {question_particle(words[-1])} demek istedin?"


def dont_know(concept, suggestions=()):
    """Not knowing, plus the known words it might have been — as a question.

    The suggestion never becomes an answer: the system still says it does not
    know, and a guess that has to be confirmed cannot turn into a belief.
    """
    plain = f"bilmiyorum. {concept} hakkında bunu bana öğretir misin?"
    if not suggestions:
        return plain
    return f"{plain} {did_you_mean(suggestions)}"


def need_first(question, suggestions=()):
    """It has a goal but no ground to stand on yet."""
    plain = ("bunu bilmiyorum. cevaplayabilmem için önce şunu öğrenmem lazım: "
             f"{question}")
    if not suggestions:
        return plain
    return f"{plain} {did_you_mean(suggestions)}"


def climbing(concept, ancestor, question):
    """It has a goal and knows exactly which fact would unlock it."""
    return f"bunu bilmiyorum ama {concept} bir {ancestor}. {question}"


def now_i_can(answer):
    """Returning to the question that was waiting."""
    return f"şimdi ilk soruna dönebilirim: {answer}"


def capitalize(text):
    """Turkish capitalisation: "içer" starts a sentence as "İçer", not "Icer"."""
    if not text:
        return text
    first = "İ" if text[0] == "i" else text[0].upper()
    return first + text[1:]


def predicate(relation, target):
    """A clause with the subject left out, the way Turkish drops it.

    "uçar", "tüylüdür" — so a paragraph can say "kuş olduğu için uçar ve
    tüylüdür" instead of repeating the name in every clause.
    """
    if relation == IS_A:
        return f"bir {target}{copula(target)}"
    if relation == NOT_A:
        return f"bir {target} değildir"
    if relation == HAS_PROPERTY:
        return f"{target}{copula(target)}"
    if relation == LACKS_PROPERTY:
        return f"{target} değildir"
    return verb_form(target, relation == CAN)


def describe(concept, relation, target):
    """A fact stated as a Turkish sentence, whatever its relation."""
    if relation == IS_A:
        return is_a_clause(concept, target)
    if relation == NOT_A:
        return is_not_a_clause(concept, target)
    if relation in (HAS_PROPERTY, LACKS_PROPERTY):
        return property_clause(concept, target, relation == HAS_PROPERTY)
    return ability_clause(concept, target, relation == CAN)
