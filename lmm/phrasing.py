"""Turkish surface language: the lexicon and every user-facing wording.

All Turkish lives here. The other organs speak in concepts and relations, and
this module is the only place that turns them into sentences — so the internal
labels never leak into what the system says, and a second language would mean
swapping this file alone.
"""
from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                           LACKS_PROPERTY, HAS_PART, LACKS_PART, PLACE, SOURCE)
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


def case_form(word, role):
    """Put the case ending back on when speaking: kutup + yer -> "kutupta".

    Two-way harmony for the vowel, voicing for the consonant — the same rules
    the copula follows, because they are the language's rules and not this
    suffix's.
    """
    if role not in (PLACE, SOURCE):
        return word
    vowels = [c for c in word if c in "aeıioöuü"]
    last = vowels[-1] if vowels else "a"
    vowel = "a" if last in "aıou" else "e"
    consonant = "t" if word and word[-1] in _VOICELESS else "d"
    suffix = consonant + vowel + ("n" if role == SOURCE else "")
    return word + suffix


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


def property_clause(concept, prop, positive=True, object=None, role=None):
    """kar, beyaz -> "kar beyazdır"; with a second concept, "kartal serçeden
    büyüktür"."""
    middle = f"{case_form(object, role)} " if object else ""
    if positive:
        return f"{concept} {middle}{prop}{copula(prop)}"
    return f"{concept} {middle}{prop} değildir"


_SOFTENS = {"k": "ğ", "p": "b", "ç": "c", "t": "d"}


def genitive(word):
    """The possessor's ending: kuş -> kuşun, kedi -> kedinin, balık -> balığın.

    Harmony picks the vowel, whether the word ends in one picks the -n-, and a
    final voiceless consonant softens before it.
    """
    if not word:
        return word
    stem = word
    if stem[-1] in _SOFTENS and len(stem) > 2:
        stem = stem[:-1] + _SOFTENS[stem[-1]]
    buffer = "n" if word[-1] in "aeıioöuü" else ""
    return stem + buffer + _harmony_vowel(word) + "n"


def part_clause(concept, part, positive=True, object=None, role=None):
    """kuş, kanadı -> "kuşun kanadı var" / "kuşun kanadı yok"."""
    return f"{genitive(concept)} {part} {'var' if positive else 'yok'}"


def property_question(concept, prop, object=None, role=None):
    """kar, beyaz -> "kar beyaz mı?"; with a comparison, "kuş serçeden büyük mü?"."""
    middle = f"{case_form(object, role)} " if object else ""
    return f"{concept} {middle}{prop} {question_particle(prop)}?"


def part_question(concept, part):
    """kuş, kanadı -> "kuşun kanadı var mı?"."""
    return f"{genitive(concept)} {part} var mı?"


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


def ability_clause(concept, action, positive, object=None, role=None):
    """penguen, uçmak, False -> "penguen uçamaz"; with a second concept,
    "kedi fare yakalar" or "penguen kutupta yaşar" depending on its role."""
    if object:
        return f"{concept} {case_form(object, role)} {verb_form(action, positive)}"
    return f"{concept} {verb_form(action, positive)}"


def definition_question(concept):
    """penguen -> "penguen nedir?"."""
    return f"{concept} nedir?"


def ability_question(concept, action, object=None, role=None):
    """penguen, yüzmek -> "penguen yüzer mi?"; with a second concept,
    "penguen kutupta yaşar mı?"."""
    verb = verb_form(action, True)
    middle = f"{case_form(object, role)} " if object else ""
    return f"{concept} {middle}{verb} {question_particle(verb)}?"


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

# Örnek cümleler GRAFTAN kuruluyor, elle yazılmıyor. Eskiden sabit sekiz
# penguen cümlesi vardı ve sistem 486 kavram bilirken hep onları gösteriyordu —
# kendini olduğundan aptal gösteren bir hata mesajı. Şimdi gerçekten bildiği
# bir kavramla örnek veriyor.
SHAPE_MOULDS = {
    "ASK_DEFINITION": "{} nedir",
    "ASK_ABILITY": "{} {} mı",
    "ASK_PROPERTY": "{} {} mı",
    "ASK_ABILITIES": "{} ne yapabilir",
    "ASK_PROPERTIES": "{} nasıldır",
    "ASK_DESCRIBE": "{} anlat",
    "ASK_WHY": "{} neden {}",
    "TEACH_TYPE": "{} bir {}dır",
}


def _example(kind, memory):
    """Grafın gerçekten bildiği bir kavramla örnek cümle. Yoksa None."""
    mould = SHAPE_MOULDS.get(kind)
    if not mould or memory is None:
        return None
    from lmm.relations import CAN, HAS_PROPERTY, IS_A
    wanted = {"ASK_ABILITY": CAN, "ASK_WHY": CAN,
              "ASK_PROPERTY": HAS_PROPERTY, "TEACH_TYPE": IS_A}.get(kind)
    for edge in memory.edges:
        if wanted and edge.relation != wanted:
            continue
        if mould.count("{}") == 2:
            if not edge.target:
                continue
            # Fiilin MASTARI değil çekimi yazılır: "kuş uçmak mı" değil
            # "kuş uçar mı". Sözlük zaten yüzey biçimini biliyor.
            target = edge.target
            if wanted == CAN and getattr(memory, "lexicon", None):
                target = memory.lexicon.surface(edge.target, True)
            made = mould.format(edge.concept, target)
            return _harmonised(made)
        return mould.format(edge.concept)
    return None


# Soru ekinin ünlüsü son heceye uyar. Örnek cümle üretirken "tüylü mı"
# yazmak, sistemin kendi dilini bilmediğini gösterir.
BACK, FRONT = "aıou", "eiöü"


def _harmonised(sentence):
    """Sondaki soru ekini son ünlüye uydurur."""
    words = sentence.split()
    if len(words) < 2 or words[-1] not in ("mı", "mi", "mu", "mü"):
        return sentence
    vowels = [c for c in words[-2] if c in BACK + FRONT]
    if not vowels:
        return sentence
    last = vowels[-1]
    if last in "aı":
        words[-1] = "mı"
    elif last in "ei":
        words[-1] = "mi"
    elif last in "ou":
        words[-1] = "mu"
    else:
        words[-1] = "mü"
    return " ".join(words)


def known_shapes(memory=None, count=6):
    """Sistemin gerçekten cevaplayabildiği örnek cümleler."""
    found = []
    for kind in SHAPE_MOULDS:
        made = _example(kind, memory)
        if made:
            found.append(f"'{made}'")
        if len(found) >= count:
            break
    if found:
        return ", ".join(found)
    return "'<kavram> nedir', '<kavram> ne yapabilir', '<kavram> nasıldır'"


def not_understood(resembles=None, memory=None):
    """What to say when nothing parsed — with a guess, if there is one."""
    example = _example(resembles, memory) or SHAPE_EXAMPLES.get(resembles)
    if example:
        return (f"bunu anlamadım. '{example}' gibi bir şey mi demek istedin? "
                f"kelimelerinden birini bilmiyor olabilirim.")
    return f"bunu anlamadım. şunları sorabilirsin: {known_shapes(memory)}."


def which_reading(word, readings):
    """Two ways to read a possessor, and no way to choose without being told."""
    return (f"'{word}' iki türlü okunabilir: {listing(list(readings))}. "
            f"Hangisini kastettiğini bilmiyorum — önce onu bana tanıtır mısın?")


def teach_me_the_word(word):
    """Asking for vocabulary the way it asks for anything else it lacks."""
    return (f"'{word}' kelimesini bilmiyorum. şöyle öğretebilirsin: "
            f"kelime: <mastar> = {word} / <olumsuzu>")


def learned_word(infinitive, positive, negative):
    return f"kelimeyi öğrendim: {positive} / {negative} ({infinitive})."


def wondering(question):
    """How the system voices a gap it noticed in itself."""
    return f"bu arada, bunu hiç öğrenmedim: {question}"


# Sistemin "cevap veremedim" dediği hâllerin ortak işareti. Tek bir yerde
# duruyor çünkü iki yer bunu bilmek zorunda: söyleyişi öğrenen organ, bir
# okumanın gerçekten işe yarayıp yaramadığını buradan anlıyor. Bunlar kullanıcı
# metnine bakan anahtar kelimeler değil — KENDİ çıktımızın imzası.
REFUSALS = ("bilmiyorum", "anlamadım", "öğrenmedim", "duymadım",
            "öğrenmem lazım", "öğretir misin")


def is_a_refusal(said):
    """Bu cevap aslında bir cevap mı, yoksa cevap verememe mi?"""
    lowered = (said or "").lower()
    return not said or any(mark in lowered for mark in REFUSALS)


def compared(first, second, shared, only_first, only_second, kinds=None):
    """İki kavramın karşılaştırması: ortak yan, sonra ayrım.

    İlişki adları ham hâlleriyle yazılmıyor — "uçmak (cannot)" bir cevap değil
    bir döküm. İlişki kaydı zaten her ilişkinin okunabilir etiketini taşıyor
    (`lmm/kinds.py`), o kullanılıyor.

    Hiçbir şey bulunamazsa uydurulmuyor: karşılaştırma da bir iddiadır.
    """
    parts = []
    if shared:
        parts.append(f"ikisi de {' ve '.join(shared)} ile ilgili")
    for concept, traits in ((first, only_first), (second, only_second)):
        if traits:
            parts.append(f"{concept}: " + ", ".join(
                _trait(relation, target, kinds) for relation, target in traits))
    if not parts:
        return f"{first} ile {second} arasında kayıtlı bir fark bulamadım."
    return capitalize("; ".join(parts) + ".")


def _trait(relation, target, kinds=None):
    """Bir kaydı okunur hâle getirir: "uçamaz", "hızlı", "tüye sahip"."""
    if relation == "property":
        return target
    label = None
    if kinds is not None:
        kind = kinds.by_name.get(relation)
        label = kind.label if kind else None
    return f"{target} {label}" if label else f"{target} ({relation})"


def certainty(edges, kinds=None):
    """"emin misin" — güven ve kanıtın kendisi.

    Yeni bilgi gerektirmiyor: güven, kaynak ve kaç tanığın söylediği zaten her
    kenarda duruyor. Yalnızca sorulmuyordu.

    Bir LLM bu soruyu cevaplayamaz — ne güveni sayılabilir ne kaynağı vardır.
    Bizim için bedava, çünkü kayıt zaten künyeli.
    """
    if not edges:
        return "henüz bir şey söylemedim ki."
    edge = edges[0]
    witnesses = len(getattr(edge, "sources", None) or [edge.source])
    parts = [f"güvenim %{edge.confidence * 100:.0f}"]
    if witnesses > 1:
        parts.append(f"{witnesses} ayrı kaynak söylüyor")
    else:
        parts.append(f"tek kaynak: {edge.source}")
    if edge.is_exception:
        parts.append("bu bir istisna, kalıtımı bozuyor")
    if getattr(edge, "disputed", False):
        parts.append("tartışmalı — kaynaklar çelişiyor")
    return capitalize(", ".join(parts) + ".")


def whence(edges):
    """"nereden biliyorsun" — künyenin kendisi.

    Bu sorunun cevaplanabilmesi projenin dördüncü şartının somut hâli: her
    olgu nereden geldiğini taşır ve insan gidip bakabilir.
    """
    if not edges:
        return "henüz bir şey söylemedim ki."
    seen = []
    for edge in edges:
        for source in (getattr(edge, "sources", None) or [edge.source]):
            if source not in seen:
                seen.append(source)
    if len(seen) == 1:
        return f"kaynağım: {seen[0]}."
    return capitalize(f"kaynaklarım: {listing(seen)}.")


def no_opinion(concept=None):
    """Görüş sorusu. Anlamadım demekle görüşüm yok demek aynı şey değil.

    Görüşün dayanağı olmaz; dayanaksız cümle kurmayan bir sistemin görüşü de
    olamaz. Bunu söylemek bir eksiklik itirafı değil, mimarinin sonucu.
    """
    if concept:
        return (f"bu bir görüş sorusu ve benim görüşüm yok. {concept} hakkında "
                f"bildiklerimi sorabilirsin.")
    return "bu bir görüş sorusu ve benim görüşüm yok — yalnız bildiklerimi söylerim."


def inventory(memory, count=8):
    """Sistemin gerçekten ne bildiği — sayarak, uydurmadan.

    "ne biliyorsun" sorusunun cevabı bir tanıtım metni değil, bir DÖKÜM
    olmalı. Sistem 486 kavram bilirken "bunu anlamadım" diyordu; şimdi
    saydığını söylüyor.
    """
    import collections
    from lmm.relations import IS_A, SAME_AS
    concepts = [c for c in memory.concepts()]
    kinds = collections.Counter(
        e.target for e in memory.edges if e.relation == IS_A)
    sources = collections.Counter(
        e.source for e in memory.edges if e.relation != SAME_AS)
    if not concepts:
        return "henüz hiçbir şey bilmiyorum. bana bir şey öğretebilirsin."
    parts = [f"{len(memory.edges)} bilgi, {len(concepts)} kavram"]
    if kinds:
        top = ", ".join(f"{t} ({n})" for t, n in kinds.most_common(count // 2))
        parts.append(f"en çok bildiğim türler: {top}")
    if sources:
        where = ", ".join(f"{s} ({n})" for s, n in sources.most_common(3))
        parts.append(f"kaynaklarım: {where}")
    return capitalize(". ".join(parts) + ".")


def nothing_more(concept):
    """Söylenecek bir şey kalmadığında. Uydurmak yerine bitirmek."""
    if not concept:
        return "henüz bir şeyden konuşmadık, önce bir şey sor."
    return f"{concept} hakkında bildiğim başka bir şey yok."


def talked_about(topics):
    """Sohbette nelerden konuşulduğu. Hiçbiri yoksa uydurulmuyor."""
    if not topics:
        return "henüz bir şey konuşmadık."
    if len(topics) == 1:
        return f"{topics[0]} hakkında konuşuyorduk."
    return capitalize(", ".join(topics[:-1]) + f" ve {topics[-1]} "
                      "hakkında konuşuyorduk.")


def learned_wording(sentence):
    """Yeni bir söyleyiş öğrendiğini söylemesi.

    Söylenmesi önemli: sistem sessizce kural biriktirmemeli. Öğrendiği şey
    görünür olmalı ki yanlışsa silinebilsin.
    """
    return f"(bu söyleyişi de öğrendim: \"{sentence}\")"


def went_and_read(source, learned, refused=0):
    """Bilmediğini fark edip gidip okuduğunu söylemesi.

    Künye söyleniyor çünkü asıl mesele o: "öğrendim" demek kolay, nereden
    öğrendiğini gösterebilmek zor. Reddedilen varsa o da söyleniyor — kaynağın
    her dediğini almadığı, sistemin en çok gösterilmesi gereken davranışı.
    """
    if not learned:
        return f"bilmiyordum, {source} sayfasını okudum ama bir şey çıkaramadım."
    said = f"bilmiyordum, {source} sayfasını okudum: {learned} bilgi öğrendim"
    if refused:
        said += f" ({refused} iddiayı almadım)"
    return said + "."


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


def disagreement(statement, others):
    """Two sources of equal standing disagree, and that is what gets said."""
    return (f"kaynaklar anlaşmıyor: {listing(list(others))} tersini söylüyor. "
            f"'{statement}' iddiasını da tartışmalı olarak kaydettim.")


def how_many(concept, target, positive, yes, no, rule):
    """Answer a question about how much of a kind something covers."""
    verb = verb_form(target, positive)
    counted = yes if positive else no
    others = no if positive else yes
    contrast = verb_form(target, not positive)
    if counted and others:
        return f"evet, {listing(counted)} {verb} ama {listing(others)} {contrast}."
    if counted:
        return f"evet, hepsi — {listing(counted)} {verb}."
    if others and rule is not None and rule == positive:
        return (f"kural olarak {concept} {verb}, ama bildiğim "
                f"{listing(others)} {contrast}.")
    if rule is not None and rule == positive:
        return f"bildiğim kadarıyla hepsi — {concept} {verb}."
    if others:
        return f"hayır, bildiğim {listing(others)} " \
               f"{verb_form(target, not positive)}."
    return f"{concept} hakkında bunu bilmiyorum."


def branch_frozen(branch):
    """A branch is refusing too much of what arrives, so it stops growing."""
    return (f"'{branch}' dalında son zamanlarda çok fazla çelişki çıktı, "
            f"o yüzden makine kaynaklarına kapattım. Sen öğretebilirsin; "
            f"gözden geçirip açmak istersen söyle.")


def disputed_note():
    return "kaynaklar bu konuda anlaşmıyor"


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


def predicate(relation, target, object=None, role=None):
    """A clause with the subject left out, the way Turkish drops it.

    "uçar", "tüylüdür" — so a paragraph can say "kuş olduğu için uçar ve
    tüylüdür" instead of repeating the name in every clause.
    """
    if relation == IS_A:
        return f"bir {target}{copula(target)}"
    if relation == NOT_A:
        return f"bir {target} değildir"
    if relation == HAS_PART:
        return f"{target} var"
    if relation == LACKS_PART:
        return f"{target} yok"
    middle = f"{case_form(object, role)} " if object else ""
    if relation == HAS_PROPERTY:
        return f"{middle}{target}{copula(target)}"
    if relation == LACKS_PROPERTY:
        return f"{middle}{target} değildir"
    return f"{middle}{verb_form(target, relation == CAN)}"


# Kendi cümle biçimi olan ilişkiler. Gerisi ilişki kaydının etiketiyle
# söyleniyor.
KNOWN_CLAUSES = (IS_A, NOT_A, HAS_PROPERTY, LACKS_PROPERTY, HAS_PART,
                 LACKS_PART, CAN, CANNOT)


def describe(concept, relation, target, object=None, role=None, kinds=None):
    """A fact stated as a Turkish sentence, whatever its relation.

    Tanınmayan bir ilişki YETENEK sanılıyordu ve sessizce bozuk cümle
    üretiyordu: "kuş kanat gerektirir" bilgisi "öğrendim: kuş kanat" diye
    onaylanıyordu. Oysa ilişki kaydı her ilişkinin okunabilir etiketini zaten
    tutuyor (`lmm/kinds.py`) — projenin "ilişkiler veri, kod değil" iddiasının
    kaçırdığı yer burasıydı. Yeni bir ilişki eklendiğinde artık bu dosya
    değişmiyor.
    """
    if kinds is not None and relation not in KNOWN_CLAUSES:
        kind = kinds.by_name.get(relation)
        if kind is not None:
            return f"{concept} {target} {kind.label}"
    if relation == IS_A:
        return is_a_clause(concept, target)
    if relation == NOT_A:
        return is_not_a_clause(concept, target)
    if relation in (HAS_PROPERTY, LACKS_PROPERTY):
        return property_clause(concept, target, relation == HAS_PROPERTY,
                               object, role)
    if relation in (HAS_PART, LACKS_PART):
        return part_clause(concept, target, relation == HAS_PART)
    return ability_clause(concept, target, relation == CAN, object, role)


def which_meaning(name, readings):
    """Bir ad birden çok şeye işaret ediyorsa, hepsini say ve sor.

    Belirsizlik bir bilgisizlik değil; sistemin iki şeyi birden bilmesidir.
    Birini seçip emin görünmek, bildiğinden azını söylemek olur.
    """
    listed = listing([f"bir {reading}" for reading in readings])
    return (f"{name} birden fazla şeye işaret ediyor: {listed} olabilir. "
            f"hangisini soruyorsun?")
