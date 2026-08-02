"""Where a fact came from, and how much that counts for.

A memory fed by several kinds of source needs to know which one to believe when
they disagree. A person speaking directly outranks a document they handed over,
which outranks a language model's output, which outranks the system's own
generalisation.

The ranking is what lets correction happen without an argument: a fact from a
higher source simply replaces a belief held on a lower one, and the system says
so. Between equals it does not choose — it asks, because picking a winner
between two sources of equal standing is guessing.
"""

HUMAN = 4
DOCUMENT = 3
DISTILLED = 2       # produced by a language model
INFERRED = 1        # the system worked it out itself

TEACHER = "sen"
INFERENCE = "çıkarım"
DISTILLED_PREFIX = "llm:"

# Her basamağın kendi ŞERİDİ var ve şeritler örtüşmüyor: bir basamaktaki
# kayıtlar ne kadar çoğalırsa çoğalsın bir üsttekinin tabanına ulaşamaz.
# İnsan ile belge eskiden aynı sayıdaydı (0,60) ve sonuç ölçüldü: iki dil
# modeli çıktısı 0,65 ile bir insanı geçiyordu. `arbitrate` sırayı mutlak
# sayarken güven sayısı onu delip geçiyordu — aynı dosyada iki farklı doğru.
CONFIDENCE = {HUMAN: 0.75, DOCUMENT: 0.6, DISTILLED: 0.5, INFERRED: 0.45}
# Bir tanık daha, kalan kuşkunun ne kadarını kapatır. Toplamsal değil, çünkü
# toplamsalken dördüncü belge tavanı dolduruyordu: 0,98 "ulaşılmaz" diye
# yazılmıştı ve dört kaynakta ulaşılıyordu.
CORROBORATION = 0.15
CEILING = 0.98          # certainty is never reached, only approached


def distilled_source(model_name):
    return f"{DISTILLED_PREFIX}{model_name}"


def level(source):
    if source == INFERENCE:
        return INFERRED
    if source.startswith(DISTILLED_PREFIX):
        return DISTILLED
    if source == TEACHER:
        return HUMAN
    return DOCUMENT


def confidence_for(source):
    return CONFIDENCE[level(source)]


def outranks(source, other):
    return level(source) > level(other)


CANDIDATE = "aday"      # the newcomer wins
INCUMBENT = "yerleşik"  # what is held wins
DISPUTED = "tartışmalı"  # neither, and that is recorded rather than hidden


def reputation_of(reputation, source):
    """A source's record: how often it agreed with the rest, how often it did not.

    Truth-discovery research compared a dozen elaborate schemes and found plain
    majority voting almost impossible to beat, at a ninth to a hundredth of the
    cost — several of the clever ones were not even reproducible run to run. So
    this stays a counter, not a model.
    """
    record = reputation.get(source, {})
    agreed = record.get("agreed", 0)
    disputed = record.get("disputed", 0)
    if agreed + disputed == 0:
        return 0.5              # no record yet: neither trusted nor suspected
    return agreed / (agreed + disputed)


def note(reputation, source, agreed):
    record = reputation.setdefault(source, {"agreed": 0, "disputed": 0})
    record["agreed" if agreed else "disputed"] += 1


def arbitrate(incumbent, candidate, reputation=None):
    """Which of two clashing claims to hold, or neither.

    Rank first, because a person outranks a document whatever either has said
    before. Then how many independent sources back each side — that is the
    majority vote the literature keeps finding sufficient. Then each side's
    record. If nothing separates them, Wikidata's answer: keep both and mark the
    disagreement rather than pretending one of them won.
    """
    reputation = reputation if reputation is not None else {}
    if candidate.source in incumbent.sources:
        # One voice cannot dispute itself. A document that says birds fly and
        # that a penguin cannot is incoherent, not two sources disagreeing, and
        # the old answer — refuse and report — is the right one.
        return INCUMBENT
    if outranks(candidate.source, incumbent.source):
        return CANDIDATE
    if outranks(incumbent.source, candidate.source):
        return INCUMBENT
    here, there = len(set(incumbent.sources)), len(set(candidate.sources))
    if here != there:
        return INCUMBENT if here > there else CANDIDATE
    mine = max(reputation_of(reputation, s) for s in incumbent.sources)
    theirs = max(reputation_of(reputation, s) for s in candidate.sources)
    if abs(mine - theirs) > 0.1:
        return INCUMBENT if mine > theirs else CANDIDATE
    return DISPUTED


def ceiling_above(rank):
    """Bu basamağın ulaşamayacağı sınır: bir üstünün tabanı, en tepede tavan.

    Şeritler bu yüzden örtüşmüyor. Sıralamanın "her ne demiş olursa olsun"
    mutlak olması, ancak sayının da onu bozmamasıyla anlam taşıyor.
    """
    above = [CONFIDENCE[other] for other in CONFIDENCE if other > rank]
    return min(above) if above else CEILING


def confidence_from(sources):
    """How sure to be, given everyone who has said it.

    The best source sets the floor and each further *independent* one raises it.
    Hearing the same thing twice from the same place is not corroboration; that
    is how a single mistake becomes a consensus of one.

    Tanıklar tabanı kendi şeridi içinde yukarı taşır ve her yenisi kalanın
    payını alır — yani sınır yaklaşılır, geçilmez. Ölçülen eski hâl: 1 insan
    0,60 / 2 dil modeli 0,65 / 4 belge 0,98. Yeni hâl: 0,75 / 0,52 / 0,66.

    KENDİ ÇIKARIMI TANIK DEĞİL. Bir insanın söylediğinden sistemin kendi
    ürettiği sonuç, o insanı doğrulamaz; ölçüldü, insan+çıkarım 0,60'tan
    0,75'e çıkıyordu. Bir şeyin kendi türevi, kendisine kanıt olamaz.
    """
    if not sources:
        return CONFIDENCE[INFERRED]
    distinct = list(dict.fromkeys(sources))
    witnesses = [s for s in distinct if level(s) != INFERRED] or distinct
    rank = max(level(source) for source in witnesses)
    base = CONFIDENCE[rank]
    room = ceiling_above(rank) - base
    closed = 1 - (1 - CORROBORATION) ** (len(witnesses) - 1)
    return min(CEILING, base + room * closed)
