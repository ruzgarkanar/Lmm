"""Induction: forming knowledge nobody stated.

Until now the system only knew what it had been told, and chaining given facts
is not the same as learning. An LLM's real advantage is that it generalises —
it answers about things nobody wrote down, because it absorbed patterns.

This does the same thing, in the open. When several children of a type share a
trait and none contradict it, the trait is proposed for the type itself. The
resulting fact is written like any other but sourced to the system rather than a
teacher, kept at a confidence low enough that answers hedge, and beaten by any
human correction without argument.

That is the part an LLM cannot do: it generalises too, but it cannot tell you
that it did, cannot show you the examples that convinced it, and cannot be
corrected on one point without retraining.
"""
import collections

from lmm.memory import (Edge, IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                        LACKS_PROPERTY, INFERRED, INFERRED_CONFIDENCE,
                        INHERITING)

MINIMUM_EXAMPLES = 2
# Bir aile en çok bu kadar üyeyle taranır. Ölçüldü: `bakım bir işlemdir`
# öğretmek 1,0 saniye sürüyordu ve saniyenin 2,9'u burada geçiyordu —
# `işlem` ailesinin 13.322 üyesi, her nitelik için üç kez taranıyor ve
# 892.574 kez `_opinionated` çağrılıyor. Sohbetle beslemeyi imkânsız kılan
# yavaşlık buydu.
#
# Kesmek bir kayıp ve kaydı düşülmeli: on üç bin üyeli bir aile hakkında
# kural önerilmiyor artık. Ama o aileler zaten kural vermiyor — üyeleri
# birbirine benzemiyor, `işlem` altında ne varsa var. Kural, ancak dar ve
# tutarlı bir ailede anlamlı.
LARGEST_FAMILY = 400
MINIMUM_TRAITS = 2      # one shared habit is a coincidence


def succession(agreeing, consulted):
    """Laplace's rule of succession: the odds the rest of the family agrees too.

    After k members are seen to share a trait and none seen to lack it, the
    chance that a further member shares it is (k+1)/(k+2), and the chance that
    *all* of the remaining n-k do is (k+1)/(n+1). Laplace derived it in 1774 for
    exactly this question — how far a run of agreeing observations licenses a
    claim about the ones you have not looked at.

    This replaces a threshold we tuned by hand, and it earns its place because
    it separates the cases we spent a night arguing with: three cold-blooded
    animals out of eight scores 0.44, three running mammals out of three scores
    1.00, and two black birds out of twenty — the very first over-generalisation
    this system ever made — scores 0.14. One formula, no dial.
    """
    return (agreeing + 1) / (consulted + 1)


# A rule has to be likelier than not to hold of the members nobody described,
# with room to spare. Below this the honest move is to keep the examples and
# decline the generalisation.
SUPPORT = 0.75


class Hypothesis:
    # Yerleştirme mi genelleme mi. İkisi ayrı türden çıkarım ve ayrı
    # güvenilirlikte: GENELLEME sayıma dayanıyor ("bu ailenin şu kadar üyesi
    # bunu yapıyor"), YERLEŞTİRME şekle ("bu şey şuna benziyor"). Birincisi
    # ölçülebilir kanıt taşıyor, ikincisi taşımıyor.
    guessed = False

    def __init__(self, concept, relation, target, examples, support=None):
        self.concept = concept
        self.relation = relation
        self.target = target
        self.examples = examples    # the children that suggested it
        self.support = support      # how far the evidence reaches, 0..1

    def as_edge(self):
        # A guess whose evidence reaches further is held more firmly, but never
        # as firmly as something it was actually told.
        confidence = INFERRED_CONFIDENCE
        if self.support is not None:
            confidence = min(INFERRED_CONFIDENCE * self.support * 2,
                             INFERRED_CONFIDENCE)
        return Edge(self.concept, self.relation, self.target,
                    source=INFERRED, confidence=confidence)


class Induction:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning

    def propose(self, about=None):
        """The first generalisation memory supports but nobody has stated.

        Placing a stranger comes before generalising over a family: knowing what
        something *is* unlocks everything its kind already knows, so it is worth
        more than one more rule about a type we have already placed.
        """
        # YERLEŞTİRME artık önerilmiyor. Ölçüldü ve öncülü bu veride
        # geçersiz: "aynı davranışı paylaşıyor, öyleyse aynı şeydir" ancak
        # davranış o türü TANIMLIYORSA geçerli. Vikipedi'nin konum olguları
        # hiçbir şey tanımlamıyor ve üç ayrı sıkılaştırmadan sonra bile
        # üretilen her öneri saçmaydı:
        #
        #   > zurnabalık bir kuştur
        #   öğrendim ... şunu fark ettim: ankara ve beypazarı —
        #   sanırım MENÇELER BİR KAPLANDIR
        #
        # Yazmayı zaten durdurmuştuk; söylemek de gürültü. Mekanizma duruyor
        # (`placements()` çağrılabilir) ama sohbete kendiliğinden girmiyor:
        # ortak nitelik taşıyan bir derlemde yeniden açılabilir.
        for hypothesis in self._candidates(about):
            return hypothesis
        return None

    # Bir niteliğin kavramların bu kadarından fazlasında bulunması, o niteliğin
    # hiçbir şey ayırt etmediği anlamına gelir. Oran seçildi, sayı değil: graf
    # büyüdükçe mutlak sayı anlamını yitirir, oran yitirmez.
    TELLING = 0.005

    def _holders(self):
        """Her niteliği HANGİ kavramların taşıdığı — kenar sayısına bağlı önbellek.

        Anahtar `_behaviour` ile BİREBİR aynı dörtlü olmalı. İlk yazışta ikili
        kullandım ve her arama 0 dönüp denetim sessizce geçti — kapı kurulmuş
        ama kapanmamış oluyordu.
        """
        marker = (getattr(self.memory, 'revision', 0),
                  getattr(self.memory, 'purges', 0))
        cached = getattr(self, "_holders_cache", None)
        if cached is not None and cached[0] == marker:
            return cached[1]
        held = {}
        for edge in self.memory.edges:
            if edge.target and edge.source != INFERRED:
                key = (edge.relation, edge.target, edge.object, edge.role)
                held.setdefault(key, set()).add(edge.concept)
        self._holders_cache = (marker, held)
        return held

    def placements(self):
        """Concepts with no place, put where their behaviour says they belong.

        A stranger that flies and has feathers is probably a bird. Nobody said
        so, and it may be wrong — a bat behaves like that too — so the guess is
        written as the system's own and yields to the first person who corrects
        it. What it buys is everything else birds know.
        """
        placed = []
        for concept in self.memory.concepts():
            if self.memory.query(concept, IS_A):
                continue                    # already knows what it is
            traits = self._behaviour(concept)
            if len(traits) < MINIMUM_TRAITS:
                continue                    # too little behaviour to go on
            fitting = []
            for category in self._categories():
                if category == concept:
                    continue
                shared = traits & self._behaviour(category)
                if len(shared) < MINIMUM_TRAITS or shared != traits:
                    continue
                # Ortak niteliklerin BİLGİ TAŞIMASI şart. Ölçüldü: 128 binlik
                # grafta `bağlı` 3.724 kavramda (%6,4), `ilçe` 1.870'te — ve
                # tam bu ikisi yüzünden her öğretme turunda grafa uydurma bir
                # olgu yazılıyordu:
                #
                #   > glorp bir kuştur
                #   öğrendim ... sanırım ÜÇOBALAR BİR KILIÇTIR
                #   yazılan: üçobalar --type--> kılıç  kaynak=çıkarım
                #
                # Öğretilen şeyle hiç ilgisi yok: `propose` 58 bin kavramı
                # tarayıp graftaki İLK sahipsizi döndürüyor ve o da hep aynı
                # köy adı. Oran %100 — her tur bir saçmalık.
                #
                # Ölçüt projenin kendi ilkesi (`gate._telling` ile aynı): bir
                # olgunun bilgi değeri, onu kaç kavramın paylaştığıyla ters
                # orantılı. Herkeste olan, kimse hakkında bilgi değildir.
                # Gerçek ayırt ediciler karşılaştırma için: `uçmak` 12
                # kavramda, `tüylü` 7, `hızlı` 72.
                # Sorulan şey "en seyrek nitelik ne kadar seyrek" DEĞİL, "bu
                # niteliklerin HEPSİNE birden kaç kavram sahip". İlk yazışta
                # en seyreğe baktım ve yetmedi: `bağlı` 102 kavramda, tavanın
                # altında, ama `bağlı`+`ilçe` ikilisi binlerce köy adında ve
                # ayırt ettiği hiçbir şey yok.
                fitting.append((category, shared))
                if len(fitting) > 1:
                    break           # birden çok raf uyuyor: seçmek uydurmaktır
            # BİR raf uyuyorsa yerleştirme bir çıkarımdır; İKİ raf uyuyorsa
            # seçim keyfîdir ve keyfî seçim uydurmadır. Kapı bunu tanım
            # sorusunda zaten söylüyor ("birden çok karşılık varsa seçmek
            # uydurmaktır") — burada söylenmiyordu ve `break` ilk uyanı
            # alıyordu.
            #
            # Ölçüldü: 128 binlik grafta her öğretme turunda grafa uydurma bir
            # olgu yazılıyordu — `glorp bir kuştur` deyince `üçobalar bir
            # kılıçtır`. Öğretilenle ilgisi yok; `bağlı`+`ilçe` davranışını 96
            # kavram paylaşıyor ve onlarca raf eşit derecede uyuyor.
            if len(fitting) != 1:
                continue
            category, shared = fitting[0]
            # Ayrıca ortak davranış BİLGİ TAŞIMALI: onu paylaşan kalabalıksa
            # ayırt ettiği bir şey yok. Ölçüt projenin kendi ilkesi
            # (`gate._telling` ile aynı).
            held = self._holders()
            together = None
            for trait in shared:
                owners = held.get(trait, set())
                together = owners if together is None else together & owners
            ceiling = max(2, int(len(self.memory.concepts()) * self.TELLING))
            if len(together or ()) > ceiling:
                continue
            placed.append(Hypothesis(concept, IS_A, category,
                                     sorted(t[1] for t in shared)))
        return placed

    def _categories(self):
        """Shelves: concepts something is already said to be, tightest first.

        Any concept with two habits in common was too loose a bar. Fire and the
        sun are both hot and both warm things, so a stranger that behaved like
        either was filed under the other — "sanırım ateş bir güneştir". A shelf
        has to be a category, and what makes a concept a category is that
        something already belongs to it.
        """
        kinds = self.memory.kinds.hierarchical()
        shelves = {edge.target for edge in self.memory.edges
                   if edge.relation == kinds}
        ranked = [(len(self._behaviour(shelf)), shelf) for shelf in shelves
                  if len(self._behaviour(shelf)) >= MINIMUM_TRAITS]
        ranked.sort()           # the tightest fit is the most informative one
        return [shelf for _, shelf in ranked]

    def _behaviour(self, concept):
        """What a concept is known to do — evidence only, never guesses.

        A guess must not become the ground for the next guess. If the system's
        own inferences counted as evidence, one wrong placement would breed a
        rule, the rule would breed another placement, and the chain would look
        exactly as confident as anything it was actually told. Inference reads
        from what it was given; it never reads from itself.
        """
        return {(edge.relation, edge.target, edge.object, edge.role)
                for edge in self.memory.query(concept)
                if edge.relation not in (IS_A, NOT_A)
                and edge.source != INFERRED}

    def still_open(self, hypothesis):
        """Whether a proposal is still unsettled — an earlier one may have closed it."""
        if hypothesis.relation == IS_A:
            return not self.memory.query(hypothesis.concept, IS_A)
        lookup = (self.reasoning.has_property
                  if hypothesis.relation in (HAS_PROPERTY, LACKS_PROPERTY)
                  else self.reasoning.can_do)
        return lookup(hypothesis.concept, hypothesis.target)[0] is None

    def learn(self, hypothesis):
        return self.memory.write(hypothesis.as_edge())

    def candidates(self):
        """Every generalisation the memory supports right now, in one pass."""
        return list(self._candidates())

    def _candidates(self, about=None):
        # Aileler TEK geçişte kuruluyor. Önce her kavram için tüm kenarlar
        # taranıyordu ve bu karesel: 16.774 kavram x 16.164 kenar = 271 milyon
        # işlem. Ölçüldü: bir öğretme turu 5,90 saniye, bir soru turu 0,03 —
        # 197 kat. Üstelik bu geçiş üretim grafında SIFIR öneri veriyor, yani
        # bedel karşılıksız.
        #
        # Bu, ölçek iddiasının tam kalbinde duran cinsten bir hata: 16 milyon
        # olguda aynı döngü dakikalarca sürer. Algoritma değişmiyor, yalnız
        # aynı bilgi bir kez toplanıyor.
        families = {}
        for edge in self.memory.edges:
            if edge.relation == IS_A and edge.target:
                families.setdefault(edge.target, []).append(edge.concept)
        # `about` verilirse yalnız O KAVRAMIN aileleri taranıyor. Tam tarama
        # 128 binlik grafta ilk adayı bulmak için 19,2 saniye sürüyor ve her
        # öğretme turunda çalışıyordu — yerleştirme kısa devre yaptığı için
        # şimdiye kadar görünmemişti.
        #
        # Daraltma hız için değil DOĞRULUK için de: "kartal bir kuştur"
        # dendikten sonra kuşlar hakkında bir kural aramak anlamlı, grafın
        # öbür ucundaki bir aile hakkında aramak değil. Bu, bugün ölçülen
        # kusurun aynısıydı — `propose` öğretilenle ilgisiz şeyler öneriyordu.
        if about:
            near = set(self.memory.query(about, IS_A) and
                       [edge.target for edge in self.memory.query(about, IS_A)])
            near |= set(self.reasoning.ancestors(about))
            families = {name: kids for name, kids in families.items()
                        if name in near}
        for parent, children in families.items():
            if not MINIMUM_EXAMPLES <= len(children) <= LARGEST_FAMILY:
                continue
            for target, affirms, denies, lookup in self._traits(children):
                if lookup(parent, target)[0] is not None:
                    continue                    # the type is already settled
                # Only what the system was told counts as evidence, never what
                # it worked out itself.
                agree = [c for c in children
                         if self._stated(c, target, affirms)]
                disagree = [c for c in children
                            if self._stated(c, target, denies)]
                # "Most of the family" has to mean most of the family that
                # has an opinion. Counting silent children as dissent kills good
                # rules — three mammals run and the other twelve were never
                # described. The opinionated ones are the ones that were
                # actually consulted, so they are the denominator, and how far
                # their agreement reaches is arithmetic rather than a dial.
                #
                # Note this cannot be done by spotting opposites instead:
                # measured over 1288 facts, only 5% of trait pairs ever co-occur
                # and "sıcak" and "soğuk" are among the pairs that do.
                # Complementary distribution finds suffix families; it does not
                # find antonyms.
                consulted = [c for c in children if self._opinionated(c, affirms)]
                for holds, relation in ((agree, affirms), (disagree, denies)):
                    other = disagree if holds is agree else agree
                    if other or len(holds) < MINIMUM_EXAMPLES:
                        continue
                    support = succession(len(holds), len(consulted))
                    if support >= SUPPORT:
                        yield Hypothesis(parent, relation, target, holds, support)

    def _opinionated(self, concept, relation):
        """Whether this child was ever described in this respect at all."""
        return any(edge.source != INFERRED and edge.quantifier in INHERITING
                   for edge in self.memory.query(concept)
                   if edge.relation in self.memory.kinds.pair(relation))

    def _stated(self, concept, target, relation):
        """Evidence for a rule about everyone must itself be about everyone.

        "bazı kuşlar yüzer" was being counted towards "hayvan yüzer", which is
        the existence claim leaking into a universal by the back door — the very
        thing the quantifier was added to stop.
        """
        edge = self.memory.direct(concept, relation, target)
        return (edge is not None and edge.source != INFERRED
                and edge.quantifier in INHERITING)

    def _children(self, parent):
        return [edge.concept for edge in self.memory.edges
                if edge.relation == IS_A and edge.target == parent]

    def _traits(self, among=None):
        """Denenecek nitelikler. `among` verilirse yalnız o ailede GEÇENLER.

        Önceden grafın bildiği her nitelik her aile için deneniyordu. Bir
        ailede hiç kimsede olmayan nitelik kural veremez — `agree` boş çıkar
        ve `MINIMUM_EXAMPLES` eşiğini geçemez — ama denenmesi bedava değil:
        ölçüldü, 2.231 nitelik x 400 üye = 892.574 tarama, ve bir cümle
        öğretmek 1,0 saniye sürüyordu.

        Daraltma sonucu değiştirmiyor, yalnız imkânsızı denemeyi bırakıyor.
        """
        if among is None:
            for action in self.memory.actions():
                yield action, CAN, CANNOT, self.reasoning.can_do
            for prop in self.memory.properties():
                yield prop, HAS_PROPERTY, LACKS_PROPERTY, self.reasoning.has_property
            return
        actions, properties = set(), set()
        for child in among:
            for edge in self.memory.query(child):
                if edge.target is None:
                    continue
                if edge.relation in (CAN, CANNOT):
                    actions.add(edge.target)
                elif edge.relation in (HAS_PROPERTY, LACKS_PROPERTY):
                    properties.add(edge.target)
        for action in actions:
            yield action, CAN, CANNOT, self.reasoning.can_do
        for prop in properties:
            yield prop, HAS_PROPERTY, LACKS_PROPERTY, self.reasoning.has_property
