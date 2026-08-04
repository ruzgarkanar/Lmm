"""Exposition: composing a paragraph out of what is known.

A language model writes fluently by sampling likely words, which is the same
machine that makes it invent. This composes instead: it decides what is worth
saying about a concept, puts it in an order a person would use, and joins it
with the connectives the meaning calls for.

Nobody wrote the paragraph that comes out, so it is generative in the sense that
matters — but every clause in it can be traced to a fact, and the exceptions get
the "ama" they deserve because the reasoning knows they are exceptions.

What it will not do is write a poem. Composition from knowledge is a different
machine from composition from a probability distribution, and this is the one
that cannot make things up.
"""
import collections

from lmm.relations import (IS_A, CAN, CANNOT, HAS_PROPERTY, LACKS_PROPERTY,
                           HAS_PART, LACKS_PART)
from lmm.trust import INFERENCE, STRANGER, level
from lmm import phrasing

OPPOSITES = {CAN: CANNOT, CANNOT: CAN,
             HAS_PROPERTY: LACKS_PROPERTY, LACKS_PROPERTY: HAS_PROPERTY}


# Bir cümleye en çok kaç öge sığar. Fazlası okunmuyor: göz zinciri kaybediyor.
MOST_IN_A_SENTENCE = 6

# Olgu türünden anlatı öbeğine. İlişki adları `lmm/relations.py`'de ve bu
# eşleme oradaki türlerin ANLATIDAKİ karşılığı — hangi cümlede söylenecekler.
GROUPS = {HAS_PROPERTY: "nasıl", LACKS_PROPERTY: "nasıl",
          CAN: "ne yapar", CANNOT: "ne yapar",
          HAS_PART: "nesi var", LACKS_PART: "nesi var"}

# Öbeğin AÇILIŞ sözcükleri artık `lmm/phrasing.py`'de. Burada duruyorlardı ve
# gerekçesi "iskelet burada kuruluyor" idi; ama iskeleti kuran şey öbeklerin
# SIRASI, açılış sözcüğü değil. Sıra burada kaldı, sözcük dile gitti: ikinci bir
# dil aynı sırayı başka sözcüklerle açar.
OTHER = "other"


class Exposition:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning
        self.told_most = None       # kapı kurar; None ise kesme yok

    def describe(self, concept, sense=None, focus_rank=None):
        """Everything worth saying about a concept, as connected prose.

        Bir kavramı anlatmak eskiden grafın tamamını ÜÇ kez tarıyordu — bir
        kez istisnalar, bir kez kendi olguları, bir kez atasının özellikleri
        için. Oysa üçü de tek bir kavramın kenarlarını istiyor ve bellek onu
        zaten indeksliyor. Ölçüldü: 16 bin olguda 0,51 ms, 323 binde 22,05 ms
        — anlatılan kavram büyümediği hâlde.
        """
        # Seçilen anlamın tür kaydı yoksa KALITIM DA susuyor. Tür ve kalıtım
        # aynı kaydın iki yüzü: "sos bir oyundur" susturulup "oyun olduğu için
        # eğlencelidir" bırakılırsa hiçbir şey kazanılmıyor, yalnız kaynağı
        # gizlenmiş oluyor.
        speaks = self._sense_speaks(concept, sense)
        parts = [self._identity(concept, sense), self._exceptions(concept),
                 self._inherited(concept) if speaks else "",
                 self._own(concept, sense, focus_rank)]
        said = [part for part in parts if part]
        if not said:
            return phrasing.dont_know(concept)
        return " ".join(said)

    def _sense_speaks(self, concept, sense):
        """Seçilen anlamın kendi tür kaydı var mı — yoksa soyağacı susmalı."""
        if not sense:
            return True
        edges = self.memory.query(concept, IS_A)
        if not edges:
            return True
        return any(edge.context in (None, sense) for edge in edges)

    def _identity(self, concept, sense=None):
        """Kavramın ne olduğu — SORULAN anlamda.

        Güvene bakmak sabit bir cevap veriyordu: `cumhuriyet` her zaman
        gazete, ne sorulursa sorulsun. Oysa graf yönetim anlamını da biliyor
        ve ikisi ayrı cümleden geldiği için künyeleri de ayrı. Anlam
        seçildiğinde önce o anlamın tür kaydı konuşuyor; o anlamın tür kaydı
        yoksa (yemek anlamındaki `sos` gibi) güvene dönülüyor, yani hiçbir
        bilgi kaybolmuyor.
        """
        edges = self.memory.query(concept, IS_A)
        if not edges:
            return ""
        chosen = [edge for edge in edges if edge.context == sense] if sense else []
        if sense and not chosen:
            # Seçilen anlamın tür kaydı yok. Damgasız kayıt her anlamla
            # uyumlu, o konuşabilir; ama her tür kaydı BAŞKA bir cümleden
            # geliyorsa susmak gerekiyor:
            #
            #     "Kerevizli dip sos nasıl yapılır?" -> "Sos bir OYUNDUR..."
            #
            # O cümle uydurma değil, grafta duruyor. Ama sorulan anlamda
            # doğru değil, ve yanlış anlamda söylenen doğru cümle yanlıştır.
            # Susulduğunda kavramın kendi olguları konuşuyor (`_own` onları
            # seçilen anlama göre zaten öne alıyor), yani cevap kaybolmuyor.
            if not self._sense_speaks(concept, sense):
                return ""
        best = max(chosen or edges, key=lambda e: e.confidence)
        sentence = phrasing.capitalize(phrasing.is_a_clause(concept, best.target))
        ancestors = self.reasoning.ancestors(concept)
        if len(ancestors) > 1:
            sentence += (f", {best.target} {phrasing.clitic_da(best.target)} "
                         f"{phrasing.predicate(IS_A, ancestors[-1])}")
        return sentence + "."

    def _exceptions(self, concept):
        """Where the concept breaks its own family's rule — the interesting part."""
        said = []
        for edge in self.memory.query(concept):
            if not self._contradicts_family(edge):
                continue
            family = self._family_clause(edge)
            own = phrasing.predicate(edge.relation, edge.target, edge.object, edge.role)
            # "ama" bir bağlaçtı ve motorun ortasında duruyordu; istisnayı
            # kuran mantık burada, söyleyişi `lmm/phrasing.py`'de.
            said.append(phrasing.exception_clause(family, concept, own))
        return " ".join(said)

    def _inherited(self, concept):
        """What it gets from what it is, named as coming from there."""
        parent = self._nearest_parent(concept)
        if parent is None:
            return ""
        clauses = []
        for target, relation, obj, role in self._traits_of(parent):
            if self._speaks_for_itself(concept, relation, target):
                continue        # already told, either as its own or as an exception
            clauses.append(phrasing.predicate(relation, target, obj, role))
        if not clauses:
            return ""
        # Kalıtım da bölünüyor. On sekiz öge tek cümlede sayıldığında
        # ("Hayvan olduğu için bir bitki değildir, yer, uyur, solur, ... ve
        # varlık var") okuyan zinciri kaybediyor. Kesme burada da var:
        # atadan gelen, kavramın KENDİ bilgisinden daha az söz hakkı almalı.
        if self.told_most is not None:
            clauses = clauses[:max(MOST_IN_A_SENTENCE, self.told_most // 2)]
        # "... olduğu için ..." ve onu sürdüren "Yine" birer bağlaçtı ve burada
        # yazılıydı. Kaç cümleye bölüneceği burada kalıyor (o bir okunurluk
        # kararı), cümlenin kendisi dile gitti.
        said = [phrasing.because_it_is(
            parent, phrasing.listing(clauses[:MOST_IN_A_SENTENCE]))]
        for at in range(MOST_IN_A_SENTENCE, len(clauses), MOST_IN_A_SENTENCE):
            said.append(phrasing.because_it_is(
                parent, phrasing.listing(clauses[at:at + MOST_IN_A_SENTENCE]),
                again=True))
        return " ".join(said)

    def _own(self, concept, sense=None, focus_rank=None):
        """Facts stated about it directly, minus the exceptions already told.

        Seçilen anlamın olguları öne alınıyor, ötekiler ATILMIYOR: damgasız
        kayıtlar her anlamla uyumlu ve elemek en güvenilir bilgiyi atmak
        olurdu — bu bir kez ölçüldü ve sınavı düşürmüştü.
        """
        edges = list(self.memory.query(concept))
        # GÜVEN sıralaması en altta duruyor: üstüne anlam ve odak sıralaması
        # gelecek ve onlar daha güçlü ölçüt. Ama eşitlikte, daha güvenilir
        # kaynak önce konuşsun — yoksa sırayı kayıt zamanı belirliyor.
        edges.sort(key=lambda edge: -edge.confidence)
        if sense:
            edges.sort(key=lambda edge: (edge.context is not None
                                         and edge.context != sense))
        # Sorunun kelimelerine yakın olgular başa: "dalağın alınması ne zaman
        # ZARARLI olur" ile "dalak nedir" artık aynı dökümü vermesin. Kapı
        # sıralayıcıyı veriyor (`gate._focus_rank`); soru sinyalsizse boş
        # döner ve sıra değişmez.
        if focus_rank is not None:
            close = focus_rank(concept, [edge.target for edge in edges
                                         if edge.target is not None])
            if close:
                edges.sort(key=lambda edge: -close.get(edge.target, 0.0))
        # Olgular TÜRLERİNE göre öbekleniyor ve her öbek ayrı bir cümle
        # kuruyor. Öncesinde hepsi tek bir virgül zinciriydi ve ölçüldü:
        #
        #   "Ayrıca uçar, tüylüdür, sıcakkanlıdır, yumurtlar, öter, kanadı
        #    var, gagası var, tüyü var, dişi yok, yüzer, eskidir, ... ve
        #    kanat var."  (kırk öge, tek cümle)
        #
        # Bir dil modelinin cevabını nesir yapan şey süslü kelimeler değil,
        # BÖLÜNME: ne olduğu ayrı, ne yaptığı ayrı, nesi olduğu ayrı cümlede
        # söyleniyor. Bölecek bilgi bizde zaten var — olgunun ilişkisi.
        # Yeni veri gerekmiyor, yalnız var olanı okumak.
        held = collections.OrderedDict()
        for edge in edges:
            if edge.relation == IS_A or self._contradicts_family(edge):
                continue
            clause = phrasing.predicate(edge.relation, edge.target,
                                        edge.object, edge.role)
            # Künyeler (`(sanırım)`, `(birinin söylediği...)`) dile taşındı:
            # hangi olgunun künye alacağı bir GÜVEN kararı ve burada kalıyor,
            # künyenin nasıl söyleneceği ise söyleyişe ait.
            if edge.source == INFERENCE:
                clause = phrasing.guessed(clause)
            elif level(edge.source) <= STRANGER and not edge.sources[1:]:
                # Bir yabancının, başka kimsenin doğrulamadığı sözü. Atılmıyor
                # — kaynağı yazılı ve kapıdan geçti — ama aynı sesle
                # söylenmiyor. Ölçüldü: "kediler uçar" diyen biri, cevabı
                # "kedi uçar, koşar, tırmanır" hâline getirebiliyordu ve
                # okuyan hangisinin nereden geldiğini göremiyordu.
                clause = phrasing.unconfirmed(clause)
            held.setdefault(GROUPS.get(edge.relation, OTHER),
                            []).append(clause)
        total = sum(len(items) for items in held.values())
        if not total:
            return ""
        # Kesme öbeklerin ÜSTÜNDE: pay her öbeğe büyüklüğüne göre düşüyor,
        # yoksa ilk öbek bütçeyi yiyor ve "ne yapabilir" hiç söylenmiyordu.
        budget = self.told_most if self.told_most is not None else total
        said = []
        for name, items in held.items():
            share = max(1, round(budget * len(items) / total))
            for at in range(0, min(len(items), share), MOST_IN_A_SENTENCE):
                piece = items[at:at + MOST_IN_A_SENTENCE]
                opening = phrasing.group_opening(name if at == 0 else None)
                said.append(f"{opening} {phrasing.listing(piece)}.")
        return " ".join(said)

    def _speaks_for_itself(self, concept, relation, target):
        """Does the concept hold its own view on this — agreeing or not?"""
        opposite = OPPOSITES.get(relation, relation)
        return (self.memory.direct(concept, relation, target) is not None
                or self.memory.direct(concept, opposite, target) is not None)

    def _contradicts_family(self, edge):
        opposite = OPPOSITES.get(edge.relation)
        if opposite is None:
            return False
        return any(self.memory.direct(ancestor, opposite, edge.target)
                   for ancestor in self.reasoning.ancestors(edge.concept))

    def _family_clause(self, edge):
        opposite = OPPOSITES[edge.relation]
        for ancestor in self.reasoning.ancestors(edge.concept):
            if self.memory.direct(ancestor, opposite, edge.target):
                return phrasing.describe(ancestor, opposite, edge.target)
        return ""

    def _nearest_parent(self, concept):
        ancestors = self.reasoning.ancestors(concept)
        return ancestors[0] if ancestors else None

    def _traits_of(self, parent):
        for edge in self.memory.query(parent):
            if edge.relation != IS_A:
                yield edge.target, edge.relation, edge.object, edge.role
