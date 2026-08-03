"""Reasoning Engine: inheritance, exceptions and conflict detection over memory.

Every answer it produces can name the chain it came from, so nothing the system
says is unexplainable.
"""
from lmm.memory import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY, LACKS_PROPERTY,
                        HAS_PART, LACKS_PART, TYPE_RELATIONS, ABILITY_RELATIONS,
                        PROPERTY_RELATIONS, PART_RELATIONS, INHERITING)
from lmm.phrasing import (ability_clause, property_clause, part_clause,
                          is_a_clause, is_not_a_clause, attribution,
                          disputed_note, is_a_step, own_inference)
from lmm.trust import INFERENCE, level


def _clause_for(relation):
    """How a relation reads as a sentence. Language, kept out of the reasoning."""
    if relation in (HAS_PROPERTY, LACKS_PROPERTY):
        return property_clause
    if relation in (HAS_PART, LACKS_PART):
        return part_clause
    return ability_clause


class Reasoning:
    def __init__(self, memory):
        self.memory = memory

    # Bir kavramın DOĞRUDAN türleri arasında bu kadarlık bir kanıt farkı,
    # ikisinin aynı şey hakkında olmadığını söyler. Aynı pay `gate._definition`
    # içinde de var ama orası yalnız TANIM sorusunu koruyordu; kalıtım
    # korumasızdı ve asıl zarar oradan geliyordu.
    SENSE_MARGIN = 0.1

    # İki kayıt ZIT şey söylüyorsa ve kanıt farkı bu kadarsa, biri ötekini
    # yenmiş sayılır. Aynı büyüklükte olması tesadüf değil: `SENSE_MARGIN` bir
    # kavramın iki ANLAMINI ayırıyor, bu ise aynı anlam içinde iki TANIĞI —
    # ikisi de "bu fark bir şey söyler mi" sorusunun cevabı.
    EVIDENCE_MARGIN = 0.1

    def ancestors(self, concept):
        """Type ancestors, nearest first, along whichever relation builds them.

        Zayıf kanıtlı bir tür, güçlüsünün yanında YÜRÜNMÜYOR. Sebebi ölçüldü:
        Vikipedi'deki "Kuş, bir mizah dergisidir" cümlesi grafa
        `kuş --type--> komedi` yazmış (güven 0,6), oysa `kuş --type--> hayvan`
        kayıtlı derlemden geliyor (0,9). Kalıtım ikisini de yürüyünce:

            penguen ataları -> [kuş, hayvan, KOMEDİ, canlı, varlık, DERGİSİ...]
            > penguen bir komedi mi
            evet, penguen bir komedidir.

        Kapının merkez sözü tam burada kırılıyordu ve kapının kendi kusuru
        değildi: graf gerçekten öyle diyordu. Ama "graf öyle diyor" savunması
        ancak grafın SÖYLEDİĞİ tek bir şey varken geçerli; iki anlam bir düğüme
        çökmüşse, ikisini birden yürümek bir şey söylemek değil, iki şeyi
        karıştırmaktır.

        Ölçüldü: 16.774 kavramın 308'i birden çok tür taşıyor ve bunların
        533'ü Vikipedi hasadından geliyor — yani bu tek bir kavramın tuhaflığı
        değil, hasadın sistematik yan ürünü. Ölçek büyüdükçe oran da artıyor
        (%8 @35.234 kavram, ölçüm ayrı raporda).

        Bu bir anlam ayrımı ÇÖZÜMÜ değil, kirliliğin yayılmasını durduran bir
        kapı. Gerçek çözüm düğüm kimliğine anlam eklemek ve o daha büyük bir iş.
        """
        return [name for step in self.lineage(concept) for name in step]

    def _hierarchy_edges(self, concept):
        """Bir kavramın doğrudan türleri, kanıt payı uygulandıktan sonra."""
        edges = self.memory.query(concept, self.memory.kinds.hierarchical())
        if len(edges) > 1:
            strongest = max(edge.confidence for edge in edges)
            edges = [edge for edge in edges
                     if strongest - edge.confidence <= self.SENSE_MARGIN]
        return edges

    def _reach(self, concept):
        """Bir kavramın üstündeki her şey, HİÇBİR süzgeç uygulanmadan.

        Yalnız "şu iki tür birbiriyle ilgili mi" sorusuna bakar; cevap vermez.
        Süzgeçsiz olması kasıtlı: anlam ayrımını kuran testin kendisi anlam
        ayrımına dayanamaz, yoksa sonsuz özyineleme olur. Kenar sayısına bağlı
        önbellek, çünkü bu test grafın büyük olduğu yerde çağrılıyor.
        """
        # Ölçüt `revision`, kenar SAYISI değil. Sayı net-sıfır değişimde
        # yanılıyor: bir kavram unutulup aynı sayıda yeni olgu yazılınca sayı
        # geri aynı olur ve önbellek bayat kalır. Ölçüldü — `forget(kartal)`
        # 11 kenar sildi, 11 yeni olgu yazıldı, sayı 128.485'e döndü ve
        # önbellek silineni hâlâ biliyordu. `revision` her yazmada, `purges`
        # her unutmada artıyor ve ikisi birlikte geri saymıyor.
        marker = (getattr(self.memory, 'revision', 0),
                  getattr(self.memory, 'purges', 0))
        cache = getattr(self, "_reach_cache", None)
        if cache is None or cache[0] != marker:
            cache = (marker, {})
            self._reach_cache = cache
        known = cache[1]
        if concept in known:
            return known[concept]
        hierarchy = self.memory.kinds.hierarchical()
        above, queue = set(), [concept]
        while queue:
            for edge in self.memory.query(queue.pop(0), hierarchy):
                if edge.target not in above and edge.target != concept:
                    above.add(edge.target)
                    queue.append(edge.target)
        known[concept] = above
        return above

    def _related(self, one, other):
        """Bu iki tür aynı şey hakkında olabilir mi.

        Üç yoldan biri yeterli: biri ötekinin atası, ortak bir ataları var, ya
        da aralarında herhangi bir bağ duruyor. Üçü de "hayır" derse aralarında
        grafın kurduğu hiçbir köprü yok demektir.
        """
        mine, theirs = self._reach(one), self._reach(other)
        if other in mine or one in theirs or (mine & theirs):
            return True
        return (any(edge.target == other for edge in self.memory.query(one))
                or any(edge.target == one for edge in self.memory.query(other)))

    def _senses(self, targets):
        """Türleri, birbirine bağlı olanlar bir arada olacak şekilde öbekle."""
        groups = []
        for target in targets:
            joined = [g for g in groups
                      if any(self._related(target, other) for other in g)]
            merged = [target]
            for group in joined:
                merged.extend(group)
                groups.remove(group)
            groups.append(merged)
        return groups

    def lineage(self, concept):
        """Ataları BASAMAK BASAMAK: [[en yakınlar], [bir üsttekiler], ...]

        İKİ ŞEY İÇİN basamaklı: birincisi, aynı derinlikteki iki ata ZIT şey
        söylediğinde hakem gerekiyor ve hakemin kimin aynı uzaklıkta olduğunu
        bilmesi şart (`_resolve`). Düz liste bunu gizliyordu ve cevabı
        `memory.edges` ekleme sırası belirliyordu — aynı bilgi, iki öğretme
        sırası, zıt cevap. Ölçüldü:

            yarasa->memeli, yarasa->uçucu   "yarasa uçar mı" -> hayır
            yarasa->uçucu, yarasa->memeli   "yarasa uçar mı" -> evet

        İkincisi ANLAM AYRIMI. Kanıt payı yalnız kanıt FARKI varken ayırıyor;
        eşit güvenli iki anlam ayrılmıyordu. Ölçüldü: `varlık --type--> dergisi`
        ile `varlık --type--> kaynak`, ikisi de 0,6 — biri edebiyat dergisi,
        öteki felsefi kavram. Kalıtım ikisini birden yürüyordu, `penguen`
        `varlık`a kadar tırmanıp "kuş da bir kaynaktır" diyordu.

        Bir düğümün doğrudan türleri BİRBİRİNE HİÇ BAĞLI DEĞİLSE o düğüm tek
        bir şey değil, aynı ada çökmüş iki şeydir. Kalıtımla oraya VARILDIYSA
        hangi anlamdan girildiği bilinmiyor: ikisini birden yürümek iki şeyi
        karıştırmak, birini seçmek atmak olur. Üçüncü şık dürüst olanı —
        yürüme, orada dur. Sorulan kavramın KENDİSİNDE durmuyoruz; soru zaten
        onun hakkında ve `gate` orada "hangi anlamda" diye soruyor.

        Ölçüldü (16.774 kavram): 308'i çok türlü, 273'ünde türler kanıt payını
        geçecek kadar yakın. Yani bu tek bir kavramın tuhaflığı değil.

        SORULAN kavramda da durmayı denedim ve ölçüp geri aldım: `zürafa`nın
        `taş` anlamı hâlâ yürünüyor, ama kökü de kapatmak sınavı 300 soruda
        %95,7'den %91,3'e ve %96,3'ten %92,3'e düşürdü (isabet değişmedi —
        kaybedilen doğru cevaplar, kazanılan bir dürüstlük değil). Kökte
        seçmemek için bir sebep de yok: soru zaten o kavram hakkında ve
        `gate._definition` orada "hangi anlamda" diye soruyor. Kalıtımla
        VARILAN düğüm başka: oraya hangi anlamdan girildiğini kimse söylemedi.
        """
        steps, current, seen = [], [concept], {concept}
        while current:
            step = []
            for node in current:
                edges = self._hierarchy_edges(node)
                if node != concept and len(edges) > 1:
                    targets = [edge.target for edge in edges]
                    if len(self._senses(targets)) > 1:
                        continue        # iki anlam, hangisinden girildiği belirsiz
                for edge in edges:
                    if edge.target not in seen:
                        seen.add(edge.target)
                        step.append(edge.target)
            if step:
                steps.append(step)
            current = step
        return steps

    def _equivalents(self):
        """Aynı şeyi gösteren adların kümeleri — künyeden okunarak.

        `SAME_AS` künyesi `transitive=True` ve `inverse_of='same_as'` diyor,
        yani bir DENKLİK ilişkisi. İki bayrak da envanterde duruyordu ve motorda
        hiçbir yerden okunmuyordu; ölçüldü:

            otomobil --property--> hızlı
            araba    --same_as---> otomobil
            has_property('araba','hızlı') -> (None, [])

        Yani sistem, kendi kaydettiği eşitliği kullanamıyordu. Bu, "ilişkiler
        veri" iddiasının karşılıksız kaldığı yerdi: yeni bir denklik ilişkisi
        veri olarak bildirilse de hiçbir şey olmuyordu.

        Denklik BURADA da ada bakılarak değil künyeden tanınıyor: geçişli ve
        kendi tersi olan her ilişki bir denkliktir. Yeni bir tane eklenirse bu
        kod değişmeden çalışır.

        Önbellek kenar sayısına bağlı: graf büyürken her sorguda tüm kenarları
        taramak ölçekte kabul edilemez ve tümevarımda tam bu hata ölçülmüştü.
        """
        # Ölçüt `revision`, kenar SAYISI değil. Sayı net-sıfır değişimde
        # yanılıyor: bir kavram unutulup aynı sayıda yeni olgu yazılınca sayı
        # geri aynı olur ve önbellek bayat kalır. Ölçüldü — `forget(kartal)`
        # 11 kenar sildi, 11 yeni olgu yazıldı, sayı 128.485'e döndü ve
        # önbellek silineni hâlâ biliyordu. `revision` her yazmada, `purges`
        # her unutmada artıyor ve ikisi birlikte geri saymıyor.
        marker = (getattr(self.memory, 'revision', 0),
                  getattr(self.memory, 'purges', 0))
        cached = getattr(self, "_alias_cache", None)
        if cached is not None and cached[0] == marker:
            return cached[1]
        names = {name for name, kind in self.memory.kinds.by_name.items()
                 if kind.transitive and kind.inverse_of == name}
        groups = {}
        if names:
            for edge in self.memory.edges:
                if edge.relation in names and edge.target:
                    groups.setdefault(edge.concept, set()).add(edge.target)
                    groups.setdefault(edge.target, set()).add(edge.concept)
        self._alias_cache = (marker, groups)
        return groups

    def aliases(self, concept):
        """Bu kavramla aynı şeyi gösteren diğer adlar, geçişli kapanışıyla."""
        groups = self._equivalents()
        if concept not in groups:
            return []
        found, queue, seen = [], [concept], {concept}
        while queue:
            for other in sorted(groups.get(queue.pop(0), ())):
                if other not in seen:
                    seen.add(other)
                    found.append(other)
                    queue.append(other)
        return found

    # Bir tahminin söylenebilmesi için kardeşlerin bu kadarında bulunması
    # gerekiyor. Eşik ÖLÇÜLDÜ, seçilmedi — 600 saklı kenarda:
    #
    #     ilk 1 önerinin isabeti   %18,6      (tek başına söylenemez)
    #     kardeşlerin %30'unda     %40,9
    #     kardeşlerin %50'sinde    %83,3      <- seçilen
    #     kardeşlerin %70'inde     %80,0      (daha seyrek, daha iyi değil)
    #
    # %83 isabet, "muhtemelen" diyerek söylenebilir bir sayı. Daha düşük eşik
    # daha çok şey söyletir ama söylenenlerin yarısı yanlış olur.
    LIKELY = 0.5
    LEAST_SIBLINGS = 4

    def likely(self, concept, relation, most=3):
        """Kardeşlerinden ÇIKARILAN, ama söylenmemiş nitelikler.

        Graftaki düzenlilik yazılmamış bir kenarı tahmin etmeye yetiyor mu
        diye ölçüldü ve yetiyor: saklanan bir kenar, yalnız yapıya bakılarak
        rastgelenin 2,2 katı doğrulukla bulunuyor (ilk 20'de %38,0 / %17,5).
        Yani graf bir liste değil — yazılmamış olan, yazılanlardan çıkıyor.
        LLM'in "ara değer bulma" yeteneğinin graf karşılığı bu.

        Dönen şey bir OLGU DEĞİL, bir tahmindir ve öyle söylenmeli. Grafa
        yazılamaz: bu gece ölçüldü, yazılan tahmin hafızayı kirletiyor ve
        biriktikçe cevapları bozuyor. Söylenebilir, çünkü gerekçesi var ve
        gerekçe cevapla birlikte veriliyor.
        """
        hierarchy = self.memory.kinds.hierarchical()
        siblings = set()
        for edge in self.memory.query(concept, hierarchy):
            if edge.target:
                siblings.update(other.concept for other
                                in self.memory.incoming(edge.target, hierarchy))
        siblings.discard(concept)
        if len(siblings) < self.LEAST_SIBLINGS:
            return []
        held = {edge.target for edge in self.memory.query(concept, relation)}
        counted = {}
        for name in siblings:
            for edge in self.memory.query(name, relation):
                if edge.target and edge.target not in held:
                    counted[edge.target] = counted.get(edge.target, 0) + 1
        found = [(target, times / len(siblings))
                 for target, times in counted.items()
                 if times / len(siblings) >= self.LIKELY]
        found.sort(key=lambda item: (-item[1], item[0]))
        return found[:most]

    def about(self, concept, relation, target, object=None, role=None):
        """Any relation at all, read from the registry rather than a branch.

        This is what lets a new kind of fact arrive as data: nothing here knows
        what "has" means, only that it denies "has_not" and carries down.
        """
        affirms, denies = self.memory.kinds.pair(relation)
        clause = _clause_for(relation)
        answer, chain, _ = self._resolve(concept, target, affirms, denies,
                                         clause, object, role)
        return answer, chain

    def can_do(self, concept, action, object=None, role=None):
        """Returns (True | False | None, explanation chain).

        None means memory holds nothing on this — the caller must not guess.
        """
        answer, chain, _ = self._resolve(concept, action, CAN, CANNOT,
                                         ability_clause, object, role)
        return answer, chain

    def has_property(self, concept, prop, object=None, role=None):
        """Same shape as can_do, for "kar beyazdır" style knowledge."""
        answer, chain, _ = self._resolve(concept, prop, HAS_PROPERTY,
                                         LACKS_PROPERTY, property_clause,
                                         object, role)
        return answer, chain

    def basis(self, candidate):
        """The edge behind the current belief about a candidate's claim.

        Lets a caller ask *why* it believes something — in particular whether a
        belief came from a teacher or from the system's own generalisation.
        """
        if candidate.relation in ABILITY_RELATIONS:
            return self._resolve(candidate.concept, candidate.target, CAN,
                                 CANNOT, ability_clause, candidate.object,
                                 candidate.role)[2]
        if candidate.relation in PROPERTY_RELATIONS:
            return self._resolve(candidate.concept, candidate.target,
                                 HAS_PROPERTY, LACKS_PROPERTY, property_clause,
                                 candidate.object, candidate.role)[2]
        return self.memory.direct(candidate.concept,
                                  IS_A if candidate.relation == NOT_A else NOT_A,
                                  candidate.target)

    def _resolve(self, concept, target, affirms, denies, clause, object=None,
                 role=None):
        """Direct knowledge first, then the type hierarchy, nearest ancestor first.

        A direct fact always beats an inherited one — that is exactly what makes
        an exception an exception.
        """
        # Both polarities may be on record when sources disagreed. A settled
        # claim outranks one still marked disputed, so arbitration actually
        # changes the answer instead of leaving the loser to speak first. Among
        # equally settled claims the stronger voice speaks: a person correcting
        # a document was being accepted, thanked, and then ignored, because the
        # document's edge simply came first in the list.
        held = [(polarity, self.memory.direct(concept, relation, target, object,
                                              role))
                for polarity, relation in ((False, denies), (True, affirms))]
        held = [(polarity, edge) for polarity, edge in held if edge]
        # Sıradaki üçüncü anahtar GÜVEN. Yoksa aynı basamaktaki iki kayıt
        # arasında hakem kalmıyor ve `memory.edges` sırası konuşuyor; ölçüldü:
        # `grep -n confidence lmm/reasoning.py` yalnız `ancestors`ı gösteriyordu,
        # yani kanıt gücü alanı çıkarım zincirinde hiç okunmuyordu.
        held.sort(key=lambda pair: (pair[1].disputed, -level(pair[1].source),
                                    -pair[1].confidence))
        if held:
            polarity, edge = held[0]
            said = clause(concept, target, polarity, object, role)
            note = disputed_note() if edge.disputed else attribution(edge.source)
            return polarity, [f"{said} ({note})"], edge
        # Denk adlar, ATALARDAN ÖNCE. "araba" ile "otomobil" aynı şeyse
        # otomobil hakkında bilinen doğrudan bilgidir; atadan miras değil.
        # Zincirdeki bağ `=` ile yazılıyor: bu dosyada Türkçe metin biriktirmek
        # mimari ihlali ve `=` her dilde aynı şeyi söylüyor.
        for other in self.aliases(concept):
            for polarity, relation in ((False, denies), (True, affirms)):
                edge = self.memory.direct(other, relation, target, object, role)
                if edge:
                    said = clause(other, target, polarity, object, role)
                    note = (disputed_note() if edge.disputed
                            else attribution(edge.source))
                    return polarity, [f"{concept} = {other}",
                                      f"{said} ({note})"], edge
        for step in self.lineage(concept):
            found = []
            for ancestor in step:
                for polarity, relation in ((False, denies), (True, affirms)):
                    edge = self.memory.direct(ancestor, relation, target,
                                              object, role)
                    if edge and not self.memory.kinds.inherits(relation):
                        continue    # some relations simply do not carry down
                    if edge and edge.quantifier not in INHERITING:
                        # "bazı kuşlar uçmaz" says nothing about this bird.
                        # Letting it inherit would turn an existence claim into
                        # a universal.
                        continue
                    if edge:
                        found.append((polarity, ancestor, edge))
            if not found:
                continue
            # AYNI UZAKLIKTAKİ atalar arasında hakem: önce tartışmasız olan,
            # sonra kaynak sırası, sonra kanıt gücü, en sonda ad. Son anahtar
            # süs değil: onsuz eşitlikte `memory.edges` sırası konuşur ve aynı
            # bilgiden iki farklı cevap çıkar. Bir sistemin öğretme sırasına
            # göre fikir değiştirmesi, uydurmaktan farksızdır.
            found.sort(key=lambda item: (item[2].disputed,
                                         -level(item[2].source),
                                         -item[2].confidence, item[1]))
            polarity, ancestor, edge = found[0]
            rivals = [item for item in found if item[0] is not polarity]
            if rivals and not self._outweighs(edge, rivals[0][2]):
                # Elmas kalıtım: iki ata zıt şey söylüyor ve kanıt ikisini
                # ayırmıyor. Birini seçmek sıraya bakmak olurdu; hangi sırayla
                # öğretilirse öğretilsin cevap aynı: bilmiyorum.
                said = clause(ancestor, target, polarity, object, role)
                other = clause(rivals[0][1], target, rivals[0][0], object, role)
                return None, [is_a_step(concept, ancestor), said,
                              is_a_step(concept, rivals[0][1]), other,
                              disputed_note()], None
            inherited = clause(ancestor, target, polarity, object, role)
            if edge.source == INFERENCE:
                # An inherited guess is still a guess, and must say so.
                inherited = own_inference(inherited)
            return polarity, [is_a_step(concept, ancestor), inherited], edge
        return None, [], None

    def _outweighs(self, edge, rival):
        """Bu kayıt ötekini gerçekten yeniyor mu — sırayla değil, kanıtla.

        Tartışmalı olmamak, daha üst bir kaynaktan gelmek ya da aynı kaynak
        basamağında ölçülebilir bir güven farkı taşımak. Hiçbiri yoksa kimse
        kazanmamıştır ve bunu söylemek, birini seçmekten dürüsttür.
        """
        if edge.disputed != rival.disputed:
            return not edge.disputed
        if level(edge.source) != level(rival.source):
            return level(edge.source) > level(rival.source)
        return edge.confidence - rival.confidence > self.EVIDENCE_MARGIN

    def abilities(self, concept):
        """[(action, True|False)] for every action this memory knows about."""
        return self._known_of(self.memory.actions(), self.can_do, concept)

    def properties(self, concept):
        """[(property, True|False)] for every property this memory knows about."""
        return self._known_of(self.memory.properties(), self.has_property, concept)

    def _known_of(self, targets, lookup, concept):
        found = []
        for target in targets:
            known, _ = lookup(concept, target)
            if known is not None:
                found.append((target, known))
        return found

    def who_can(self, action, positive=True):
        """Every concept known to do (or known not to do) an action."""
        return [c for c in self.memory.concepts()
                if self.can_do(c, action)[0] is positive]

    def survey(self, concept, target, relation_pair):
        """Which members of a kind are known to do this, and which are not.

        "Bazı kuşlar uçmaz" does not have to be stored to be answered: the
        exceptions already on record say it. An existence claim is a reading of
        the memory, not another fact in it.
        """
        lookup = (self.has_property if relation_pair[0] == HAS_PROPERTY
                  else self.can_do)
        yes, no = [], []
        for other in self.memory.concepts():
            if other == concept or concept not in self.ancestors(other):
                continue
            # Inherited counts: a sparrow flies because it is a bird, and the
            # question is about the members, not about who was told what.
            known, _ = lookup(other, target)
            if known is True:
                yes.append(other)
            elif known is False:
                no.append(other)
        return yes, no

    def find_conflict(self, candidate):
        """Explanation string if the candidate edge conflicts with what we know."""
        if candidate.relation in TYPE_RELATIONS:
            return self._type_conflict(candidate)
        affirms, denies = self.memory.kinds.pair(candidate.relation)
        if denies is None:
            return None
        known, chain = self.about(candidate.concept, candidate.relation,
                                  candidate.target, candidate.object,
                                  candidate.role)
        claimed = candidate.relation == affirms
        # İki VAROLUŞSAL iddia çelişmez: "bazı kuşlar uçar" ile "bazı kuşlar
        # uçmaz" mantıkta birlikte doğrudur (I ile O tutarlıdır) ve Türkçede
        # de öyle. Nicelik hesaba katılmadığı için bunlar çelişki sayılıyordu
        # ve öğretmene gereksiz bir onay sorusu soruluyordu.
        #
        # Çelişki karşıtlık karesinde EVRENSEL uçta doğar: "kuşlar uçar" ile
        # "bazı kuşlar uçmaz" gerçekten çelişir. Onun için en az birinin
        # kalıtan (evrensel) bir iddia olması gerekiyor.
        if (candidate.quantifier not in INHERITING
                and self._only_partial(candidate, affirms, denies)):
            return None
        if known is not None and known != claimed:
            return "şu an bildiğim: " + " çünkü ".join(chain)
        return None

    def _only_partial(self, candidate, affirms, denies):
        """Karşı taraftaki kayıtların hepsi varoluşsal mı — yani evrensel yok mu."""
        for relation in (affirms, denies):
            # `quantities()` KENAR döndürüyor, nicelik değil — adı yanıltıcı ve
            # ilk yazışta dizgi sandım: karşılaştırma hiç tutmadı, her
            # varoluşsal iddia çelişkisiz sayıldı ve gerçek çelişki kayboldu.
            for edge in self.memory.quantities(
                    candidate.concept, relation, candidate.target,
                    candidate.object, candidate.role):
                if edge.quantifier in INHERITING:
                    return False
        return True

    def _type_conflict(self, candidate):
        """"penguen bir kuş değildir" against a hierarchy that says it is."""
        opposite = NOT_A if candidate.relation == IS_A else IS_A
        edge = self.memory.direct(candidate.concept, opposite, candidate.target)
        if edge is not None:
            clause = (is_a_clause if opposite == IS_A else is_not_a_clause)
            return (f"şu an bildiğim: {clause(candidate.concept, candidate.target)} "
                    f"(kaynak: {edge.source})")
        if (candidate.relation == NOT_A
                and candidate.target in self.ancestors(candidate.concept)):
            return ("şu an bildiğim: "
                    + is_a_clause(candidate.concept, candidate.target))
        return None
