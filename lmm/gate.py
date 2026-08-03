"""Epistemic Gate: answers are produced from memory alone.

There is no path from this class to a sentence that memory does not support, so
hallucination is not filtered out — it is unreachable.
"""
from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                           HAS_PART, LACKS_PART, PLACE, REQUIRES, ALL)
from lmm.phrasing import (related_instead, likely_traits,
                          is_a_clause, is_not_a_clause, denied, ability_clause,
                          property_clause,
                          dont_know, how_many, which_meaning, compared,
                          because, because_chain, actually, affirmed,
                          inherited_clause, definition_answer, requirements,
                          unknown_requirement, more_so, both_but_unranked,
                          only_one_has, no_comparison, never_learned_properties,
                          properties_answer, never_learned_abilities,
                          abilities_answer, never_heard_action, nobody_does,
                          who_answer, never_heard_property, nobody_is,
                          who_is_answer)
from lmm.similarity import nearest
from lmm.intuition import (ASK_WHO, ASK_ABILITIES, ASK_WHY, ASK_PROPERTIES,
                           ASK_DESCRIBE, ASK_HOW_MANY, ASK_WHERE,
                           ASK_COMPARE, ASK_WHICH_MORE,
                           ASK_REQUIREMENT)
from lmm.exposition import Exposition

import collections

HEDGE_THRESHOLD = 0.5
# Bir listede en çok kaç şey söylenir. Fazlası cevap değil döküm oluyor ve
# okuyan kaybediyor. Kalanı "başka ne biliyorsun" ile alınabilir.
#
# SABİT DEĞİL, VARSAYILAN. Sabit olduğu sürece cevabın uzunluğu soruya
# bakmıyordu: "kalp nedir" ile "kalp anlat" aynı genişlikte çıkıyor, "uzun
# anlat" dense bile değişmiyordu. Ölçüldü ve bedeli de görüldü — grafa 686
# doğru olgu eklendiğinde sınav %97,5'ten %95,8'e düştü, çünkü doğru cevap
# beş kişilik listeye giremedi. Sekize çıkarınca eski seviyeye döndü, yani
# kayıp bilgide değil kesmedeydi.
# Sekiz, beş değil: ölçüldü. Beşteyken grafa doğru olgu eklemek sınavı
# DÜŞÜRÜYORDU (%97,5 -> %95,8), çünkü doğru cevap listeye giremiyordu —
# bilgi arttıkça cevabın kötüleşmesi, kesmenin yanlış yerde olduğunun kanıtı.
# Sekizde eski seviye geri geliyor ve on ikide bir şey değişmiyor, yani
# doğru yer burası: döküme kaçmadan önceki en geniş nokta.
MOST_TOLD = 8
# Anlatma isteyen soru daha geniş: "anlat" diyen döküm değil ANLATI istiyor.
# Dokuz denendi ve ölçüm "sabit" dedi: "nedir" 114 karakter, "anlat" 130 —
# 1,3 kat. Aynı ölçümde dil modeli 4,8 kat açılıyor (278 -> 1348). Bir cevabın
# uzunluğu soruya bakmıyorsa sistem soruyu değil kendini konuşuyor.
DESCRIBING = 24
# Yönergeyle istenen uzunluklar. LLM'de bunu istem yapıyor; burada oturum
# hatırlıyor ve her cevaba uygulanıyor.
TOLD_BY_LENGTH = {"short": 3, "long": 30}
# Bir anlam öbeğinin sorunun kelimelerine ne kadar yakın durması, güvenin
# sabit seçimini bozmaya yetsin. Eşik olmadan gürültü seçiyor: her öbekte bir
# kelime bir kelimeye biraz benzer ve en yüksek gürültü kazanıyor.
SENSE_MARGIN = 0.35
# Bir olgunun, sorunun kelimelerine "yakın" sayılması için eşik. Eşiğin altı
# gürültü: her kelime her kelimeye biraz benzer ve en yüksek gürültüyü öne
# almak sıralamayı bozar.
FOCUS_MARGIN = 0.35
# Karşılaştırmada kaç ortak ata söylenir. İkiden fazlası hiyerarşinin
# tepesine tırmanıyor ("varlık", "şey") ve orada her şey ortaktır.
CLOSEST_SHARED = 2
# Bu kadar yakın iki karşılık arasında seçim yapmak, bilmediğini uydurmaktır.
AMBIGUITY_MARGIN = 0.1


class EpistemicGate:
    def __init__(self, memory, reasoning):
        self.memory = memory
        self.reasoning = reasoning
        self.exposition = Exposition(memory, reasoning)
        # O ANKİ CÜMLENİN kelimeleri. Anlam seçimi buna bakıyor; boşken kapı
        # eski davranışına düşüyor, yani bir oturum bunu hiç kurmasa da
        # çalışır. Kolaylık, bağımlılık değil.
        self.focus_words = ()
        # Kaç şey söylenecek. Oturum yönergeden kurar; kurulmazsa varsayılan.
        self.told_most = MOST_TOLD

    def answer(self, intent):
        if intent.kind == ASK_WHO:
            return self._who(intent.target, intent.relation == CAN,
                             intent.relation)
        if intent.kind == ASK_ABILITIES:
            return self._abilities(intent.concept)
        if intent.kind == ASK_WHY:
            return self._why(intent)
        if intent.kind == ASK_DESCRIBE:
            # ANLATMA sorusu daha geniş: "anlat" diyen döküm değil anlatı
            # istiyor. Yönerge verilmişse o kazanır — kullanıcının söylediği,
            # sistemin varsayımını her zaman yener.
            # ANLAT geniş konuşur — ama istenen kısaysa istenen kazanır.
            # Kullanıcının söylediği, sistemin varsayımını her zaman yener.
            self.exposition.told_most = (max(self.told_most, DESCRIBING)
                                         if self.told_most >= MOST_TOLD
                                         else self.told_most)
            return self.exposition.describe(
                intent.concept, self._sense(intent.concept, self.focus_words),
                self._focus_rank)
        if intent.kind == ASK_COMPARE:
            return self._compare(intent.concept, intent.target)
        if intent.kind == ASK_REQUIREMENT:
            return self._requirement(intent.concept, intent.target)
        if intent.kind == ASK_WHICH_MORE:
            return self._which_more(intent.concept, intent.object,
                                    intent.target)
        if intent.kind == ASK_WHERE:
            return self._where(intent)
        if intent.kind == ASK_HOW_MANY:
            return self._how_many(intent)
        if intent.kind == ASK_PROPERTIES:
            return self._all_properties(intent.concept)
        if intent.relation == IS_A:
            if intent.target:
                return self._is_a(intent.concept, intent.target)
            return self._definition(intent.concept)
        if intent.relation == CAN:
            return self._ability(intent.concept, intent.target, intent.object,
                                 intent.role)
        if intent.relation in (HAS_PART, LACKS_PART):
            return self._about(intent)
        if intent.relation == HAS_PROPERTY:
            return self._property(intent.concept, intent.target, intent.object,
                                  intent.role)
        return self._dont_know(intent.concept)

    def _compare(self, first, second):
        """İki kavramı karşılaştırır: ortak yanları ve ayrıldıkları yer.

        Yayılım organı (`lmm/spreading.py`) bunun için kurulmuştu — bir
        kavramdan komşularına dağılan ilgi, iki kavramın ortak atalarını
        düz aramanın bulamayacağı yerde buluyor. Ama hiçbir yerden
        çağrılmıyordu; kurulmuş, test edilmiş ve bağlanmamış duruyordu.

        Ayrım, ortak atadan gelmeyen kayıtlar: "kuş uçar ama penguen uçamaz"
        cümlesindeki asıl bilgi budur. Hiçbiri bulunamazsa uydurulmuyor.
        """
        from lmm import spreading
        if not first or not second:
            return self._dont_know(first or second)
        for concept in (first, second):
            if not self.memory.query(concept):
                return self._dont_know(concept)
        # Ortak ATA önce gelir. Yayılım köprüsü doğru çalışıyor ama "ortak yan"
        # için fazla gevşek: kartal ve penguen ikisi de uçmakla ilişkili, uçak
        # da öyle — köprü `uçak` diyordu. Oysa iki kuşun ortak yanı uçak değil,
        # KUŞ olmaları. Hiyerarşi, komşuluktan güçlü kanıttır.
        #
        # Yayılım yalnız ortak ata YOKSA devreye giriyor: o zaman elde başka
        # bir bağ yok demektir ve zayıf bir bağ, hiç bağ olmamasından iyidir.
        # Ortak atalar YAKINDAN uzağa sıralı geliyor ve ilk olanlar en
        # özgül olanlar. "kuş" ile "hayvan" anlamlı; listenin dibindeki
        # "komedi" ise başka bir anlamdan sızıyor — *Penguen* aynı zamanda bir
        # mizah dergisi ve graf iki anlamı ayırmıyor.
        #
        # Çok anlamlılık burada çözülmüyor (grafın kendi eksiği) ama en yakın
        # ataları almak, uzaktaki yanlış anlamı büyük ölçüde eliyor.
        mine = self.reasoning.ancestors(first)
        theirs = set(self.reasoning.ancestors(second))
        shared = [step for step in mine if step in theirs][:CLOSEST_SHARED]
        if not shared:
            links = spreading.neighbours(self.memory)
            shared = spreading.bridge(self.memory, first, second, links)
        mine = self._traits(first)
        theirs = self._traits(second)
        only_mine = [t for t in mine if t not in theirs]
        only_theirs = [t for t in theirs if t not in mine]
        return compared(first, second, shared, only_mine[:4],
                        only_theirs[:4], self.memory.kinds)

    def _requirement(self, concept, action):
        """"bir kuşun uçabilmesi için ne gerekir" — kayıtlı önkoşullar.

        Önkoşul kalıtılıyor: kuş için geçerli olan kartal için de geçerli.
        Hiçbiri kayıtlı değilse uydurulmuyor — bir önkoşulu tahmin etmek,
        cevabı uydurmakla aynı şey.
        """
        if not concept:
            return self._dont_know(concept)
        if not self.memory.query(concept):
            return self._dont_know(concept)
        found = []
        for step in [concept] + self.reasoning.ancestors(concept):
            for edge in self.memory.query(step, REQUIRES):
                if action and edge.object and edge.object != action:
                    continue
                if edge.target not in found:
                    found.append(edge.target)
        if not found:
            return unknown_requirement(concept)
        return requirements(concept, found)

    def _which_more(self, first, second, trait):
        """"hangisi daha hızlı, kartal mı penguen mi" — sıralama, tahmin değil.

        İlişki grafta zaten duruyordu: "kartal serçeden hızlıdır" cümlesi
        `kartal --hızlı--> [çıkış: serçe]` diye yazılıyor. Eksik olan yalnızca
        soru biçimi ve karşılaştırma işlemiydi.

        Kayıtlı bir üstünlük yoksa uydurulmuyor. İkisinde de nitelik varsa
        "ikisi de" denir; hiçbirinde yoksa bilinmediği söylenir. Sıralamayı
        tahmin etmek, cevabı uydurmakla aynı şey.
        """
        if not (first and second and trait):
            return self._dont_know(first or second)
        for concept in (first, second):
            if not self.memory.query(concept):
                return self._dont_know(concept)
        ahead = self._beats(first, second, trait)
        behind = self._beats(second, first, trait)
        if ahead and not behind:
            return more_so(first, second, trait)
        if behind and not ahead:
            return more_so(second, first, trait)
        mine = self._has_trait(first, trait)
        theirs = self._has_trait(second, trait)
        if mine and theirs:
            return both_but_unranked(trait)
        if mine:
            return only_one_has(first, second, trait)
        if theirs:
            return only_one_has(second, first, trait)
        return no_comparison(first, second, trait)

    def _beats(self, concept, other, trait):
        """Kayıtlı bir üstünlük var mı: "X, Y'den TRAIT'tir"."""
        for step in [concept] + self.reasoning.ancestors(concept):
            for edge in self.memory.query(step, HAS_PROPERTY):
                if edge.target == trait and edge.object == other:
                    return True
        return False

    def _has_trait(self, concept, trait):
        for step in [concept] + self.reasoning.ancestors(concept):
            for edge in self.memory.query(step, HAS_PROPERTY):
                if edge.target == trait:
                    return True
        return False

    def _traits(self, concept):
        """Bu kavram için geçerli olan şeyler — kendi kayıtları ve kalıtımı."""
        # Kavramın KENDİ kayıtlarında etkin anlam süzgeci uygulanıyor;
        # atalardan gelenler zaten o anlama ait, çünkü ata seçimi anlam
        # ayrımından geçmiş durumda (`reasoning.lineage`). Süzgeç olmadan
        # karşılaştırma iki anlamı karıştırıyordu:
        #
        #   kartal ile penguen arasındaki fark ne
        #   -> "kartal: avlanmak yapabilir, hızlı, İSTANBUL SAHİP, kurul"
        split = self._many_senses(concept)
        sense = self._sense(concept, self.focus_words) if split else None
        found = []
        for at, step in enumerate([concept] + self.reasoning.ancestors(concept)):
            for edge in self.memory.query(step):
                if edge.relation in (IS_A, NOT_A):
                    continue
                if (split and at == 0 and edge.context is not None
                        and edge.context != sense):
                    continue        # karşılaştırmada öteki anlam gürültüdür
                mark = (edge.relation, edge.target)
                if mark not in found:
                    found.append(mark)
        return found

    def _where(self, intent):
        """"penguen nerede yaşar" — the places recorded for this action."""
        found = []
        for concept in [intent.concept] + self.reasoning.ancestors(intent.concept):
            for edge in self.memory.query(concept, CAN):
                if edge.target == intent.target and edge.role == PLACE:
                    found.append((concept, edge.object))
        if not found:
            return self._dont_know(intent.concept)
        concept, place = found[0]
        clause = ability_clause(intent.concept, intent.target, True, place, PLACE)
        return inherited_clause(clause, intent.concept,
                                concept if concept != intent.concept else None)

    def _how_many(self, intent):
        """"bazı kuşlar uçar mı" — read off the members, not stored as a fact.

        Sorunun niceliği buraya kadar geliyor ama kapı onu atıyordu: "bazı",
        "her" ve "hiçbir" üçü de aynı cevabı alıyordu. Ölçülen hali: "hiçbir
        kuş uçar mı" -> "evet, kartal ve serçe uçar". Kanıt aynı olsa da
        sorulan nicelik değişince cevap değişir, o yüzden nicelik söyleyişe
        veriliyor — karşılaştırmayı orası yapıyor, çünkü cevap dile bağlı.
        """
        positive = intent.relation == CAN
        yes, no = self.reasoning.survey(intent.concept, intent.target,
                                        (CAN, CANNOT))
        rule, _ = self.reasoning.can_do(intent.concept, intent.target)
        return how_many(intent.concept, intent.target, positive, yes, no, rule,
                        getattr(intent, "quantifier", ALL) or ALL)

    def _is_a(self, concept, target):
        """"kalp bir organ mı" — a yes/no about a place in the hierarchy."""
        if self.memory.direct(concept, NOT_A, target) is not None:
            return denied(concept, target)
        ancestors = self.reasoning.ancestors(concept)
        # Ret de kalıtılır: "organ bir canlı değildir" + "kalp bir organdır"
        # ⇒ kalp bir canlı değildir. Hiyerarşi tam da bunun için var. Eskiden
        # yalnızca kavramın kendi kaydına bakılıyordu ve bu olumsuz, kapalı
        # dünya varsayımının yan ürünü olarak geliyordu — yani doğru cevap
        # yanlış sebeple veriliyordu.
        for step in ancestors:
            if self.memory.direct(step, NOT_A, target) is not None:
                return denied(concept, target, [is_a_clause(concept, step)])
        if target in ancestors:
            chain = [is_a_clause(concept, step) for step in ancestors
                     if step == target or ancestors.index(step) == 0]
            return affirmed(chain[-1])
        return self._apart(concept, ancestors, target) or self._dont_know(concept)

    def _apart(self, concept, ancestors, target):
        """İki kavramın ayrı olduğunun KANITI, yoksa None.

        Burada eskiden ortak kök yetiyordu: iki kavram aynı hiyerarşiye
        yerleşmişse ve biri diğerinin atası değilse "ayrı dallardadır"
        sayılıyordu. Bu, hiyerarşinin bir BÖLÜMLEME olduğunu varsayar; değil.
        Ölçülen çürütme: "memeli bir hayvandır" + "omurgalı bir hayvandır" +
        "insan bir memelidir" grafında sistem "insan bir omurgalı değildir"
        diyordu — iki kategori örtüşür, kardeş olmaları ayrı olmalarını
        göstermez. Üretim grafında aynı hata "penguen bir yırtıcı değildir"
        oluyordu. Yolun yokluğu olumsuzun kanıtı değildir; bilmediğini
        "değildir" diye söyleyen bir sistem uydurmuyor sayılmaz.

        Kalan kanıt gerçek olan: kayıtlı bir ret. Ret simetriktir ve iki
        taraftan da kalıtılır — "kuş bir memeli değildir" kaydı, kartal (bir
        kuş) ile fare (bir memeli) arasındaki soruyu da cevaplar. Eskiden
        yalnız kavramın kendi tarafına bakılıyordu.
        """
        mine = [concept] + ancestors
        theirs = [target] + self.reasoning.ancestors(target)
        for step in mine:
            for other in theirs:
                for first, second in ((step, other), (other, step)):
                    if self.memory.direct(first, NOT_A, second) is None:
                        continue
                    reasons = []
                    if step != concept:
                        reasons.append(is_a_clause(concept, step))
                    reasons.append(is_not_a_clause(first, second))
                    if other != target:
                        reasons.append(is_a_clause(target, other))
                    return denied(concept, target, reasons)
        return None

    def _about(self, intent):
        """Any relation the registry knows — nothing here is per-relation code."""
        known, chain = self.reasoning.about(intent.concept, intent.relation,
                                            intent.target, intent.object,
                                            intent.role)
        if known is None:
            return self._dont_know(intent.concept)
        return because(known, chain)

    def _property(self, concept, prop, object=None, role=None):
        known, chain = self.reasoning.has_property(concept, prop, object, role)
        if known is None:
            return self._dont_know(concept)
        return because(known, chain)

    def _many_senses(self, concept):
        """Bu kavram gerçekten birden çok ANLAM taşıyor mu.

        Birden çok BAĞLAM, birden çok anlam demek değil: aynı şey birkaç
        cümlede anlatılmış olabilir. İlk yazışta bunları karıştırdım ve süzgeç
        doğru olguları kesti — sınav %97,3'ten %95,7'ye düştü, yanlış cevap
        4'ten 9'a çıktı.

        Ayıran şey türlerin İLİŞKİSİ: biri ötekinin atası mı, ortak ataları
        var mı. Hepsi ilgiliyse tek anlamdır ve süzgecin işi yoktur.

        Ölçüt `ancestors`, `_related` DEĞİL. İkincisi HAM erişime bakıyor ve
        128 binlik kirli grafta neredeyse her şey her şeye bağlı çıkıyor —
        "tavla" yeniden tek anlamlı sanıldı. `ancestors` anlam süzgecinden
        geçiyor; `mahalle in ancestors("oyun")` -> False.
        """
        targets = [edge.target for edge in self.memory.query(concept, IS_A)
                   if edge.target]
        for at, one in enumerate(targets):
            above = self.reasoning.ancestors(one)
            for other in targets[at + 1:]:
                theirs = self.reasoning.ancestors(other)
                if (other in above or one in theirs
                        or (set(above) & set(theirs))):
                    continue
                return True
        return False

    def _sense(self, concept, focus=()):
        """Kavramın ETKİN anlamı — sorunun işaret ettiği cümle.

        Aynı cümleden çıkan olgular aynı anlama aittir ve bu, künyede duran
        bir olgu; tahmin değil. Ölçüldü:

            kartal  (damgasız)  -> kuş, avlanmak, hızlı
                    2421167707  -> ilçe, banliyö          (İstanbul)
                    2561926007  -> kasaba, kurul, nüfus   (Macaristan)
            tavla   778856018   -> oyun, iki
                    1038044999  -> mahalle, hatay, defne

        Vektörle ayırmak da denendi ve zayıf kaldı (altı nitelikten beşi doğru,
        `hızlı` yanlış). Provenans hiçbir tahmin gerektirmiyor.
        """
        edges = self.memory.query(concept, IS_A)
        if not edges:
            return None
        strongest = max(edges, key=lambda edge: edge.confidence).context
        if not focus:
            return strongest
        # SORUNUN KENDİSİ ANLAMI SEÇER. Güvene bakmak sabit bir seçimdir:
        # `sos` her zaman oyun, `maya` her zaman paket çıkıyordu, cümlede ne
        # yazarsa yazsın. Ölçüldü, 29 cevabın 8'i tam buradan bozuluyordu:
        #
        #     "Kerevizli Dip Sos nasıl yapılır?"  -> "Sos bir oyundur"
        #     "Cumhuriyet Partisi..."             -> "Cumhuriyet bir gazetedir"
        #
        # Her anlam öbeği, sorunun kelimelerine ne kadar yakın durduğuyla
        # yarışıyor. Yakınlık gömmeden geliyor — `kerevizli` ile `yiyecek`
        # aynı çevrede geçer, `oyun` ile geçmez.
        #
        # LLM'de bunun adı dikkat: bağlam, hangi temsilin okunacağını
        # ağırlıklandırıyor. Aynı iş, ayrık grafta: bağlam, hangi PROVENANS
        # öbeğinin konuşacağını ağırlıklandırıyor. Fark şu ki burada seçilen
        # şey bir vektör değil, kaynağı yazılı bir cümle.
        #
        # Geometri yalnız SEÇİYOR, hiçbir olgu YAZMIYOR — sınır burada da
        # duruyor. Yakınlık hiçbir öbeği öne çıkarmazsa güvene dönülüyor.
        vectors = getattr(self.memory, "vectors", None)
        if vectors is None:
            return strongest
        groups = self._sense_groups(concept)
        if len(groups) < 2:
            return strongest
        best, mark = 0.0, strongest
        for context, targets in groups.items():
            score = 0.0
            for word in focus:
                if word == concept:
                    continue
                close = vectors.nearest(word, targets, 1)
                if close:
                    score = max(score, close[0][1])
            if score > best:
                best, mark = score, context
        return mark if best >= SENSE_MARGIN else strongest

    def _sense_groups(self, concept):
        """Anlam öbekleri: künye -> o cümleden çıkan hedefler.

        Çokanlamlılığın ölçüsü budur, tür kaydı değil. `_many_senses` yalnız
        `type` bağlarına bakıyor ve ölçüldü — kaçırıyor:

            sos  368270650   has:soya, has:mirin, property:tatlı   (yemek)
                 1947393639  type:oyun, property:iki               (oyun)

        Yemek anlamının hiç tür kaydı yok, o yüzden `sos` "tek anlamlı"
        sayılıyor ve soru ne olursa olsun oyun anlamı konuşuyordu. Ayrı
        cümleden gelmek, ayrı anlam olmak için yeterli kanıt.
        """
        groups = {}
        for edge in self.memory.query(concept):
            if edge.target is not None:
                groups.setdefault(edge.context, []).append(str(edge.target))
        groups.pop(None, None)          # damgasızlar her anlamla uyumlu
        return groups

    def _in_sense(self, concept, found):
        """Yalnız etkin anlama ait olgular.

        Damgasız kayıtlar her anlamla uyumlu sayılıyor: elle derlenen
        dosyalardan gelenlerin bağlamı yok ve onları elemek, en güvenilir
        bilgiyi atmak olurdu.

        ATMIYOR, SIRALIYOR. Önce elemeyi denedim ve ölçüm durdurdu: sınav
        %97,3'ten %96,7'ye düştü, çünkü öteki anlama ait olgular tamamen
        kayboluyordu —

            soru : taşlıca nasıldır      kayıt: taşlıca --property--> kâhtada
            cevap: taşlıca küçüktür.     (başka anlamın nitelikleri)

        O olgu ne yanlış ne uydurma; yalnız başka bir anlama ait. Atmak
        bilgiyi kaybetmek, sıralamak ise doğru anlamı öne almak. Kesme
        (`_telling`) zaten baştan alıyor, yani etkin anlam konuşuyor ve öteki
        kaybolmuyor.

        Kavram tek anlamlıysa bu sıralama hiçbir şey yapmıyor — 47.872
        kavramın yalnız 2.788'i birden çok cümleden besleniyor.
        """
        marks = {}
        for edge in self.memory.query(concept):
            if edge.target is not None:
                marks.setdefault(edge.target, edge.context)
        # İki ayrı kanıt, ikisi de çokanlamlılığı gösterebilir: ayrı türler
        # (`_many_senses`) ya da ayrı cümleler (`_sense_groups`). İlki tek
        # başınayken yemek anlamı olan `sos`u kaçırıyordu.
        if not (self._many_senses(concept)
                or len(self._sense_groups(concept)) > 1):
            return found                # tek anlamlı: süzgecin işi yok
        sense = self._sense(concept, self.focus_words)
        # Anahtar HEDEF. `reasoning.properties` (hedef, kutup) döndürüyor,
        # `abilities` de öyle — ilk yazışta (ilişki, hedef) anahtarı kullandım,
        # hiçbir arama tutmadı ve süzgeç sessizce her şeyi geçirdi. Kurulmuş
        # ama kapanmamış bir kapı, hiç olmamasından kötüdür: çalıştığı
        # sanılır.
        return sorted(found, key=lambda item:
                      0 if marks.get(item[0], None) in (None, sense) else 1)

    def _all_properties(self, concept):
        found = self._in_sense(concept, self.reasoning.properties(concept))
        if not found:
            # Bilinmiyorsa kardeşlerinden ÇIKARILABİLİR mi. Eşik ölçüldü:
            # kardeşlerin yarısından fazlasında varsa isabet %83,3. Daha
            # düşük eşikte daha çok şey söylenir ama %29'a düşer ve onda
            # yedisi yanlış olur — gerekçeli bile olsa söylenemez.
            #
            # Kapsam dürüstçe DÜŞÜK: kavramların %1,2'sinde çalışıyor. Sebebi
            # ölçüldü — Vikipedi kardeşlere ortak nitelik değil, her birine
            # kendine özgü ayrıntı veriyor. Yapı sağlam, veri seyrek.
            return self._likely(concept, HAS_PROPERTY) \
                or never_learned_properties(concept)
        return properties_answer(concept, self._telling(concept, found))

    def _telling(self, concept, found, most=None):
        """Bilgi taşıyanları öne al, kalabalığı kes.

        "kartal ne yapabilir" sorusuna dokuz şey saymak teknik olarak doğru ama
        cevap değil: `büyür`, `ölür`, `yer` her canlı için doğrudur ve kartal
        hakkında hiçbir şey söylemez. Bilgi taşıyan `avlanır` ve `uçar`.

        İlk ölçütüm yanlıştı: "kaç kavram paylaşıyor" saydım ve tam tersini
        elde ettim — `uçmak` çok paylaşılıyor ama kuş için TANIMLAYICI olan o.
        Doğru ölçüt paylaşım değil MESAFE: kavramın kendi kaydı, atasından
        gelenden daha çok şey söyler. `avlanır` kartalın kendi kaydı,
        `büyür` canlıdan miras.

        Kesme sessiz değil: kalanı "başka ne biliyorsun" ile sorulabilir.
        """
        most = most if most is not None else self.told_most
        if len(found) <= most:
            return found
        steps = [concept] + self.reasoning.ancestors(concept)
        rank = {}
        for distance, step in enumerate(steps):
            for edge in self.memory.query(step):
                rank.setdefault(edge.target, distance)
        # ANLAM birincil ölçüt, mesafe ikincil. Sıralamayı `_in_sense`
        # yapıyordu ama burası mesafeye göre yeniden sıralayıp onu eziyordu:
        # "kartal nasıldır" cevabında Macaristan kasabasının `kurul` niteliği
        # geri geliyordu. İki sıralama üst üste binerse sonuncusu kazanır.
        other = self._other_sense(concept)
        # SORU birincil ölçüt. Sorunun kelimelerine yakın olgu, mesafesi ne
        # olursa olsun önce konuşur — "dalak ne zaman ZARARLI olur" sorusunda
        # `zararlı`ya yakın hedefler, dökümün başına gelir. Soru sinyal
        # vermiyorsa ("dalak nedir") sıralama eskisi gibi: anlam, sonra mesafe.
        focus = self._focus_rank(concept, [item[0] for item in found])
        return sorted(found, key=lambda item: (item[0] in other,
                                               -focus.get(item[0], 0.0),
                                               rank.get(item[0], 99)))[:most]

    def _enriched(self, concept, relation, found):
        """(eylem, kutup) ikililerine, grafta duruyorsa NESNEYİ ekler.

        `reasoning.abilities` yalnız eylemi ve kutbu döndürüyor — nesne graf
        kenarında duruyor ama cevaba hiç ulaşmıyordu. Kalıtımla gelen eylemin
        nesnesi yok (kavramın kendi kaydı değil), o zaman ikili kalıyor.
        """
        richer = []
        for action, positive in found:
            edge = next((one for one in self.memory.query(concept, relation)
                         if one.target == action and one.object), None)
            richer.append((action, positive, edge.object, edge.role) if edge
                          else (action, positive))
        return richer

    def _focus_rank(self, concept, targets):
        """Hedef -> sorunun kelimelerine yakınlık. Yakın olan konuşsun.

        Cümlenin öznesi çıkarıldıktan sonra kalan kelimeler ÇÖPE GİDİYORDU ve
        ölçüldü — sonuç, farklı soruya aynı cevap:

            "dalağın alınması ne zaman ZARARLI olur"  6 kelimeden 1'i kullanıldı
            "dalak nedir"                             ikisine de aynı döküm

        LLM'de atılan kelime yok: cevap cümlenin tamamına şartlanır. Buradaki
        karşılığı bu sıralama — soru kelimeleri, düğümün olguları arasından
        hangilerinin söyleneceğini ağırlıklandırıyor. Anlam seçimiyle
        (`_sense`) aynı mekanizmanın ikinci yarısı: o hangi ANLAM konuşacak
        diye soruyordu, bu hangi OLGULAR.

        Geometri yine yalnız SEÇİYOR: sıralanan her olgu grafta duruyor,
        kaynağıyla. Hiçbir şey eklenmiyor, hiçbir şey atılmıyor — kesilen,
        "başka ne biliyorsun" ile alınabilir.
        """
        vectors = getattr(self.memory, "vectors", None)
        if vectors is None or not self.focus_words:
            return {}
        from lmm import frequency
        from lmm.verbs import is_structural
        counts = frequency.counts()
        asked = [word for word in self.focus_words
                 if word not in concept and concept not in word
                 and not is_structural(word, counts)]
        if not asked:
            return {}
        scored = {}
        for target in targets:
            close = vectors.nearest(str(target), asked, 1)
            if close and close[0][1] >= FOCUS_MARGIN:
                scored[target] = close[0][1]
        return scored

    def _other_sense(self, concept):
        """Etkin anlama AİT OLMAYAN hedefler — çok anlamlı kavramlarda."""
        if not self._many_senses(concept):
            return frozenset()
        sense = self._sense(concept, self.focus_words)
        return frozenset(edge.target for edge in self.memory.query(concept)
                         if edge.target is not None
                         and edge.context is not None
                         and edge.context != sense)

    def _who(self, action, positive, relation=CAN):
        if relation == HAS_PROPERTY:
            return self._who_is(action)
        if action not in self.memory.actions():
            return never_heard_action(action)
        found = self.reasoning.who_can(action, positive)
        if not found:
            return nobody_does(action, positive)
        return who_answer(found, action, positive)

    def _who_is(self, prop):
        """"kimler beyaz" — everything known to carry a property."""
        if prop not in self.memory.properties():
            return never_heard_property(prop)
        found = [c for c in self.memory.concepts()
                 if self.reasoning.has_property(c, prop)[0] is True]
        if not found:
            return nobody_is(prop)
        return who_is_answer(found, prop)

    def _likely(self, concept, relation):
        """Kardeşlerden çıkan tahmin — varsa cümlesi, yoksa None."""
        guesses = self.reasoning.likely(concept, relation)
        if not guesses:
            return None
        family = next((edge.target for edge
                       in self.memory.query(concept, IS_A)), None)
        if not family:
            return None
        return likely_traits(concept, [name for name, _ in guesses], family)

    def _abilities(self, concept):
        found = self._in_sense(concept, self.reasoning.abilities(concept))
        if not found:
            return self._likely(concept, CAN) \
                or never_learned_abilities(concept)
        return abilities_answer(concept,
                                self._enriched(concept, CAN,
                                               self._telling(concept, found)))

    def _why(self, intent):
        """Why questions work the same for what a thing does and how it is."""
        about_property = intent.relation == HAS_PROPERTY
        lookup = (self.reasoning.has_property if about_property
                  else self.reasoning.can_do)
        clause = property_clause if about_property else ability_clause
        known, chain = lookup(intent.concept, intent.target)
        if known is None:
            return self._dont_know(intent.concept)
        if known is not True:
            return actually(clause(intent.concept, intent.target, known))
        return because_chain(chain)

    def _definition(self, concept):
        """Bir adın birden çok karşılığı varsa, seçmek uydurmaktır.

        Wikipedia'dan derlenen 47.736 kavramın %9,6'sında birden fazla tür
        vardı: "tavla" hem bir mahalle hem bir oyun. Sistem en yüksek güvenli
        olanı sessizce seçiyordu ve emin görünüyordu — bir dil modelinin
        yapacağı şeyin aynısı. Oysa burada bilinmeyen bir şey yok; iki cevap da
        biliniyor ve doğru davranış ikisini de söyleyip sormak.
        """
        edges = self.memory.query(concept, IS_A)
        if not edges:
            return self._dont_know(concept)
        # Aynı zincirdeki iki tür BELİRSİZLİK DEĞİLDİR, özgülük farkıdır.
        # "insan bir memelidir" ile "insan bir canlıdır" ikisi de doğru ve
        # `canlı`, `memeli`nin atası. Ölçüt yalnız güven farkıyken ikisi de
        # 0,75 olduğu için ayrılamıyordu ve sistem olmayan bir belirsizlik
        # bildiriyordu:
        #
        #   insan nedir -> "birden fazla şeye işaret ediyor: bir memeli ve
        #                   bir canlı olabilir. hangisi?"
        #
        # Graf 128 bine çıkınca bu her yerde patladı; küçük grafta iki tür
        # nadirdi. Aynı zincirdeyse EN ÖZGÜL olan seçiliyor — seçmek burada
        # uydurmak değil, çünkü öteki de aynı şeyin daha genel adı.
        # Ölçüt `ancestors`, `_reach` DEĞİL. İlki anlam süzgecinden geçiyor,
        # ikincisi ham. Ham kümeyle denendi ve gerçek belirsizliği YUTTU:
        # 128 binlik grafta `mahalle` neredeyse her şeyin ham erişiminde
        # (`mahalle in _reach("oyun")` -> True) ve "tavla" yeniden tek anlamlı
        # sanıldı — bu projenin klasik belirsizlik örneği sessizce kayboldu.
        # Anlam süzgeçli zincirde `mahalle in ancestors("oyun")` -> False.
        settled = []
        for edge in sorted(edges, key=lambda e: -e.confidence):
            if any(edge.target in self.reasoning.ancestors(other.target)
                   for other in settled):
                continue                # zaten söylenenin ATASI: daha genel
            settled = [held for held in settled
                       if held.target
                       not in self.reasoning.ancestors(edge.target)]
            settled.append(edge)
        edges = settled or edges
        best = max(edges, key=lambda e: e.confidence)
        rivals = [edge for edge in edges
                  if edge.target != best.target
                  and best.confidence - edge.confidence <= AMBIGUITY_MARGIN]
        if rivals:
            readings = [best.target] + [edge.target for edge in rivals]
            return which_meaning(concept, readings)
        return definition_answer(concept, best.target, best.source,
                                 sure=best.confidence >= HEDGE_THRESHOLD)

    def _ability(self, concept, action, object=None, role=None):
        known, chain = self.reasoning.can_do(concept, action, object, role)
        if known is None:
            return self._dont_know(concept)
        return because(known, chain)

    def _dont_know(self, concept):
        known = self.memory.concepts()
        close = nearest(concept, known) if concept else ()
        if close:
            return dont_know(concept, close)
        # Yazım komşusu yoksa ANLAM komşusuna bakılır. İkisi ayrı şey: "kus"
        # ile "kuş" harf komşusu, "glokom" ile "katarakt" anlam komşusu ve
        # ikincisini yalnız dağılım verebilir.
        #
        # Ölçüldü: 400 gerçek sorunun %56,1'inde grafın hiç duymadığı bir
        # kavram var. Bunların çoğunda yazım komşusu da yok, yani cevap düz
        # "bilmiyorum" oluyordu. Elde ne olduğunu söylemek bir iddia değil ve
        # kullanıcıya sorusunu yeniden sorma imkânı veriyor.
        #
        # Vektör hiçbir zaman bir OLGU üretmiyor; yalnız hangi kavramların
        # gösterileceğini seçiyor. Kapı bu noktada zaten "bilmiyorum" demiş.
        neighbours = self._related_known(concept)
        if neighbours:
            return related_instead(concept, neighbours)
        return dont_know(concept, ())

    def _related_known(self, concept, count=3):
        """Anlamca yakın ve grafın BİLDİĞİ kavramlar — vektör varsa."""
        vectors = getattr(self.memory, "vectors", None)
        if vectors is None or not concept:
            return ()
        try:
            close = vectors.similar(concept, 40)
        except Exception:                                   # noqa: BLE001
            return ()
        known = set(self.memory.concepts())
        found = []
        for word, _ in close:
            if word in known and word != concept and self.memory.query(word):
                found.append(word)
            if len(found) >= count:
                break
        return tuple(found)
