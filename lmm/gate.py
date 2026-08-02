"""Epistemic Gate: answers are produced from memory alone.

There is no path from this class to a sentence that memory does not support, so
hallucination is not filtered out — it is unreachable.
"""
from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                           HAS_PART, LACKS_PART, PLACE, REQUIRES)
from lmm.phrasing import (is_a_clause, is_not_a_clause, ability_clause,
                          property_clause,
                          who_clause, ability_summary, property_summary,
                          verb_form, dont_know, attribution, how_many,
                          listing, copula, which_meaning, compared)
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
MOST_TOLD = 5
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

    def answer(self, intent):
        if intent.kind == ASK_WHO:
            return self._who(intent.target, intent.relation == CAN,
                             intent.relation)
        if intent.kind == ASK_ABILITIES:
            return self._abilities(intent.concept)
        if intent.kind == ASK_WHY:
            return self._why(intent)
        if intent.kind == ASK_DESCRIBE:
            return self.exposition.describe(intent.concept)
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
            return (f"{concept} için ne gerektiğini bilmiyorum. "
                    f"bana öğretir misin?")
        return f"{concept} için {listing(found)} gerekir."

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
            return f"{first}, çünkü {first} {second}den {trait}dır."
        if behind and not ahead:
            return f"{second}, çünkü {second} {first}den {trait}dır."
        mine = self._has_trait(first, trait)
        theirs = self._has_trait(second, trait)
        if mine and theirs:
            return (f"ikisi de {trait}, ama hangisinin daha {trait} olduğunu "
                    f"bilmiyorum.")
        if mine:
            return f"{first} {trait}, {second} için bunu bilmiyorum."
        if theirs:
            return f"{second} {trait}, {first} için bunu bilmiyorum."
        return f"{first} ile {second} arasında {trait} karşılaştırması bilmiyorum."

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
        found = []
        for step in [concept] + self.reasoning.ancestors(concept):
            for edge in self.memory.query(step):
                if edge.relation in (IS_A, NOT_A):
                    continue
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
        if concept != intent.concept:
            return f"{clause}, çünkü {intent.concept} bir {concept}."
        return clause + "."

    def _how_many(self, intent):
        """"bazı kuşlar uçar mı" — read off the members, not stored as a fact."""
        positive = intent.relation == CAN
        yes, no = self.reasoning.survey(intent.concept, intent.target,
                                        (CAN, CANNOT))
        rule, _ = self.reasoning.can_do(intent.concept, intent.target)
        return how_many(intent.concept, intent.target, positive, yes, no, rule)

    def _is_a(self, concept, target):
        """"kalp bir organ mı" — a yes/no about a place in the hierarchy."""
        if self.memory.direct(concept, NOT_A, target) is not None:
            return f"hayır, {is_not_a_clause(concept, target)}."
        ancestors = self.reasoning.ancestors(concept)
        # Ret de kalıtılır: "organ bir canlı değildir" + "kalp bir organdır"
        # ⇒ kalp bir canlı değildir. Hiyerarşi tam da bunun için var. Eskiden
        # yalnızca kavramın kendi kaydına bakılıyordu ve bu olumsuz, kapalı
        # dünya varsayımının yan ürünü olarak geliyordu — yani doğru cevap
        # yanlış sebeple veriliyordu.
        for step in ancestors:
            if self.memory.direct(step, NOT_A, target) is not None:
                return (f"hayır, {is_not_a_clause(concept, target)}, "
                        f"çünkü {is_a_clause(concept, step)}.")
        if target in ancestors:
            chain = [is_a_clause(concept, step) for step in ancestors
                     if step == target or ancestors.index(step) == 0]
            return f"evet, {chain[-1]}."
        if not self.memory.query(concept, IS_A):
            return self._dont_know(concept)
        # Yolun yokluğu olumsuzun kanıtı DEĞİLDİR. Burada eskiden kavramın
        # herhangi bir tür bağı olması yetiyordu ve sistem bilmediği şeyi
        # olumsuzluyordu: ansiklopediden "vaşak bir hayvan türüdür" okuduktan
        # hemen sonra "vaşak bir hayvan değildir" diyordu. Sürekli okuyarak
        # büyüyen bir grafta hiyerarşiyi tamam saymak temelsiz.
        #
        # Olumsuz ancak kanıtla söylenir: iki kavram da AYNI hiyerarşiye
        # yerleşmişse ve hiçbiri diğerinin üstünde değilse, ayrı dallardadırlar
        # ve bu gerçek bir ayrılık kanıtıdır — "penguen bir memeli değildir",
        # çünkü ikisi de hayvanın altında ve biri diğerinin atası değil.
        # Ortak kök yoksa cevap bilmemektir.
        mine = set(ancestors) | {concept}
        theirs = set(self.reasoning.ancestors(target)) | {target}
        if not mine & theirs:
            return self._dont_know(concept)
        return f"bildiğim kadarıyla {is_not_a_clause(concept, target)}."

    def _about(self, intent):
        """Any relation the registry knows — nothing here is per-relation code."""
        known, chain = self.reasoning.about(intent.concept, intent.relation,
                                            intent.target, intent.object,
                                            intent.role)
        if known is None:
            return self._dont_know(intent.concept)
        return f"{'evet' if known else 'hayır'}, çünkü {' ve '.join(chain)}."

    def _property(self, concept, prop, object=None, role=None):
        known, chain = self.reasoning.has_property(concept, prop, object, role)
        if known is None:
            return self._dont_know(concept)
        prefix = "evet" if known else "hayır"
        return f"{prefix}, çünkü {' ve '.join(chain)}."

    def _all_properties(self, concept):
        found = self.reasoning.properties(concept)
        if not found:
            return f"{concept} nasıldır, bunu hiç öğrenmedim."
        return property_summary(concept, self._telling(concept, found)) + "."

    def _telling(self, concept, found, most=MOST_TOLD):
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
        if len(found) <= most:
            return found
        steps = [concept] + self.reasoning.ancestors(concept)
        rank = {}
        for distance, step in enumerate(steps):
            for edge in self.memory.query(step):
                rank.setdefault(edge.target, distance)
        return sorted(found, key=lambda item: rank.get(item[0], 99))[:most]

    def _who(self, action, positive, relation=CAN):
        if relation == HAS_PROPERTY:
            return self._who_is(action)
        if action not in self.memory.actions():
            return f"{verb_form(action, True)} diye bir şeyi hiç duymadım."
        found = self.reasoning.who_can(action, positive)
        if not found:
            return (f"bildiğim hiçbir şeyin {verb_form(action, positive)}ini "
                    f"öğrenmedim.")
        return who_clause(found, action, positive) + "."

    def _who_is(self, prop):
        """"kimler beyaz" — everything known to carry a property."""
        if prop not in self.memory.properties():
            return f"'{prop}' diye bir niteliği hiç duymadım."
        found = [c for c in self.memory.concepts()
                 if self.reasoning.has_property(c, prop)[0] is True]
        if not found:
            return f"bildiğim hiçbir şeyin {prop} olduğunu öğrenmedim."
        return f"{listing(found)} {prop}{copula(prop)}."

    def _abilities(self, concept):
        found = self.reasoning.abilities(concept)
        if not found:
            return f"{concept} ne yapabilir, bunu hiç öğrenmedim."
        return ability_summary(concept, self._telling(concept, found)) + "."

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
            return f"aslında {clause(intent.concept, intent.target, known)}."
        return "çünkü " + " ve ".join(chain) + "."

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
        best = max(edges, key=lambda e: e.confidence)
        rivals = [edge for edge in edges
                  if edge.target != best.target
                  and best.confidence - edge.confidence <= AMBIGUITY_MARGIN]
        if rivals:
            readings = [best.target] + [edge.target for edge in rivals]
            return which_meaning(concept, readings)
        answer = f"{is_a_clause(concept, best.target)} ({attribution(best.source)})."
        if best.confidence < HEDGE_THRESHOLD:
            return "emin değilim ama " + answer
        return answer

    def _ability(self, concept, action, object=None, role=None):
        known, chain = self.reasoning.can_do(concept, action, object, role)
        if known is None:
            return self._dont_know(concept)
        prefix = "evet" if known else "hayır"
        return f"{prefix}, çünkü {' ve '.join(chain)}."

    def _dont_know(self, concept):
        known = self.memory.concepts()
        return dont_know(concept, nearest(concept, known) if concept else ())
