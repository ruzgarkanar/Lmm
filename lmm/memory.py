"""Living Memory: LMM's knowledge organ.

Knowledge does not live in weights here — it lives in this graph, writable at any
moment, persisted to disk, and always carrying its source and confidence.
"""
import gzip
import json
import os
import time

from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,  # noqa: F401
                           LACKS_PROPERTY, HAS_PART, LACKS_PART, TYPE_RELATIONS,
                           ABILITY_RELATIONS, PROPERTY_RELATIONS, PART_RELATIONS,
                           PLACE, SOURCE, OBJECT, ALL, MOST, SOME, NO,
                           QUANTIFIERS, INHERITING)

from lmm.lexicon import Lexicon, current
from lmm.trust import INFERENCE, confidence_for, confidence_from, outranks
from lmm.kinds import Kinds

FORMAT_VERSION = 3

# The source on facts nobody stated — the system worked them out itself.
INFERRED = INFERENCE
INFERRED_CONFIDENCE = confidence_for(INFERENCE)


class CycleError(Exception):
    """Raised when a write would create a cycle in the type hierarchy."""


class Names(list):
    """Adların sırası — düz liste değil, üzerine indeks iliştirilebilen bir liste.

    NEDEN: `similarity.nearest` her "bilmiyorum" cevabında tüm kavramları
    tarıyor ve maliyeti graf boyutuyla doğrusal — ölçüldü: 16 bin kavramda
    40,9 ms, 335 binde 808,7 ms. İndekslenmesi gerekiyordu ve asıl soru
    indeksin NEREDE yaşayacağıydı.

    Modül düzeyinde `id(liste)` ile anahtarlanan bir önbellek denenebilirdi;
    tehlikeli, çünkü çöp toplayıcı bir kimliği geri kullanınca aynı uzunluktaki
    başka bir listeye BAŞKASININ indeksi verilir — bu projede bayat önbellek
    demek, sessiz yanlış cevap demek. İndeksi listenin kendisine asmak bu soruyu
    ortadan kaldırıyor: nesne yaşadığı sürece indeks geçerli, kavram eklenince
    `concepts()` YENİ bir Names döndürüyor ve eski indeks nesnesiyle birlikte
    düşüyor. Geçersizleştirme bir kural değil, bir sonuç.

    Liste alt sınıfı olması davranışı değiştirmiyor: eşitlik, dilimleme,
    `set(...)`, sıralama — hepsi listenin kendisi.
    """


def coverage(quantifier):
    """Bir iddianın türün ne kadarına ulaştığı. Evrensel > varsayılan > varoluşsal.

    Sayı değil sıra: Cyc'in 0-100 aralığından kaçınmanın bedeli, nicelikleri
    yine de karşılaştırabilmek. Üç basamak yetiyor, çünkü dilin ayırdığı da bu.

    NEDEN evrensel ile "hiçbiri" aynı basamakta: bu gösterimde olumsuzluk
    yüklemde duruyor, nicelikte değil. "hiçbir kuş uçmaz" -> (kuş, cannot,
    uçmak, hiçbiri); "tüm kuşlar uçmaz" -> (kuş, cannot, uçmak, hepsi). İkisi
    de aynı evrensel olumsuz iddia — Türkçedeki olumsuzluk uyumu yüzünden
    "hiç" ile "-mez" tek bir olumsuzluk. Mantıktan gelen okur NO'nun kutbu
    çevirmesini bekler; burada çevirmez, INHERITING = (ALL, NO) da bunu söylüyor.
    """
    if quantifier in INHERITING:
        return 2
    return 1 if quantifier == MOST else 0


class Edge:
    def __init__(self, concept, relation, target, object=None, role=None,
                 source="unknown", confidence=None, is_exception=False,
                 timestamp=None, sources=None, disputed=False, quantifier=ALL,
                 context=None):
        self.concept = concept
        self.relation = relation        # IS_A | CAN | CANNOT
        self.target = target
        self.object = object            # a second concept, if the sentence had one
        self.role = role                # what that concept is doing there
        self.source = source
        # Everyone who has independently said this. The strongest of them is
        # what answers cite; how many there are is what confidence rests on.
        self.sources = list(sources) if sources else [source]
        # Unless a caller insists, how sure to be follows from who said it.
        self.confidence = (confidence if confidence is not None
                           else confidence_from(self.sources))
        self.is_exception = is_exception
        self.disputed = disputed        # sources disagree and neither prevailed
        self.quantifier = quantifier    # how much of the kind this covers
        # Bu olgunun hangi CÜMLEDEN geldiği. Kaynaktan ayrı bir şey: kaynak
        # "kim söyledi", bağlam "hangi söyleyişte".
        #
        # Anlam ayrımının en güçlü sinyali burada ve atılıyordu. Aynı cümleden
        # çıkan olgular aynı ANLAMA aittir; ölçüldü:
        #
        #   tavla  cümle 1 -> oyun, iki, zar, pul, platform
        #          cümle 2 -> mahalle, hatay, defne
        #   kartal cümle 1 -> ilçe, banliyö        (İstanbul)
        #          cümle 3 -> kasaba, kurulu, nüfus (Macaristan)
        #
        # Vektörle ayırmayı denedim ve 20 bin cümlelik gömme 6 nitelikten
        # 5'ini doğru ayırdı, `hızlı`yı kaçırdı. Künye zaten kusursuz ayırıyor
        # ve hiçbir tahmin gerektirmiyor — geometri değil PROVENANS.
        self.context = context
        self.timestamp = timestamp if timestamp is not None else time.time()

    def corroborate(self, source):
        """Another voice for a fact already held. Independent ones count."""
        if source not in self.sources:
            self.sources.append(source)
        if outranks(source, self.source):
            self.source = source        # answers cite the strongest voice
        self.confidence = confidence_from(self.sources)
        return self

    def to_dict(self):
        return {"concept": self.concept, "relation": self.relation,
                "target": self.target, "object": self.object, "role": self.role,
                "source": self.source, "sources": self.sources,
                "confidence": self.confidence, "is_exception": self.is_exception,
                "disputed": self.disputed, "quantifier": self.quantifier,
                "context": self.context, "timestamp": self.timestamp}

    @staticmethod
    def from_dict(data):
        return Edge(**data)


class Memory:
    """Facts the system knows, plus the questions it has already put to a teacher.

    Both are memory: one is what it believes, the other is what it has already
    wondered aloud. Without the second, curiosity would ask the same question
    forever — which is the opposite of learning.
    """

    def __init__(self, lexicon=None):
        # Its own vocabulary, so two memories in one process never teach each
        # other words. A service holds more than one at a time.
        self.lexicon = lexicon if lexicon is not None else Lexicon()
        self.edges = []
        self.asked = set()
        self.vocabulary = []    # words learned beyond the core lexicon
        self.patterns = []      # ways of saying things, learned beyond the core
        self.reputation = {}    # each source's record of agreeing and disagreeing
        self.health = {}        # per branch: how much is accepted, how much refused
        self.kinds = Kinds()    # what each relation does — data, not code
        self._rebuild()

    def _rebuild(self):
        """Indexes over the edges. Everything here is derivable from self.edges.

        Without them every question scanned every fact, which is fine at three
        hundred and hopeless at a million — and curiosity and induction sweep the
        whole memory, so the cost was quadratic where it hurt most.
        """
        self._by_concept = {}
        self._by_target = {}
        self._exact = {}
        self._quantities = {}
        self._concepts = []
        self._concept_set = set()
        self._actions, self._action_set = [], set()
        self._properties, self._property_set = [], set()
        # Ad listelerinin dondurulmuş hâli; ilk istendiğinde kurulur, graf
        # değişince düşer. Grafın kaçıncı hâlinde olduğumuzu ise dışarıdaki
        # önbellekler soruyor (`spreading.neighbours`).
        self._names = {}
        self.revision = getattr(self, "revision", 0) + 1
        self.purges = getattr(self, "purges", 0)
        for edge in self.edges:
            self._index(edge)

    def _index(self, edge):
        self.revision += 1
        if self._names:
            self._names = {}     # yeni ad gelmiş olabilir: dondurulmuş liste bayat
        self._by_concept.setdefault(edge.concept, []).append(edge)
        # Ters yön de indeksli: "bu kavrama KİM işaret ediyor" sorusu eskiden
        # tüm kenarları tarıyordu. Ölçüldü — eş anlamlı araması (`cli._meanings`)
        # tek bir cümlede yüzlerce kez bu taramayı yapıyordu ve 10 kat büyümüş
        # grafta 20 sorunun 34,5 saniyesinin 15,4'ü buradaydı.
        if edge.target:
            self._by_target.setdefault(edge.target, []).append(edge)
        # Nicelik kimliğin parçası. Değilken "bazı kuşlar uçmaz" ile "hiçbir kuş
        # uçmaz" aynı kenara düşüyordu ve ikincisi birincisini corroborate() ile
        # GÜÇLENDİRİYORDU: iki ayrı iddia tek kayda çöküyor, evrensel olan da
        # yutuluyordu. Ölçüldü: SOME(a) sonra NO(b) yazınca tek kenar kalıyor ve
        # güven 0.60 -> 0.75'e çıkıyordu — hiç söylenmemiş bir mutabakat.
        slot = (edge.concept, edge.relation, edge.target, edge.object, edge.role)
        self._exact[slot + (edge.quantifier,)] = edge
        self._quantities.setdefault(slot, []).append(edge)
        self._note_concept(edge.concept)
        if edge.relation in TYPE_RELATIONS:
            self._note_concept(edge.target)
        elif edge.relation in ABILITY_RELATIONS:
            self._note(edge.target, self._actions, self._action_set)
        elif edge.relation in PROPERTY_RELATIONS:
            self._note(edge.target, self._properties, self._property_set)

    def _note_concept(self, name):
        self._note(name, self._concepts, self._concept_set)

    @staticmethod
    def _note(name, ordered, seen):
        if name not in seen:
            seen.add(name)
            ordered.append(name)

    def learn_word(self, infinitive, positive, negative, lexicon=None):
        """Add a verb to what this memory knows how to say, and can hear."""
        entry = {"infinitive": infinitive, "positive": positive,
                 "negative": negative}
        if entry not in self.vocabulary:
            self.vocabulary.append(entry)
        (lexicon or self.lexicon).learn_verb(infinitive, positive, negative)
        return entry

    def learn_pattern(self, pattern):
        """Remember a new way of saying something, so it survives a restart.

        A pattern is knowledge like a fact or a word: the model file carries
        everything the system worked out, including how to be spoken to.
        """
        entry = pattern.to_dict()
        if entry not in self.patterns:
            self.patterns.append(entry)
        return entry

    def mark_asked(self, key):
        self.asked.add(key)

    def has_asked(self, key):
        return key in self.asked

    def query(self, concept, relation=None):
        found = self._by_concept.get(concept, ())
        if relation is None:
            return list(found)
        return [e for e in found if e.relation == relation]

    def direct(self, concept, relation, target, object=None, role=None,
               quantifier=None):
        """Aranan olgu. Nicelik verilmezse o yuvadaki en geniş kapsamlı iddia.

        İKİ NİCELİK AYNI ANDA DURABİLİR Mİ? Evet — ve bu bir çelişki değil.
        Aynı yüklem kutbunda "hepsi", "çoğu" ve "bazı" altbağlılık (subalternation)
        ilişkisindedir: hepsi -> çoğu -> bazı. Evrensel doğruysa varoluşsal da
        doğrudur; ikisi birlikte tutarlıdır. Bu yüzden ikisini de saklıyoruz.

        Mantığın karşıtlık karesinde asıl çelişki KUTUPLAR ARASINDA: evrensel-P
        ile varoluşsal-¬P (A ile O). O çatışmayı zaten reasoning.find_conflict
        yakalıyor — ölçtüm: "kuşlar uçar" + "bazı kuşlar uçmaz" bugün de
        "bir çelişki fark ettim" diyor, çünkü karşıt bağıntıya bakıyor. Burada
        yapılması gereken, o denetime yuvanın EN GÜÇLÜ iddiasını göstermek.

        Nicelik sormayan çağrı — kodun çoğu — "bu yuvada ne biliyorum" diye
        sorar; doğru cevap en geniş kapsamlı olandır, çünkü evrensel iddia
        varoluşsalı zaten içerir. Aksi hâli ölçtüm: "bazı kuşlar uçmaz" sırf
        önce yazıldığı için sonradan gelen "hiçbir kuş uçmaz"ı sonsuza dek
        gölgeliyor, INHERITING dışı kaldığı için de kalıtımı kapatıyordu —
        "penguen uçar mı" cevapsız kalıyordu. Şimdi evrensel olan iniyor.

        Eşitlikte ilk yazılan konuşur: max() ilk en büyüğü döndürür, yani
        kayıt sırası kararlı kalır ve aynı ölçüm iki kez aynı sonucu verir.
        """
        slot = (concept, relation, target, object, role)
        if quantifier is not None:
            return self._exact.get(slot + (quantifier,))
        held = self._quantities.get(slot)
        if not held:
            return None
        return max(held, key=lambda edge: coverage(edge.quantifier))

    def quantities(self, concept, relation, target, object=None, role=None):
        """Aynı yuvada duran bütün nicelikler, yazılma sırasıyla.

        Bir kenarın artık tek bir niceliği olmadığını görmek isteyen için:
        cevaplayan direct()'in seçtiğini alır, denetleyen hepsini görmelidir.
        """
        return list(self._quantities.get(
            (concept, relation, target, object, role), ()))

    def write(self, edge):
        if edge.relation == IS_A and self._creates_cycle(edge):
            raise CycleError(f"{edge.concept} -> {edge.target} creates a cycle")
        # Pekişme yalnız AYNI nicelik için. Farklı nicelik yeni bir iddiadır;
        # onu tanık saymak, kimsenin söylemediği bir mutabakatı uydurmaktı.
        existing = self.direct(edge.concept, edge.relation, edge.target,
                               edge.object, edge.role, edge.quantifier)
        if existing is not None:
            existing.corroborate(edge.source)
            return existing
        self.edges.append(edge)
        self._index(edge)
        return edge

    def _named(self, key, ordered):
        """Sıralı adların dondurulmuş hâli — graf değişene kadar aynı nesne.

        Eskiden her çağrı listeyi KOPYALIYORDU. Tek başına ucuz görünüyor
        (16 bin kavramda 0,034 ms) ama çağıranı sayınca değişiyor:
        `grammar.known()` ve `cli._meanings` bunu cümle başına yüzlerce kez
        soruyor. Ölçüldü: 20 kat büyümüş grafta üçlü çağrı 1,72 ms, yani tek
        bir cümlede saniyeler.

        Dönen liste PAYLAŞILIYOR — çağıran onu değiştirmemeli. Kopya vermenin
        bedeli buydu ve karşılığında hiçbir çağıran o kopyayı değiştirmiyordu.
        Buna karşılık dönen şey bir enstantane: graf büyüyünce bu liste
        büyümez, bir sonraki çağrı yenisini alır — yani üzerinde gezinirken
        öğrenmek listeyi altından çekmiyor.
        """
        found = self._names.get(key)
        if found is None:
            found = self._names[key] = Names(ordered)
        return found

    def concepts(self):
        """Everything that behaves like a thing, in the order it was learned."""
        return self._named("concepts", self._concepts)

    def actions(self):
        """Every action this memory has ever heard of, in learning order."""
        return self._named("actions", self._actions)

    def properties(self):
        """Every property this memory has ever heard of, in learning order."""
        return self._named("properties", self._properties)

    def incoming(self, target, relation=None):
        """Bu ada HEDEF olarak işaret eden kenarlar — ters yön, indeksli.

        `query()` bir kavramdan çıkanı verir; bu, ona geleni. İkisi olmadan
        "buna kim eş anlamlı" gibi her soru tüm kenarları tarıyordu.
        """
        found = self._by_target.get(target, ())
        if relation is None:
            return list(found)
        return [e for e in found if e.relation == relation]

    def forget(self, concept):
        """Erase everything known about a concept, in or out. Returns the count.

        Selective deletion from a trained model's weights is famously impractical;
        here it is a filter over a list, and afterwards the system genuinely does
        not know — it goes back to saying "bilmiyorum".

        Bilinmeyen bir ad artık indeksten anlaşılıyor ve hiçbir şeye
        dokunulmuyor. Ölçüldü: 20 kat büyümüş grafta olmayan bir kavramı
        unutmak 349,2 ms sürüyordu — tamamı boşuna, çünkü silinecek bir şey
        yoktu. Gerçekten silme yine tam yeniden kurulum: ad listeleri
        adların KAÇ kenardan geldiğini saymıyor, sayarsak silme ucuzlar ama
        her yazma pahalılaşır ve yazma yüz kat daha sık.
        """
        if concept not in self._by_concept and concept not in self._by_target:
            return 0
        remaining = [e for e in self.edges
                     if e.concept != concept and e.target != concept]
        removed = len(self.edges) - len(remaining)
        self.edges = remaining
        self.purges += 1
        self._rebuild()
        return removed

    def _creates_cycle(self, edge):
        # Walk type edges up from the target: can we reach the concept again?
        queue, seen = [edge.target], set()
        while queue:
            current = queue.pop(0)
            if current == edge.concept:
                return True
            if current in seen:
                continue
            seen.add(current)
            queue.extend(e.target for e in self.query(current, IS_A))
        return False

    @staticmethod
    def _opener(path):
        """A .lmm file is the same JSON, gzipped — around twenty times smaller.

        Compression is a storage detail, not a change of kind: gunzip it and you
        are looking at the same readable facts. Nothing becomes a black box.
        """
        return (gzip.open, "wt", "rt") if path.endswith(".lmm") else (open, "w", "r")

    def save(self, path):
        payload = {"format": FORMAT_VERSION,
                   "edges": [e.to_dict() for e in self.edges],
                   "asked": sorted(self.asked),
                   "vocabulary": self.vocabulary,
                   "patterns": self.patterns, "reputation": self.reputation,
                   "health": self.health, "kinds": self.kinds.to_list()}
        opener, write_mode, _ = self._opener(path)
        temp = path + ".tmp"
        with opener(temp, write_mode, encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        os.replace(temp, path)  # atomic: a half-written file never appears

    @staticmethod
    def load(path):
        memory = Memory()
        try:
            opener, _, read_mode = Memory._opener(path)
            with opener(path, read_mode, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):        # format 1: a bare list of edges
                memory.edges = [Edge.from_dict(d) for d in data]
                memory._rebuild()
            else:
                memory.edges = [Edge.from_dict(d) for d in data["edges"]]
                memory._rebuild()
                memory.asked = set(data.get("asked", []))
                memory.patterns = data.get("patterns", [])
                memory.reputation = data.get("reputation", {})
                memory.health = data.get("health", {})
                memory.kinds.load(data.get("kinds", []))
                for word in data.get("vocabulary", []):
                    memory.learn_word(**word)   # words come back with the facts
        except (OSError, ValueError, KeyError, TypeError):
            pass  # missing or corrupt file: start with an empty memory
        return memory
