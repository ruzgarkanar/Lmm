"""Grammar as data, not as code.

Sentence patterns used to be a chain of `if len(tokens) == 3 and ...` in the
parser, which meant the system could learn any fact but not a single new way of
saying one — and a second language would have meant a second parser.

A pattern is knowledge like any other, so it lives in a list that can be added
to, shipped in a pack, or worked out from an example. Nothing here is Turkish
except the entries themselves; the machinery matches slots against tokens and
knows nothing about which language it is looking at.

A slot is one of:
    KAVRAM   a thing being spoken about, plural stripped
    TUR      a type name, its copula stripped if it carries one
    NITELIK  a property, which must carry a copula
    SOZ      any word at all
    FIIL     a verb the lexicon knows, in either polarity
    SORU     a question particle
    KIM      an interrogative
Anything else in a pattern is a literal that must appear exactly.
"""
from lmm.relations import (IS_A, NOT_A, CAN, CANNOT, HAS_PROPERTY,
                           LACKS_PROPERTY, OBJECT, ALL)

KAVRAM = "{kavram}"
TUR = "{tür}"
NITELIK = "{nitelik}"
SOZ = "{söz}"
FIIL = "{fiil}"
SORU = "{soru}"
KIM = "{kim}"
ROL = "{rol}"        # a second concept wearing a case ending
NICEL = "{nicel}"    # how much of a kind: bütün / çoğu / bazı / hiçbir
SAHIP = "{sahip}"    # a possessor, marked as one by the language
NESNEL = "{nesnel}"  # a concept wearing an accusative ending: "pengueni"

SLOTS = (KAVRAM, TUR, NITELIK, SOZ, FIIL, SORU, KIM, ROL, NICEL, SAHIP,
         NESNEL)

FROM_VERB = "fiilden"       # the relation follows the verb's own polarity

# Bir kavram öbeğinin içine hangi kelimelerin giremeyeceği DİLE aittir ve
# biçimbilimden sorulur. Bu gece ilk yazışta listeler buraya konmuştu ve bu
# bir mimari ihlaldi: `grammar.py` hangi dile baktığını bilmemeli, yoksa
# ikinci bir dil ikinci bir dilbilgisi motoru demek olur.
# Ünsüz yumuşaması: gövde sonundaki sert ünsüz, ünlüyle başlayan ek alınca
# yumuşar. "bahset" + "er" -> "bahsed"er. Düz önek karşılaştırması bunu
# kaçırıyor ve doğru eşleşmeyi reddediyordu.
SOFTENS = {"t": "d", "k": "ğ", "p": "b", "ç": "c"}


def _starts_with(token, stem):
    """Kelime bu gövdeyle başlıyor mu — ünsüz yumuşaması sayılarak."""
    if not stem or token.startswith(stem):
        return bool(stem)
    soft = SOFTENS.get(stem[-1])
    return bool(soft) and token.startswith(stem[:-1] + soft)


def _blocked(morphology):
    """Öbeğe giremeyecek kelimeler — dilin kendi bildirdikleri."""
    found = set()
    # Pekiştireç burada YOK: yeri yalnızca kavram yuvası. Her yuvada
    # yasaklamak "kartal çok hızlıdır" gibi geçerli cümleleri düşürüyor.
    # Soru sözcüğü de kavram öbeğine giremez. "kartal ne tür bir hayvandır"
    # cümlesinde kavram `kartal ne tür` diye okunuyordu — grafta öyle bir şey
    # yok ve cümle bildirme sanılıp öğretme dalına düşüyordu. Bir soru
    # sözcüğü, hakkında konuşulan şeyin parçası olamaz.
    for name in ("denials", "postpositions", "correlatives", "joiners",
                 "clitics", "interrogatives", "question_particles"):
        found.update(getattr(morphology, name, ()))
    return found


def _blocked_in_concepts(morphology):
    """Yalnızca KAVRAM ve TÜR yuvasında yasak olanlar.

    Kural kelimenin kendisinde değil YERİNDE: "çok hızlı" geçerli bir
    niteliktir ama "kartal çok çok" bir kavram değildir.
    """
    found = set()
    for name in ("intensifiers",):
        found.update(getattr(morphology, name, ()))
    return found

# Bağlaçlar da bir öbeğin içine giremez: "penguen ile kartal aynı mı"
# cümlesinde kavram `penguen ile kartal` diye okunuyordu — iki kavram tek
# kavram sanılıyor ve soru cevapsız kalıyordu. `değil` ile aynı hata, farklı
# kelime sınıfı. Bağlaç kapalı bir sınıf ve zaten cümlecik bölmede sayılı.

# Concepts, types and properties are routinely more than one word — "müşteri
# bakiyesi", "bilgi türü", "çok hızlı". Verbs and particles never are.
PHRASE_SLOTS = (KAVRAM, TUR, NITELIK, SOZ)
MAX_PHRASE = 3


def _widths(slot):
    """How many tokens to try for a slot, in the order worth trying.

    A term is longer than a word — "müşteri bakiyesi gizlidir" is one thing
    being called one word, not one thing being called two — so a concept
    reaches for the longest span it can and everything else takes the shortest
    that works.

    Uzunluğu ARAMAK ile uzun olanı KABUL ETMEK ayrı işler: ikincisinin denetimi
    `_names_one_thing`'de. Uzun süre denetim yoktu ve bedeli ölçüldü; bkz.
    oradaki not.
    """
    if slot == KAVRAM:
        return range(MAX_PHRASE, 0, -1)
    if slot in PHRASE_SLOTS:
        return range(1, MAX_PHRASE + 1)
    return (1,)


class Pattern:
    def __init__(self, tokens, kind, relation=None, concept=None, target=None,
                 name="", object=None, quantifier=None, fallback=False):
        self.tokens = tokens
        self.kind = kind
        self.relation = relation
        self.concept = concept      # slot index, or None if the sentence omits it
        self.target = target
        self.object = object        # where the object sits — a matter of language
        self.quantifier = quantifier    # slot holding "how much of the kind"
        # Toplu çıkarımla, tek tek gözden geçirilmeden gelen kalıp SON ÇARE
        # sayılır: yalnız hiçbir özenli kalıp tutmadığında denenir.
        #
        # Ölçüldü ve fark büyüktü. 84 çıkarılmış kalıp öne eklenince gerçek
        # cümlelerde okuma %28,7 -> %79,3 çıktı ama sınav doğruluğu %95,7 ->
        # %61,0 düştü, isabet %99,3 -> %92,4. Yani kapsam kazanılırken zaten
        # DOĞRU okunan cümleler kaçırıldı — genel kalıp özel olanın önüne
        # geçti. Kapsam ile doğruluk arasındaki takas gerçek, ama sırayı doğru
        # kurunca takas ortadan kalkıyor: son çare hiçbir şeyi elinden almaz.
        self.fallback = fallback
        self.name = name or " ".join(tokens)

    def to_dict(self):
        return {"tokens": self.tokens, "kind": self.kind,
                "relation": self.relation, "concept": self.concept,
                "target": self.target, "object": self.object,
                "quantifier": self.quantifier, "name": self.name,
                "fallback": self.fallback}

    @staticmethod
    def from_dict(data):
        return Pattern(**data)


class Grammar:
    """An ordered list of patterns and the machinery to match tokens against it.

    Order carries meaning: "kim uçar" has the same shape as "kuşlar uçar", so
    the question has to be tried before the lesson.
    """

    def __init__(self, patterns, morphology, known=(), meanings=None):
        self.patterns = list(patterns)
        self.morphology = morphology    # supplies the language's suffix rules
        # Live, not a snapshot: a concept taught mid-conversation must be
        # recognisable in the very next sentence.
        self._known = known
        # Bir kelimenin başka nasıl söylendiği. Graftan geliyor, listeden değil:
        # eş anlamlılık bir olgudur ve künyesiyle durur. Verilmezse kalıplar
        # yalnız birebir sözcükle eşleşir — eskisi gibi.
        self._meanings = meanings
        # İkinci geçişte açılır: bkz. `match`.
        self._on_faith = False

    def _conditional(self, token, lexicon):
        """Bu kelime bir KOŞUL taşıyor mu — yani cümle bir iddia değil mi.

        "yağmur yağarsa ıslanırsın" cümlesi yağmur hakkında hiçbir şey
        söylemez; iki olay arasında bir bağ kurar. Ayrıştırıcı bunu bildirme
        sanıyor ve grafa olmayan bir olgu yazıyordu:

            kenar: yağmur --can--> ıslanmak (nesne: yağarsa)

        Ek DİLDEN soruluyor (`conditional_suffixes`), burada yazılı değil;
        dilini bildirmeyen bir organ boş küme alır ve bu kapı sessizce kapanır.

        Yanlış pozitif tehlikesi gerçek: "masa", "kasa", "elbise" de aynı
        harflerle biter. Ayrım sayıma bırakılıyor — koşul eki bir FİİLE
        binebilir. `yağarsa` - `sa` = `yağar`, derlem onu fiil biliyor.
        `masa` - `sa` = `ma`, bilmiyor. Bu, bu dosyanın her yerindeki disiplin:
        kural değil tanık karar verir.
        """
        endings = getattr(self.morphology, "conditional_suffixes", ())
        for ending in endings:
            if not token.endswith(ending) or len(token) - len(ending) < 3:
                continue
            stem = token[: -len(ending)]
            if lexicon is not None and lexicon.reading(stem) is not None:
                return True
            from lmm import frequency
            if stem in frequency.verbs():
                return True
        return False

    def known(self):
        """Grafın kavramları, küme olarak.

        Küme her çağrıda yeniden kuruluyordu ve bu işlev cümle başına ~110 kez
        çağrılıyor. Kaynak listenin kimliği değişmediyse küme de değişmemiştir
        — `Session._concepts` aynı listeyi kenar sayısı değişene dek geri
        veriyor, yani `is` karşılaştırması hem doğru hem bedava.

        Kimliğe bakmak eşitliğe bakmaktan ucuz ve burada YETERLİ: liste
        değişmişse yeni bir nesnedir. Bayat kalma riski yok; öğrenilen bir
        kavram bir sonraki cümlede görünüyor.
        """
        source = self._known() if callable(self._known) else self._known
        cached = getattr(self, "_known_cache", None)
        if cached is not None and cached[0] is source:
            return cached[1]
        found = set(source)
        self._known_cache = (source, found)
        return found

    def add(self, pattern, first=False):
        self.patterns.insert(0, pattern) if first else self.patterns.append(pattern)
        return pattern

    def match(self, tokens, lexicon):
        """First pattern that fits, with the slots it captured.

        İki geçiş. Birincisinde çok kelimeli bir kavram ancak TANIKLI ise
        kabul edilir (`_names_one_thing`); ikincisinde iki kelimelik bir ad
        güvene alınır. Sıra bir ilkeyi kuruyor: **uzun okuma son çaredir.**
        Cümle kavramı kısa tutarak okunabiliyorsa öyle okunur — çünkü
        "penguen uçamayan bir kuştur" cümlesinin iki okuması var ve kısası
        penguen hakkında konuşuyor, uzunu ise olmayan bir şey hakkında.
        Cümle başka türlü hiç okunamıyorsa ("tool broker bir güven kapısıdır")
        iki kelime bir ad sayılır: orada kısa okuma diye bir şey yok.
        """
        for on_faith in (False, True):
            self._on_faith = on_faith
            try:
                for pattern in self.patterns:
                    captured = self._fit(pattern, tokens, lexicon)
                    if captured is not None:
                        return pattern, captured
            finally:
                self._on_faith = False
        return None, None

    def _fit(self, pattern, tokens, lexicon):
        """Match slots against tokens, allowing a concept to span several words.

        "müşteri bakiyesi" and "kredi tahsisi" are one thing each, and a domain
        is mostly made of terms like them. Shorter spans are tried first, so a
        sentence that fit before fits the same way now.
        """
        return self._fit_from(pattern.tokens, 0, tokens, 0, lexicon, [])

    def _fit_from(self, slots, slot_at, tokens, token_at, lexicon, captured):
        if slot_at == len(slots):
            return list(captured) if token_at == len(tokens) else None
        slot = slots[slot_at]
        widths = _widths(slot)
        for width in widths:
            if token_at + width > len(tokens):
                continue        # widths are not always ascending — never break
            value = self._capture_span(slot, tokens[token_at:token_at + width],
                                       lexicon)
            if value is None:
                continue
            captured.append(value)
            found = self._fit_from(slots, slot_at + 1, tokens, token_at + width,
                                   lexicon, captured)
            captured.pop()
            if found is not None:
                return found
        return None

    def _capture_span(self, slot, span, lexicon):
        """A slot's value over one or more tokens; suffixes ride the last word."""
        if len(span) == 1:
            return self._capture(slot, span[0], lexicon)
        if slot not in PHRASE_SLOTS:
            return None
        # A case-marked word is doing its own job in the sentence. Letting one
        # into a phrase turned "kartal serçeden büyüktür" into the concept
        # "kartal serçeden" being "büyük", written to memory without a murmur.
        if any(self.morphology.is_oblique(token) for token in span):
            return None
        # Olumsuzluk kelimesi bir öbeğin İÇİNE giremez: "penguen bir kuş değil
        # mi" cümlesinin hedefi "kuş değil" değil, "kuş"tur ve olumsuzluk
        # cümlenin kutbunu çevirir. Yutulduğunda graf "kuş değil" diye bir
        # kavram arıyor ve bulamıyordu — soru sessizce cevapsız kalıyordu.
        blocked = _blocked(self.morphology)
        if any(token in blocked for token in span):
            return None
        # Zamir tek başına özne yerine geçebilir ("o uçar mı") ama bir kavram
        # öbeğinin İÇİNE giremez: "bunu bana anlatır mısın" cümlesinde kavram
        # `bunu bana` diye okunuyordu ve grafta öyle bir şey yok.
        if any(token in getattr(self.morphology, "pronouns", ())
               for token in span):
            return None
        # Pekiştireç yalnızca KAVRAM ve TÜR yuvasında yasak: "çok hızlı"
        # geçerli bir niteliktir ama "kartal çok çok" bir kavram değildir.
        # İlk yazışta her yuvada yasaklamıştım ve geçerli cümleleri
        # düşürüyordu — kural kelimenin kendisinde değil, YERİNDE.
        if slot in (KAVRAM, TUR):
            only_here = _blocked_in_concepts(self.morphology)
            if any(token in only_here for token in span):
                return None
        # Bir kavram öbeğinin içinde FİİL olamaz. "penguen uçar ve uçamaz"
        # cümlesinde kavram `penguen uçar` diye okunup grafa yazılıyordu —
        # cümlenin yüklemi kavramın parçası sanılıyordu. Yazılan saçmalıkların
        # en tehlikelisi: hem kavramı hem olguyu bozuyor.
        if slot in (KAVRAM, TUR) and any(lexicon.knows(token)
                                         for token in span):
            return None
        tail = self._capture(slot, span[-1], lexicon)
        if tail is None:
            return None
        # Buraya kadarki denetimlerin hepsi "bu kelime öbeğe giremez" diyor.
        # Hiçbiri "bu kelimeler BİRLİKTE bir ad mı" diye sormuyordu ve yuva
        # kalan her şeyi tek bir kavram adına çeviriyordu:
        #     "penguen uçamayan bir kuştur" -> penguen uçamayan --tür--> kuş
        # Penguen hakkında hiçbir şey yazılmıyor; üstelik uydurma düğüm gerçek
        # kavramı gölgeliyor ve "penguen uçar mı" sorusu "yoksa penguen
        # uçamayan mı demek istedin" diye dönüyordu. Sistem "öğrendim" diyor,
        # öğrendiği şey çöp.
        if slot in (KAVRAM, TUR) and not self._names_one_thing(span, tail):
            return None
        return " ".join(list(span[:-1]) + [tail])

    def _names_one_thing(self, span, tail):
        """Bu kelimeler tek bir şeyin adı mı, yoksa ad + onu niteleyen mi?

        Varsayılan HAYIR. Bileşik ad ("müşteri bakiyesi", "kara delik") ile
        nitelenmiş ad ("penguen uçamayan") aynı dizilişte durur, o yüzden fark
        tanıklıkla kapanır — tahminle değil. Yanlış bir şey öğrenmektense hiç
        öğrenmemek yeğ: tanık yoksa öbek dağılır, kavram tek kelimeye iner ve
        niteleme yüklem tarafında kalır.

        Dört tanık, ucuzdan pahalıya — graf, biçimbilim, derlem, ve son çare.
        """
        words = list(span[:-1]) + [tail]
        known = self.known()
        joined = " ".join(words)
        # 1. Graf. Bir kez öğrenilmiş terim ikinci cümlede tartışılmaz.
        if joined in known or self.morphology.strip_plural(joined) in known:
            return True
        # 2. Biçimbilim. Dil bileşik adın başını EKLE işaretliyorsa mesele yok:
        # Türkçe'de "bakiye-si", "tür-ü", "cism-i". `getattr` ile soruluyor
        # çünkü her dilde böyle bir ek yok — olmayan dilde bu tanık susar,
        # dilbilgisi motoru hangi dile baktığını bilmemeye devam eder.
        if self._compound_head(tail) or self._compound_head(span[-1]):
            return True
        # 3. Derlem. Ekle işaretlenmeyen bileşikler ("kara delik") ancak
        # sayımla ayrılır ve ölçüm de oradan geldi: nitelenmiş adın niteleyeni
        # ya derecelenerek geçiyor (sıfat), ya sıfat-fiil, ya da ilk üç yüz
        # kelimede (yapısal). Bileşiğin her parçası ise düpedüz isim.
        if self._witnessed(words):
            return True
        # 4. Son çare, yalnız ikinci geçişte ve yalnız İKİ kelime için: cümle
        # başka hiçbir türlü okunamıyorsa iki kelime bir ad sayılır. Üç kelime
        # burada durmuyor çünkü ölçüldü — tanıksız üç kelimelik yuvalar bileşik
        # ad değil, niteleyici zinciriydi: "devekuşu kırmızı olmayan".
        return self._on_faith and len(words) == 2

    def _compound_head(self, word):
        """Dilin kendi işareti: bu kelime bir tamlamanın başı mı?"""
        marks = getattr(self.morphology, "possessive_suffixes", ())
        return any(word.endswith(mark) and len(word) > len(mark) + 1
                   for mark in marks)

    def _witnessed(self, words):
        """Derlem bu kelimeleri bir arada tek ad olarak taşıyor mu?

        Testler `lmm/verbs.py`'de ve sayımdan geliyor — burada dile ait tek bir
        sözcük yok. Derlem yoksa tanık susar ve cevap HAYIR olur: körken uzun
        öbek uydurmaktansa kısa kavramda kalmak.
        """
        from lmm import frequency
        from lmm.verbs import (is_adjective, is_noun, is_participle,
                               is_structural)
        seen = frequency.counts()
        if not seen:
            return False
        graded, verbs = frequency.grades(), frequency.verbs()
        for word in words:
            if is_participle(word, verbs) or is_structural(word, seen) \
                    or is_adjective(word, graded, seen):
                return False
        return is_noun(words[-1], seen)

    def _bare(self, token):
        """Koşaç eki soyulmuş hâli — ek yoksa kelimenin kendisi.

        "mu" ile "mudur", "neler" ile "nelerdir" aynı şeyi sorar. Liste
        uzatmak yerine ek soyuluyor: koşaç ekleri zaten türetilmiş ve soru
        sözcükleri kapalı sınıf olarak duruyor.
        """
        morphology = self.morphology
        return (morphology.strip_copula(token) if morphology.has_copula(token)
                else token)

    def _peel(self, token):
        """Ek soyulunca bilinen bir kavram çıkıyorsa o kavram, yoksa None."""
        from lmm.frames import case_of
        known = self.known()
        morphology = self.morphology
        seen, edge = {token}, [token]
        for _ in range(4):
            following = []
            for form in edge:
                # İyelik eki de soyuluyor: "penguenin özellikleri neler"
                # sorusunda kavram `penguenin` diye kalıyordu ve grafta
                # bulunamıyordu. Biçimbilim onu zaten çözebiliyordu; yuva
                # sormuyordu.
                # Biçimbilim dile göre değişir ve her dilde iyelik eki
                # olmayabilir: makine dile özgü yöntem VARSAYMAMALI. İlk
                # yazışta varsaymıştım ve İngilizce dilbilgisi testleri
                # düştü — bu projenin dilden bağımsızlık iddiası tam da
                # orada sınanıyor.
                genitive = getattr(morphology, "strip_genitive", None)
                for candidate in (morphology.strip_plural(form),
                                  genitive(form, known) if genitive else None,
                                  case_of(form)[0]):
                    if not candidate or candidate in seen:
                        continue
                    if candidate in known:
                        return candidate
                    seen.add(candidate)
                    following.append(candidate)
            if not following:
                return None
            edge = following
        return None

    def _asks(self, token):
        """Bu kelime, hangi çekimde olursa olsun, soru soruyor mu?

        Çıplak ek, koşaçlı hâl ve çekimli hâl — üçü de. Çekimli hâller
        derlemden keşfedildi; bkz. `lmm/asking.py`.
        """
        from lmm import asking
        return asking.asks(token, self.morphology)

    def _capture(self, slot, token, lexicon):
        """What this token means in this slot, or None if it does not fit."""
        morphology = self.morphology
        # Tek kelimelik yapısal sözcük de bir kavram olamaz. Öbek denetimi
        # yalnızca ÇOK kelimeli yuvalarda çalışıyordu ve "kartal da uçar mı"
        # cümlesinde "da" tek başına kavram yerine geçiyordu.
        if slot in PHRASE_SLOTS and slot != SOZ:
            if token in _blocked(morphology):
                return None

        if slot not in SLOTS:
            if token == slot:
                return slot
            # Dilin kendi bildirdiği eşdeğerler: "neden" kalıbı "niçin"i de
            # eşleştirir. Aynı soru için ikinci bir kalıp yazmak, kalıp
            # sayısını şişirmekten başka bir şey yapmıyordu.
            for group in getattr(morphology, "groups", ()):
                if slot in group and token in group:
                    return slot
            # "bahseder misin" ile "anlat" aynı şeyi istiyor. Bunu bilmek için
            # kalıp eklemek gerekmiyor; grafta "anlat = bahset" yazması yeterli.
            #
            # Bakılan yön önemli: KALIPTAKİ sözcüğün eş anlamlılarına bakılıyor,
            # gelen kelimeninkine değil. Tersi yazılmıştı ve çalışmıyordu —
            # gelen kelimenin çekimli hâli grafta hiç durmuyor, gövdesi duruyor.
            # Çekim gövdeye önek olarak uyuyor: "bahseder" -> "bahset".
            for other in (self._meanings(slot) if self._meanings else ()):
                if _starts_with(token, other):
                    return slot
            # Soru sözcüğü eş anlamlıları: "niçin" ile "neden" aynı şeyi
            # sorar. Kalıp çoğaltmak yerine graftan okunuyor — eş anlamlılık
            # bir olgu ve yeri graf.
            if self._meanings and token in self._meanings(slot):
                return slot
            return None
        if slot in (KAVRAM, TUR, NITELIK, NESNEL) and self._conditional(token,
                                                                       lexicon):
            return None
        if slot == KAVRAM:
            # A closed-class word is never the thing being talked about.
            # "bazı kuşlar uçmaz" was read as the concept "bazı" doing something.
            if token in getattr(morphology, "quantifiers", ()):
                return None
            if self._asks(token):
                return None
            plain = morphology.strip_plural(token)
            known = self.known()
            if plain in known:
                return plain
            # Durum eki taşıyan bir kavram da kavramdır: "bana PENGUENLERDEN
            # bahseder misin" cümlesinin konusu penguendir. Ek soyulmadığında
            # öğrenilen kalıp eşleşiyor ama grafta karşılığı bulunamıyordu —
            # kalıp çalışıyor sanılıyordu, oysa cevap başka yerden geliyordu.
            # Yalnızca soyulmuş hâli GERÇEKTEN bilinen bir kavramsa yapılıyor;
            # yoksa ek, kelimenin parçası olabilir ("banka" / "bank'a").
            peeled = self._peel(token)
            if peeled is not None:
                return peeled
            # Çoğul EKİ ile çoğul ADI ayrı şeyler. "kuşlar" bir ekten ibaret,
            # ama "maldivler", "bahamalar", "etçiller", "çift çenekliler"
            # kendileri çoğuldur ve tekilleri yoktur. Ek koşulsuz soyulunca
            # grafta duran olgu ulaşılmaz oluyordu:
            #
            #   grafta: alkinler --type--> bileşik
            #   > alkinler nedir   -> "bunu bilmiyorum ... alkin nedir?"
            #
            # Ölçüldü: 16.774 kavramın 316'sı (%1,9) böyle çoğul adlı ve
            # hepsi sessizce erişilemezdi. Sohbet sınavı bunu takip turunda
            # yakaladı — tek soruluk ölçüt göremiyordu.
            #
            # Karar kurala değil GRAFA bırakılıyor: soyulmuş hâli bilinmiyor
            # ama kelimenin kendisi biliniyorsa, ek kelimenin parçasıdır.
            if token in known:
                return token
            return plain
        if slot == SOZ:
            return token
        if slot == TUR:
            if self._asks(token):
                return None
            return morphology.strip_copula(token)
        if slot == NITELIK:
            if not morphology.has_copula(token):
                return None
            # Soru eki koşaç alınca nitelik olmaz. "mudur" koşaç taşıdığı için
            # bu yuvaya giriyordu ve "penguen mutlu mudur" cümlesi
            # [KAVRAM=penguen mutlu, NITELIK=mu] diye okunup TEACH sayılıyordu:
            # soru, grafa `sence penguenler mutlu —property→ mu` diye
            # yazılıyordu. Cevap uydurulmuyordu ama hafıza kirleniyordu — ki
            # hafıza bu projenin tek varlığı.
            if self._asks(token):
                return None
            return morphology.strip_copula(token)
        if slot == FIIL:
            reading = lexicon.reading(token)
            if reading is None:
                # Sözlük yalnız ÖĞRETİLEN fiilleri bilir. Derlemde milyonlarca
                # kez geçen `çalışır` orada yoksa bu yuva boş dönüyor, cümle
                # nitelik kalıbına düşüyor ve `insan çalışır mı` sorusu
                # `insan --property--> çalışır` diye okunuyordu. Graf `insan
                # --can--> çalışmak` kaydını tutarken cevap "bilmiyorum"du:
                # bilinen bir şeyi bilmiyor demek, kapının en pahalı hatası.
                #
                # Derlem üçüncü tanık ve yalnız TANIKLIK ediyor: kural
                # üretmiyor, sayımla çıkarılmış bir eşlemeye bakıyor. Sözlükten
                # sonra sorulur — öğretilmiş bilgi gözlemi ezer.
                from lmm import frequency
                reading = frequency.verbs().get(token)
            return reading if reading is not None else None
        if slot == SORU:
            from lmm import asking
            return asking.particle_of(token, morphology)
        if slot == KIM:
            from lmm import asking
            return asking.interrogative_of(token, morphology)
        if slot == NESNEL:
            # Only strip an accusative when what is left is a concept we know:
            # "kedi" ends in the same letter and is not "ked".
            stem = morphology.strip_accusative(token)
            if stem != token and stem in self.known():
                return stem
            return token if token in self.known() else None
        if slot == SAHIP:
            stem = morphology.strip_genitive(token, self.known())
            return stem if stem != token else None
        if slot == NICEL:
            return morphology.quantifiers.get(token)
        if slot == ROL:
            stem, role = morphology.role_of(token)
            return (stem, role) if role else None
        return None

    def read(self, pattern, captured):
        """Turn a match into (kind, relation, concept, target, object)."""
        concept = self._slot_value(pattern, captured, pattern.concept)
        target = self._slot_value(pattern, captured, pattern.target)
        obj, role = self._object_and_role(pattern, captured)
        relation = pattern.relation
        if relation == FROM_VERB:
            relation = CAN if self._polarity(pattern, captured) else CANNOT
        return (pattern.kind, relation, concept, target, obj, role,
                self._slot_value(pattern, captured, pattern.quantifier) or ALL)

    def _object_and_role(self, pattern, captured):
        """The second concept and what it is doing, when a pattern captured one."""
        if pattern.object is None:
            return None, None
        value = captured[pattern.object]
        if isinstance(value, tuple):
            return value[0], value[1]
        return value, OBJECT

    def _slot_value(self, pattern, captured, index):
        if index is None:
            return None
        value = captured[index]
        return value[0] if isinstance(value, tuple) else value

    @staticmethod
    def _polarity(pattern, captured):
        for slot, value in zip(pattern.tokens, captured):
            if slot == FIIL:
                return value[1]
        return True


def learn_pattern(tokens, kind, relation, concept_index, target_index,
                  lexicon, morphology):
    """Work out a reusable pattern from one labelled sentence.

    Given "kediler zıplar" labelled as a lesson about ability, this decides
    which words were the example and which were the shape, and returns a pattern
    that will match the next sentence built the same way.
    """
    slots = []
    for index, token in enumerate(tokens):
        if index in (concept_index, target_index):
            slots.append(_slot_for(token, index, target_index, relation,
                                   lexicon, morphology))
        elif token in morphology.question_particles:
            slots.append(SORU)
        elif token in morphology.interrogatives:
            slots.append(KIM)
        elif lexicon.knows(token):
            slots.append(FIIL)
        else:
            slots.append(token)         # a function word: part of the shape
    return Pattern(slots, kind, relation, concept_index, target_index)


def induce(pairs, parser, morphology, lexicon, max_length=6, evidence=2):
    """Work patterns out of prose, with nobody labelling anything.

    Each pair holds a passage a person wrote and the plain sentences a model
    restated it as. We already know what the plain sentences mean, because our
    own parser reads them — so every pair is a labelled example that no human
    had to label.

    Alignment is by content: the prose sentence that mentions both the concept
    and the target is the one that said it. Sentences longer than a handful of
    words are left alone, since a pattern taken from a twenty-word sentence
    would only ever match that sentence again.

    A shape has to turn up more than once before it is believed. One occurrence
    is a coincidence; the same shape twice is a rule — the same standard we hold
    facts to.
    """
    seen = {}
    for pair in pairs:
        prose = _sentences(pair.get("düzyazı", ""))
        for plain in pair.get("sade", "").splitlines():
            intent = parser(plain)
            if intent.concept is None or intent.target is None:
                continue
            for sentence in prose:
                tokens = _tokens(sentence, morphology)
                if not 2 < len(tokens) <= max_length:
                    continue
                concept_at = _where(tokens, intent.concept, morphology)
                target_at = _where(tokens, intent.target, morphology)
                if concept_at is None or target_at is None or concept_at == target_at:
                    continue
                pattern = learn_pattern(tokens, intent.kind, intent.relation,
                                        concept_at, target_at, lexicon, morphology)
                key = (tuple(pattern.tokens), pattern.kind, pattern.relation,
                       pattern.concept, pattern.target)
                seen[key] = seen.get(key, 0) + 1
                break
    return [Pattern(list(tokens), kind, relation, concept, target)
            for (tokens, kind, relation, concept, target), count in seen.items()
            if count >= evidence]


def _sentences(text):
    found, current = [], ""
    for character in text:
        if character in ".!?\n;":
            if current.strip():
                found.append(current.strip())
            current = ""
        else:
            current += character
    if current.strip():
        found.append(current.strip())
    return found


def _tokens(sentence, morphology):
    cleaned = "".join(c for c in sentence if c not in ".,!?;:\"'")
    return cleaned.replace("İ", "i").replace("I", "ı").lower().split()


def _where(tokens, word, morphology):
    """Which token carried this concept, allowing for the suffixes it wears."""
    for index, token in enumerate(tokens):
        stem = morphology.strip_copula(morphology.strip_plural(token))
        if stem == word or token == word or token.startswith(word):
            return index
    return None


def _slot_for(token, index, target_index, relation, lexicon, morphology):
    if lexicon.knows(token):
        return FIIL
    if index == target_index and relation in (IS_A, NOT_A):
        return TUR
    if index == target_index and relation in (HAS_PROPERTY, LACKS_PROPERTY):
        return NITELIK if morphology.has_copula(token) else SOZ
    return KAVRAM
