"""Language Intuition Core: sentence -> Intent.

Carries no knowledge of the world, and no knowledge of Turkish either. It
tokenises, hands the tokens to a grammar, and turns whatever matched into an
intent. The patterns are a list and the morphology is a set of suffix rules —
both data, both replaceable, neither written into this file.

That is what makes a second language a second data module rather than a second
parser, and what lets the system be taught a new way of saying something while
it is running.
"""
from lmm.lexicon import ACTIVE
from lmm.relations import ALL
from lmm.language import LanguageOrgan
from lmm.turkish import (turkish, TurkishMorphology, TEACH, ASK, ASK_WHO,  # noqa: F401
                         ASK_ABILITIES, ASK_WHY, ASK_PROPERTIES, ASK_DESCRIBE,
                         ASK_HOW_MANY, ASK_WHERE, ASK_COMPARE,
                         ASK_THREAD, ASK_WHICH_MORE, ASK_REQUIREMENT,
                         ASK_MORE, ASK_INVENTORY, ASK_CERTAINTY,
                         ASK_SOURCE, ASK_OPINION)

PUNCTUATION = ".,!?;:\"'"

UNKNOWN = "UNKNOWN"
UNKNOWN_WORD = "UNKNOWN_WORD"
AMBIGUOUS = "AMBIGUOUS"

_MORPHOLOGY = TurkishMorphology()
QUESTION_PARTICLES = _MORPHOLOGY.question_particles
COPULA_SUFFIXES = _MORPHOLOGY.copula_suffixes
PLURAL_SUFFIXES = _MORPHOLOGY.plural_suffixes
INTERROGATIVES = _MORPHOLOGY.interrogatives
PRONOUNS = _MORPHOLOGY.pronouns


def lower(text):
    """Turkish lowercasing.

    Python maps "İ" to "i" plus a combining dot, not to "i", so "İnsanlar"
    silently becomes a different word from "insan". Every sentence starting
    with İ was quietly learned under a concept nobody could ever ask about.
    """
    return text.replace("İ", "i").replace("I", "ı").lower()


def tokenize(sentence):
    cleaned = "".join(c for c in sentence if c not in PUNCTUATION)
    return lower(cleaned).split()


def _has_suffix(word, suffixes):
    return any(word.endswith(s) and len(word) > len(s) + 1 for s in suffixes)


def _strip_suffix(word, suffixes):
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) > len(suffix) + 1:
            return word[: -len(suffix)]
    return word


class Intent:
    def __init__(self, kind, concept=None, relation=None, target=None,
                 object=None, role=None, quantifier=ALL, confidence=1.0):
        self.kind = kind
        self.concept = concept
        self.relation = relation
        self.target = target
        self.object = object        # a second concept, if the sentence had one
        self.role = role            # what that concept is doing there
        self.quantifier = quantifier    # how much of the kind is meant
        self.confidence = confidence
        self.resembles = None       # shape the network saw, when nothing matched
        self.resemblance = 0.0


def _peeled(word):
    """Kelimenin çekim katmanları soyulmuş hâlleri, en azdan çoğa."""
    found = [word]
    for ending in ("sınız", "siniz", "sunuz", "sünüz", "sın", "sin", "sun",
                   "sün", "ım", "im", "um", "üm", "ız", "iz", "uz", "üz",
                   "lar", "ler", "dır", "dir", "dur", "dür"):
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            found.append(word[: -len(ending)])
    return found


def _a_known_verb(word):
    """Derlem bu kelimeyi bir fiil çekimi olarak tanıyor mu.

    "Öğretilmemiş" ile "Türkçe'de olmayan" ayrı şeyler. Oturumun sözlüğü yalnız
    öğretilmiş fiilleri bilir ve doğru olan budur — o, sistemin öğrendiği
    dağarcık. Ama "bu kelime bir fiil mi" sorusu daha geniş ve cevabını derlem
    veriyor: 807 fiil sayımla çıkarılmış.

    Tablo yoksa False döner ve sistem eski davranışına düşer — kolaylık,
    bağımlılık değil.
    """
    from lmm import frequency
    known = frequency.verbs()
    if not known:
        return False
    if word in known:
        return True
    # Çekim katmanları: kişi eki, çoğul, koşaç dışta durur.
    for ending in ("sınız", "siniz", "sunuz", "sünüz", "sın", "sin", "sun",
                   "sün", "ım", "im", "um", "üm", "ız", "iz", "uz", "üz",
                   "lar", "ler", "dır", "dir", "dur", "dür"):
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            if word[: -len(ending)] in known:
                return True
    return False


class Intuition(LanguageOrgan):
    def __init__(self, network=None, lexicon=None, grammar=None, words=None,
                 known=(), memory=None, meanings=None):
        self.network = network      # MiniNetwork supplies the resemblance hint
        # Bir bellek verilirse, rakip okumalar arasında seçim ona göre yapılır:
        # en az yeni varsayım gerektiren okuma kazanır. Verilmezse eski davranış
        # sürer — ilk uyan kalıp.
        self.memory = memory
        self.lexicon = lexicon or ACTIVE
        self.grammar = grammar or turkish(words, known, meanings)

    def _without_clitics(self, tokens, morphology):
        """Pekiştirme parçacıkları atılmış hâller — en azdan çoğa.

        Ayrı yazılan "da"/"de" Türkçe'de her zaman pekiştirmedir; ek olduğunda
        bitişik yazılır ("kartalda" bulunma hâli, "kartal da" pekiştirme).
        Yazım bu ayrımı zaten yapıyor, biz yalnızca okuyoruz.

        Muhatap zamirleri de aynı kapıdan geçiyor: "bana"/"bize" bir soruda
        cümlenin konusu değil, kime söylendiğidir — ve LMM her zaman muhatap.
        "bunu bana anlatır mısın" ile "bunu anlatır mısın" aynı sorudur.

        Sırayla veriliyor: önce hiç atmadan, sonra atarak. Böylece parçacığın
        gerçekten kalıbın parçası olduğu bir cümle bozulmuyor.
        """
        atilabilir = set(getattr(morphology, "clitics", ("da", "de")))
        atilabilir |= set(getattr(morphology, "addressees", ()))
        if not any(token in atilabilir for token in tokens):
            return
        kept = [token for token in tokens if token not in atilabilir]
        if len(kept) >= 2:
            yield kept

    def _without_filler(self, tokens, morphology):
        """Son kelime söylem parçacığıysa onsuz hâli, değilse None.

        Ölçüt kelimenin kimliği değil taşıdığı yük: fiil değilse, bilinen bir
        kavram değilse, ek taşımıyorsa ve cümlede en az iki kelime kalıyorsa
        atılabilir. Yanlış atarsak zaten kalıp eşleşmez ve hiçbir şey
        kazanmayız — bu yüzden kapı açık bırakmıyor.
        """
        if len(tokens) < 3:
            return None
        last = tokens[-1]
        if self.lexicon.knows(last):
            return None
        if last in self.grammar.known():
            return None
        if morphology.has_copula(last) or morphology.is_oblique(last):
            return None
        from lmm import asking
        if asking.asks(last, morphology):
            return None
        return tokens[:-1]

    def _learn_from_corpus(self, word):
        """Derlemin bildiği bir fiili oturumun sözlüğüne alır.

        Derlem bir kaynak gibi davranıyor: fiil keşfi ölçülmüş bir yöntemle
        (olumlu/olumsuz çift testi) çıkarılmış ve künyesi belli. Öğrenilen şey
        fiilin ANLAMI değil, fiil OLDUĞU — cümlenin yapısını çözmek için
        gereken tam olarak bu.
        """
        from lmm import frequency
        known = frequency.verbs()
        if not known:
            return False
        for candidate in _peeled(word):
            found = known.get(candidate)
            if found is None:
                continue
            infinitive = found[0]
            positive = negative = None
            for surface, (name, is_positive) in known.items():
                if name != infinitive:
                    continue
                if is_positive and positive is None:
                    positive = surface
                elif not is_positive and negative is None:
                    negative = surface
            if positive and negative:
                self.lexicon.learn_verb(infinitive, positive, negative)
                return True
        return False

    def _unknown_verb(self, word, morphology):
        """Whether a final word is a verb we were never taught.

        "Öğretilmemiş" ile "Türkçe'de olmayan" ayrı şeyler. Oturumun sözlüğü
        yalnız öğretilmiş fiilleri bilir (98 tane) ve doğru olan budur — o,
        sistemin öğrendiği dağarcık. Ama "bu kelime bir fiil mi" sorusu daha
        geniş ve cevabını derlem veriyor: 807 fiil sayımla çıkarılmış.

        Ayrım olmadan sistem "penguen konusunda ne BİLİYORSUN" cümlesine
        "'biliyorsun' kelimesini bilmiyorum" diyordu — oysa o kelime Türkçe'de
        gayet var, yalnız bu grafta öğretilmemiş. Cümlenin yapısını anlamak
        için fiilin ne olduğunu bilmek yetiyor; ne anlama geldiğini bilmek
        gerekmiyor.
        """
        if self.lexicon.knows(word) or morphology.has_copula(word):
            return False
        # Derlem bu kelimeyi fiil olarak tanıyorsa SÖZLÜĞE ÖĞRET ve öyle
        # devam et. "Bilmiyorum, öğret bana" demek doğruydu ama artık gereksiz:
        # derlem 807 fiili sayımla çıkarmış ve öğrenmek bedava.
        #
        # Yalnızca "bu bir fiil" demek yetmiyordu — denendi ve daha kötü oldu:
        # sistem "öğret bana" demeyi bıraktı ama cümleyi de anlayamadı, çünkü
        # kalıplar sözlüğe bakıyor. Bilgiyi kullanmak, bilgiye sahip olmaktan
        # ayrı bir iş.
        if self._learn_from_corpus(word):
            return False        # artık biliyoruz; çağıran cümleyi yeniden okur
        if word in morphology.question_particles or word in morphology.openers:
            return False
        if word in getattr(morphology, "interrogatives", ()):
            return False
        return word not in self.grammar.known()

    def understand(self, sentence):
        """A matched pattern is the evidence; the network speaks when none matched.

        The grammar is deterministic, so overriding a successful match with a
        network score only ever invents doubt — an earlier version refused a
        perfectly good sentence because its subject was a word it had never met.
        """
        tokens = tokenize(sentence)
        morphology = self.grammar.morphology
        while tokens and tokens[0] in morphology.openers:
            tokens = tokens[1:]         # "peki uçar mı" is "uçar mı"
        # Açıcı SONDA da durabilir: "nasıldır peki", "ne yapar peki". Yalnız
        # baştan soyulduğunda o cümleler okunamıyordu ve sistem `peki`yi bir
        # nitelik sanıyordu — "'peki' diye bir niteliği hiç duymadım".
        # Ölçüldü: uzun sohbet sınavında eksiltili takiplerin üçte biri
        # buradan kayboluyordu.
        while len(tokens) > 1 and tokens[-1] in morphology.openers:
            tokens = tokens[:-1]
        # Söylem araları çok kelimeli olabilir ("bu arada"), o yüzden tek tek
        # kelime soymak onları yakalayamıyordu — `bu` özne sanılıyordu.
        for aside in getattr(morphology, "asides", ()):
            parts = aside.split()
            if tokens[:len(parts)] == parts and len(tokens) > len(parts):
                tokens = tokens[len(parts):]
        # Muhatap zamiri de baştan ayıklanıyor: bir soruda "bana"/"bize"
        # cümlenin konusu değil, kime söylendiğidir — ve LMM her zaman
        # muhatap. Sonradan atmak yetmiyordu, çünkü yanlış bir okuma önce
        # başarılı oluyor ve atma yoluna hiç gelinmiyordu.
        addressees = getattr(morphology, "addressees", ())
        if addressees and len(tokens) > 2:
            kept = [token for token in tokens if token not in addressees]
            if len(kept) >= 2:
                tokens = kept

        # İlk uyan kalıp yerine EN UCUZ okuma. Bir cümleye birden çok kalıp
        # uyabilir ve bugüne kadar aralarında seçim yapan bir şey yoktu; ilki
        # kazanıyordu. Bildiklerimize en az aykırı düşen okumayı seçmek,
        # Hobbs'un 1988'de "yorumlama kaçınımdır" diye adlandırdığı şey.
        if self.memory is not None:
            from lmm.abduction import best
            reading = best(self.grammar, tokens, self.lexicon, self.memory)
            if reading is not None:
                (kind, relation, concept, target, obj, role,
                 quantifier) = reading.parts
                return Intent(kind, concept, relation, target, obj, role,
                              quantifier)
        pattern, captured = self.grammar.match(tokens, self.lexicon)
        if pattern is not None:
            (kind, relation, concept, target, obj, role,
             quantifier) = self.grammar.read(pattern, captured)
            return Intent(kind, concept, relation, target, obj, role, quantifier)

        # A subject followed by something that is neither a known verb nor a
        # property is almost certainly a verb nobody taught us. Saying so is
        # more useful than shrugging, and it is how vocabulary grows.
        #
        # This used to fire only on two-word sentences, which meant "penguenler
        # suya girer" and "kuşlar sabah öter" got a shrug instead of the name of
        # the one word standing in the way. Whatever else a Turkish sentence
        # carries, its verb is last, so the test belongs on the last token
        # regardless of how many complements come before it. It must not fire on
        # a predicate that is a *property* — "kar çok beyazdır" ends in a copula,
        # not a verb — nor on a question, which is a different failure.
        # Söylem parçacığı: atılınca cümlenin yapısını bozmayan son kelime.
        # "penguen bir kuş değil mi YANİ" cümlesindeki "yani" hiçbir şey
        # eklemiyor ama ayrıştırıcı onu bilinmeyen bir fiil sanıp cümleyi
        # düşürüyordu.
        #
        # Liste yazılmıyor — sınanıyor: kelime atılınca cümle okunuyorsa ve
        # kelime ne fiil ne kavram ne de ek taşıyorsa, taşıdığı şey yapı değil
        # söylemdir. Bu, "peki"/"ya" için zaten yapılanın cümle SONUNA
        # uygulanmış hâli.
        # Pekiştirme parçacığı cümlenin ORTASINDA da durabilir: "kartal DA mı
        # uçuyor". Türkçe'de "da"/"de" ayrı bir kelime olarak yazıldığında her
        # zaman pekiştirmedir — ek olduğunda kelimeye bitişik yazılır. Yani
        # ayrı duran bir "da" hiçbir zaman içerik taşımaz ve atılması güvenli.
        for trimmed in self._without_clitics(tokens, morphology):
            pattern, captured = self.grammar.match(trimmed, self.lexicon)
            if pattern is not None:
                (kind, relation, concept, target, obj, role,
                 quantifier) = self.grammar.read(pattern, captured)
                return Intent(kind, concept, relation, target, obj, role,
                              quantifier)

        trimmed = self._without_filler(tokens, morphology)
        if trimmed is not None:
            pattern, captured = self.grammar.match(trimmed, self.lexicon)
            if pattern is not None:
                (kind, relation, concept, target, obj, role,
                 quantifier) = self.grammar.read(pattern, captured)
                return Intent(kind, concept, relation, target, obj, role,
                              quantifier)

        # Two words leave nothing else the stranger could be, so it is named
        # whatever the subject is. Past two words the sentence has room for
        # genuine nonsense, so the subject has to be something we know before
        # its last word is treated as a missing verb rather than as noise.
        subject = morphology.strip_plural(tokens[0]) if tokens else ""
        long_enough = len(tokens) == 2 or (len(tokens) > 2
                                           and subject in self.grammar.known())
        if long_enough and tokens:
            before = len(self.lexicon.verbs)
            unknown = self._unknown_verb(tokens[-1], morphology)
            if len(self.lexicon.verbs) > before:
                # Derlemden fiil öğrenildi: sözlük büyüdü, kalıplar artık
                # uyabilir. Öğrenip denememek, öğrenmemekle aynı kapıya çıkar.
                pattern, captured = self.grammar.match(tokens, self.lexicon)
                if pattern is not None:
                    (kind, relation, concept, target, obj, role,
                     quantifier) = self.grammar.read(pattern, captured)
                    return Intent(kind, concept, relation, target, obj, role,
                                  quantifier)
            if unknown:
                return Intent(UNKNOWN_WORD, concept=subject,
                              target=tokens[-1])

        # A possessor that splits two ways and matches nothing known is a real
        # ambiguity of the language, not a failure to parse. Saying which two
        # readings they are lets one sentence settle it.
        readings = getattr(morphology, "genitive_readings", lambda w: [])(tokens[0]) \
            if tokens else []
        if len(readings) > 1 and not (set(readings) & self.grammar.known()):
            intent = Intent(AMBIGUOUS, concept=tokens[0])
            intent.readings = readings
            return intent

        intent = Intent(UNKNOWN, confidence=0.0)
        if self.network is not None and tokens:
            intent.resembles, intent.resemblance = self.network.predict(
                tokens, self.lexicon)
        return intent
