"""LMM v0 chat interface. Usage: python3 -m lmm.cli [memory_file]"""
import sys

from lmm.memory import Memory
from lmm.relations import HAS_PROPERTY, IS_A, SAME_AS
from lmm import lexicon
from lmm.reasoning import Reasoning
from lmm.gate import EpistemicGate
from lmm.intuition import (Intuition, Intent, TEACH, ASK, ASK_WHO, UNKNOWN,
                           UNKNOWN_WORD, AMBIGUOUS, PRONOUNS, ASK_THREAD,
                           ASK_WHY, ASK_PROPERTIES, ASK_MORE,
                           ASK_INVENTORY, ASK_CERTAINTY, ASK_SOURCE,
                           ASK_OPINION, ASK_DESCRIBE, ASK_COMPARE, lower,
                           tokenize)
from lmm import social
from lmm import coordination
from lmm.thread import Thread
from lmm import frames
from lmm import wording
try:
    from core.intent import reading_of
except Exception:                                           # noqa: BLE001
    def reading_of(name, has_target=True):   # torch yoksa: kalıplarla çalışır
        return None
from lmm.distill import split_words
from lmm.verbs import is_structural
from lmm import frequency
from lmm.grammar import Pattern
from lmm.discovered import words_of
from lmm import arithmetic
from lmm.learning import (LearningLoop, CONFLICT, LEARNED, CORRECTED,
                          DISPUTE, FROZEN)
from lmm.induction import Induction
from lmm.network import MiniNetwork
from lmm.curiosity import Curiosity
from lmm.pursuit import Pursuit
from lmm import phrasing
from lmm.phrasing import (forgotten, wondering, not_understood, now_i_can, generalising,
                          describe, teach_me_the_word, learned_word, computed,
                          cannot_compute, which_reading, went_and_read,
                          learned_wording, is_a_refusal, talked_about,
                          nothing_more, sentences, inventory,
                          certainty, whence, no_opinion,
                          exception_learned, not_learned, help_text, status,
                          opened, inquiry_open, intent_network, wording_open,
                          dictionary_open, voice_state, help_hint,
                          saved_and_gone)

# A custom LanguageOrgan may report graded confidence; ours parses or does not.
CONFIDENCE_THRESHOLD = 0.35
RESEMBLANCE = 0.5       # below this, guessing at what you meant is noise
# Ağın tahmini bu güvenin altındaysa hiç denenmiyor.
#
# Eşik 0,4'tü ve gerekçesiz bir sabitti. Ölçüldü: "penguen konusunda ne
# biliyorsun" cümlesinde ağ %37 güvenle DOĞRU okumayı veriyor ve eşikten
# kılpayı düşüyordu. Oysa güven, doğruluğun ölçüsü değil — DENETİM o.
#
# Düşük güvenli tahmin zaten iki kapıdan geçiyor: okuma grafta cevaplanacak,
# ve okuma cümleye uyacak. Geçemeyen atılıyor. Eşiği düşürmek yanlış cevabı
# değil, yalnız deneme sayısını artırıyor.
NEURAL_THRESHOLD = 0.15
# Çıplak soru sözcüğü hangi soruyu sürdürüyor. Kapalı bir eşleme: sözcükler
# zaten bildirilmiş kapalı sınıftan, buradaki yalnızca hangi niyete
# karşılık geldikleri.
FOLLOW_UPS = {"neden": ASK_WHY, "niye": ASK_WHY, "niçin": ASK_WHY,
              "nasıl": ASK_PROPERTIES, "kim": ASK_WHO, "kimler": ASK_WHO}
# Sözcük listeleri artık `lmm/turkish.py`'de: çıktı değil GİRDİ oldukları için
# yerleri `phrasing.py` değil dilin tanımı. Buradaki kopyalar UYUŞMUYORDU da —
# `openers` burada altı sözcüktü, dilde sekiz; "acaba kartal", "hem kartal" ve
# "işte kartal" eksiltili takipleri sessizce anlaşılmıyordu.
# Öznesi olmayan niyetler. Bunlar bağlamdan özne almazlar ve almadıkları için
# "anlaşılmadı" sayılmamalılar — sohbetin kendisi hakkında bir sorunun öznesi
# yoktur. ASK_THREAD burada olmadığı için cevap üretiliyor ama sessizce
# "anlamadım"a çevriliyordu.
SUBJECTLESS = (ASK_WHO, UNKNOWN, UNKNOWN_WORD, AMBIGUOUS, ASK_THREAD,
               ASK_MORE, ASK_INVENTORY, ASK_CERTAINTY, ASK_SOURCE)


_HELD_VECTORS = {}


def _vectors(path="models/gomme.json"):
    """Kayıtlı gömme vektörleri — yoksa None ve sistem onsuz çalışır.

    SÜREÇ BAŞINA bir kez yükleniyor. İlk yazışta her `Session` kendi kopyasını
    okuyordu ve 20 MB'lık dosya test takımını 5,8 saniyeden 44,8'e çıkardı —
    702 test, her biri bir oturum. Vektörler salt okunur ve derleme bağlı,
    yani paylaşmak güvenli; graf gibi oturuma özel değiller.
    """
    if path in _HELD_VECTORS:
        return _HELD_VECTORS[path]
    import json
    import os
    found = None
    if os.path.exists(path):
        try:
            from lmm.vectors import Vectors
            with open(path, encoding="utf-8") as handle:
                found = Vectors.from_dict(json.load(handle))
        except Exception:                                   # noqa: BLE001
            found = None
    _HELD_VECTORS[path] = found
    return found


class Session:
    """One conversation: the five organs wired together over a memory file."""

    def __init__(self, path, language=None, inquiry=None, wording=None,
                 intent=None, voice=None, dictionary=None, vectors=None):
        self.path = path
        self.memory = Memory.load(path)
        # Gömme vektörleri: varsa bağlanır, yoksa sistem eskisi gibi çalışır.
        # Niyet ağıyla aynı disiplin — kolaylık, bağımlılık değil.
        #
        # Ne işe yaradığı dar ve kasıtlı: kapı "bilmiyorum" dediğinde ve yazım
        # komşusu da bulunmadığında, ANLAMCA yakın BİLİNEN kavramları
        # gösteriyor. Bir olgu üretmiyor, bir iddiada bulunmuyor; yalnız elde
        # ne olduğunu söylüyor. Dağılımsal benzerlik zıtları ayıramaz ("sıcak"
        # ile "soğuk" aynı çevrede geçer, bu projede ölçüldü) — o yüzden
        # geometri ADAY bulur, kararı kapı verir.
        self.memory.vectors = vectors if vectors is not None else _vectors()
        lexicon.use(self.memory.lexicon)    # this session speaks its own words
        reasoning = Reasoning(self.memory)
        self.gate = EpistemicGate(self.memory, reasoning)
        # any LanguageOrgan fits here; the other organs never see a sentence
        # Suffix rules come from the words this memory has met, falling back
        # to the declared lists while it is still small.
        # NOT — `memory=` BİLEREK geçilmiyor ve bu bir eksiklik değil, ölçülmüş
        # bir karar. `intuition.understand` bu parametreyi alınca `abduction`
        # devreye giriyor: "birden çok kalıp uyduğunda bildiklerimize en az
        # aykırı okumayı seç". Kulağa doğru geliyor, testleri var, ve sohbette
        # hiç çalışmamış.
        #
        # Bağlandı ve ÖLÇÜLDÜ: 16 test kırıldı. Örnek — "penguen neden uçar"
        # sorusunda beklenen "aslında penguen uçamaz" (yanlış öncülü düzeltmek)
        # yerine "bilmiyorum" geliyor, çünkü abduction cevaplanamayan bir
        # okumayı daha ucuz buluyor.
        #
        # Yani eksik olan kablo değil, organın kendisi: maliyet işlevi gerçek
        # akışta yanlış seçiyor. Bağlamadan önce düzeltilmeli.
        self.language = language or Intuition(network=MiniNetwork.default(),
                                              lexicon=self.memory.lexicon,
                                              words=words_of(self.memory),
                                              known=self._concepts,
                                              meanings=self._meanings)
        for entry in self.memory.patterns:      # ways of speaking it was taught
            pattern = Pattern.from_dict(entry)
            # Sohbette tek tek öğrenilen kalıp öne, toplu çıkarılan sona.
            # İkisi ayrı şeyler: birincisi bir insanın onayladığı bir okuma,
            # ikincisi sayımın önerdiği bir şekil.
            self.language.grammar.add(pattern, first=not pattern.fallback)
        self.learning = LearningLoop(self.memory, reasoning)
        self.curiosity = Curiosity(self.memory, reasoning)
        self.pursuit = Pursuit(self.memory, reasoning)
        self.induction = Induction(self.memory, reasoning)
        self.pending = None   # an edge awaiting the teacher's confirmation
        self.last = None      # son soru — "peki ya kartal" onu tekrarlar
        self.goal = None      # a question it is still trying to earn the answer to
        self.goal_steps = set()
        self.focus = None     # what "o" and a bare question refer to
        # Sohbet geçmişi: grafa yazılmaz, oturumla birlikte gider. Bir sohbetin
        # neden bahsettiği dünya hakkında bir olgu değil.
        self.thread = Thread()
        # Hangi kavram hakkında ne söylendiği — "başka ne biliyorsun"
        # sorusunun tekrar etmemesi için. Grafa yazılmaz, oturumla gider.
        self.told = {}
        # Son cevabın dayandığı kenarlar — "emin misin" ve "nereden
        # biliyorsun" sorularının cevabı burada. Yeni bilgi değil, zaten
        # tutulan künyenin sorulabilir hâle gelmesi.
        self.grounds = []
        # Oturumun yönergeleri — LLM'deki system prompt'un karşılığı. Prompt
        # her istekte yeniden gönderilir; yönerge bir kez söylenir ve oturum
        # hatırlar. Boşken hiçbir şey değişmiyor.
        self.directives = {}
        # Akıcı ağız: grafın cevabını çekirdeğe söyletir, ama üretilen cümle
        # geri okunup grafla karşılaştırılır. Verilmezse şablonlar kullanılır —
        # akıcılık isteğe bağlı, doğruluk değil.
        self.voice = voice
        # Soruşturma: bilmediğini fark edip gidip okuma. İsteğe bağlı, çünkü
        # ağ gerektiriyor ve sistemin çalışması ona bağlı olmamalı.
        self.inquiry = inquiry
        self.investigated = set()   # aynı kavram için ikinci kez ağa çıkılmaz
        # Söyleyiş öğrenme: anlaşılmayan cümleden kalıcı kalıp çıkarma.
        # Soruşturma gibi isteğe bağlı — kapalıyken davranış değişmiyor.
        self.wording = wording
        self.asked_about = set()    # aynı cümle için ikinci kez sorulmaz
        # Eş anlamlı arama: tanımadığı bir sözcük için sözlüğe gitmek.
        # `synonyms` kurulmuştu ama üretimde HİÇ import edilmiyordu — grafta
        # duran eş anlamlılıklar kullanılıyor, yenisi asla keşfedilmiyordu.
        # Soruşturma gibi isteğe bağlı: ağ gerektiriyor.
        self.dictionary = dictionary
        self.looked_up = set()      # aynı kelime için ikinci kez sözlüğe gidilmez
        # Eğitilmiş niyet okuyucusu: kalıpların YEDEĞİ, yerine geçeni değil.
        # Kalıp eşleşirse buraya hiç gelinmiyor.
        self.intent = intent

    def _meanings(self, word):
        """Bu kelimenin grafta kayıtlı eş anlamlıları.

        Ayrıştırıcı tanımadığı bir sözcükle karşılaşınca buraya bakar: "bahset"
        ile "anlat" aynı şeyi istiyorsa kalıp eklemeye gerek yok, grafta o
        kayıt vardır — künyesiyle, denetlenebilir, düzeltilebilir.

        Bu, eş anlamlılığı ağırlığa ya da kümeye gömmemenin karşılığı.
        Dağılımsal kümeleme denendi ve ölçülüp çürütüldü: işlevsel kelimelerde
        aynı gruptaki çift, farklı gruptakinin altında kalıyordu.
        """
        found = {edge.target for edge in self.memory.query(word, SAME_AS)}
        found.update(edge.concept
                     for edge in self.memory.incoming(word, SAME_AS))
        return found

    def _concepts(self):
        """Grafın kavramları — kopyası kenar sayısına bağlı saklanıyor.

        `grammar.known()` bunu cümle başına ~110 kez çağırıyor ve her çağrıda
        yeni bir liste kopyalanıp kümeye çevriliyordu. Ölçüldü: 10 katlık
        grafta ayrıştırma süresinin %30'u. 17 katta kabul edilemez.
        """
        # Ölçüt `revision`, kenar SAYISI değil. Sayı net-sıfır değişimde
        # yanılıyor: bir kavram unutulup aynı sayıda yeni olgu yazılınca sayı
        # geri aynı olur ve önbellek bayat kalır. Ölçüldü — `forget(kartal)`
        # 11 kenar sildi, 11 yeni olgu yazıldı, sayı 128.485'e döndü ve
        # önbellek silineni hâlâ biliyordu. `revision` her yazmada, `purges`
        # her unutmada artıyor ve ikisi birlikte geri saymıyor.
        marker = (getattr(self.memory, 'revision', 0),
                  getattr(self.memory, 'purges', 0))
        cached = getattr(self, "_concepts_cache", None)
        if cached is not None and cached[0] == marker:
            return cached[1]
        found = self.memory.concepts()
        self._concepts_cache = (marker, found)
        return found

    def _aimed(self, tokens, relation):
        """Cümlenin sorduğu HEDEF: yetenek sorusunda fiil, nitelikte nitelik.

        Ağ niyet SINIFINI söylüyor ("bu bir yetenek sorusu") ama hangi yetenek
        olduğunu söylemiyor — sınıflandırıcı odur, çıkarıcı değil. Hedef
        bulunamayınca `reading_of(has_target=False)` tekil soruyu çoğula
        düşürüyor: "penguen uçar mı" -> "penguen ne yapabilir".

        Bu sabit `False`, ağın ürettiği iki sınıfı (ASK_ABILITY, ASK_PROPERTY)
        sohbette ULAŞILMAZ kılıyordu. Ağ onları ayırt etmeyi öğrendi ve o
        öğrenme hiç kullanılmıyordu.

        Hedefi ağ değil graf ve derlem veriyor, yani uydurma yok: fiil
        `predicate_of` ile okunuyor (sözlük, sonra derlem tanıklığı), nitelik
        ise grafın DUYDUĞU nitelikler arasında aranıyor. Bulunamazsa eski
        davranışa düşülüyor — genişletme, daraltma değil.
        """
        if relation == "can":
            for token in reversed(tokens):
                stem, tense = frames.predicate_of(token, self.memory.lexicon)
                if stem:
                    return stem
            return None
        if relation == "property":
            morphology = self.language.grammar.morphology
            heard = set(self.memory.properties())
            for token in reversed(tokens):
                for form in (token, morphology.strip_copula(token)):
                    if form in heard:
                        return form
        return None

    def _unplaced(self, tokens, at):
        """Cümlede yeri doldurulamayan bir içerik sözcüğü var mı.

        Yapısal sözcük (sıklık sıralamasının tepesi), fiil, sözlükte olan ya da
        grafın tanıdığı her kelimenin bir yeri var. Kalanı içeriktir ve okuma
        onu açıklayamıyorsa cümle anlaşılmamıştır.
        """
        from lmm import frequency
        from lmm.verbs import is_structural
        counts = frequency.counts()
        verbs = frequency.verbs()
        for index, token in enumerate(tokens):
            if index == at or len(token) < 3:
                continue
            if token in verbs or self.memory.query(token):
                continue
            if self.memory.lexicon.reading(token) is not None:
                continue
            if counts and is_structural(token, counts):
                continue
            return True
        return False

    def _neural_reading(self, line):
        """Kalıplar yetmediğinde eğitilmiş ağa sorar — ve denetler.

        Ağ yalnızca niyet TÜRÜNÜ söyler. Kavramı uydurmaz: cümlede grafta
        karşılığı olan ilk kelime alınır. Sonra çıkan okuma grafta gerçekten
        cevaplanıyor mu diye bakılır; cevap üretemiyorsa okuma atılır.

        Bu denetim olmadan bağlantı bir gerileme olurdu — ağ her cümleye bir
        niyet uydurabilir ve uydurma niyet, uydurma cevaba açılan kapıdır.

        Geçen okuma kalıba çevrilip kalıcı yazılıyor: ağ bir kez katkı yapar,
        sonrası hızlı, denetlenebilir ve ağsız çalışır.
        """
        if self.intent is None or line in self.asked_about:
            return None
        try:
            name, confidence = self.intent(line)
        except Exception:                                   # noqa: BLE001
            return None
        if not name or confidence < NEURAL_THRESHOLD:
            return None
        tokens = tokenize(line)
        # Önce hedefli okumayı dene. Hedef yoksa çoğula düşer — eski davranış.
        aimed = reading_of(name)
        target = self._aimed(tokens, aimed[1]) if aimed else None
        found = reading_of(name, has_target=target is not None)
        if found is None:
            return None
        kind, relation = found
        collapsed = kind != aimed[0]
        if collapsed:
            target = None       # çoğula düşüldü: hedef artık anlamsız
        morphology = self.language.grammar.morphology
        at, concept = wording.concept_in(tokens, self.memory, morphology,
                                         self.focus)
        if concept is None:
            return None
        # Çoğula düşmek, SORULANDAN BAŞKA bir soruyu cevaplamaktır ve bu ancak
        # cümlede yeri doldurulamamış bir içerik sözcüğü YOKSA kabul edilebilir.
        # Ölçüldü:
        #
        #   insan bir omurgalı mı -> "insan tüylü, hareketli ve sıcakkanlıdır"
        #                            ve o okuma kalıcı kalıp olarak yazıldı
        #
        # `omurgalı` grafın hiç duymadığı bir kelime; sistem onu sessizce atıp
        # başka bir soruyu cevapladı. Kapının aynı gün düzelttiği dürüst
        # "bilmiyorum", ağ yolundan kendinden emin bir hatayla değiştiriliyordu
        # — bu mimaride yapılabilecek en kötü takas.
        #
        # "kartal nasıl bir hayvan" bundan etkilenmiyor: oradaki her kelimenin
        # yeri var (`hayvan` grafta duruyor), yani çoğula düşmek sorunun makul
        # okuması.
        if collapsed and self._unplaced(tokens, at):
            return None
        said = self.gate.answer(Intent(kind, concept, relation, target))
        if is_a_refusal(said):
            return None
        # Cevap üretebilmek yetmiyor: okuma cümleye de uymalı. Bu kapı olmadan
        # "bir kuşun uçabilmesi için ne gerekir" cümlesi "kuş ne yapabilir"
        # diye okunuyor, graf cevaplıyor ve YANLIŞ kalıp kalıcı yazılıyordu.
        # Okumanın kendi kullandığı kelimeler kapıya sorulmaz: kavram ve
        # hedef tanım gereği açıklanmıştır.
        used = {at} | ({tokens.index(target)} if target in tokens else set())
        if not wording.accounted_for(kind, tokens, self.memory.lexicon,
                                     self.language.grammar.patterns,
                                     self._meanings, consumed=used):
            return None
        self.asked_about.add(line)
        reading = wording.Reading(kind, relation, at,
                                  tokens.index(target) if target in tokens
                                  else None)
        pattern = wording.learn(line, reading, self.memory.lexicon, morphology)
        if pattern is None:
            return said
        pattern.name = f"ağdan öğrenildi: {line}"
        self.language.grammar.add(pattern, first=True)
        self.memory.patterns.append(pattern.to_dict())
        return f"{said} {learned_wording(line)}"

    def _look_up(self, word):
        """Tanımadığı bir sözcüğün eş anlamlısını sözlükten arar ve grafa yazar.

        Kalıplar bir sözcüğü bilmiyorsa, o sözcüğün bildiğimiz bir sözcükle
        aynı şeyi söylemesi mümkün — "bahseder" ile "anlat" gibi. Sözlük bunu
        biliyor ve okumak bir kez sürüyor; sonrası grafta, künyesiyle.

        Ölçülmüş bir kısıt: aynı kelime için ikinci kez ağa çıkılmıyor.
        """
        if self.dictionary is None or word in self.looked_up:
            return False
        self.looked_up.add(word)
        try:
            from lmm import synonyms
            written = synonyms.into(self.memory, [word] + list(self._pattern_words()),
                                    self.dictionary)
        except Exception:                                   # noqa: BLE001
            return False
        return written > 0

    def _pattern_words(self):
        """Kalıplarda geçen sabit sözcükler — eş anlamlısı aranacak hedefler."""
        from lmm.grammar import SLOTS
        found = set()
        for pattern in self.language.grammar.patterns:
            for token in pattern.tokens:
                if token not in SLOTS and len(token) > 2:
                    found.add(token)
        return sorted(found)[:12]

    def _learn_wording(self, line):
        """Anlaşılmayan cümleden kalıcı bir kalıp çıkarmayı dener.

        Üç kapı, üçü de olgular için kurulmuş olanların aynısı: dil modeli
        okur ama karar vermez; çıkan okuma grafta gerçekten cevaplanmıyorsa
        kalıp YAZILMAZ; yazılan kalıp hangi cümleden geldiğini taşır.

        Uydurma bir okuma cevap üretemez — denetim tam da bu yüzden işliyor.
        """
        if self.wording is None or line in self.asked_about:
            return None
        self.asked_about.add(line)
        try:
            reading = wording.ask_model(line, self.wording)
        except Exception:                                   # noqa: BLE001
            return None       # model yoksa sohbet durmaz
        if reading is None:
            return None
        tokens = tokenize(line)
        morphology = self.language.grammar.morphology
        works, said = wording.answerable(reading, tokens, self.memory,
                                         self.gate, morphology)
        if not works:
            return None       # işe yaramayan okuma kalıp olmaz
        pattern = wording.learn(line, reading, self.memory.lexicon,
                                self.language.grammar.morphology)
        if pattern is None:
            return said       # cevabı ver ama kalıbı saklama (cümle çok uzun)
        pattern.name = f"öğrenildi: {line}"
        self.language.grammar.add(pattern, first=True)
        self.memory.patterns.append(pattern.to_dict())
        return f"{said} {learned_wording(line)}"

    def respond(self, line):
        said = self._respond(line)
        # Yönergeler söyleyişe EN SONDA uygulanır — hangi yoldan çıkarsa
        # çıksın. İçerik graftan gelmiş ve kapıdan geçmiş durumda; burada
        # değişen yalnız uzunluk ve kaynak gösterimi.
        return phrasing.directed(said, self.directives)

    def _directive(self, line):
        """Cümle bir yönerge mi? Kapalı sınıftan, dizgi karşılaştırmasıyla."""
        grammar = getattr(self.language, "grammar", None)
        morphology = getattr(grammar, "morphology", None)
        table = getattr(morphology, "directives", None) or {}
        text = lower(line).strip(" .,!?")
        for kind, forms in table.items():
            if any(text == form or text.endswith(" " + form)
                   or text.startswith(form + " ") for form in forms):
                return kind
        return None

    def _respond(self, line):
        directive = self._directive(line)
        if directive is not None:
            if directive == "short":
                self.directives["length"] = "short"
            elif directive == "long":
                self.directives["length"] = "long"
            elif directive == "plain":
                self.directives["sources"] = False
            elif directive == "cited":
                self.directives["sources"] = True
            return phrasing.directive_set(directive)
        if self.pending is not None:
            settled = self._resolve_pending(line)
            if settled is not None:
                return settled
        if arithmetic.looks_like_a_sum(line):
            # Computed, not recalled: a sum has no place in memory and no
            # business being guessed at.
            try:
                return computed(arithmetic.normalise(line),
                                arithmetic.evaluate(line))
            except arithmetic.Undecidable as reason:
                return cannot_compute(reason)
        words, rest = split_words(line)
        if words:                       # "kelime: uçmak = uçar / uçamaz"
            for infinitive, positive, negative in words:
                self.memory.learn_word(infinitive, positive, negative)
            infinitive, positive, negative = words[-1]
            return learned_word(infinitive, positive, negative)
        # Sosyal alışveriş, bilgi sorusu değildir: "nasılsın" sorusuna cevap
        # vermek için hiçbir şey bilmek gerekmez. Bunlar grafta aranınca
        # bulunamıyor ve reddediliyordu; sekiz turluk bir sohbette üç tur
        # böyle kayboldu.
        exchange = social.recognise(line)
        if exchange is not None:
            return social.reply(exchange, self.memory)
        # "peki ya kartal" — soru tekrar edilmiyor, yalnızca öznesi değişiyor.
        # Ayrıştırıcı bu cümleyi hiç tanımıyordu, dolayısıyla zaten var olan
        # bağlam mekanizmasına ulaşamıyordu.
        # İki özneli soru: "penguen ve kartal kuş mu". Cümle bölünüyor, her
        # özne için aynı soru NORMAL yoldan soruluyor, cevaplar birleştiriliyor.
        # Kalıp eklemek yerine bölmek, üç özneli soruda da çalışır ve hiçbir
        # denetimi atlamaz — parçalar yine kapıdan geçiyor.
        # Çoğul gönderme: "ikisi de kuş mu" — özneler cümlede değil, az önce
        # konuşulanlar. Sohbet geçmişi zaten tutuluyor; eksik olan onu ÖZNE
        # olarak kullanmaktı. Bu olmadan sistem "ikisi nedir?" diye soruyordu.
        # "sence"/"bence" ile başlayan cümle bir GÖRÜŞ sorusudur. Açılış
        # sözcüğü diye ayıklanıyordu ve cümle bilgi sorusu sanılıyordu —
        # "sence penguenler mutlu mu" grafta aranıp bulunamıyordu. Oysa
        # cevabı graf değil, mimarinin kendisi veriyor: görüşüm yok.
        if self._an_opinion(line):
            return no_opinion(self.focus)

        forgotten_now = self._forget(line)
        if forgotten_now is not None:
            return forgotten_now
        corrected = self._correction(line)
        if corrected is not None:
            line = corrected
        referred = self._referred(line)
        if referred is not None:
            return referred

        together = self._coordinated(line)
        if together is not None:
            return together

        carried = self._bare_follow_up(line) or self._carried(line)
        intent = self._in_context(
            self._typed(carried or self.language.understand(line)))
        if intent.kind == AMBIGUOUS:
            return which_reading(intent.concept, intent.readings)
        if intent.kind == UNKNOWN_WORD:
            # Tanımadığı sözcük için önce SÖZLÜĞE bak: "bahseder" ile "anlat"
            # aynı şeyi istiyorsa kalıp eklemeye gerek yok, grafta o kayıt
            # olması yeter. Bulursa cümleyi yeniden okur.
            if intent.target and self._look_up(intent.target):
                again = self.language.understand(line)
                if again.kind not in (UNKNOWN, UNKNOWN_WORD, AMBIGUOUS):
                    return self._question(self._in_context(again))
            # Öğrenme yolları burada da denenmeli. Yalnız UNKNOWN dalında
            # çağrılıyorlardı ve "kartal hakkında ne SÖYLEYEBİLİRSİN" cümlesi
            # UNKNOWN_WORD veriyor — bilinmeyen bir kelime yüzünden cümlenin
            # tamamı öğrenilemez sayılıyordu.
            #
            # Oysa bir kelimeyi bilmemek, cümlenin ne istediğini bilmemek
            # değil: ağ o cümleye %98 güvenle ASK_DESCRIBE diyor.
            for attempt in (self._neural_reading, self._learn_wording):
                learned = attempt(line)
                if learned is not None:
                    return learned
            return teach_me_the_word(intent.target)
        if intent.kind == UNKNOWN or intent.confidence < CONFIDENCE_THRESHOLD:
            # Anlamadıysa öğrenmeyi dener: önce eğitilmiş ağ (bedava, yerel),
            # sonra dil modeli (ağ gerektirir, ücretli). İkisi de aynı denetimden
            # geçiyor — okuma grafta cevaplanmıyorsa kalıp yazılmıyor.
            for attempt in (self._neural_reading, self._learn_wording):
                learned = attempt(line)
                if learned is not None:
                    return learned
            resembles = intent.resembles if intent.resemblance >= RESEMBLANCE else None
            tokens = tokenize(line)
            spotted = [word for word in tokens
                       if len(self.memory.query(word)) >= 3][:2]
            return not_understood(resembles, self.memory, spotted,
                                  long=len(tokens) > 4)
        if intent.kind == ASK_THREAD:
            return talked_about(self.thread.recent())
        if intent.kind == ASK_MORE:
            return self._more()
        if intent.kind == ASK_INVENTORY:
            return inventory(self.memory)
        if intent.kind == ASK_CERTAINTY:
            return certainty(self.grounds, self.memory.kinds)
        if intent.kind == ASK_SOURCE:
            return whence(self.grounds)
        if intent.kind == ASK_OPINION:
            return no_opinion(intent.concept or self.focus)
        # Yalnızca grafın tanıdığı kavramlar konu sayılıyor. Başarısız bir
        # ayrıştırma "sence penguen" gibi bir şey üretebiliyor ve o sohbet
        # geçmişine sızıyordu — geçici bir kayıt bile olsa, olmayan bir şeyi
        # konuşmuş gibi göstermek uydurmadır.
        if intent.concept and self.memory.query(intent.concept):
            self.thread.note(intent.concept)
        if intent.kind != TEACH:
            self.last = intent          # eksiltili sorular buna dayanacak
            # Cümlenin kendi kelimeleri kapıya veriliyor: çokanlamlı bir
            # kavramda hangi anlamın konuşacağını soru belirlesin.
            self.gate.focus_words = tuple(tokenize(line))
            said = self._fluent(intent, line, self._question(intent))
            # ÇOKLU HİPOTEZ. İlk okuma cevap üretmediyse ve kavram grafta
            # yoksa, adaylar sırayla deneniyor. Bu, sistemin tek sert kararını
            # geri alınabilir yapıyor: ayrıştırıcının yanılması artık yolun
            # sonu değil.
            #
            # Geri izleme bu projede zaten var — `grammar._fit_from` kalıp
            # eşleştirirken tam bunu yapıyor. Kavram çözümlemesinde yoktu ve
            # ölçüldü: 200 gerçek sorunun %47'si orada kayboluyordu.
            #
            # Kazananı biz seçmiyoruz: grafta CEVAP ÜRETEN kazanıyor, ve kapı
            # yine doğruluyor. Aday sayısı artıyor, kapı gevşemiyor.
            # Koşul "kavram grafta yok" DEĞİL, "graf adayı daha iyi
            # tanıyor". İlk hâli yoklukla sınırlıydı ve bir eş adlılık onu
            # kapattı: `kuşlar` grafta VAR — Bursa'nın köyü olarak, 3 olgu.
            # Kavram "var" sayıldı, aday yolu açılmadı ve sistem kuş yerine
            # "kuşlar nedir, öğret" dedi. Ölçüt destek: `kuş` 15 olgu > köy 3.
            if is_a_refusal(said) and intent.concept:
                # İki ŞEKİL deneniyor, çünkü özne yanlış okunduğunda cümlenin
                # gerisi de yanlış okunuyor. Ölçüldü: "Ormanların önemi ve
                # faydaları nelerdir" cümlesi `ASK(ormanların önemi, type,
                # fay)` diye okunmuştu — "orman bir FAY mıdır" sorusu. Adayı
                # o bozuk şeklin içine koymak işe yaramaz.
                #
                #   ÖZGÜN    ilişki doğru okunmuş olabilir, önce o denenir
                #   ANLAT    okunamadıysa graf o kavram hakkında ne biliyorsa
                #
                # ANLAT bir tahmin değil: her cümlesi graftan geliyor ve kapı
                # yine önünde. Sorulmayan soruya cevap verme riski var, ve
                # sınır tam burada — bu yola YALNIZCA kavram grafta
                # bulunamadığında giriliyor, yani başka türlü sessizlik olacak
                # yerde. Özne cevapta açıkça geçiyor ("Orman bir yerdir"), o
                # yüzden yanlış anladıysak kullanıcı görüyor.
                held = len(self.memory.query(intent.concept))
                for name, same_thing in self._candidates(intent.concept):
                    if len(self.memory.query(name)) <= held:
                        continue        # aday, eldekinden iyi tanınmıyor
                    shapes = [Intent(intent.kind, name, intent.relation,
                                     intent.target, intent.object,
                                     intent.role, intent.quantifier)]
                    # ANLAT yalnız kimliği koruyan adaya açık. Gömme komşusu
                    # sorulan şeyin kendisi değil, o yüzden onun hakkında
                    # konuşmak soruyu cevaplamaz. Özgün şekli deneyebilir —
                    # orada okunan ilişkinin de tutması gerekiyor, çok daha sıkı.
                    if same_thing:
                        shapes.append(Intent(ASK_DESCRIBE, name))
                    for shape in shapes:
                        other = self._typed(shape)
                        tried = self._fluent(other, line,
                                             self._question(other))
                        if not is_a_refusal(tried):
                            intent, said = other, tried
                            self.last = intent
                            break
                    else:
                        continue
                    break
            self._remember_grounds(intent)
            # Ayrıştırıcı güvenle YANLIŞ okuyabiliyor: "bana penguenlerden
            # bahseder misin" cümlesini %100 güvenle "bana" hakkında bir soru
            # sanıyordu. O yüzden öğrenme yalnız "anlamadım"a bağlı olamaz —
            # ölçüt, yapılanın işe YARAYIP yaramadığı.
            if is_a_refusal(said):
                # Önce eğitilmiş ağ — bedava ve yerel. Sonra dil modeli, çünkü
                # o ağ gerektiriyor ve ücretli.
                for attempt in (self._neural_reading, self._learn_wording):
                    found = attempt(line)
                    if found is not None:
                        return found
            return said
        # Soru sözcüğü taşıyan bir cümle bildirme olamaz. Ayrıştırıcı
        # yanılabilir ve yanıldığında bedeli ağır: soru grafa olgu diye
        # yazılır ve bir daha çıkmaz. Bu kapı yanılmayı cevapsızlığa
        # çeviriyor — hafızayı kirletmektense anlamamak.
        if self._asks_something(line):
            return not_understood(None)
        status, message, edge = self.learning.teach(intent)
        if status == CONFLICT:
            self.pending = edge
            return message
        if status in (DISPUTE, FROZEN):
            return message
        if status in (LEARNED, CORRECTED):
            return f"{message} {self._after_learning()}".strip()
        return message

    def _candidates(self, concept, most=5):
        """Kavram için ADAYLAR — ağırlıklı, en desteklenen önce.

        Bu, sistemin tek SERT kararını yumuşatıyor. Ayrıştırıcı cümleden bir
        kavram çıkarıyor ve ondan sonraki her şey ona bağlı; yanılırsa graf
        boşuna aranıyor ve dönüş yok. Ölçüldü — 200 gerçek sorunun %47'si tam
        burada kayboluyordu ve sebep grafın bilmemesi DEĞİLDİ:

            "Ormanların önemi ve faydaları nelerdir?"
              ayrıştırıcı  -> 'ormanların önemi'   grafta yok
              gerçek özne  -> 'orman'              grafta 15 olgu VAR

        LLM'de bu adım hiç yok: dağıtık temsilde `orman` ile `ormanların
        önemi` zaten aynı bölgede, ayrı şey değiller. Bizde ayrı düğümler —
        biri var biri yok.

        Üç kaynaktan aday üretiliyor, hepsi elimizde olan şeylerden:

            ALT ÖBEK      "ormanların önemi" -> "ormanların" -> "orman"
            EK SOYMA      biçimbilim zaten yapıyor, öbek için yapılmıyordu
            ANLAM KOMŞUSU gömme, bilinen kavramlar arasından

        Ağırlık graftan geliyor: bir aday hakkında graf kaç olgu biliyorsa o
        kadar desteklidir. Dikkatin (attention) buradaki karşılığı bu — kararı
        biz vermiyoruz, GRAFIN DESTEĞİ veriyor.

        Uydurma riski artmıyor: aday sayısı artıyor, kapı gevşemiyor. Kazanan
        aday da seçilmiyor, grafta CEVAP ÜRETEN kazanıyor.

        Her aday `(ad, aynı_şey_mi)` ikilisiyle dönüyor ve bu ayrım ölçümle
        geldi. Alt öbek ve ek soyma KİMLİĞİ KORUR: "ormanların önemi" ile
        "orman" aynı şeydir, başka söylenişi. Gömme komşusu korumaz — "fosfor"
        ile "kükürt" iki ayrı elementtir. Ayrım yapılmadığında olan şuydu:

            "Fosfor hangi besinlerde bulunur?" -> "Kükürt bir elementtir..."
            "Kaza namazı nasıl kılınır?"       -> "Olay bir kavramdır..."

        Hiçbiri uydurma değil, her cümle graftan geliyor. Ama sorulmayan
        soruya cevap vermek, cevap vermemekten kötüdür.
        """
        if not concept:
            return []
        # YAPI SÖZCÜĞÜ KAVRAM DEĞİLDİR. Ayrıştırıcı özneyi bulamadığında sık
        # sık kapalı sınıftan bir kelime tutuyor: `bu`, `böyle`, `olan`,
        # `biraz`, `fazla`. Bunlara aday üretmek zararlı — gömme `bu` için
        # `vankomisin` ve `trakeostomi` öneriyordu, çünkü dağılımsal benzerlik
        # her yerde geçen bir kelimeye hiçbir şey söyleyemez.
        #
        # Ölçüt kelime listesi değil, SIRA: bir dilin en sık kelimeleri her
        # derlemde aynı türdendir. `lmm/verbs.is_structural` bunu sayımdan
        # okuyor. Zamir zaten bağlamdan çözülür, gömmeden değil.
        counts = frequency.counts()
        if all(is_structural(word, counts) for word in concept.split()):
            return []
        found, seen = [], {concept}
        same = set()                # kimliği koruyanlar
        words = concept.split()
        # Alt öbekler: sondan ve baştan daralt.
        for size in range(len(words) - 1, 0, -1):
            for at in (0, len(words) - size):
                piece = " ".join(words[at:at + size])
                if piece and piece not in seen:
                    seen.add(piece)
                    same.add(piece)
                    found.append(piece)
        # Ek soyma: biçimbilim tek kelimede yapıyor, öbeğin başında yapmıyordu.
        morphology = getattr(getattr(self.language, "grammar", None),
                             "morphology", None)
        if morphology is not None:
            for piece in list(found) + [concept]:
                head = piece.split()[0]
                # Kesme işareti bir EK sınırı: "urfa’da" -> "urfa". Özel ad
                # eklerini kesmeyle ayırıyor ve gövde solda kalıyor.
                for mark in ("’", "'"):
                    if mark in head:
                        head = head.split(mark)[0]
                # Durum eki taşıyan kelime kavramın kendisi değil, ROLÜ:
                # "kadında", "turizmle", "urfa’da". Biçimbilim bunu zaten
                # biliyor (`role_of`) ve tek kelimede kullanılıyordu; öbeğin
                # başında kullanılmıyordu ve 43 soru tam burada kayboldu.
                stem, _ = morphology.role_of(head)
                # Ekler KATMAN KATMAN soyuluyor: "ormanların" bir kez soyunca
                # "ormanlar" ve graf onu bilmiyor — `orman`ı biliyor. Tek
                # geçişte kalmak, kavramı gömme komşusu sanmaktı (kimlik
                # kaybolup ANLAT yolu kapanıyordu).
                genitive = morphology.strip_genitive(stem, self._concepts())
                for peeled in (head, stem,
                               morphology.strip_plural(stem),
                               morphology.strip_accusative(stem),
                               genitive,
                               morphology.strip_plural(genitive)):
                    if peeled and peeled not in seen:
                        seen.add(peeled)
                        same.add(peeled)
                        found.append(peeled)
        # Anlam komşusu: gömme varsa, BİLİNEN kavramlar arasından.
        vectors = getattr(self.memory, "vectors", None)
        if vectors is not None:
            for word in ([concept] + words)[:2]:
                try:
                    close = vectors.similar(word, 12)
                except Exception:                           # noqa: BLE001
                    continue
                for name, _ in close:
                    if name not in seen and self.memory.query(name):
                        seen.add(name)
                        found.append(name)
        # AĞIRLIK: graf kaç olgu biliyorsa o kadar destekli.
        found = [name for name in found if self.memory.query(name)]
        found.sort(key=lambda name: -len(self.memory.query(name)))
        return [(name, name in same) for name in found[:most]]

    def _correction(self, line):
        """"hayır penguen uçamaz" -> düzeltilmiş cümle, değilse None.

        Ret sözcüğüyle aynı kelimeler ama işlevi başka: tek başına duruyorsa
        ret, arkasından cümle geliyorsa DÜZELTMEDİR. Bildirilmediği için
        ayrıştırıcı "hayır"ı kavram sanıyordu ve grafa `hayır penguen` diye
        bir düğüm yazılıyordu:

            > hayır penguen uçamaz
            öğrendim: hayır penguen uçamaz. bunu hiç öğrenmedim: HAYIR nedir?

        Bir sistemin yanlışını düzeltmek, ona bir şey öğretmek kadar temel ve
        bunun DOSYA AÇMADAN yapılabilmesi gerekiyor.
        """
        # Başka bir dil organı takılıysa `grammar` ya da `morphology`
        # olmayabilir; sistem o zaman da çalışmalı.
        grammar = getattr(self.language, "grammar", None)
        marks = getattr(getattr(grammar, "morphology", None), "corrections", ())
        tokens = tokenize(line)
        if len(tokens) < 2 or tokens[0] not in marks:
            return None
        return " ".join(tokens[1:])

    def _forget(self, line):
        """"penguen hakkında bildiklerini unut" -> silme, değilse None.

        `memory.forget` yazılmıştı ve sohbetten erişilemiyordu — dosyayı elle
        açmak bir arayüz değildir. Silinebilir olmak bu mimarinin LLM
        karşısındaki en somut farkı; erişilemezse yok sayılır.
        """
        grammar = getattr(self.language, "grammar", None)
        marks = getattr(getattr(grammar, "morphology", None),
                        "forget_words", ())
        tokens = tokenize(line)
        if not marks or not tokens or tokens[-1] not in marks:
            return None
        known = self._concepts()
        for size in range(min(3, len(tokens) - 1), 0, -1):
            for at in range(len(tokens) - size):
                name = " ".join(tokens[at:at + size])
                if name in known:
                    return forgotten(name, self.memory.forget(name))
        return None

    def _an_opinion(self, line):
        """Bu cümle görüş mü soruyor.

        Ayrım kelimede değil ne istendiğinde: "sence X mi" bilgi değil kanaat
        istiyor ve kanaatin dayanağı olmaz. Dayanaksız cümle kurmayan bir
        sistemin görüşü de olamaz — bunu söylemek bir eksiklik itirafı değil,
        mimarinin doğrudan sonucu.
        """
        # Başka bir dil organı takılıysa `grammar` ya da `morphology`
        # olmayabilir; sistem o zaman da çalışmalı.
        grammar = getattr(self.language, "grammar", None)
        morphology = getattr(grammar, "morphology", None)
        marks = getattr(morphology, "opinion_marks", ()) or ()
        tokens = tokenize(line)
        return bool(marks) and bool(tokens) and tokens[0] in marks

    def _remember_grounds(self, intent):
        """Cevabın dayandığı kayıtlar — "emin misin" bunlara bakacak.

        Kalıtımla gelen cevabın dayanağı atadaki kayıttır: "kartal uçar"
        cevabının kaynağı `kuş --can--> uçmak`. Zinciri izlemezsek "nereden
        biliyorsun" sorusuna boş dönerdik.
        """
        if not intent.concept:
            return
        found = []
        steps = [intent.concept] + self.gate.reasoning.ancestors(intent.concept)
        for step in steps:
            for edge in self.memory.query(step, intent.relation):
                if intent.target and edge.target != intent.target:
                    continue
                found.append(edge)
        if not found:                       # ilişki eşleşmediyse kavramın kendisi
            found = list(self.memory.query(intent.concept))
        if found:
            self.grounds = found

    def _kept(self, intent, concept):
        """Cevaplanan bir soruyu sohbet geçmişine işler.

        Ana yol (`_answer`) bir soruya cevap verdikten sonra üç şey yapar:
        konuyu iplikçiğe not eder, niyeti `self.last`a koyar, dayanakları
        saklar. Eksiltili takipler ("neden", "emin misin", "nereden
        biliyorsun") tam bu üçüne dayanıyor.

        `_referred` ve `_coordinated` cevabı üretip bu üçünü ATLIYORDU.
        Ölçüldü: "penguen ve kartal kuş mu" doğru cevaplanıyor, hemen ardından
        "neden" -> "bunu anlamadım", "emin misin" -> "henüz bir şey söylemedim
        ki". Sistem söylediğini unutuyordu — sohbetin en temel sözleşmesi.

        Bu, doğru cevabın yanına doğru hatırayı da koyuyor.
        """
        if concept and self.memory.query(concept):
            self.thread.note(concept)
        if intent is not None:
            self.last = intent
            self.focus = concept or self.focus
            self._remember_grounds(intent)

    def _referred(self, line):
        """"ikisi de kuş mu" — özneler sohbetten gelir, cümleden değil.

        Gönderme zamiri kaç şeye işaret ediyorsa o kadar konu alınıyor ve
        her biri için aynı soru normal yoldan soruluyor. Uydurma yok: sohbette
        yeterli konu yoksa bu yol hiç açılmıyor ve cümle olağan akışa düşüyor.
        """
        grammar = getattr(self.language, "grammar", None)
        morphology = getattr(grammar, "morphology", None)
        marks = set(getattr(morphology, "plural_pronouns", ()) or ())
        if not marks:
            return None
        tokens = tokenize(line)
        # Zamir çekimli de gelebilir: "İKİSİNİN farkı ne". Ölçüt yine kapalı
        # sınıf — yalnız dilin gönderme zamirleri, gövdesinden tanınıyor.
        at = next((i for i, token in enumerate(tokens)
                   if token in marks
                   or any(token.startswith(mark) and len(token) - len(mark) <= 4
                          for mark in marks)), None)
        if at is None:
            return None
        tail = [token for token in tokens[at + 1:]
                if token not in marks
                and token not in getattr(morphology, "clitics", ())]
        if not tail:
            return None
        topics = self.thread.recent(2)
        if len(topics) < 2:
            return None
        # Önce İKİLİ okuma: "ikisinin farkı ne" iki konuyu tek soruda istiyor,
        # konu başına aynı soruyu sormak değil. İki konu dilin kendi
        # bağlayıcılarıyla birleştirilip olağan yoldan okutulur — cümleyi
        # yine dilbilgisi anlar, burada hiçbir kalıp yazılı değil. Yalnız
        # karşılaştırma okuması kabul ediliyor; olmadıysa tek tek yol sürüyor.
        links = (tuple(getattr(morphology, "postpositions", ())[:1])
                 + tuple(getattr(morphology, "joiners", ())))
        for link in links:
            both = " ".join([topics[1], link, topics[0]] + tail)
            found = self.language.understand(both)
            if found.kind == ASK_COMPARE:
                answer = self.gate.answer(self._typed(found))
                if not is_a_refusal(answer):
                    self._kept(found, topics[1])
                    return answer
        answers = []
        for topic in reversed(topics):
            said = None
            indefinite = getattr(morphology, "indefinite", "")
            for words in ([topic] + tail, [topic, indefinite] + tail):
                found = self.language.understand(" ".join(words))
                if found.kind in (UNKNOWN, UNKNOWN_WORD, AMBIGUOUS):
                    continue
                answer = self.gate.answer(self._typed(found))
                if not is_a_refusal(answer):
                    said = answer
                    break
            if said is None:
                return None
            answers.append(said)
            self._kept(found, topic)
        return coordination.combine(answers)

    def _coordinated(self, line):
        """"penguen ve kartal kuş mu" — iki özne, iki soru, tek cevap.

        Bölünmüş parçaların her biri normal yoldan, kapıdan geçerek
        cevaplanıyor; birleştirme yalnızca söyleyişte. Yani bu yol hiçbir
        denetimi atlamıyor.
        """
        grammar = getattr(self.language, "grammar", None)
        morphology = getattr(grammar, "morphology", None)
        joiners = set(getattr(morphology, "joiners", ()) or ())
        if not joiners:
            return None         # başka bir dil organı: bağlaçlarını bilmiyoruz
        tokens = tokenize(line)
        subjects, tail = coordination.split_subjects(tokens, joiners,
                                                     grammar.known())
        if not subjects:
            return None
        answers = []
        for subject in subjects:
            # Kuyruk iki türlü kurulabiliyor: "penguen kuş mu" ve "penguen BİR
            # kuş mu". İkisi de Türkçe ve hangisinin kalıbı olduğunu önceden
            # bilmiyoruz.
            #
            # Ölçüt "anlaşıldı mı" DEĞİL, "cevap üretti mi". İlki yetmiyordu:
            # "penguen kuş mu" cümlesi NİTELİK sorusu diye ayrıştırılıyor —
            # yanlış ama anlaşılmış sayılıyor — ve döngü orada durup cevapsız
            # kalıyordu. Aynı disiplin sistemin her yerinde: bir okuma ancak
            # işe yarıyorsa kabul edilir.
            said = None
            for words in ([subject] + tail,
                          [subject, getattr(morphology, "indefinite", "")] + tail):
                found = self.language.understand(" ".join(words))
                if found.kind in (UNKNOWN, UNKNOWN_WORD, AMBIGUOUS):
                    continue
                found = self._typed(found)
                answer = self.gate.answer(found)
                if not is_a_refusal(answer):
                    said = answer
                    break
            if said is None:
                return None
            answers.append(said)
            # `_typed()` burada da uygulanıyor. Önce yoktu ve iki kardeş yol
            # aynı soruyu farklı ilişkiyle okuyordu: "ikisi de kuş mu" tür
            # sorusu, "penguen ve kartal kuş mu" nitelik sorusu sayılıyordu.
            self._kept(found, subject)
        return coordination.combine(answers)

    def _fluent(self, intent, line, said):
        """Cevabı akıcı söyletmeyi dener. Denetimi geçmezse şablon kalır.

        Kayıp yok: ağız düşerse grafın düz cevabı zaten elde. Kazanç ise
        çekirdeğin kurduğu cümle — kalıp değil.
        """
        if self.voice is None or is_a_refusal(said) or not intent.concept:
            return said
        try:
            return self.voice(said, line, self.memory, self.gate.reasoning,
                              (intent.concept,))
        except Exception:                                   # noqa: BLE001
            return said

    def _more(self):
        """"başka ne biliyorsun" — o an konuşulan kavram hakkında henüz
        söylenmemiş olan.

        Uydurma yok: söylenecek bir şey kalmadıysa öyle deniyor. Sohbetin
        neresinde olduğunu bilmek, bilgiyi tekrar etmemek demek — ve bir
        insan sohbeti en çok bununla ilerliyor.
        """
        if not self.focus:
            return nothing_more(None)
        said = self.told.setdefault(self.focus, set())
        fresh = []
        for edge in self.memory.query(self.focus):
            mark = (edge.relation, edge.target)
            if mark in said or edge.relation == SAME_AS:
                continue
            said.add(mark)
            fresh.append(describe(self.focus, edge.relation, edge.target,
                                  edge.object, edge.role, self.memory.kinds))
            if len(fresh) >= 3:
                break
        if not fresh:
            return nothing_more(self.focus)
        return sentences(fresh)

    def _asks_something(self, line):
        """Cümlede soru sözcüğü var mı — varsa bildirme sayılamaz.

        Dil organı takılabilir olduğu için biçimbilim garanti değil; yoksa
        denetim yapılmıyor ve davranış eskisi gibi kalıyor.
        """
        from lmm import asking
        grammar = getattr(self.language, "grammar", None)
        morphology = getattr(grammar, "morphology", None)
        if morphology is None:
            return False
        # Soru İŞARETİ en açık kanıt ve hiç bakılmıyordu — `tokenize`
        # noktalamayı attığı için görünmez. Ölçüldü: 400 gerçek Türkçe cümlede
        # grafa yazılan 10 kaydın 10'u soru işaretiyle bitiyordu.
        # İşaretin kendisi dilden soruluyor: İspanyolca "¿", Yunanca ";"
        # kullanır ve kod hangisi olduğunu bilmemeli.
        if line.rstrip().endswith(tuple(getattr(morphology,
                                                "question_marks", ()))):
            return True
        for token in tokenize(line):
            if asking.interrogative_of(token, morphology) is not None:
                return True
            if asking.particle_of(token, morphology) is not None:
                return True
        return False

    def _bare_follow_up(self, line):
        """Tek kelimelik eksiltili soru: "neden", "nasıl", "kim".

        "penguen uçar mı" -> "hayır" -> "neden" dendiğinde sorulan şey
        "penguen neden uçamaz"dır. Bağlam mekanizması "peki ya kartal" için
        vardı ama çıplak soru sözcüğünü kapsamıyordu — ve insan sohbette en
        çok onu kullanıyor.

        Önceki soru yoksa hiçbir şey uydurulmuyor.
        """
        if self.last is None:
            return None
        words = tokenize(line)
        if len(words) != 1:
            return None
        from lmm import asking
        morphology = self.language.grammar.morphology
        bare = asking.interrogative_of(words[0], morphology)
        if bare is None:
            return None
        kind = FOLLOW_UPS.get(bare)
        if kind is None:
            return None
        return Intent(kind, self.last.concept, self.last.relation,
                      self.last.target, self.last.object, self.last.role,
                      self.last.quantifier)

    def _carried(self, line):
        """Eksiltili soru: önceki sorunun aynısı, yeni bir özneyle.

        "penguen uçar mı" dedikten sonra "peki ya kartal" demek, kartal için
        aynı soruyu sormaktır. Bunu anlamak için son sorunun TÜRÜNÜ hatırlamak
        gerekiyor; sistem yalnızca son KAVRAMI hatırlıyordu.
        """
        if self.last is None:
            return None
        words = tokenize(line)
        if not words or len(words) > 4:
            return None
        rest = [w for w in words
                if w not in getattr(self.language.grammar.morphology,
                                    "openers", ())]
        if len(rest) != 1 or len(rest) == len(words):
            return None                 # eksiltme işareti yok
        # Biçimbilim grammar'da duruyor. Burada `self.memory.morphology`
        # aranıyordu ve o öznitelik hiç var olmadı: koşul her zaman False
        # dönüyor, çoğul hiç soyulmuyordu. "peki ya kartallar" tanınmıyordu.
        concept = self.language.grammar.morphology.strip_plural(rest[0])
        if concept not in set(self.memory.concepts()):
            return None                 # tanımadığı bir şey için tekrarlama
        repeated = Intent(self.last.kind, concept, self.last.relation,
                          self.last.target, self.last.object, self.last.role,
                          self.last.quantifier)
        return repeated

    def _typed(self, intent):
        """Hedef bilinen bir TÜR ise soru nitelik değil tür sorusudur.

        "penguen kuş mudur" cümlesinde "bir" yok, o yüzden ayrıştırıcı nitelik
        okuyordu. Küçük grafta bu fark etmiyordu; graf büyüyünce sistem okumanın
        yanlış olduğunu görüp reddetti ve — yerine tür okumasını denemediği için
        — cümleyi hiç anlamadı. Yani BÜYÜME bir soruyu bozmuştu.

        Ayrımı yapacak bilgi zaten grafta: bir şeyin ALTINDA başka şeyler varsa
        o bir türdür. "kuş"un altında penguen ve kartal var; "hızlı"nın altında
        kimse yok. Kural yazmaya gerek yok, sayması yeterli.
        """
        if intent.relation != HAS_PROPERTY or not intent.target:
            return intent
        if self.memory.direct(intent.concept, HAS_PROPERTY, intent.target):
            return intent           # gerçekten nitelik: kayıtta öyle duruyor
        below = any(edge.target == intent.target and edge.relation == IS_A
                    for edge in self.memory.edges)
        if below:
            intent.relation = IS_A
        return intent

    def _in_context(self, intent):
        """Fill in what the sentence left out, and remember what it was about.

        Nothing here is a context window: knowledge is already permanent. This
        only tracks the thread of the conversation, so "peki yüzer mi" means
        what a person would take it to mean.
        """
        if intent.concept is None or intent.concept in PRONOUNS:
            intent.concept = self.focus
        elif intent.concept:
            self.focus = intent.concept
        if intent.concept is None and intent.kind not in SUBJECTLESS:
            # Nothing said, nothing to carry on from: better to admit it than
            # to answer about None.
            return Intent(UNKNOWN, confidence=0.0)
        return intent

    def _investigate(self, intent):
        """Bilmediği bir kavram sorulduysa gidip okur. Ne yaptığını döndürür.

        Bu, üçüncü şartın edilgen hâlinden etkin hâline geçtiği yer. Bugüne
        kadar graf yalnızca biri bir şey öğretirse büyüyordu; burada sistem
        kendi eksiğini fark edip kaynağa gidiyor. Ağırlıklar değişmiyor,
        yeniden eğitim yok — bilgi zaten ağırlıkta değil.

        Soruşturma kapalıysa hiçbir şey olmaz: ağ isteğe bağlı, davranışın
        varsayılanı değil.
        """
        if self.inquiry is None or not intent.concept:
            return ""
        concept = intent.concept
        if self.memory.query(concept) or concept in self.investigated:
            return ""       # bilineni yeniden okumak grafı büyütmez, yavaşlatır
        self.investigated.add(concept)
        try:
            report = self.inquiry(self.memory, concept)
        except Exception:                                   # noqa: BLE001
            return ""       # ağ yoksa sohbet durmaz; bilmemek bir cevaptır
        if not report.get("kaynak"):
            return ""
        return went_and_read(report["kaynak"], report.get("yazılan", 0),
                             report.get("çelişen", 0) + report.get("uydurma", 0))

    def _question(self, intent):
        read = self._investigate(intent)
        if read:
            # Okuduktan sonra soru yeniden değerlendirilir: cevap artık grafta
            # olabilir ve öyleyse insana sormaya gerek yok.
            answer = (self.gate.answer(intent)
                      if self.pursuit.resolved(intent)
                      else self.pursuit.opening(intent))
            self.goal = None if self.pursuit.resolved(intent) else intent
            return f"{read} {answer}".strip()
        if intent.kind == ASK and not self.pursuit.resolved(intent):
            self.goal = intent          # hold it; a plan beats a shrug
            self.goal_steps = set()
            opening = self.pursuit.opening(intent)
            step = self.pursuit.next_step(intent)
            if step is not None:
                self.goal_steps.add(step.key)
            return opening
        self.goal = None
        return self.gate.answer(intent)

    def _after_learning(self):
        """A goal in hand outranks idle wondering."""
        if self.goal is not None:
            if self.pursuit.resolved(self.goal):
                answer = self.gate.answer(self.goal)
                self.goal = None
                return now_i_can(answer)
            step = self.pursuit.next_step(self.goal)
            if step is not None and step.key not in self.goal_steps:
                self.goal_steps.add(step.key)
                return step.text
            return ""
        # Çıkarım SÖYLENİYOR ama YAZILMIYOR. Ölçüldü: 128 binlik grafta her
        # öğretme turu grafa bir uydurma olgu yazıyordu —
        #
        #   > glorp bir kuştur
        #   öğrendim ... sanırım ÜÇOBALAR BİR KILIÇTIR
        #   yazılan: üçobalar --type--> kılıç  kaynak=çıkarım
        #
        # Öğretilenle ilgisi yok: `propose` bütün grafı tarayıp ilk sahipsizi
        # döndürüyor. Üç ayrı sıkılaştırma denendi (bilgi değeri eşiği, ortak
        # sahip sayısı, tek raf şartı) ve hiçbiri yetmedi; yalnız saçmalığın
        # türü değişti ("mençeler bir kaplandır", çünkü ikisi de `ankara` ve
        # `beypazarı` taşıyor).
        #
        # Mekanizmanın ÖNCÜLÜ bu veride geçersiz: "aynı davranışı paylaşıyor,
        # öyleyse aynı şeydir" ancak davranış o türü TANIMLIYORSA geçerli, ve
        # Vikipedi'den gelen konum olguları hiçbir şey tanımlamıyor.
        #
        # Yazmayı durdurmak yeteneği kapatmak değil, hafızayı korumak: tahmin
        # söyleniyor, insan onaylarsa öğretebiliyor. Uydurmamak bu mimarinin
        # tek şartı ve bir tahmin, kalıcı hafızaya kendi başına giremez.
        # Öğretilen kavramın ailesi taranıyor, tüm graf değil.
        hypothesis = self.induction.propose(getattr(self.last, "concept", None)
                                            or self.focus)
        if hypothesis is not None:
            # GENELLEME yazılıyor, YERLEŞTİRME yazılmıyor. İkisi ayrı türden:
            # genelleme sayıma dayanıyor ("bu ailenin şu kadar üyesi bunu
            # yapıyor") ve `support` taşıyor; yerleştirme şekle dayanıyor
            # ("bu şey şuna benziyor") ve hiçbir kanıt taşımıyor.
            if not getattr(hypothesis, "guessed", False):
                self.induction.learn(hypothesis)
            return generalising(hypothesis.examples,
                                describe(hypothesis.concept, hypothesis.relation,
                                         hypothesis.target))
        question = self.curiosity.next_question()
        return wondering(question.text) if question is not None else ""

    def _resolve_pending(self, line):
        """Bekleyen onayın cevabı — ya da cevap değilse None.

        Önce "evet değilse ret" sayılıyordu ve bekleyen onay varken gelen HER
        satır yutuluyordu. Ölçüldü, iki şey birden kaybediliyordu:

            > penguen uçamaz          -> "bir çelişki fark ettim ... öğreneyim mi?"
            > bazı kuşlar uçar mı     -> "tamam, öğrenmedim."

        Kullanıcının SORUSU cevapsız kaldı ve istisna da hiç öğrenilmedi. Bir
        soru, bir evet/hayır sorusunun cevabı değildir.

        Ölçüt sözcük listesi değil YAPI: satır onaysa onay, retse ret, ikisi de
        değilse bu bir cevap değildir ve olağan yoldan işlenmeli. Bekleyen soru
        duruyor — öğretmen açıklayıcı bir soru sorup sonra "evet" diyebilir ve
        istisna hâlâ kaydedilir.
        """
        morphology = self.language.grammar.morphology
        spoken = lower(line).strip(" .,!?")
        if spoken in getattr(morphology, "affirmations", ()):
            edge, self.pending = self.pending, None
            self.learning.confirm_exception(edge)
            return exception_learned()
        if spoken in getattr(morphology, "refusals", ()):
            self.pending = None
            return not_learned()
        return None

    def save(self):
        self.memory.save(self.path)


def _status(session):
    """Sayıları oturumdan okur, söylemeyi söyleyişe bırakır."""
    memory = session.memory
    return status(len(memory.edges), len(memory.concepts()),
                  len(memory.vocabulary), len(memory.kinds.known()))


def _inquirer():
    """Soruşturma açıksa: bilmediğini fark edince gidip okuyan işlev.

    Ağ ve bir okuyucu gerektiriyor, o yüzden isteğe bağlı ve kapalı gelir.
    Kurulamıyorsa sohbet soruşturmasız sürer — bilmemek de bir cevaptır.
    """
    from lmm import inquiry
    from lmm.compiler import CompileError, model_reader
    try:
        reader = model_reader(temperature=0.0)
    except CompileError:
        reader = None       # okuyucu yoksa kendi çerçeve okuyucumuz devreye girer
    return lambda memory, concept: inquiry.investigate(memory, concept,
                                                       reader=reader)


def _reader(backbone="cekirdek"):
    """Eğitilmiş niyet okuyucusu. Yüklenemezse None — sistem kalıplarla sürer.

    Torch `lmm/` içine hiç girmiyor: yükleme `core/intent.py`'de ve o dosya
    olmadığında da her şey çalışıyor. Sıfır bağımlılık sınırı burada korunuyor.
    """
    try:
        from core.intent import Reader
    except Exception:                                       # noqa: BLE001
        return None
    found = Reader(backbone)
    return found.read if found.ready else None


def _dictionary():
    """Vikisözlük getiricisi. Ağa nazik davranmak için araya bekleme koyar."""
    import time
    from lmm import synonyms

    def polite(word):
        found = synonyms.fetch(word)
        time.sleep(1.1)
        return found
    return polite


def _wording():
    """Söyleyiş öğrenici. Dil modeli yoksa None — sistem kalıplarla sürer."""
    try:
        from lmm.compiler import model_reader
        from lmm import wording as _w
        return model_reader(temperature=0.0, brief=_w.BRIEF)
    except Exception:                                       # noqa: BLE001
        return None


def _voice():
    """Akıcı ağız. Yüklenemezse None — sistem şablonlarla sürer."""
    try:
        from core.bridge import load
        from lmm import registry
        bridge, _ = load(registry.where("core"))
    except Exception:                                       # noqa: BLE001
        return None
    return lambda said, line, memory, reasoning, concepts: bridge.express(
        said, line, memory, reasoning, concepts=concepts)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    asking = "--soruştur" in sys.argv or "--sorustur" in sys.argv
    # Niyet ağı artık VARSAYILAN. İlke: elde olanı kullan, olmayanı arama.
    #
    # Bayrağın arkasında olmasının bedeli ölçüldü — kimse açmadıkça sistem
    # aynı cümleye sonsuza kadar "anlamadım" diyordu, oysa yükleme 2,1 sn ve
    # soru başına 14 ms. Yokluğunda 40 sorunun 4'ü kayboluyor.
    #
    # Sıfır bağımlılık iddiası bozulmuyor: torch ya da model yoksa `_reader`
    # sessizce None dönüyor ve sistem kalıplarla çalışmaya devam ediyor. Bunu
    # kapatmak isteyen `--kalıpsız` diyebilir.
    understanding = not ("--kalıpsız" in sys.argv or "--kalipsiz" in sys.argv)
    fluent = "--akıcı" in sys.argv or "--akici" in sys.argv
    looking_up = "--sözlük" in sys.argv or "--sozluk" in sys.argv
    # Söyleyiş öğrenme DIŞARIYA çıkıyor: her anlaşılmayan cümlede bir dil
    # modeline istek gidiyor. Yerel ve bedava olan şeyler varsayılan olabilir,
    # dışarı çıkan ve para harcayan şeyler olamaz — istenmeden yapılmamalı.
    teaching = "--öğren" in sys.argv or "--ogren" in sys.argv
    backbone = "hazir" if "--hazir" in sys.argv else "cekirdek"
    path = args[0] if args else "memory.json"
    reader = _reader(backbone) if understanding else None
    # `wording` (söyleyiş öğrenme) hiç geçirilmiyordu: `Session` parametreyi
    # kabul ediyor, `main()` vermiyordu. Kurulmuş, testli bir organ yalnızca
    # testlerden çağrılıyordu — `abduction` ve `spreading` ile aynı hikâye.
    #
    # Aynı bayrağa bağlı: `--anla` zaten dil modeli gerektiriyor, söyleyiş
    # öğrenme de onu gerektiriyor.
    session = Session(path, inquiry=_inquirer() if asking else None,
                      intent=reader, voice=_voice() if fluent else None,
                      wording=_wording() if teaching else None,
                      dictionary=_dictionary() if looking_up else None)
    print(opened(path))
    print(f"  {_status(session)}")
    if asking:
        print(inquiry_open())
    if understanding:
        print(intent_network(backbone, reader))
        if session.wording:
            print(wording_open())
    if looking_up:
        print(dictionary_open())
    if fluent:
        print(voice_state(session.voice))
    print(help_hint())
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        morphology = self.language.grammar.morphology
        if lower(line) in getattr(morphology, "exit_words", ()):
            break
        if lower(line) in getattr(morphology, "help_words", ()):
            print(help_text())
            continue
        if lower(line) in getattr(morphology, "status_words", ()):
            print("  " + _status(session))
            continue
        print(session.respond(line))
        session.save()
    session.save()
    print(saved_and_gone())


if __name__ == "__main__":
    main()
