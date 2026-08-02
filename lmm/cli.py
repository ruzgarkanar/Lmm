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
                           ASK_OPINION, lower, tokenize)
from lmm import social
from lmm import coordination
from lmm.thread import Thread
from lmm import wording
try:
    from core.intent import reading_of
except Exception:                                           # noqa: BLE001
    def reading_of(name, has_target=True):   # torch yoksa: kalıplarla çalışır
        return None
from lmm.distill import split_words
from lmm.grammar import Pattern
from lmm.discovered import words_of
from lmm import arithmetic
from lmm.learning import (LearningLoop, CONFLICT, LEARNED, CORRECTED,
                          DISPUTE, FROZEN)
from lmm.induction import Induction
from lmm.network import MiniNetwork
from lmm.curiosity import Curiosity
from lmm.pursuit import Pursuit
from lmm.phrasing import (wondering, not_understood, now_i_can, generalising,
                          describe, teach_me_the_word, learned_word, computed,
                          cannot_compute, which_reading, went_and_read,
                          learned_wording, is_a_refusal, talked_about,
                          nothing_more, capitalize, inventory,
                          certainty, whence, no_opinion)

# A custom LanguageOrgan may report graded confidence; ours parses or does not.
CONFIDENCE_THRESHOLD = 0.35
RESEMBLANCE = 0.5       # below this, guessing at what you meant is noise
# Ağın tahmini bu güvenin altındaysa hiç denenmiyor. Denetim zaten var ama
# düşük güvenli tahminler denetimi boşuna meşgul ediyor.
NEURAL_THRESHOLD = 0.4
# Çıplak soru sözcüğü hangi soruyu sürdürüyor. Kapalı bir eşleme: sözcükler
# zaten bildirilmiş kapalı sınıftan, buradaki yalnızca hangi niyete
# karşılık geldikleri.
FOLLOW_UPS = {"neden": ASK_WHY, "niye": ASK_WHY, "niçin": ASK_WHY,
              "nasıl": ASK_PROPERTIES, "kim": ASK_WHO, "kimler": ASK_WHO}
AFFIRMATIVE = ("evet", "e")
EXIT_WORDS = ("çık", "cik", "exit")
# Öznesi olmayan niyetler. Bunlar bağlamdan özne almazlar ve almadıkları için
# "anlaşılmadı" sayılmamalılar — sohbetin kendisi hakkında bir sorunun öznesi
# yoktur. ASK_THREAD burada olmadığı için cevap üretiliyor ama sessizce
# "anlamadım"a çevriliyordu.
SUBJECTLESS = (ASK_WHO, UNKNOWN, UNKNOWN_WORD, AMBIGUOUS, ASK_THREAD,
               ASK_MORE, ASK_INVENTORY, ASK_CERTAINTY, ASK_SOURCE)


class Session:
    """One conversation: the five organs wired together over a memory file."""

    def __init__(self, path, language=None, inquiry=None, wording=None,
                 intent=None, voice=None, dictionary=None):
        self.path = path
        self.memory = Memory.load(path)
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
                                              known=self.memory.concepts,
                                              meanings=self._meanings)
        for entry in self.memory.patterns:      # ways of speaking it was taught
            self.language.grammar.add(Pattern.from_dict(entry), first=True)
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
        found = set()
        for edge in self.memory.query(word, SAME_AS):
            found.add(edge.target)
        for edge in self.memory.edges:
            if edge.relation == SAME_AS and edge.target == word:
                found.add(edge.concept)
        return found

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
        found = reading_of(name, has_target=False)
        if found is None:
            return None
        kind, relation = found
        morphology = self.language.grammar.morphology
        at, concept = wording.concept_in(tokens, self.memory, morphology,
                                         self.focus)
        if concept is None:
            return None
        said = self.gate.answer(Intent(kind, concept, relation))
        if is_a_refusal(said):
            return None
        # Cevap üretebilmek yetmiyor: okuma cümleye de uymalı. Bu kapı olmadan
        # "bir kuşun uçabilmesi için ne gerekir" cümlesi "kuş ne yapabilir"
        # diye okunuyor, graf cevaplıyor ve YANLIŞ kalıp kalıcı yazılıyordu.
        if not wording.accounted_for(kind, tokens, self.memory.lexicon,
                                     self.language.grammar.patterns,
                                     self._meanings):
            return None
        self.asked_about.add(line)
        reading = wording.Reading(kind, relation, at, None)
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
        if self.pending is not None:
            return self._resolve_pending(line)
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
            return not_understood(resembles, self.memory)
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
            said = self._fluent(intent, line, self._question(intent))
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
        at = next((i for i, token in enumerate(tokens) if token in marks), None)
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
        answers = []
        for topic in reversed(topics):
            said = None
            for words in ([topic] + tail, [topic, "bir"] + tail):
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
            for words in ([subject] + tail, [subject, "bir"] + tail):
                found = self.language.understand(" ".join(words))
                if found.kind in (UNKNOWN, UNKNOWN_WORD, AMBIGUOUS):
                    continue
                answer = self.gate.answer(found)
                if not is_a_refusal(answer):
                    said = answer
                    break
            if said is None:
                return None
            answers.append(said)
            self.thread.note(subject)
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
        return capitalize(". ".join(fresh) + ".")

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
        for token in tokenize(line):
            if asking.interrogative_of(token, morphology) is not None:
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
        opener = {"peki", "ya", "yaa", "e", "ee", "pekiya"}
        rest = [w for w in words if w not in opener]
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
        hypothesis = self.induction.propose()
        if hypothesis is not None:
            self.induction.learn(hypothesis)
            return generalising(hypothesis.examples,
                                describe(hypothesis.concept, hypothesis.relation,
                                         hypothesis.target))
        question = self.curiosity.next_question()
        return wondering(question.text) if question is not None else ""

    def _resolve_pending(self, line):
        edge, self.pending = self.pending, None
        if lower(line) in AFFIRMATIVE:
            self.learning.confirm_exception(edge)
            return "öğrendim (istisna olarak işledim)."
        return "tamam, öğrenmedim."

    def save(self):
        self.memory.save(self.path)


HELP = """
ÖĞRETMEK                          SORMAK
  penguen bir kuştur                penguen nedir
  penguen bir memeli değildir       penguen bir kuş mu
  kuşlar uçar                       penguen uçar mı
  penguen uçamaz                    kimler uçar
  kuşlar tüylüdür                   penguen tüylü mü
  bazı kuşlar uçmaz                 bazı kuşlar uçar mı
  kuşun kanadı var                  kuşun kanadı var mı
  penguen kutupta yaşar             penguen neden uçamaz
  kediler fare yakalar              penguen ne yapabilir
  kartal serçeden büyüktür          penguen nasıldır
  kelime: koşmak = koşar / koşamaz  penguen anlat
                                    17 çarpı 43 kaç

  Konuşmayı sürdürür: "peki yüzer mi", "anlat", "nasıldır"
  Komutlar: yardım · durum · çık
"""


def _status(session):
    memory = session.memory
    return (f"{len(memory.edges)} bilgi · {len(memory.concepts())} kavram · "
            f"{len(memory.vocabulary)} öğrenilmiş kelime · "
            f"{len(memory.kinds.known())} ilişki türü")


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
    print(f"LMM — yaşayan bellek: {path}")
    print(f"  {_status(session)}")
    if asking:
        print("  soruşturma açık: bilmediğim bir şey sorulursa gidip okurum.")
    if understanding:
        print(f"  niyet ağı: {'açık (' + backbone + ')' if reader else 'yok'}"
              " — kalıp yetmediğinde devreye giriyor.")
        if session.wording:
            print("  söyleyiş öğrenme: açık"
                  " — anlaşılmayan cümleden kalıcı kalıp çıkarır.")
    if looking_up:
        print("  sözlük: açık — tanımadığı sözcüğün eş anlamlısını arar.")
    if fluent:
        print(f"  akıcı ağız: {'açık' if session.voice else 'YÜKLENEMEDİ'}"
              " — cevap çekirdeğe söyletilir, geri okunup denetlenir.")
    print("  'yardım' yazarsan ne söyleyebileceğini gösteririm.")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if lower(line) in EXIT_WORDS:
            break
        if lower(line) in ("yardım", "yardim", "help", "?"):
            print(HELP)
            continue
        if lower(line) in ("durum", "istatistik"):
            print("  " + _status(session))
            continue
        print(session.respond(line))
        session.save()
    session.save()
    print("bellek kaydedildi. hoşça kal.")


if __name__ == "__main__":
    main()
