"""Oturum: bütün organları tek akışta birleştiren yer.

Buraya kadar altı parça ayrı ayrı kuruldu ve her biri kendi başına sınandı.
Bu dosya onları birbirine bağlıyor ve mimarinin tamamı ilk kez uçtan uca
çalışıyor:

    cümle
      -> OKUYUCU AĞI      işlem üretir (PASS · ASK · WRITE)
      -> GEOMETRİ         etiketleri kimliğe çevirir, çağrışımla kayıt toplar
      -> KAPI             yazacaksa denetler · söyleyecekse destekleri arar
      -> TARTIM           konuşmadan önce N adayı üç teraziden geçirir
      -> KONUŞUCU AĞI     kayıtlardan cümle kurar
      -> KAPI (çıkış)     geri okunmayan cümle DÜŞER
      -> YAŞANTI          ne söylendi, ne oldu — şaşkınlıkla ağırlıklı

İç tartım, "böyle mi desem şöyle mi" adımının karşılığıdır ve üç terazisi
vardır:

    DESTEK    her iddianın kaydı var mı (kapı)
    YAŞANTI   benzerini söylediğimde ne olmuştu (geçmiş sonuçlar)
    KAPSAM    sorulan kayıtların ne kadarını söylüyor

Ağlar yoksa sistem susmaz, ham kayıtları döker: konuşamaz ama yalan da
söylemez. Dördüncü şart her durumda geçerlidir.

Dile ait hiçbir şey yoktur.
"""
from v3 import dynamics, geometry
from v3.gate import Gate, SPEAK
from v3.memory import Memory, OPERATOR, STRANGER
from v3.reader import ASK, PASS, Reader, WRITE
from v3.speaker import Speaker, prompt_of

# Okuyucunun işlemi uygulanmadan önce taşıması gereken en az güven. Altındaki
# okuma, okuma sayılmaz — emin olmadığını yazmak, uydurmanın kapısıdır.
CERTAIN = 0.5

# Bir cevapta en çok kaç kayıt konuşur. Fazlası cevap değil döküm olur.
MOST = 8


class Session:
    """Bir konuşma. Belleği taşır, organları bağlar, yaşantısını kaydeder."""

    def __init__(self, path=None, speaker_name=None):
        self.path = path
        self.memory = Memory.load(path) if path else Memory()
        self.gate = Gate(self.memory)
        self.reader = Reader()
        self.speaker = Speaker()
        # Kim konuşuyor: işletmeci mi, tanınmayan biri mi. Yazılan her kaydın
        # basamağı buradan gelir ve bir yabancı, doğrulanmış bilgiyi ezemez.
        # AD ile BASAMAK ayrı: ad kayda yazılır ("ali", "#operator"),
        # basamak hakemliğe girer. Duman testi ‹4› basana kadar tek alandaydı.
        self.level = STRANGER if speaker_name else OPERATOR
        self.who = speaker_name or "#operator"
        self.focus = []             # son konuşulan kimlikler — bağlam
        self.turns = 0              # auto-uyku sayacı
        # Geometri tablosu: varsa her kimlik doğarken vektörünü alır. Yoksa
        # sistem çalışır ama çokanlamlılık çözümü zayıflar — kolaylık,
        # bağımlılık değil.
        self.vectors = {}
        root = __import__("os").path.dirname(
            __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
        table = __import__("os").path.join(root, "models/v3/vectors.json")
        try:
            import json, os
            if os.path.exists(table):
                self.vectors = json.load(open(table, encoding="utf-8"))
                from v3 import embed
                embed.attach(self.memory, self.vectors)
        except Exception:                                   # noqa: BLE001
            self.vectors = {}
        if self.memory.self_key is None:
            self.memory.self_key = self.memory.identify("#self")

    # --- ana akış -------------------------------------------------------

    def respond(self, line):
        """Bir cümleye cevap. Dönen daima metin, asla desteksiz iddia."""
        # AUTO-UYKU: belge §117 uyku turunu canlıya bağlar. Denetim yakaladı —
        # solma/damıtma/hakemlik yalnız elle çağrılıyordu, sistem hiç
        # "uyumuyordu". Her 50 turda bir damıtır, sönümler, hakemler.
        self.turns += 1
        if self.turns % 50 == 0:
            dynamics.sleep(self.memory)
        operation = self.reader.read(line)
        # YAZIM yolu ancak yazacak bir şey VARSA açılır. Denetim yakaladı:
        # okuyucu soruyu WRITE sanınca (ASK verisi olmadan eğitilmişti)
        # `_write` boş özne/değerle boş dizgi dönüyordu ve kullanıcı soruya
        # hiç cevap alamıyordu. Eksik parçalı WRITE artık toplama yoluna
        # düşer — yanlış okumanın bedeli sessizlik değil, deneme olur.
        # Yazım için özne + değer ŞART (ikisi de metinde, hizalanabilir).
        # YÜKLEM şart DEĞİL: ölçüldü, yüklem cümlede yalnız %0,5 geçiyor
        # ("tür", "özellik" birer çıkarım, metinde yok) — yüklemi şart koşmak
        # öğretmenin %99,5'ini düşürüyordu. Yüklem, değerin türünden graf
        # tarafında belirlenecek; okuyucudan metinde olmayan bir şey istemek
        # yanlıştı.
        #
        # "penguen nedir" olgu diye yazılmasın diye asıl kapı GERİ OKUMA:
        # kurulacak kayıt serimlenip geri okunuyor, aynı WRITE çıkmıyorsa
        # yazılmıyor. Bu dil bilmez — okuyucunun kendi tutarlılığına bakar.
        if (operation.kind == WRITE and operation.confidence >= CERTAIN
                and operation.subject and operation.value):
            return self._write(operation, line)
        # PASS + özne yok = sohbeti süren söz ("selam", "naber"). Cevabı
        # KONUŞUCU üretir — girdiyi diyalog bağlamı olarak alıp karşılığını
        # kurar. Denetim yakaladı: buradan konuşucuya HİÇ yol yoktu, sabit
        # "[·]" dönüyordu; konuşucu diyaloğu öğrense bile çağrılmıyordu.
        #
        # Kapı yine önünde: sohbet cevabı bir OLGU iddiası taşımamalı. İçinde
        # belleğe yazılabilir bir üçlü varsa (uydurma) düşer; taşımıyorsa
        # (selamlaşma, tepki) geçer — çünkü doğru/yanlış olamaz.
        if operation.kind == PASS and not operation.subject:
            said = self._chat(line)
            self._live(line, said, [])
            return said or "[·]"
        records = self._gather(operation, line)
        said = self._speak(records, line)
        self._live(line, said, records)
        # Boş cevap sessizlik değil, açık RET imidir.
        return said or "[?]"

    # --- yazma ----------------------------------------------------------

    def _write(self, operation, line):
        """Öğretilen bir şeyi belleğe koyar — kapıdan geçerek."""
        if not (operation.subject and operation.value):
            return ""
        subject = self._identity(operation.subject)
        predicate = self._identity(operation.predicate or "")
        value = self._identity(operation.value)
        # Yazmadan ÖNCE bellekte ne olduğunu bil — tahmin. Yeni olgu onunla
        # çelişirse ŞAŞKINLIK. Belge §110: tahmin tutmadıysa yaşantı pahalı.
        # Kullanıcı tepkisini beklemeden, çelişkinin kendisi tahmin hatasıdır
        # ve otomatik girer (denetim: reacted hiç çağrılmıyordu, surprise hep
        # 0'dı).
        expected = self.gate._contradiction(subject, predicate, value)
        record, why = self.gate.admit(subject, predicate, value,
                                      self.who, self.level)
        if record is None:
            return ""
        surprise = 1.0 if (why == 1 or expected is not None) else 0.0
        # Pekiştirme BURADA YOK: `memory.write` yinelenen olguyu zaten
        # `strengthen(source)` ile karşılıyor ve o koruma kaynak kümesine
        # bakıyor. Buradaki ikinci çağrı, tek öğretmeyi iki tanık sayıyordu
        # — denetim ölçtü ve kaldırıldı.
        self.focus = [subject]
        # TÜRETME: yeni olgu, bilinenlerle ZİNCİRLENİR — dümdüz ezber değil,
        # çıkarım. "kartal→kuş" + "kuş→hayvan" ⊢ "kartal→hayvan". Türetilen
        # kayıt `#inference` kaynağıyla, DÜŞÜK güvenle yazılır ve gerekçesine
        # bağlanır; söylenirken açıkça çıkarım olduğu belli olur, uydurma
        # değil. Bu, mimarinin 5. maddesi — geometri/zincir önerir, kapı
        # doğrular. Kendi yorumunu katması buradan.
        self._learn_transitive(predicate)
        if predicate in self.memory.transitive:
            self._derive(subject, predicate, value)
        # #BEN: öğrenilen her olgu #self'e de bağlanır — sistem "ne öğrendim"
        # sorusuna kendi geçmişinden cevap verebilsin (belge §98).
        self.last = self.memory.lived(line, outcome=1.0, surprise=surprise,
                                      about=[subject, self.memory.self_key])
        return self._render([record])

    def _learn_transitive(self, predicate):
        """Bu yüklem geçişli mi — graf kapalı bir ÜÇGEN gördü mü.

        Geçişlilik elle bildirilmez, veriden öğrenilir: aynı yüklemle
        A→B, B→C VE A→C üçü de TANIKsa (öğretilen/okunan, çıkarım değil), o
        yüklem geçişlidir. "tür" bir is-a örneğinden öğrenir; "sever" asla —
        çünkü sevgi zinciri grafta kapanmaz. Kural, verinin kendisinden.
        """
        if predicate in self.memory.transitive:
            return
        edges = [r for records in self.memory.by_subject.values()
                 for r in (self.memory.records[k] for k in records)
                 if r.predicate == predicate and r.source != "#inference"]
        forward = {}
        for r in edges:
            forward.setdefault(r.subject, set()).add(r.value)
        for a, bs in forward.items():
            for b in bs:
                for c in forward.get(b, ()):
                    if c in bs:                 # A→B, B→C ve A→C hepsi tanık
                        self.memory.transitive.add(predicate)
                        return

    def _derive(self, subject, predicate, value):
        """Yeni olgu (özne→değer) çevresinde İKİ YÖNLÜ zincirleme çıkarım.

        Geçişli bir yüklemde (A tür B, B tür C ⊢ A tür C) yeni bir kenar iki
        yönde zincir açar:

            İLERİ    değer→W varsa  ⊢ özne→W    (kuş→hayvan biliniyorsa
                                                 kartal→kuş yazınca kartal→hayvan)
            GERİ     X→özne varsa   ⊢ X→değer   (kartal→kuş biliniyorsa
                                                 kuş→hayvan yazınca kartal→hayvan)

        Tek yön yetmiyordu — hangi olgunun önce öğretildiğine göre zincir
        kopuyordu. Türetilen kayıt `#inference`, düşük güven, gerekçeye bağlı;
        yanlış zincir görünür ve olgu diye söylenmez.
        """
        forward = [(subject, r.value, [r.key])
                   for r in self.memory.about(value, touch=False)
                   if r.predicate == predicate]
        # GERİ: değeri ÖZNE olan kayıtlar (X → özne) ⊢ X → değer.
        # İlk yazışta `r.value == value` yazmıştım — değere işaret edeni
        # arıyordu, oysa ÖZNEYE işaret edeni aramak gerekiyor.
        incoming = [(r.subject, value, [r.key])
                    for records in self.memory.by_subject.values()
                    for r in (self.memory.records[k] for k in records)
                    if r.predicate == predicate and r.value == subject
                    and r.subject != value]
        for who, what, because in forward + incoming:
            if who != what and not self.gate.behind(who, predicate, what):
                self.gate.inferred(who, predicate, what, because)

    # --- okuma ----------------------------------------------------------

    def _gather(self, operation, line):
        """Soruya ilgili kayıtları toplar — geometri bulur, yayılım getirir."""
        labels = [one for one in (operation.subject, operation.value) if one]
        if not labels and self.focus:
            # Bağlam düşüşü yalnız TANIDIK cümlede: içinde ne derlemde ne
            # bellekte olan bir kelime varsa ("zzzq nedir") son konuya
            # düşmek, bilinmeyene bilinenle cevap vermektir — kapatıldı.
            from v3.dataset import fold
            for word in line.split():
                piece = fold("".join(ch for ch in word if ch.isalnum()))
                if (piece and piece not in self.vectors
                        and not self.memory.candidates(piece)):
                    return []
            return self._focused()          # özne yoksa son konudan devam
        # Özne SÖYLENMİŞ ama hiçbir kimliğe çözülemiyorsa cevap RET olmalı.
        # Ölçüldü: "zzzq nedir" saf vektör çağrışımıyla kartal kaydını
        # getiriyordu — bilinmeyene bilinen şeyle cevap vermek uydurmadır.
        if labels and not any(self.memory.candidates(one) for one in labels):
            return []
        weights = geometry.recall(self.memory, vector=self._vector_of(line),
                                  labels=labels, most=MOST * 3)
        if not weights:
            return []
        records = [self.memory.records[key] for key in weights
                   if key in self.memory.records]
        records.sort(key=lambda one: (-weights.get(one.key, 0.0), -one.trust))
        self.focus = [one.subject for one in records[:1]] or self.focus
        return records[:MOST]

    def _focused(self):
        """Öznesi söylenmemiş soru: son konuşulan kimlikten devam."""
        records = []
        for key in self.focus:
            records.extend(self.memory.about(key))
        records.sort(key=lambda one: -one.trust)
        return records[:MOST]

    # --- konuşma --------------------------------------------------------

    def _chat(self, line):
        """Sohbet cümlesine karşılık — konuşucudan, kapıdan geçerek.

        Girdi diyalog bağlamı olarak veriliyor ("önceki söz\n\n"); konuşucu
        eğitiminde diyalog çiftleri tam bu biçimde. Üretilen aday, olgu
        iddiası taşıyorsa reddedilir: sohbet karşılığı bilgi ÜRETMEZ, yalnız
        konuşmayı sürdürür.
        """
        if not self.speaker.ready:
            return ""
        for candidate in self.speaker.say(line + "\n\n", count=3):
            claim = self.reader.read(candidate)
            # Aday bir OLGU yazmaya kalkıyorsa (özne+değer) kapı reddeder:
            # sohbet cevabı graf iddiası olamaz, kaydı olmayan iddia söylenmez.
            if claim.subject and claim.value and self._known(claim.value):
                if not self.gate.behind(self._known(claim.subject),
                                        self._known(claim.predicate or ""),
                                        self._known(claim.value)):
                    continue
            return candidate
        return ""

    def _speak(self, records, line):
        """Adayları üretir, tartar, kapıdan geçirir. Hiçbiri geçmezse ham."""
        supported = [one for one in records if one.trust >= SPEAK]
        if not supported:
            return ""
        if not self.speaker.ready:
            return self._render(supported)
        candidates = self.speaker.say(prompt_of(supported, self.memory))
        best, score = None, -1.0
        for candidate in candidates:
            weighed = self._weigh(candidate, supported)
            if weighed > score:
                best, score = candidate, weighed
        if best is None or score <= 0.0:
            return self._render(supported)
        return best

    def _weigh(self, candidate, records):
        """İç tartım: destek · yaşantı · kapsam. Negatifse aday düşer.

        İki düzeltme, ikisi de satır satır kıyas denetiminden:

        ÇOK İDDİA — aday cümle cümle bölünüp HER parçası okunur. Tek okuma,
        çok cümleli adayın yalnız ilk iddiasını denetliyordu; kapı çıkışı
        vaat edilenden zayıftı.

        KAPSAM — eski ölçü `len(aday) / (40 · kayıt)` idi: 40 ölçülmemiş bir
        sabitti ve UZUNLUĞU ödüllendiriyordu — uzun saçmalayan aday, kısa
        doğru adayı geçebilirdi. Yeni ölçü desteklenen iddia sayısının
        istenen kayıtlara oranı: kapsama, harfle değil İDDİAYLA ölçülür.
        """
        claims = []
        for piece in candidate.replace("!", ".").replace("?", ".").split("."):
            piece = piece.strip()
            if not piece:
                continue
            operation = self.reader.read(piece)
            if operation.subject and operation.value:
                claim = (self._known(operation.subject),
                         self._known(operation.predicate or ""),
                         self._known(operation.value))
                if all(part is not None for part in claim):
                    claims.append(claim)
                else:
                    return -1.0     # bilinmeyen ada iddia: uydurma şüphesi
        passed, dropped = self.gate.supported(claims)
        if dropped:
            return -1.0             # desteksiz iddia: cümle düşer
        support = len(passed) / max(len(claims), 1) if claims else 0.0
        history = self._history(records)
        covered = len(passed) / max(len(records), 1)
        return support + history + covered

    def _history(self, records):
        """Benzerini söylediğimde ne olmuştu — yaşantıdan gelen fren."""
        about = [one.subject for one in records]
        lived = self.memory.recall(about, most=6)
        if not lived:
            return 0.0
        return sum(one.outcome * (0.5 + one.surprise)
                   for one in lived) / len(lived)

    # --- yaşantı --------------------------------------------------------

    def _live(self, line, said, records):
        """Söylediğini kaydeder. Tepki sonra gelir, `reacted` ile bağlanır."""
        about = [one.subject for one in records[:3]]
        about.append(self.memory.self_key)
        self.last = self.memory.lived(said or line, outcome=0.0,
                                      surprise=0.0, about=about)

    def reacted(self, outcome, expected=0.0):
        """Kullanıcının tepkisi geldi: yaşantıyı ŞAŞKINLIKLA ağırlıklandır.

        Tahmin tuttuysa ucuz, tutmadıysa pahalı ve kalıcı — sobaya bir kez
        dokunmanın bin tekrardan güçlü olmasının sebebi.
        """
        held = getattr(self, "last", None)
        if held is None:
            return None
        held.outcome = float(outcome)
        held.surprise = self.gate.weigh(expected, outcome)
        return held

    # --- yardımcılar ----------------------------------------------------

    def _vector_of(self, text):
        """Cümlenin kaba vektörü: kelimelerinin ortalaması.

        Denetim yakaladı: `resolve` ve `recall` oturumdan hiç vektörsüz
        çağrılıyordu — sürekli katman çalışma anında ulaşılamaz koddu ve
        çokanlamlılık yine popülerliğe düşüyordu. Bağlam vektörü buradan
        gelir; sıra kaybeder, ilk eleme için yeter (yerini okuyucu ağının
        temsili alacak).
        """
        from v3.dataset import fold
        held = []
        for word in text.split():
            start, stop = 0, len(word)
            while start < stop and not word[start].isalnum():
                start += 1
            while stop > start and not word[stop - 1].isalnum():
                stop -= 1
            piece = fold(word[start:stop]) if stop > start else ""
            if piece and piece in self.vectors:
                held.append(self.vectors[piece])
        return geometry.mean(held)

    def _identity(self, label):
        """Etiketi kimliğe çevirir; yoksa açar. Bağlam hangisi olduğunu seçer."""
        if not label:
            return None
        key, score = geometry.resolve(self.memory, label,
                                      self.vectors.get(label))
        if key is not None:
            return key
        return self.memory.identify(label, vector=self.vectors.get(label))

    def _known(self, label):
        """Etiketi VAR OLAN bir kimliğe çevirir — yenisini AÇMAZ.

        Tartımda kullanılır: üretici yeni bir ad uydurduysa karşılığı
        bulunamaz ve iddia desteksiz sayılır.
        """
        if not label:
            return None
        key, _ = geometry.resolve(self.memory, label,
                                   self.vectors.get(label))
        return key

    def _render(self, records):
        """Ağ yokken ham kayıt dökümü — konuşamaz ama yalan söylemez."""
        lines = []
        for record in records[:MOST]:
            subject = self._label(record.subject)
            predicate = self._label(record.predicate)
            value = self._label(record.value)
            mark = " [~]" if self.gate.doubtful(record) else ""
            lines.append(f"{subject} → {predicate} → {value}"
                         f" ‹{record.source}›{mark}")
        return " · ".join(lines)

    def _label(self, key):
        held = self.memory.identities.get(key) if isinstance(key, int) else None
        if held is not None and held.labels:
            return held.labels[0]
        return str(key)

    # --- bakım ----------------------------------------------------------

    def sleep(self):
        """Uyku turu: damıt, sönümle, hakemle. Dönen sayım sözlüğü."""
        return dynamics.sleep(self.memory)

    def curious(self, most=10):
        """Belleğin boşlukları — nereye bakmalı."""
        return dynamics.gaps(self.memory, most)

    def tension(self):
        """Çelişki basıncı — neye rahatsız olmalı."""
        return dynamics.pressure(self.memory)

    def save(self):
        if self.path:
            self.memory.save(self.path)
