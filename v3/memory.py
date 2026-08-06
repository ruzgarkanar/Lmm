"""Yaşayan belleğin çekirdeği: kimlik, kayıt, bağ, yaşantı.

Bu dosyada dile ait TEK BİR ŞEY YOKTUR. Ne kelime listesi, ne ek, ne hece, ne
kalıp, ne şablon, ne ünlü kuralı. Buradaki her ad bir VERİ ALANI adıdır —
veritabanının sütun adı gibi. Dil sisteme yalnız iki eğitilmiş ağdan girer.

Eski belleğin kırıldığı yer düğümün bir DİZGİ olmasıydı: `kartal` hem kuş hem
ilçeydi ve ikisi aynı düğümde çarpışıyordu. Ölçüldü — 1.500 çok anlamlı
durumda doğru anlamı seçme oranı %21-24, rastgele %14,4. Yani anlam ayrımı
neredeyse hiç çalışmıyordu ve sebebi mekanizma değil, KİMLİĞİN YOKLUĞUYDU.

Burada kavram bir SAYI kimliktir. Kelime yalnız o kimliğe takılmış bir
etikettir ve bir kimliğin birden çok dilde etiketi olabilir — bilgi bir kez
öğrenilir, her dilde konuşulur.

    #1  etiketler: kartal, eagle      kayıtlar: tür → #2
    #7  etiketler: kartal             kayıtlar: tür → #9
                                      ikisi AYRI VARLIK, aynı yazılış

Kayıt da düğümdür: kaynağı, güveni, zamanı, tanık sayısı ve YAŞANTISI olan bir
şey. Kayıtlar birbirine bağlanabilir (neden, sonra, koşul) — düz üçlünün
yapamadığı ve uzun anlatının iskeletini veren şey budur.
"""
import json
import time

# Bağ türleri. Bunlar İLİŞKİ KİMLİĞİDİR, dil değil: hangi cümlenin hangi bağı
# taşıdığını okuyucu ağı öğrenir, burada yazılı bir liste eşleştirmez.
CAUSE, THEN, IF, CONTRA = 1, 2, 3, 4

# Kaynak basamakları. Sayı büyükse söz daha ağır basar. Bir yabancının tek
# cümlesi, doğrulanmış bir belgeyi ezmemeli — eski bellekte bu ölçülmüş ve
# ezebildiği görülmüştü.
OPERATOR, DOCUMENT, DISTILLED, INFERRED, STRANGER = 5, 4, 3, 2, 1


class Identity:
    """Bir kavram. Kelime değil — kelimeler yalnız etiketi."""

    __slots__ = ("key", "labels", "vector", "seen")

    def __init__(self, key, labels=(), vector=None):
        self.key = key
        self.labels = list(labels)      # hangi dilde hangi yazılış
        self.vector = vector            # sürekli katman: bulmak için
        self.seen = 0                   # kaç kez erişildi (solma için)

    def to_dict(self):
        return {"key": self.key, "labels": self.labels,
                "vector": self.vector, "seen": self.seen}

    @classmethod
    def from_dict(cls, held):
        found = cls(held["key"], held.get("labels", ()), held.get("vector"))
        found.seen = held.get("seen", 0)
        return found


class Record:
    """Bir olgu — ve olgunun kendisi bir düğüm.

    Düz üçlüde kayıt bir kenardı ve kenara bir şey bağlanamıyordu. Burada
    kaydın kimliği var, dolayısıyla başka kayıtlara bağlanabiliyor: "yağmur
    yağdı" kaydı, "toprak ıslandı" kaydına NEDEN bağıyla bağlanır.
    """

    __slots__ = ("key", "subject", "predicate", "value", "source", "trust",
                 "at", "witnesses", "last_seen", "links", "episodic")

    def __init__(self, key, subject, predicate, value, source,
                 trust=0.5, at=None, episodic=True):
        self.key = key
        self.subject = subject          # kimlik anahtarı
        self.predicate = predicate      # yüklem kimliği — dağarcık AÇIK
        self.value = value              # kimlik anahtarı ya da düz değer
        self.source = source
        self.trust = trust
        self.at = at if at is not None else time.time()
        self.witnesses = 1
        self.last_seen = self.at
        self.links = []                 # [(bağ türü, kayıt anahtarı)]
        # Episodik: tek bir olaydan geldi ("Ali dedi ki"). Damıtma turunda
        # tekrarlana tekrarlana semantiğe yükselir. Beynin hipokampus/korteks
        # ayrımının bellekteki karşılığı.
        self.episodic = episodic

    def to_dict(self):
        return {"key": self.key, "subject": self.subject,
                "predicate": self.predicate, "value": self.value,
                "source": self.source, "trust": self.trust, "at": self.at,
                "witnesses": self.witnesses, "last_seen": self.last_seen,
                "links": self.links, "episodic": self.episodic}

    @classmethod
    def from_dict(cls, held):
        found = cls(held["key"], held["subject"], held["predicate"],
                    held["value"], held["source"], held.get("trust", 0.5),
                    held.get("at"), held.get("episodic", True))
        found.witnesses = held.get("witnesses", 1)
        found.last_seen = held.get("last_seen", found.at)
        found.links = [tuple(one) for one in held.get("links", ())]
        return found


class Experience:
    """Yaşantı — duygunun mühendislik karşılığı.

    Çocuk sobaya dokunur, yanar, öğrenir. Ona "ateş tehlikelidir" listesi
    verilmez; deneyim kayıt olur ve sonraki davranışı o kayıt frenler.

    Burada da öyle: sistem bir şey söyler, bir tepki alır, tepki kaydedilir.
    Sonraki benzer durumda aday cevaplar bu kayıtlara çarpılır.

    `surprise` — öngörücü kodlamanın karşılığı. Beyin sürekli tahmin eder ve
    yalnız ŞAŞIRDIĞINDA güncellenir. Tahmin tutmuşsa kayıt ucuz, tutmamışsa
    pahalı ve kalıcı. Tek seferde öğrenme (sobaya bir kez dokunmak) buradan
    çıkıyor — bin tekrar değil, bir sürpriz.
    """

    __slots__ = ("key", "said", "outcome", "surprise", "at", "about")

    def __init__(self, key, said, outcome, surprise=0.0, at=None, about=()):
        self.key = key
        self.said = said                # ne söylendi (ham)
        self.outcome = outcome          # ne oldu — sayısal işaret
        self.surprise = surprise        # tahmin ile gerçek arası fark
        self.at = at if at is not None else time.time()
        self.about = list(about)        # ilgili kimlik anahtarları

    def to_dict(self):
        return {"key": self.key, "said": self.said, "outcome": self.outcome,
                "surprise": self.surprise, "at": self.at, "about": self.about}

    @classmethod
    def from_dict(cls, held):
        return cls(held["key"], held["said"], held["outcome"],
                   held.get("surprise", 0.0), held.get("at"),
                   held.get("about", ()))


class Memory:
    """Kimlikler, kayıtlar, yaşantılar — ve kendi geçmişi.

    Dışarıdan doğrudan yazılamaz: yazmanın tek yolu `write`, ve orada kaynak
    basamağı ile güven hesabı zorunludur. Firma senaryosunda grafın
    değiştirilemez olması bu kapıya dayanır.
    """

    FORMAT = 3

    def __init__(self):
        self.identities = {}            # anahtar -> Identity
        self.records = {}               # anahtar -> Record
        self.experiences = {}           # anahtar -> Experience
        self.by_label = {}              # etiket -> [kimlik anahtarı]
        self.by_subject = {}            # kimlik anahtarı -> [kayıt anahtarı]
        self.self_key = None            # #BEN — özyaşam öyküsünün düğümü
        self._next = 1

    # --- kimlik ---------------------------------------------------------

    def _key(self):
        found = self._next
        self._next += 1
        return found

    def identify(self, label, vector=None, same_as=None):
        """Etiketi bir kimliğe bağlar; gerekirse YENİ kimlik açar.

        Aynı yazılışın iki kavram olabilmesi buranın bütün meselesi: `same_as`
        verilmezse ve etiket zaten başka bir kimliğe bağlıysa, çağıran hangi
        kimliği kastettiğini söylemek zorundadır. Karar burada verilmez —
        burası yalnız kaydı tutar; hangi anlamın kastedildiğini bulmak
        geometrinin ve bağlamın işi.
        """
        if same_as is not None:
            found = self.identities[same_as]
            if label not in found.labels:
                found.labels.append(label)
                self.by_label.setdefault(label, []).append(same_as)
            return same_as
        key = self._key()
        self.identities[key] = Identity(key, [label], vector)
        self.by_label.setdefault(label, []).append(key)
        return key

    def candidates(self, label):
        """Bu etiketi taşıyan bütün kimlikler — hangisi olduğu ayrı iş."""
        return list(self.by_label.get(label, ()))

    # --- kayıt ----------------------------------------------------------

    def write(self, subject, predicate, value, source, trust=None,
              episodic=True):
        """Bir olguyu belleğe koyar; aynısı varsa PEKİŞTİRİR.

        Pekişme, ikinci bağımsız kaynağın kalan kuşkunun bir payını
        kapatmasıdır — toplamsal değil, çünkü toplamsalken birkaç belge
        tavanı deliyordu (eski bellekte ölçüldü).
        """
        trust = self.trust_of(source) if trust is None else trust
        for key in self.by_subject.get(subject, ()):
            held = self.records[key]
            if held.predicate == predicate and held.value == value:
                if source not in str(held.source):
                    held.witnesses += 1
                    held.trust += (1.0 - held.trust) * 0.15
                    held.trust = min(held.trust, 0.98)
                held.last_seen = time.time()
                return held
        key = self._key()
        found = Record(key, subject, predicate, value, source, trust,
                       episodic=episodic)
        self.records[key] = found
        self.by_subject.setdefault(subject, []).append(key)
        return found

    def link(self, one, other, kind):
        """İki kaydı bağlar — neden, sonra, koşul, çelişki."""
        held = self.records[one]
        if (kind, other) not in held.links:
            held.links.append((kind, other))
        return held

    def about(self, subject):
        """Bir kimlik hakkındaki kayıtlar; erişim SOLMAYI geciktirir."""
        found = self.identities.get(subject)
        if found is not None:
            found.seen += 1
        return [self.records[key] for key in self.by_subject.get(subject, ())]

    @staticmethod
    def trust_of(source):
        """Kaynağın basamağından başlangıç güveni. Şeritler örtüşmez."""
        return {OPERATOR: 0.75, DOCUMENT: 0.6, DISTILLED: 0.5,
                INFERRED: 0.45, STRANGER: 0.3}.get(source, 0.3)

    # --- yaşantı --------------------------------------------------------

    def lived(self, said, outcome, surprise=0.0, about=()):
        """Bir yaşantıyı kaydeder. Şaşırtan yaşantı ağır basar."""
        key = self._key()
        found = Experience(key, said, outcome, surprise, about=about)
        self.experiences[key] = found
        return found

    def recall(self, about, most=8):
        """Bu kimliklerle ilgili yaşantılar, şaşkınlığı yüksek olan önce."""
        wanted = set(about)
        held = [one for one in self.experiences.values()
                if wanted & set(one.about)]
        held.sort(key=lambda one: -one.surprise)
        return held[:most]

    # --- dosya ----------------------------------------------------------

    def save(self, path):
        held = {"format": self.FORMAT, "next": self._next,
                "self": self.self_key,
                "identities": [one.to_dict()
                               for one in self.identities.values()],
                "records": [one.to_dict() for one in self.records.values()],
                "experiences": [one.to_dict()
                                for one in self.experiences.values()]}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(held, handle, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        found = cls()
        try:
            held = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            return found
        found._next = held.get("next", 1)
        found.self_key = held.get("self")
        for one in held.get("identities", ()):
            made = Identity.from_dict(one)
            found.identities[made.key] = made
            for label in made.labels:
                found.by_label.setdefault(label, []).append(made.key)
        for one in held.get("records", ()):
            made = Record.from_dict(one)
            found.records[made.key] = made
            found.by_subject.setdefault(made.subject, []).append(made.key)
        for one in held.get("experiences", ()):
            made = Experience.from_dict(one)
            found.experiences[made.key] = made
        return found
